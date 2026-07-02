"""
记忆关联图 —— SQLite memory_relations 表，轻量级替代 Neo4j 图数据库。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Relations 是"记忆联想引擎"——在记忆块之间建立关系边，实现链式回忆。
  当 L3 检索命中某条记忆时，通过关联图扩展出相关的记忆，模拟人类的
  "想起一件事，连带想起另一件"的联想过程。

节点 ID 格式：
  fact:{id}     —— 原子事实节点
  chunk:{id}    —— 语料块节点

边类型：
  related       —— 同源共现（来自同一段对话/同一篇语料）
  cause / effect —— 因果关系（LLM 从语料推断出的因果链）

边强度（strength）：
  0.0 ~ 1.0，expand_associative_recall 只返回 strength >= min_strength 的边。
  默认 related 强度 0.7，corpus→facts 强度 0.55，因果强度由 LLM 判定。

关联图的生命周期与 L3 语料同步：
  - L3 rollup（L2 过期 → L3）→ 不建图（无 fact 提取）
  - 用户记住 → 不建图
  - 语料导入时如有 fact 提取 → link_related_facts + link_corpus_to_facts + detect_causal_relations
============================================================================
"""

from __future__ import annotations

import json
import logging
import re
from itertools import combinations

from app.config import settings
from app.llm import chat_completion
from app.session import store

logger = logging.getLogger(__name__)

# 节点 ID 前缀常量
FACT_PREFIX = "fact:"     # 事实节点前缀
CHUNK_PREFIX = "chunk:"   # 语料块节点前缀

# 默认关联强度
_RELATED_STRENGTH = 0.7   # 同源共现的默认强度

# LLM 因果推断提示词：从语料中提取事实之间的因果关系
_CAUSAL_PROMPT = """以下是从同一段语料提取的原子事实。识别语料中**明确可推断**的因果关系（如有）。

事实列表（下标从 0 开始）：
{facts_json}

语料片段（供参照，勿超出其内容推断）：
{corpus}

只输出 JSON：
{{"relations": [
  {{"cause_index": 0, "effect_index": 1, "strength": 0.0-1.0}}
]}}

规则：
- cause_index 为「原因」事实下标，effect_index 为「结果」事实下标
- 例：「被猫抓了」→「讨厌猫」可建立因果；单纯并列经历不要建因果
- 无明确因果 → relations 为 []
- 禁止编造语料中没有的因果"""


def fact_key(fact_id: int) -> str:
    """将事实 ID 转换为图节点键：fact:{id}。"""
    return f"{FACT_PREFIX}{int(fact_id)}"


def chunk_key(chunk_id: str) -> str:
    """将语料块 ID 转换为图节点键：chunk:{id}。"""
    return f"{CHUNK_PREFIX}{str(chunk_id).strip()}"


def parse_memory_key(key: str) -> tuple[str, str]:
    """解析图节点键，返回 (类型, ID)。

    例:
      "fact:123" → ("fact", "123")
      "chunk:abc" → ("chunk", "abc")
    """
    k = str(key or "").strip()
    if k.startswith(FACT_PREFIX):
        return "fact", k[len(FACT_PREFIX):]
    if k.startswith(CHUNK_PREFIX):
        return "chunk", k[len(CHUNK_PREFIX):]
    return "unknown", k


def link_related_facts(fact_ids: list[int], *, strength: float | None = None) -> int:
    """将一组事实两两之间建立"同源共现"关系边（无向图，双向存储）。

    用于从同一段语料提取的多条事实之间建立关联。
    combinations 确保每对只建一次双向边。

    Args:
        fact_ids: 来自同一段语料的事实 ID 列表
        strength: 边强度（默认 0.7，代表同源共现的可靠程度）

    Returns:
        创建的边总数（每条事实对 × 2，因为双向存储）。
    """
    ids = [int(i) for i in fact_ids if i]
    if len(ids) < 2:
        return 0  # 单条事实无法形成任何边
    s = float(strength if strength is not None else _RELATED_STRENGTH)
    n = 0
    # combinations(sorted(set(ids)), 2)：
    # 对所有事实两两配对，每组只处理一次（无向图语义）。
    # set 去重防止同一 ID 重复出现导致自环；sorted 保证输出稳定。
    for a, b in combinations(sorted(set(ids)), 2):
        # 双向存储：A→B 和 B→A 各建一条边，使得从任意节点出发都能找到邻居
        store.upsert_memory_relation(fact_key(a), fact_key(b), "related", s)
        store.upsert_memory_relation(fact_key(b), fact_key(a), "related", s)
        n += 2
    return n


def link_corpus_to_facts(corpus_chunk_id: str, fact_ids: list[int], *, strength: float = 0.55) -> int:
    """将语料块与从其中提取的事实建立关系边。

    边方向是双向的：语料→事实 和 事实→语料。
    强度较低（0.55），因为语料块比事实更泛化。

    Args:
        corpus_chunk_id: 语料块的 chunk_id
        fact_ids:        从该语料提取的事实 ID 列表
        strength:        边强度（默认 0.55）

    Returns:
        创建的边总数。
    """
    cid = str(corpus_chunk_id or "").strip()
    if not cid:
        return 0
    ck = chunk_key(cid)
    n = 0
    for fid in fact_ids:
        if not fid:
            continue
        fk = fact_key(int(fid))
        store.upsert_memory_relation(ck, fk, "related", strength)
        store.upsert_memory_relation(fk, ck, "related", strength)
        n += 2
    return n


def detect_causal_relations(fact_rows: list[tuple[int, str]], corpus_text: str) -> int:
    """用 LLM 从语料中检测事实之间的因果关系，建立 cause/effect 边。

    仅当 facts >= 2 条时才执行（单一事实无法形成因果）。
    LLM 只推断语料中明确可推断的因果，禁止编造。

    典型因果示例：
      - 原因："被猫抓了" → 结果："讨厌猫"
      - 原因："经常加班" → 结果："感到疲惫"

    非因果示例（不应建边）：
      - "今天吃了火锅" + "今天下雨" → 仅并列，无因果

    Args:
        fact_rows:   [(fact_id, fact_text), ...] 列表
        corpus_text: 原始语料文本（供 LLM 参照，不超过 4000 字符）

    Returns:
        创建的因果边总数。
    """
    if len(fact_rows) < 2:
        return 0

    # 构建 LLM 请求：事实列表 + 语料片段
    facts_json = json.dumps(
        [{"index": i, "fact": t} for i, (_, t) in enumerate(fact_rows)],
        ensure_ascii=False,
    )
    prompt = _CAUSAL_PROMPT.format(
        facts_json=facts_json,
        corpus=str(corpus_text or "")[:4000],
    )
    raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.1)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return 0
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return 0

    n = 0
    for item in data.get("relations") or []:
        if not isinstance(item, dict):
            continue
        try:
            ci = int(item.get("cause_index"))
            ei = int(item.get("effect_index"))
        except (TypeError, ValueError):
            continue
        # 校验下标合法性：自环（ci==ei）、越界、负数均拒绝
        if ci == ei or ci < 0 or ei < 0 or ci >= len(fact_rows) or ei >= len(fact_rows):
            continue
        strength = float(item.get("strength", 0.75))
        # 强度 < 0.5 视为 LLM 认为因果不成立，不建边
        if strength < 0.5:
            continue
        cause_id, _ = fact_rows[ci]
        effect_id, _ = fact_rows[ei]
        # 双向存储但类型不同：
        # cause→effect 边类型为 "cause"（从因到果），
        # effect→cause 边类型为 "effect"（从果到因），
        # 这样在 expand_associative_recall 时可以根据边类型区分方向
        store.upsert_memory_relation(fact_key(cause_id), fact_key(effect_id), "cause", strength)
        store.upsert_memory_relation(fact_key(effect_id), fact_key(cause_id), "effect", strength)
        n += 2
    if n:
        logger.info("causal relations linked count=%d", n)
    return n


def build_relations_for_extracted_facts(
    fact_rows: list[tuple[int, str]], corpus_chunk_id: str, corpus_text: str,
) -> None:
    """一站式建图：对一组提取的事实执行三种关联建边。

    1. link_related_facts：事实之间建"related"边（同源共现）
    2. link_corpus_to_facts：语料块与事实之间建边
    3. detect_causal_relations：LLM 推断因果关系边

    这是图构建的主要入口，通常由语料导入流程调用。
    """
    if not fact_rows:
        return
    ids = [fid for fid, _ in fact_rows]
    link_related_facts(ids)
    link_corpus_to_facts(corpus_chunk_id, ids)
    if len(fact_rows) >= 2:
        detect_causal_relations(fact_rows, corpus_text)


def resolve_memory_text(memory_key: str, person_id: str = "") -> str:
    """根据节点键解析对应的记忆文本内容。

    从事实表或 chunks 表中查找并返回文本。
    用于 expand_associative_recall 中将图邻居转换为可读文本。

    这是图查询的最后一步：图算法只产生 node key，需要本函数
    把 key 翻译成实际文本才能注入 prompt。

    Args:
        memory_key: 图节点键（如 "fact:123" 或 "chunk:abc"）
        person_id:  用户 ID（预留给按用户过滤事实的场景，当前未使用）

    Returns:
        对应节点的文本内容；查不到时返回空字符串。
    """
    kind, ref = parse_memory_key(memory_key)
    if kind == "fact":
        row = store.get_fact_by_id(int(ref))
        return str(row["fact"]).strip() if row else ""
    if kind == "chunk":
        row = store.l3_get_chunk(ref)
        return str(row["text"]).strip() if row else ""
    return ""


def expand_associative_recall(
    seed_keys: list[str], *, person_id: str = "",
    min_strength: float | None = None, limit: int = 8,
) -> list[dict]:
    """基于种子节点做记忆关联图扩展，返回相关联的记忆文本。

    这是 router 调用的联想扩展入口。流程：
      1. 从 seeds 出发，在 memory_relations 表中查找邻居节点
      2. 只保留 strength >= min_strength 的边
      3. 去重后解析每个邻居的文本内容
      4. 返回 top-N 条关联记忆

    联想链示例：
      种子: L3 命中 "刘远慧在杭州实习"
      → 关联图邻居: "刘远慧喜欢喝奶茶"（related, 0.7）
      → 输出: 奶茶偏好的记忆

    Args:
        seed_keys:   种子节点键列表（通常来自 L3 匹配结果的 chunk_id）
        person_id:   用户 ID（预留）
        min_strength: 最小边强度阈值（默认取配置 memory_relation_min_strength，通常 0.6）
        limit:       最多返回条数

    Returns:
        关联记忆列表，每条含 memory_id、text、relation_type、strength、via。
    """
    threshold = (
        float(min_strength)
        if min_strength is not None
        else float(getattr(settings, "memory_relation_min_strength", 0.6))
    )
    # dict.fromkeys 去重同时保持顺序——种子可能有重复的 chunk_key
    seeds = list(dict.fromkeys(k for k in seed_keys if k))
    if not seeds:
        return []

    seen_seeds = set(seeds)
    # 取 limit * 3 条候选：图查询可能返回许多自环或非文本节点，
    # 需要额外余量保证去重和解析后仍有足够条数
    rows = store.get_memory_relations(seeds, min_strength=threshold, limit=limit * 3)
    out: list[dict] = []
    seen_text: set[str] = set()  # 文本去重——不同节点可能指向相同文本

    for row in rows:
        src = str(row.get("from_id") or "")
        dst = str(row.get("to_id") or "")
        # 确定邻居方向：由于边是双向存储的，需要判断哪一端是种子、哪一端是邻居。
        # 如果 src 在种子集中，邻居是 dst；反之邻居是 src。
        neighbor = dst if src in seen_seeds else src
        # 邻居如果是种子本身（可能由双向边导致），跳过
        if neighbor in seen_seeds:
            continue
        text = resolve_memory_text(neighbor, person_id)
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        out.append({
            "memory_id": neighbor,
            "text": text,
            "relation_type": row.get("relation_type", "related"),
            "strength": round(float(row.get("strength", 0.5)), 3),
            "via": src if src in seen_seeds else dst,  # 标注「通过哪个种子」关联到的邻居
        })
        if len(out) >= limit:
            break
    return out


def seed_keys_from_l3_matches(l3_matches: list[dict]) -> list[str]:
    """从 L3 召回结果中提取可用的种子节点键。

    优先使用匹配结果中的 chunk_id，若无则通过文本反查 chunks 表获取 ID。
    最终返回去重后的 chunk 键列表。

    Args:
        l3_matches: L3 召回结果列表（每个元素含 chunk_id 和 text）

    Returns:
        chunk 键列表（如 ["chunk:abc123", "chunk:def456"]）。
    """
    keys: list[str] = []
    for m in l3_matches:
        cid = m.get("chunk_id")
        if cid:
            keys.append(chunk_key(str(cid)))
            continue
        # 兜底：如果匹配结果没有 chunk_id（例如某些旧格式的 L3 数据），
        # 通过文本内容反查 chunks 表获取对应的 chunk_id
        text = str(m.get("text", "")).strip()
        if text:
            for ref in store.l3_find_chunks_by_text(text, limit=1):
                keys.append(chunk_key(ref["chunk_id"]))
                break
    return list(dict.fromkeys(keys))  # 去重保持顺序


def seed_keys_from_matches(
    device_id: str, person_id: str,
    fact_matches: list[dict], corpus_matches: list[dict],
) -> list[str]:
    """兼容旧接口：从匹配结果中提取种子节点键（内部调用 seed_keys_from_l3_matches）。"""
    del device_id, person_id, fact_matches  # 旧参数，当前版本不使用
    return seed_keys_from_l3_matches(corpus_matches or [])
