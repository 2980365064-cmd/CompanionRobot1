"""
记忆召回路由 —— 按用户身份决定检索策略，协调多层记忆的联合召回。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Router 是"记忆调度中心"——每轮对话开始时，根据用户身份（访客/已实名）
  决定从哪些记忆层检索、检索顺序、以及是否触发关联扩展。

召回策略：

  访客模式（tmp_* / 未实名）：
    → 仅返回 L1 工作记忆（当前会话窗口）
    → 不调用 embedding（节省 API 费用）
    → L0/L2/L3 全部为空

  已实名模式（verified person_id）：
    1. L0 全量加载（无向量门控，无条件注入）
    2. L2 向量检索 episodic；无命中 → fallback 注入最近 N 条摘要
    3. L3 混合检索（仅当 query_needs_memory_answer 为 True 时触发）
    4. 联想扩展：基于 L3 命中结果做 memory_relations 图扩展
    5. 情感轨迹：附加最近情感快照

输出结构（memory dict）：
  {
    working:     L1 消息列表
    episodic:    L2 摘要列表
    semantic:    L3 文本列表
    l3:          L3 文本列表（同上，新字段名）
    l0:          L0 核心事实列表
    l2_hit:      L2 检索是否命中
    l3_hit:      L3 检索是否命中
    memory_miss: 是否完全未命中（用于反幻觉提示）
    person_id:   活跃用户 ID
    guest_mode:  是否访客模式
    matches:     {l2: [...], l3: [...], related: [...]} 原始召回数据
  }
============================================================================
"""

from __future__ import annotations

import hashlib
import time

from app.config import settings
from app.llm import embed_texts
from app.memory.l2 import episodic_memory
from app.memory.guard import extract_self_name, is_noise_memory_for_l3, query_needs_memory_answer
from app.memory.l0 import list_l0_cached
from app.memory.l1 import working_memory
from app.memory.relations import expand_associative_recall, seed_keys_from_l3_matches
from app.memory.emotion import emotion_trajectory
from app.memory.identity import memory_scoped_to_person
from app.memory.l3 import semantic_memory
from app.session import store

# ── Embedding 查询向量缓存 ─────────────────────────────────────────────────
# LRU + TTL 缓存：key = (person_id, query_hash)，最多缓存 128 条
# TTL = 300 秒（5分钟），与 Anthropic prompt cache TTL 对齐
_EMBED_CACHE_SIZE = 128
_EMBED_CACHE_TTL = 300.0
_embed_cache: dict[str, tuple[float, list[float]]] = {}


def _cached_embed(text: str, cache_key: str = "") -> list[float]:
    """带缓存的单文本向量化。"""
    if not text:
        return []
    key = cache_key or hashlib.md5(text.encode()).hexdigest()[:12]
    now = time.monotonic()
    entry = _embed_cache.get(key)
    if entry and now - entry[0] < _EMBED_CACHE_TTL:
        return entry[1]
    embs = embed_texts([text])
    vec = embs[0] if embs else []
    if vec:
        if len(_embed_cache) >= _EMBED_CACHE_SIZE:
            oldest = min(_embed_cache, key=lambda k: _embed_cache[k][0])
            del _embed_cache[oldest]
        _embed_cache[key] = (now, vec)
    return vec


def _batch_embed(texts: list[str]) -> list[list[float]]:
    """批量向量化（跳过缓存命中项，减少 API 调用）。

    对每个 text 检查缓存，只对未命中的做批量 API 调用，
    然后合并缓存结果。
    """
    if not texts:
        return []
    now = time.monotonic()
    results: list[list[float]] = []
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    for i, t in enumerate(texts):
        if not t:
            results.append([])
            continue
        key = hashlib.md5(t.encode()).hexdigest()[:12]
        entry = _embed_cache.get(key)
        if entry and now - entry[0] < _EMBED_CACHE_TTL:
            results.append(entry[1])
        else:
            results.append([])  # placeholder
            miss_indices.append(i)
            miss_texts.append(t)
    if miss_texts:
        embs = embed_texts(miss_texts)
        for j, emb in enumerate(embs):
            idx = miss_indices[j]
            results[idx] = emb
            key = hashlib.md5(miss_texts[j].encode()).hexdigest()[:12]
            if len(_embed_cache) >= _EMBED_CACHE_SIZE:
                oldest = min(_embed_cache, key=lambda k: _embed_cache[k][0])
                del _embed_cache[oldest]
            _embed_cache[key] = (now, emb)
    return results


def _enrich_l3_query(query: str, working: list[dict]) -> str:
    """用 L1 上下文扩充 L3 检索查询——解决短追问丢失前文主题的问题。

    场景：用户先说「之前在粥顶山玩的时候...」，
    然后追问「你真的想起来了吗」。
    单独的「你真的想起来了吗」不包含任何可检索的实体词,
    需要从上下文提取「粥顶山」等关键信息拼入 query。

    Args:
        query:   当前用户消息
        working: L1 最近消息列表 [{"role": "user"/"assistant", "content": ...}, ...]

    Returns:
        扩充后的 query，无需扩充时返回原 query。
    """
    q = query.strip()
    # 消息足够长 → 自带上下文，不扩充
    if len(q) >= 18:
        return q
    # 消息虽短但包含具体实体词（地名/事件/人名）→ 不扩充
    if _has_content_entities(q):
        return q
    # 收集最近几轮用户消息（排除当前 query 本身）
    recent_user = [
        m["content"].strip()
        for m in working[-8:]
        if m.get("role") == "user" and m["content"].strip() != q
    ]
    if not recent_user:
        return q
    # 找前文中信息量最大的消息作为上下文：
    # 跳过短追问（< 12 字且无实体词），优先取包含具体内容的消息
    best = ""
    for msg in reversed(recent_user):
        if len(msg) > len(best) and not (len(msg) < 12 and not _has_content_entities(msg)):
            best = msg
        if len(best) >= 30:
            break  # 已经够长了
    if not best or len(best) <= 8:
        return q
    snippet = best[:80].strip()
    if snippet in q:
        return q
    return f"{q}（上下文：{snippet}）"


_CN_MONTHS: dict[str, int] = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


def _cn_month_to_int(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        n = int(token)
        return n if 1 <= n <= 12 else None
    if token in _CN_MONTHS:
        return _CN_MONTHS[token]
    if len(token) == 2 and token[0] == "十" and token[1] in _CN_MONTHS:
        return 10 + _CN_MONTHS[token[1]]
    return None


def _parse_month_key(query: str) -> str | None:
    """Extract YYYY-MM when user asks about a specific month."""
    import re

    q = query.strip()
    m = re.search(r"(\d{4})\s*年\s*(?:(\d{1,2})|([一二三四五六七八九十]{1,2}))月", q)
    if m:
        year = int(m.group(1))
        month = _cn_month_to_int(m.group(2) or m.group(3) or "")
        if month:
            return f"{year}-{month:02d}"
    m2 = re.search(r"(\d{4})-(\d{2})", q)
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}"
    return None


def _expand_l3_query_dates(query: str) -> str:
    """Append ISO month tags so FTS can hit ## 2024-04 style headings."""
    mk = _parse_month_key(query)
    if not mk:
        return query
    year, month = mk.split("-")
    extras = [mk, f"{year}年{int(month)}月"]
    mo = int(month)
    inv = {v: k for k, v in _CN_MONTHS.items() if v <= 12}
    if mo in inv:
        extras.append(f"{inv[mo]}月")
    return query + " " + " ".join(extras)


def _build_l3_query(query: str, working: list[dict], self_name: str | None) -> str:
    base = _enrich_l3_query(query, working)
    if self_name and self_name not in base:
        base = f"{base} {self_name}"
    return _expand_l3_query_dates(base)


def _chunk_section_month(text: str) -> str | None:
    """Primary YYYY-MM section tag from an L3 chunk heading."""
    import re

    m = re.search(r"## (\d{4}-\d{2})\b", text or "")
    return m.group(1) if m else None


def _is_month_primary_chunk(text: str, mk: str) -> bool:
    return _chunk_section_month(text) == mk


def _boost_month_l3_matches(
    query: str,
    matches: list[dict],
    *,
    person_id: str,
    persona_person_id: str,
) -> list[dict]:
    """When user names a month, prefer chunks whose ## heading matches YYYY-MM."""
    mk = _parse_month_key(query)
    if not mk:
        return matches

    def month_bucket(m: dict) -> int:
        sm = _chunk_section_month(str(m.get("text", "")))
        if sm == mk:
            return 0
        if sm:
            return 2
        return 1

    on_month = [m for m in matches if _is_month_primary_chunk(str(m.get("text", "")), mk)]
    if len(on_month) >= 2:
        neutral = [m for m in matches if month_bucket(m) == 1]
        return (on_month + neutral)[: max(5, len(matches))]

    extras = [persona_person_id] if persona_person_id else []
    fts_rows = store.l3_fts_search_pool(
        mk,
        person_id,
        extra_person_ids=extras,
        limit=12,
    )
    injected: list[dict] = []
    seen = {str(m.get("text", "")).strip() for m in matches if m.get("text")}
    for row in fts_rows:
        text = str(row.get("text", "")).strip()
        if not text or text in seen or not _is_month_primary_chunk(text, mk):
            continue
        seen.add(text)
        injected.append(
            {
                "text": text,
                "score": 0.99,
                "chunk_id": row.get("chunk_id", ""),
                "category": str(row.get("category") or ""),
                "source": str(row.get("source") or ""),
                "collection": str(row.get("collection") or "memory"),
            }
        )

    neutral = [m for m in matches if month_bucket(m) == 1]
    on_month_vec = [m for m in on_month if m not in injected]
    merged = injected + on_month_vec + neutral
    return merged[: max(5, len(matches))]


def _has_content_entities(text: str) -> bool:
    """检测短消息是否已包含具体内容实体（地名、事件、专有名词等）。
    有实体 → 不需要从上下文扩充；无实体 → 可能是追问，需要扩充。
    """
    import re
    # 地名/场所词
    if re.search(r"[一-鿿]{2,}(?:山|路|街|公园|广场|学校|医院|公司|餐厅|店|城|海|湖|河|岛)", text):
        return True
    # 具体动作+地点
    if re.search(r"(?:去|在|到|从)(?:过|了)?[一-鿿]{2,6}", text):
        return True
    # 具体事物/事件描述
    if re.search(r"(?:吃|喝|买|看|玩|唱|爬|逛|旅游|旅行|考试|面试|搬家|结婚)", text):
        return True
    # 具体时间
    if re.search(r"\d+[年月日天号]", text):
        return True
    return False


def _empty_recall(working: list[dict], *, reason: str) -> dict:
    """构建空的召回结果字典（访客模式或无数据时使用）。

    返回结构中所有 L2/L3/L0 字段均为空列表，memory_miss=False（因为
    访客模式不算"未命中"，只是不启用检索）。agent 模块检查 guest_mode
    决定是否注入记忆到 prompt。

    Args:
        working: L1 当前会话消息列表
        reason:  空召回的原因描述（如 "guest_l1_only"）
    """
    return {
        "working": working,
        "episodic": [],
        "semantic": [],
        "l3": [],
        "l0": [],
        "l2_hit": False,
        "l3_hit": False,
        "facts_hit": False,
        "corpus_triggered": False,
        "corpus_reason": [reason] if reason else [],
        "memory_miss": False,
        "person_id": None,
        "guest_mode": True,
        "matches": {"l2": [], "l3": [], "related": []},
    }


class MemoryRouter:
    """记忆召回路由器：协调 L0/L1/L2/L3 多层记忆的联合召回。

    核心设计：
      - 每条记忆层的检索都是独立的，router 负责编排它们的调用顺序
      - 访客模式下跳过所有向量检索，仅返回 L1
      - L3 检索按需触发（query_needs_memory_answer 门控），避免无效检索
    """

    def _recall_guest(self, working: list[dict]) -> dict:
        """访客模式召回：仅返回 L1 当前会话消息。

        访客模式的核心节约：不调用 embedding API，不查询任何持久化存储，
        仅从当前会话的 messages 表读取最近消息。这是系统的"浅度模式"，
        适用于未实名用户和匿名访客。
        """
        payload = _empty_recall(working, reason="guest_l1_only")
        payload["person_id"] = None
        return payload

    def recall(
        self,
        device_id: str,
        session_id: str,
        query: str,
        *,
        person_id: str | None = None,
    ) -> dict:
        """执行完整的多层记忆召回。

        召回流程：
          1. 获取 L1 工作记忆（当前会话消息）
          2. 判断是否为已实名用户：否 → 访客模式，仅返回 L1
          3. 从用户消息提取自称名，用于 L3 query 改写
          4. L0 全量加载
          5. L2 向量检索（有 query）或 fallback 最近摘要（无 query）
          6. L3 混合检索（仅 query_needs_memory_answer 为 True 时触发）
          7. 记忆关联图扩展
          8. 情感轨迹附加
          9. 汇总所有结果

        Args:
            device_id:  设备标识
            session_id: 会话标识
            query:      检索查询文本（通常是用户当前消息或改写后的）
            person_id:  用户 ID（None 或不提供则从 session 读取）

        Returns:
            完整的多层记忆召回字典。
        """
        # Step 1: L1 工作记忆
        working = working_memory.get_recent(session_id)
        pid = str(person_id or "").strip()

        # Step 2: 访客判断 → 跳过所有嵌入检索
        if not memory_scoped_to_person(pid):
            return self._recall_guest(working)

        # Step 3: query 预处理
        q = query.strip()
        # 从用户消息中提取自称名，用于 L3 检索 query 改写
        # 例如用户说"我是刘远慧"→ 用"刘远慧"做 L3 检索，提高命中率
        self_name = extract_self_name(q)
        l3_query = _build_l3_query(q, working, self_name)
        # q_emb 用于 L2 检索，l3_emb 用于 L3 检索（可能不同，因为 query 改写）
        # 批量向量化：q 和 l3_query 合并为一次 API 调用（含缓存）
        if q and l3_query and l3_query != q:
            embs = _batch_embed([q, l3_query])
            q_emb = embs[0] if embs[0] else None
            l3_emb = embs[1] if embs[1] else q_emb
        elif q:
            q_emb = _cached_embed(q) or None
            l3_emb = q_emb
        else:
            q_emb = None
            l3_emb = None
        # needs_memory：门控函数，判断用户是否在问需要记忆的东西
        # "你好"/"在吗"等寒暄不需要查 L3，节省 embedding 调用
        needs_memory = query_needs_memory_answer(query)

        # Step 4: L0 全量加载（带缓存，无条件，无门控）
        l0_rows = list_l0_cached(pid)

        # Step 5: L2 情景记忆检索
        l2_matches = episodic_memory.recall_scored(
            device_id, pid, query, settings.episodic_top_k, q_emb=q_emb
        )
        l2_hit = any(m.get("score") is not None for m in l2_matches)
        # 用户在问需要记忆的事实：禁止注入「最近几条 L2」兜底，避免无关摘要诱发编造
        if not l2_matches and not needs_memory:
            l2_matches = episodic_memory.recall_scored(
                device_id, pid, "", settings.l2_recall_recent, q_emb=None
            )
        elif needs_memory and not l2_hit:
            l2_matches = []
        episodic = [m["text"] for m in l2_matches if m.get("text")]

        # Step 6: L3 语义记忆检索（按需触发）
        # 只有 query_needs_memory_answer 返回 True 才检索 L3。
        # 门控原因：寒暄短句（"你好"/"在吗"）不需要查长期记忆，
        # 跳过 L3 检索可节省一次 embedding API 调用和一次数据库查询。
        l3_matches: list[dict] = []
        if needs_memory:
            # 附加人物画像事实的 person_id（跨用户共享的人物知识）
            # 例如 persona/corpus/ 中的通用人物百科，不属于特定用户，
            # 但对所有实名用户都有参考价值。
            persona_pid = str(getattr(settings, "persona_fact_person_id", "") or "").strip()
            l3_top_k = 5 if _parse_month_key(q) else 3
            l3_matches = semantic_memory.recall_l3_scored(
                device_id,
                pid,
                l3_query,
                l3_top_k,
                q_emb=l3_emb,
                persona_person_id=persona_pid,
            )
            l3_matches = [
                m for m in l3_matches if not is_noise_memory_for_l3(m.get("text", ""))
            ]
            l3_matches = _boost_month_l3_matches(
                q,
                l3_matches,
                person_id=pid,
                persona_person_id=persona_pid,
            )
            l3_thresh = settings.l3_sim_threshold
            l3_matches = [
                m for m in l3_matches if (m.get("score") or 0) >= l3_thresh
            ]
            mk = _parse_month_key(q)
            if mk and needs_memory:
                on_month = [
                    m
                    for m in l3_matches
                    if _is_month_primary_chunk(str(m.get("text", "")), mk)
                ]
                if not on_month:
                    l3_matches = []

        l3_hit = bool(l3_matches)
        l3_texts = [str(m["text"]) for m in l3_matches if m.get("text")]

        # Step 7: 记忆关联图扩展
        # 从 L3 命中结果提取种子节点（chunk_id），在 memory_relations 表中
        # 查找关联的记忆块（strength >= memory_relation_min_strength），
        # 扩展召回范围（链式记忆：想起一件事，带出相关的事）
        seed_keys = seed_keys_from_l3_matches(l3_matches)
        related_matches = expand_associative_recall(seed_keys, person_id=pid)
        # 去重：排除已在 L3 主结果中出现的文本
        matched_texts = {str(m.get("text", "")).strip() for m in l3_matches if m.get("text")}
        related_matches = [
            r
            for r in related_matches
            if str(r.get("text", "")).strip() not in matched_texts
        ]

        # memory_miss: 所有检索层都无命中 → 需要反幻觉提示介入
        # 告诉 Agent："我不知道，因为记忆里没有相关信息"。
        # 仅当 needs_memory=True（用户在问需要记忆的东西）且所有层都无命中时
        # 才设为 True。寒暄短句的 memory_miss 为 False，不影响。
        memory_miss = needs_memory and not l2_hit and not l3_hit and not related_matches

        # Step 8: 情感轨迹
        # 附加最近的情感快照（最近 3-5 次对话的情感状态趋势），
        # 帮助 agent 感知用户近期情绪变化，调整回复语气。
        emo_traj = emotion_trajectory(device_id, pid)

        # Step 9: 汇总结果
        return {
            "working": working,
            "episodic": episodic,
            "semantic": l3_texts,
            "l3": l3_texts,
            "l0": l0_rows,
            "l2_hit": l2_hit,
            "l3_hit": l3_hit,
            "facts_hit": l3_hit,          # 兼容旧字段名
            "corpus_triggered": l3_hit,   # 兼容旧字段名
            "corpus_reason": ["l3_unified"] if needs_memory and l3_hit else [],
            "memory_miss": memory_miss,   # 全部未命中 → Agent 应表示不知道
            "person_id": pid,
            "guest_mode": False,
            "emotion_trajectory": emo_traj,
            "matches": {
                "l2": l2_matches,
                "l3": l3_matches,
                "related": related_matches,
            },
        }


# 模块级单例，供 agent 模块调用
memory_router = MemoryRouter()
