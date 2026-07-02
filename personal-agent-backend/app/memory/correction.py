"""
记忆纠错 —— 用户纠正长期记忆：删除/修补 L3 chunks + L0，写入修正后的事实。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Correction 是"记忆修正器"——当用户指出 Agent 的记忆有误时，
  自动定位并删除/修补错误的 L3 语料块，同时写入正确的替代信息。

触发条件：
  - user_signals_memory_correction() 检测到用户纠正信号
  - 需要 settings.memory_auto_correct = true
  - 仅对已实名用户生效（访客不存储长期记忆，无需修正）

修正操作类型：
  delete_fact           —— 按 match_text 删除匹配的 L3 块和 L0 条目
  delete_l3_chunk       —— 按 chunk_id 精确删除指定 L3 块
  patch_l3_chunk        —— 将指定 chunk 的文本替换为新内容（重 embedding）
  add_fact              —— 写入一条修正事实（附带语境信息到 L3）
  add_l3_correction     —— 写入一条修正语境到 L3

LLM 驱动的修正流程：
  1. 收集本次召回的 L3 记忆和已入库 chunk 列表作为上下文
  2. 将用户原话 + 助手回复 + 上下文发给 LLM
  3. LLM 判断是否需要修正（should_correct）
  4. LLM 生成具体的修正动作列表（actions）
  5. _execute_actions 逐一执行修正操作
============================================================================
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import settings
from app.llm import chat_completion
from app.memory.guard import memory_l3_hit
from app.memory.l3 import semantic_memory
from app.session import store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 正则信号检测 —— 两套互补的正则模式，用于判断用户是否在纠正记忆
# ---------------------------------------------------------------------------

# 模式 1：记忆否认词 —— 用户明确指出记忆有误
# 匹配"记错了/没这回事/不是这样的"等表述。
# 这类词触发率高、误判率低，直接返回 True。
_MEMORY_WRONG = re.compile(
    r"记错了|记错|你记错|记混了|记混|记忆有误|记忆.*不对|长期记忆|"
    r"没这回事|没这事|不是这样的|不是那样|不是这样|你搞错|搞错了|"
    r"别记|不要记|别记住|忘了吧|其实不是|实际上不是|并不是|"
    r"纠正.*记忆|记忆.*纠正|把你记|你记成|记成.*错|说错了.*记|"
    r"不存在|从没|从来没有|没去过|没见过"
)

# 模式 2：即时纠错词 —— 用户说"打错了/说错了"，可能是自纠打字错误
# 这类词单独出现时不足以判断为记忆纠错（可能只是在纠正自己的输入错误）。
# 必须配合 memory_l3_hit() 确认本轮确实命中了 L3 记忆，才判定为记忆纠错。
_CORRECTION = re.compile(r"输错了|打错了|说错了|搞错了|打错字|写错了|刚才.*错")


def user_signals_memory_correction(user_msg: str, memory: dict | None = None) -> bool:
    """检测用户消息是否包含记忆纠错信号。

    两条检测路径：
      1. 包含"记错了/没这回事"等明显的记忆否认词 → 直接返回 True
      2. 包含"打错了/说错了"等即时纠错词 + 本轮 L3 有命中 →
         用户可能在纠正被召回的旧记忆 → 返回 True

    Args:
        user_msg: 用户当前消息
        memory:   本轮召回的记忆数据（用于判断 L3 是否命中）

    Returns:
        True 表示用户正在纠正记忆，应触发修正流程。
    """
    msg = user_msg.strip()
    if not msg:
        return False
    # 路径 1：明确的记忆否认词 → 直接判定为纠错信号
    if _MEMORY_WRONG.search(msg):
        return True
    # 路径 2：即时纠错词 + 本轮确实召回了旧记忆 → 用户可能在纠正被召回的旧记忆。
    # 需要 memory 参数不为空且 L3 确实命中，避免误判（用户说"打错字了"不代表记忆有问题）。
    if _CORRECTION.search(msg) and memory:
        if memory_l3_hit(memory) or memory.get("corpus_triggered"):
            return True
    return False


def _build_correction_context(
    device_id: str, person_id: str, memory: dict,
) -> dict[str, Any]:
    """构建记忆修正的上下文信息，供 LLM 分析。

    收集三层信息：
      1. l3_recalled_previews：本轮召回的 L3 记忆摘要（取前 320 字符）
      2. l3_chunk_refs：本轮召回的 L3 块引用（含 chunk_id 用于精确操作）
      3. stored_l3：该用户所有已入库的 L3 块列表（用于对比）

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        memory:    本轮召回的记忆数据

    Returns:
        包含上述三层信息的字典。
    """
    # 从召回结果中提取 L3 命中文本
    l3_hits = memory.get("matches", {}).get("l3") or []
    recalled_texts = [
        str(m.get("text", "")).strip()
        for m in l3_hits
        if isinstance(m, dict) and str(m.get("text", "")).strip()
    ]

    # 按文本反查 chunks 表获取 chunk_id。
    # 文本匹配可能返回多条记录（同一文本被多次导入），所以 limit=3 取足够候选，
    # seen_ids 确保同一 chunk_id 不重复出现。
    chunk_refs: list[dict] = []
    seen_ids: set[str] = set()
    for text in recalled_texts[:8]:  # 最多处理 8 条召回文本
        for ref in store.l3_find_chunks_by_text(text, limit=3):
            cid = ref["chunk_id"]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            chunk_refs.append(ref)

    # 获取该用户全部已入库 L3 块
    pid = str(person_id or "").strip()
    stored = store.l3_list_person_memory(pid, device_id=device_id, limit=20) if pid else []
    return {
        "l3_recalled_previews": [t[:320] for t in recalled_texts[:6]],
        "l3_chunk_refs": chunk_refs[:12],
        "stored_l3": [
            {
                "chunk_id": r["chunk_id"],
                "text": str(r.get("text", ""))[:320],
                "category": r.get("category", ""),
            }
            for r in stored
        ],
    }


def _execute_actions(
    device_id: str, person_id: str, session_id: str, actions: list[dict],
) -> dict[str, int]:
    """执行 LLM 生成的修正动作列表。

    支持 5 种操作类型，每种都有独立的安全校验：

    delete_fact：
      - 按 match_text 删除匹配的 L3 chunks + L0 条目
      - L3: 通过 l3_find_chunks_by_text 定位 → l3_delete_chunk
      - L0: 通过 l0_delete_matching 删除

    delete_l3_chunk：
      - 按 chunk_id 精确删除指定 L3 块

    patch_l3_chunk：
      - 取旧块的元数据（device_id/person_id/source/category）
      - 对 new_text 重新做 embedding
      - 用 l3_bulk_upsert 覆盖原块（原地替换）

    add_fact / add_l3_correction：
      - 写入修正信息到 L3，source="user_correction"
      - add_fact 附带 corpus 上下文；add_l3_correction 仅有 text

    Args:
        device_id:  设备标识
        person_id:  用户 ID
        session_id: 会话标识
        actions:    LLM 生成的修正动作列表

    Returns:
        各操作类型执行次数统计。
    """
    pid = str(person_id or "").strip()
    stats = {
        "deleted_facts": 0, "deleted_chunks": 0, "patched_chunks": 0,
        "added_facts": 0, "added_corpus": 0, "deleted_l0": 0,
    }

    for raw in actions[:6]:  # 最多执行 6 个动作，防止 LLM 过度操作
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("op", "")).lower().strip()

        # 操作 1：按文本删除 L3 chunks + L0 条目
        if op == "delete_fact":
            match = str(raw.get("match_text", "")).strip()
            if match:
                if pid:
                    stats["deleted_l0"] += store.l0_delete_matching(pid, match)
                for ref in store.l3_find_chunks_by_text(match, limit=8):
                    if store.l3_delete_chunk(ref["chunk_id"]):
                        stats["deleted_chunks"] += 1

        # 操作 2：按 chunk_id 精确删除
        elif op == "delete_l3_chunk":
            cid = str(raw.get("chunk_id", "")).strip()
            if cid and store.l3_delete_chunk(cid):
                stats["deleted_chunks"] += 1

        # 操作 3：修补 chunk（替换文本内容，重做 embedding）
        # 流程：取旧 chunk 的元数据 → 对新文本做 embedding → upsert 覆盖原记录
        # 注意：patch 保留原 chunk_id，只是更新文本和向量，所以旧引用不会断裂
        elif op == "patch_l3_chunk":
            cid = str(raw.get("chunk_id", "")).strip()
            new_text = str(raw.get("new_text", "")).strip()
            # new_text >= 4 是安全阈值：太短的文本可能来自 LLM 幻觉
            if cid and len(new_text) >= 4:
                row = store.l3_get_chunk(cid)
                if row:
                    from app.llm import embed_texts

                    emb = embed_texts([new_text])[0]
                    # upsert 覆盖：chunk_id 不变，文本和向量替换为新值。
                    # confidence=1.0 因为这是用户亲口纠正，可信度最高。
                    store.l3_bulk_upsert([
                        {
                            "chunk_id": cid, "collection": "memory",
                            "device_id": device_id, "person_id": pid,
                            "text": new_text, "embedding": emb,
                            "source": "user_correction", "category": "correction",
                            "confidence": 1.0,
                        }
                    ])
                    stats["patched_chunks"] += 1

        # 操作 4：写入一条修正事实（附带语境）
        elif op == "add_fact":
            fact = str(raw.get("fact", "")).strip()
            corpus = str(raw.get("corpus", "")).strip()
            if len(fact) >= 4 and pid:
                from app.memory.l3 import ingest_l3_text

                body = corpus or f"【用户确认的事实】{fact}"
                cid = ingest_l3_text(
                    device_id, pid, body, source="user_correction",
                    source_session=session_id, category="correction",
                )
                if cid:
                    stats["added_corpus"] += 1

        # 操作 5：写入修正语境到 L3
        elif op in ("add_corpus_correction", "add_l3_correction"):
            text = str(raw.get("text", "")).strip()
            if len(text) >= 8 and pid:
                from app.memory.l3 import ingest_l3_text

                cid = ingest_l3_text(
                    device_id, pid, text, source="user_correction",
                    source_session=session_id, category="correction",
                )
                if cid:
                    stats["added_corpus"] += 1
            elif len(text) >= 8:
                # 无 person_id 的兜底路径：仍然写入 L3 语料库，但使用
                # 基于 device_id + hash 的匿名 chunk_id，不绑定到特定用户。
                # 这样设备维度的更正不会丢失，后续实名后可关联。
                cid = f"corr-{device_id}-{abs(hash(text)) % 10**10}"
                semantic_memory.ingest_chunks([
                    {
                        "id": cid, "text": text,
                        "meta": {"source": "user_correction", "device_id": device_id},
                    }
                ])
                stats["added_corpus"] += 1

    return stats


def try_apply_memory_corrections(
    device_id: str, person_id: str, session_id: str,
    user_msg: str, assistant_msg: str, memory: dict | None,
) -> dict[str, Any] | None:
    """记忆修正主入口：检测纠正信号 → LLM 分析 → 执行修正动作。

    流程：
      1. 检查配置开关（memory_auto_correct）和用户信号
      2. 仅对已实名用户执行（访客无长期记忆）
      3. 构建修正上下文（召回的记忆 + 已入库块列表）
      4. LLM 分析：判断是否需要修正，生成修正动作
      5. 执行修正动作：删除/修补/新增
      6. 记录日志

    Args:
        device_id:      设备标识
        session_id:     会话标识
        person_id:      用户 ID
        user_msg:       用户原始消息
        assistant_msg:  助手回复内容
        memory:         本轮召回的记忆数据

    Returns:
        {"reason": "...", "stats": {...}} 如果执行了修正；
        None 如果未触发修正（配置关闭/无信号/非实名/LLM 判断无需修正）。
    """
    # 守卫 1：配置开关 —— 允许运维/测试时关闭自动纠错
    if not settings.memory_auto_correct:
        return None
    # 守卫 2：检测用户纠正信号 —— 无信号则不做任何分析
    if not memory or not user_signals_memory_correction(user_msg, memory):
        return None
    # 守卫 3：仅已实名用户 —— 访客无长期记忆，无需修正
    if not str(person_id or "").strip():
        return None

    # 构建 LLM 分析上下文
    ctx = _build_correction_context(device_id, person_id, memory)
    prompt = f"""你是长期记忆修正器。用户指出 Agent 的长期记忆（L3 语料）有误，请根据用户本轮原话生成修正动作。

规则：
1. 只依据用户**明确说的**纠正内容，禁止编造
2. 用户仅纠正称呼/名字且未否定整段经历 → 优先 add_l3_correction，慎用 delete_l3_chunk
3. 用户明确否定某段经历（没这回事、从没、记错了）→ delete_l3_chunk 或 patch_l3_chunk / add_l3_correction 写准正确版本
4. 优先用 chunk_id（来自 l3_chunk_refs）；没有 id 时用 match_text 删块（delete_fact 兼容 op）
5. 写入正确信息：优先 add_l3_correction（完整语境）或 add_fact（附带 corpus 字段，等价写入 L3）
6. 若无足够信息执行修正 → should_correct=false

本轮召回的长期记忆摘要：
{json.dumps(ctx["l3_recalled_previews"], ensure_ascii=False)}

可操作的 L3 块（含 chunk_id）：
{json.dumps(ctx["l3_chunk_refs"], ensure_ascii=False)}

该用户已入库 L3 块：
{json.dumps(ctx["stored_l3"], ensure_ascii=False)}

用户：{user_msg}
助手：{assistant_msg}

只输出 JSON：
{{
  "should_correct": true,
  "reason": "一句话",
  "actions": [
    {{"op":"delete_fact","match_text":""}},
    {{"op":"delete_l3_chunk","chunk_id":""}},
    {{"op":"patch_l3_chunk","chunk_id":"","new_text":""}},
    {{"op":"add_fact","fact":"","corpus":"完整语境","category":"correction"}},
    {{"op":"add_l3_correction","text":""}}
  ]
}}"""

    raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.1)
    # 从 LLM 返回中提取 JSON（处理可能带 markdown 代码块包裹的情况）
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    # LLM 认为不需要修正 → 不做任何操作
    if not data.get("should_correct"):
        return None

    # 确保 actions 是列表，且非空（should_correct=true 但 actions=[] 不操作）
    actions = data.get("actions") if isinstance(data.get("actions"), list) else []
    if not actions:
        return None

    stats = _execute_actions(device_id, person_id, session_id, actions)
    logger.info("memory correction: %s stats=%s", data.get("reason", ""), stats)
    return {"reason": data.get("reason", ""), "stats": stats}
