"""
记忆纠错 —— 用户纠正长期记忆：归档旧记忆项、写入修正后的事实。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Correction 是"记忆修正器"——当用户指出 Agent 的记忆有误时，
  自动定位并归档错误的记忆项，同时写入正确的替代信息。

触发条件：
  - user_signals_memory_correction() 检测到用户纠正信号
  - 需要 settings.memory_auto_correct = true
  - 仅对已实名用户生效（访客不存储长期记忆，无需修正）

修正操作类型：
  delete_fact           —— 按 match_text 搜索匹配的记忆项，归档/删除
  delete_item           —— 按 item_id 精确归档指定记忆项
  patch_item            —— 归档旧记忆项，写入更正后的新记忆项
  add_fact              —— 写入一条修正事实（kind=fact, source=user_correction）
  add_correction        —— 写入一条修正语境（kind=correction, visibility=always）

LLM 驱动的修正流程：
  1. 收集本轮召回的长期记忆和已入库记忆项列表作为上下文
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
# 必须配合本轮确实命中了长期记忆，才判定为记忆纠错。
_CORRECTION = re.compile(r"输错了|打错了|说错了|搞错了|打错字|写错了|刚才.*错")


def user_signals_memory_correction(user_msg: str, memory: dict | None = None) -> bool:
    """检测用户消息是否包含记忆纠错信号。

    两条检测路径：
      1. 包含"记错了/没这回事"等明显的记忆否认词 → 直接返回 True
      2. 包含"打错了/说错了"等即时纠错词 + 本轮长期记忆有命中 →
         用户可能在纠正被召回的旧记忆 → 返回 True

    Args:
        user_msg: 用户当前消息
        memory:   本轮召回的记忆数据（用于判断长期记忆是否命中）

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
    # 需要 memory 参数不为空且长期记忆确实命中，避免误判（用户说"打错字了"不代表记忆有问题）。
    if _CORRECTION.search(msg) and memory:
        diag = memory.get("diagnostics") or {}
        if diag.get("has_long_term") or memory.get("corpus_triggered"):
            return True
    return False


def _build_correction_context(
    device_id: str, person_id: str, memory: dict,
) -> dict[str, Any]:
    """构建记忆修正的上下文信息，供 LLM 分析。

    收集三层信息：
      1. recalled_previews：本轮召回的长期记忆摘要（取前 320 字符）
      2. memory_item_refs：本轮召回的长期记忆对应的记忆项引用（含 id 用于精确操作）
      3. stored_items：该用户所有已入库的长期记忆列表（用于对比）

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        memory:    本轮召回的记忆数据

    Returns:
        包含上述三层信息的字典。
    """
    # 从召回结果中提取长期记忆文本
    # diagnostics.long_term 由 MemoryRouter 填充，字段：text/kind/content_hash
    diag = memory.get("diagnostics") or {}
    long_term_hits = diag.get("long_term") or []
    recalled_texts = [
        str(m.get("text", "")).strip()
        or str(m.get("content", "")).strip()
        for m in long_term_hits
        if isinstance(m, dict)
        and (str(m.get("text", "")).strip() or str(m.get("content", "")).strip())
    ]

    # 按文本搜索统一记忆库获取 item_id
    pid = str(person_id or "").strip()
    item_refs: list[dict] = []
    seen_ids: set[str] = set()
    for text in recalled_texts[:8]:  # 最多处理 8 条召回文本
        if not pid:
            break
        for ref in store.search_memory_items(
            pid,
            kinds=["fact", "entity", "episode", "correction"],
            query=text,
            include_expired=True,
            limit=3,
        ):
            item_id = str(ref.get("id", "")).strip()
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            item_refs.append({
                "id": item_id,
                "text": str(ref.get("content", ""))[:320],
                "kind": ref.get("kind", ""),
            })

    # 获取该用户全部已入库长期记忆项
    stored = store.search_long_term_memory(pid, limit=20) if pid else []
    return {
        "recalled_previews": [t[:320] for t in recalled_texts[:6]],
        "memory_item_refs": item_refs[:12],
        "stored_items": [
            {
                "id": r.get("id", ""),
                "text": str(r.get("content", ""))[:320],
                "kind": r.get("kind", ""),
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
      - 按 match_text 搜索匹配的记忆项，逐一归档（archive_memory_item）

    delete_item（替代原来的 delete_long_term_chunk）：
      - 按 item_id 精确归档指定记忆项

    patch_item（替代原来的 patch_long_term_chunk）：
      - 归档旧记忆项，写入一条修正后的新记忆项（保留原 kind/visibility）

    add_fact：
      - 写入一条修正事实（kind="fact", source="user_correction", visibility="recall_only"）

    add_correction（替代原来的 add_long_term_correction）：
      - 写入一条修正语境（kind="correction", visibility="always"）

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
        "deleted_facts": 0,
        "deleted_items": 0,
        "patched_items": 0,
        "added_facts": 0,
        "added_corrections": 0,
    }

    for raw in actions[:6]:  # 最多执行 6 个动作，防止 LLM 过度操作
        if not isinstance(raw, dict):
            continue
        op = str(raw.get("op", "")).lower().strip()

        # 操作 1：按文本删除匹配的记忆项
        if op == "delete_fact":
            match = str(raw.get("match_text", "")).strip()
            if match and pid:
                for ref in store.search_memory_items(
                    pid,
                    query=match,
                    include_expired=True,
                    limit=10,
                ):
                    item_id = str(ref.get("id", "")).strip()
                    if item_id and store.archive_memory_item(item_id):
                        stats["deleted_facts"] += 1

        # 操作 2：按 item_id 精确归档
        elif op == "delete_item":
            item_id = str(raw.get("item_id", "")).strip()
            if item_id and store.archive_memory_item(item_id):
                stats["deleted_items"] += 1

        # 操作 3：修补记忆项（归档旧项，写入新项）
        elif op == "patch_item":
            item_id = str(raw.get("item_id", "")).strip()
            new_text = str(raw.get("new_text", "")).strip()
            # new_text >= 4 是安全阈值：太短的文本可能来自 LLM 幻觉
            if item_id and len(new_text) >= 4 and pid:
                old = store.get_memory_item(item_id)
                if old:
                    # 归档旧项
                    store.archive_memory_item(item_id)
                    # 对新文本做 embedding
                    from app.llm import embed_texts

                    emb = embed_texts([new_text])[0]
                    # 写入新项，保留旧项的 kind/visibility，覆盖为修正后内容。
                    # confidence=1.0 因为这是用户亲口纠正，可信度最高。
                    store.write_memory_item(
                        person_id=pid,
                        device_id=device_id,
                        kind=str(old.get("kind", "fact")),
                        source="user_correction",
                        visibility=str(old.get("visibility", "recall_only")),
                        content=new_text,
                        confidence=1.0,
                        source_table="memory_items",
                        source_id=item_id,
                        source_session=session_id,
                        embedding_json=json.dumps(emb) if emb else "[]",
                    )
                    stats["patched_items"] += 1

        # 操作 4：写入一条修正事实（附带语境）
        elif op == "add_fact":
            fact = str(raw.get("fact", "")).strip()
            corpus = str(raw.get("corpus", "")).strip()
            if len(fact) >= 4 and pid:
                body = corpus or f"【用户确认的事实】{fact}"
                from app.llm import embed_texts

                emb = embed_texts([body])[0]
                store.write_memory_item(
                    person_id=pid,
                    device_id=device_id,
                    kind="fact",
                    source="user_correction",
                    visibility="recall_only",
                    content=body,
                    confidence=1.0,
                    source_session=session_id,
                    embedding_json=json.dumps(emb) if emb else "[]",
                )
                stats["added_facts"] += 1

        # 操作 5：写入修正语境到长期记忆（替代原来的 add_long_term_correction）
        elif op == "add_correction":
            text = str(raw.get("text", "")).strip()
            if len(text) >= 8 and pid:
                from app.llm import embed_texts

                emb = embed_texts([text])[0]
                store.write_memory_item(
                    person_id=pid,
                    device_id=device_id,
                    kind="correction",
                    source="user_correction",
                    visibility="always",
                    content=text,
                    confidence=1.0,
                    source_session=session_id,
                    embedding_json=json.dumps(emb) if emb else "[]",
                )
                stats["added_corrections"] += 1

    return stats


def try_apply_memory_corrections(
    device_id: str, person_id: str, session_id: str,
    user_msg: str, assistant_msg: str, memory: dict | None,
) -> dict[str, Any] | None:
    """记忆修正主入口：检测纠正信号 → LLM 分析 → 执行修正动作。

    流程：
      1. 检查配置开关（memory_auto_correct）和用户信号
      2. 仅对已实名用户执行（访客无长期记忆）
      3. 构建修正上下文（召回的记忆 + 已入库记忆项列表）
      4. LLM 分析：判断是否需要修正，生成修正动作
      5. 执行修正动作：删除/修补/新增
      6. 记录日志

    Args:
        device_id:      设备标识
        person_id:      用户 ID
        session_id:     会话标识
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
    prompt = f"""你是长期记忆修正器。用户指出 Agent 的长期记忆（记忆项）有误，请根据用户本轮原话生成修正动作。

规则：
1. 只依据用户**明确说的**纠正内容，禁止编造
2. 用户仅纠正称呼/名字且未否定整段经历 → 优先 add_correction，慎用 delete_item
3. 用户明确否定某段经历（没这回事、从没、记错了）→ delete_item 或 patch_item / add_correction 写准正确版本
4. 优先用 item_id（来自 memory_item_refs）；没有 id 时用 match_text 删除（delete_fact 兼容 op）
5. 写入正确信息：优先 add_correction（完整语境）或 add_fact（附带 corpus 字段，等价写入长期记忆项）
6. 若无足够信息执行修正 → should_correct=false

本轮召回的长期记忆摘要：
{json.dumps(ctx["recalled_previews"], ensure_ascii=False)}

可操作的记忆项（含 item_id）：
{json.dumps(ctx["memory_item_refs"], ensure_ascii=False)}

该用户已入库的长期记忆项：
{json.dumps(ctx["stored_items"], ensure_ascii=False)}

用户：{user_msg}
助手：{assistant_msg}

只输出 JSON：
{{
  "should_correct": true,
  "reason": "一句话",
  "actions": [
    {{"op":"delete_fact","match_text":""}},
    {{"op":"delete_item","item_id":""}},
    {{"op":"patch_item","item_id":"","new_text":""}},
    {{"op":"add_fact","fact":"","corpus":"完整语境"}},
    {{"op":"add_correction","text":""}}
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
