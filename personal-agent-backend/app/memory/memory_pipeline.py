"""记忆管线（Memory Pipeline）—— 上下文压缩、会话收尾、近期记忆归档（自包含模块）。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  MemoryPipeline 是"记忆管线调度器"——负责将短期工作记忆（messages）转化为
  中长期记忆（近期摘要 → 长期语料）。函数名全量语义化，调用方无需理解
  底层分层细节。

数据流：
  ┌─────────────────┐  满 N 轮压缩   ┌──────────────┐  过期归档    ┌──────────────────┐
  │ Working Context │ ───────────→  │ Recent Memory│ ──────────→ │ Long-Term Memory │
  │   (messages)    │  删除原消息    │  (summaries) │             │    (corpus)      │
  └─────────────────┘               └──────────────┘             └──────────────────┘
       ↑
       │  finalize_session_memory / 会话收尾
       └───────────────────────────────────────────

四个核心操作：
  1. compact_working_context_batch   —— 满 N 轮时，最老一批消息 LLM 摘要写入近期记忆
  2. finalize_session_memory         —— 会话结束时，剩余工作上下文全量压缩 + 画像转正尝试
  3. archive_expired_recent_memory   —— 定时任务：过期近期记忆（>14天）→ 长期叙述块归档
  4. write_explicit_memory_request   —— 用户"记住"意图：对话原文直接写入长期记忆

关键设计：
  - 访客模式不压缩（仅已实名用户产生活动记录）
  - 摘要含 5 个维度：summary / topics / open_loops / emotion / importance
  - 会话收尾后还会尝试核心事实提取
============================================================================
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from app.config import settings
from app.llm import chat_completion
from app.memory.recent_memory import recent_memory
from app.session import store

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 公共 API（语义化入口）
# ══════════════════════════════════════════════════════════════════════════════


def compact_working_context_batch(device_id: str, session_id: str) -> bool:
    """将当前会话最老一批工作上下文压缩为统一记忆库摘要。

    触发条件：当前会话对话轮数 >= working_context_turns。
    每次压缩 context_compaction_batch_turns 轮的消息（最老一批）。

    Args:
        device_id:  设备标识
        session_id: 会话标识

    Returns:
        True 表示成功压缩了一批消息，False 表示未达到阈值或压缩失败。
    """
    # 压缩：工作上下文→统一记忆库
    return _compact_working_context_to_recent_memory(device_id, session_id)


def finalize_session_memory(device_id: str, session_id: str) -> str:
    """会话结束时的收尾操作：剩余工作上下文全量压缩 + 画像转正尝试 + 清空 messages。

    这是会话生命周期的终结点。

    Args:
        device_id:  设备标识
        session_id: 会话标识

    Returns:
        生成的摘要文本（用于后续核心事实提取）；无摘要时返回空字符串。
    """
    return consolidate_session(device_id, session_id)


def archive_expired_recent_memory(device_id: str | None = None) -> int:
    """定时任务：将过期近期记忆（>14 天）归档为长期语料块。

    Args:
        device_id: 可选，指定设备（None 表示所有设备）

    Returns:
        成功归档的记录条数。
    """
    # 归档：近期记忆→长期语料
    return _archive_expired_recent_memory(device_id)


def write_explicit_memory_request(
    device_id: str, session_id: str, user_msg: str, assistant_msg: str
) -> None:
    """用户明确说"记住"时：将对话原文直接写入长期记忆。

    触发条件：用户消息中包含"记住/别忘了/以后要/帮我记"等关键词。
    访客模式不写入（仅已实名用户可存储长期记忆）。

    Args:
        device_id:     设备标识
        session_id:    会话标识
        user_msg:      用户原始消息
        assistant_msg: 助手回复内容
    """
    _write_explicit_memory_request(device_id, session_id, user_msg, assistant_msg)


def maybe_compact_working_context(
    device_id: str, session_id: str, person_id: str = ""
) -> bool:
    """每轮对话后检查：工作上下文超过阈值时压缩最老批次到统一记忆库。

    Args:
        device_id:  设备标识
        session_id: 会话标识
        person_id:  用户 ID（可选，未提供时从会话自动获取）

    Returns:
        True 表示至少执行了一次压缩。
    """
    # 检查并压缩：工作上下文→统一记忆库
    return _maybe_compact_working_context(device_id, session_id, person_id)


# ══════════════════════════════════════════════════════════════════════════════
# 内部实现（从 extractor.py 内联嵌入）
# ══════════════════════════════════════════════════════════════════════════════


def _format_transcript(messages: list[dict]) -> str:
    """将消息字典列表格式化为 LLM 可读的对话文本。"""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


_RECENT_SUMMARY_INSTRUCTION = """将以下对话压缩为**近期情景摘要**（用于长期情感感知）。

要求：
- summary：150～280 字，写清时间线、谁说了什么、情绪、决定、待办、对方自称/关系（若有）
- topics：逗号分隔主题
- open_loops：未完结事项数组，无则 []
- emotion：mood 取 平静/开心/低落/焦虑/难过/生气/兴奋/疲惫/烦躁/害怕/轻松/期待；intensity 取 0.0~1.0；trigger 为情绪诱因；attitude 为 倾诉/依赖/敷衍/调侃/冷淡/亲密/求助
- importance：1~5（1-闲聊/琐事, 2-日常, 3-普通, 4-重要事件/情绪事件, 5-里程碑/重大决定）
- people：涉及的人物名数组（无则 []）

只输出 JSON：
{{"summary":"...","topics":"...","open_loops":[],"emotion":{{"mood":"平静","intensity":0.3,"trigger":"","attitude":""}},"importance":3,"people":[]}}"""


def _compact_working_context_to_recent_memory(device_id: str, session_id: str) -> bool:
    """工作上下文满 working_context_turns 轮时：将最老一批消息 LLM 摘要后写入统一记忆库并删除原消息。"""
    turns = store.count_turns(session_id)
    if turns < settings.working_context_turns:
        return False

    batch = settings.context_compaction_batch_turns
    msg_limit = batch * 2
    oldest = store.get_oldest_messages(session_id, msg_limit)
    if len(oldest) < 4:
        return False

    transcript = _format_transcript(oldest)
    prompt = f"""{_RECENT_SUMMARY_INSTRUCTION}

对话：
{transcript}"""
    raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.2)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return False
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return False

    summary = str(data.get("summary", "")).strip()
    if not summary:
        return False

    topics = str(data.get("topics", ""))
    loops = data.get("open_loops", [])
    open_loops = json.dumps(loops, ensure_ascii=False) if loops else ""
    emotion = json.dumps(data.get("emotion") or {}, ensure_ascii=False)
    importance = int(data.get("importance") or 3)
    importance = max(1, min(5, importance))
    raw_people = data.get("people", [])
    people = json.dumps(raw_people, ensure_ascii=False) if isinstance(raw_people, list) and raw_people else ""

    person_id = store.get_session_active_person_id(session_id) or ""
    recent_memory.save_recent_summary(
        device_id, person_id, session_id, summary, topics, open_loops,
        emotion=emotion, importance=importance, people=people,
    )
    # 清除已压缩的工作上下文消息
    store.delete_messages_by_ids([m["id"] for m in oldest])
    logger.debug("工作上下文→统一记忆库：压缩 %d 条消息 session=%s", len(oldest), session_id)
    return True


def consolidate_session(device_id: str, session_id: str) -> str:
    """会话结束时的收尾操作：剩余工作上下文全量压缩 + 画像转正尝试 + 清空 messages。"""
    from app.memory.identity import is_verified_person_id

    person_id = store.get_session_active_person_id(session_id) or ""
    if not is_verified_person_id(person_id):
        # 收尾：清空工作上下文（访客模式，不产生记忆）
        store.finalize_session(session_id)
        return ""

    if person_id:
        while store.count_turns(session_id) >= settings.working_context_turns:
            if not _compact_working_context_to_recent_memory(device_id, session_id):
                break

    messages = store.get_session_messages(session_id)
    recent_summary = ""

    if person_id and len(messages) >= 1:
        transcript = _format_transcript(messages[-40:])
        prompt = f"""{_RECENT_SUMMARY_INSTRUCTION}

{transcript}"""
        raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                recent_summary = str(data.get("summary", "")).strip()
                if recent_summary:
                    topics = str(data.get("topics", ""))
                    loops = data.get("open_loops", [])
                    open_loops = json.dumps(loops, ensure_ascii=False) if loops else ""
                    emotion = json.dumps(data.get("emotion") or {}, ensure_ascii=False)
                    recent_memory.save_recent_summary(
                        device_id, person_id, session_id, recent_summary, topics, open_loops, emotion=emotion
                    )
            except json.JSONDecodeError:
                pass

    if person_id:
        raw = store.get_person_profile(person_id)
        if raw and raw.get("provisional"):
            from app.memory.profile import try_promote_provisional_profile

            try_promote_provisional_profile(device_id, person_id, raw)

    # 收尾：清空工作上下文
    store.finalize_session(session_id)
    return recent_summary


def _archive_expired_recent_memory(device_id: str | None = None) -> int:
    """定时任务：将过期近期记忆条目（>14 天）归档为长期语料块（近期记忆→长期记忆）。"""
    from app.memory.long_term_memory import archive_recent_to_long_term

    rows = store.list_expired_recent_memory(device_id)
    if not rows:
        return 0

    by_person: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pid = str(row.get("person_id") or "").strip()
        by_person[pid].append(row)

    rolled = 0
    for person_id, items in by_person.items():
        dev_id = str(items[0].get("device_id") or "")
        try:
            archive_recent_to_long_term(dev_id, person_id, items)
            store.archive_recent_memory([str(r["id"]) for r in items])
            rolled += len(items)
        except Exception as exc:
            logger.warning("近期→长期归档失败 person=%s: %s", person_id, exc)
    return rolled


def _write_explicit_memory_request(device_id: str, session_id: str, user_msg: str, assistant_msg: str) -> None:
    """用户明确说"记住"时：将对话原文直接写入长期记忆。"""
    if not re.search(r"记住|别忘了|以后要|帮我记", user_msg):
        return

    from app.memory.identity import is_verified_person_id
    from app.memory.long_term_memory import store_long_term_text

    pid = store.get_session_active_person_id(session_id) or ""
    if not is_verified_person_id(pid):
        return

    corpus = f"用户：{user_msg}\n助手：{assistant_msg}"
    store_long_term_text(
        device_id,
        pid,
        corpus,
        source="user_remember_intent",
        source_session=session_id,
        category="remember",
    )


def _maybe_compact_working_context(device_id: str, session_id: str, person_id: str = "") -> bool:
    """每轮对话后检查：工作上下文超过阈值时压缩最老批次到统一记忆库。"""
    from app.memory.identity import is_verified_person_id

    pid = str(person_id or store.get_session_active_person_id(session_id) or "").strip()
    if not is_verified_person_id(pid):
        return False
    threshold = settings.working_context_turns
    trigger = threshold + max(6, threshold // 5)
    if store.count_turns(session_id) < trigger:
        return False
    did = False
    while store.count_turns(session_id) >= threshold:
        if not _compact_working_context_to_recent_memory(device_id, session_id):
            break
        did = True
    return did
