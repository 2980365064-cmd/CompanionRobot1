"""
L1→L2 压缩与会话收尾 —— 记忆系统的时间维度转换引擎。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Extractor 是"记忆压缩机"——负责将短期消息（L1）转化为中期摘要（L2），
  再在过期时将 L2 归档为长期语料（L3）。

数据流全景图：
  ┌──────────┐  满 N 轮压缩    ┌──────────┐  7天后过期     ┌──────────┐
  │  L1 消息  │ ───────────→   │  L2 摘要  │ ──────────→   │  L3 语料  │
  │ (messages)│ compress_l1_to │(episodic) │ rollup_expired │ (chunks)  │
  └──────────┘     _l2         └──────────┘   _l2           └──────────┘
       ↑                           │
       │      consolidate_session  │  会话收尾
       └───────────────────────────┘

四个核心函数：
  1. compress_l1_to_l2    —— L1 满 N 轮时，最老一批消息 LLM 摘要写入 L2
  2. consolidate_session   —— 会话结束时，剩余 L1 全量压缩 + 画像转正尝试
  3. rollup_expired_l2     —— 定时任务：过期 L2（>7天）→ L3 叙述块归档
  4. ingest_remember_to_l3 —— 用户"记住"意图：对话原文直接写入 L3

关键设计：
  - 访客模式不压缩（仅已实名用户产生活动记录）
  - L2 摘要含 4 个维度：summary（文本）/ topics（主题）/ open_loops（待办）/ emotion（情感）
  - 会话收尾后还会尝试 L0 提取（由 agent 异步调用 extract_l0_from_session_summary）
============================================================================
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from app.config import settings
from app.llm import chat_completion
from app.memory.l2 import episodic_memory
from app.session import store

logger = logging.getLogger(__name__)


def _format_transcript(messages: list[dict]) -> str:
    """将消息字典列表格式化为 LLM 可读的对话文本。

    格式：每行一条 `role: content`，按时间正序排列。
    """
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


# LLM 摘要指令：要求 LLM 生成 6 维度的 L2 摘要
# 新增 importance 和 people 字段以支持情感重要性加权
_L2_SUMMARY_INSTRUCTION = """将以下对话压缩为 **L2 情景记忆**（用于长期情感感知）。

要求：
- summary：150～280 字，写清时间线、谁说了什么、情绪、决定、待办、对方自称/关系（若有）
- topics：逗号分隔主题
- open_loops：未完结事项数组，无则 []
- emotion：mood 取 平静/开心/低落/焦虑/难过/生气/兴奋/疲惫/烦躁/害怕/轻松/期待；intensity 取 0.0~1.0；trigger 为情绪诱因；attitude 为 倾诉/依赖/敷衍/调侃/冷淡/亲密/求助
- importance：1~5（1-闲聊/琐事, 2-日常, 3-普通, 4-重要事件/情绪事件, 5-里程碑/重大决定）
- people：涉及的人物名数组（无则 []）

只输出 JSON：
{{"summary":"...","topics":"...","open_loops":[],"emotion":{{"mood":"平静","intensity":0.3,"trigger":"","attitude":""}},"importance":3,"people":[]}}"""


def compress_l1_to_l2(device_id: str, session_id: str) -> bool:
    """L1 满 working_memory_turns 轮时：将最老一批消息 LLM 摘要后写入 L2 并删除原消息。

    触发条件：当前会话对话轮数 >= working_memory_turns。
    每次压缩 l1_compress_batch_turns 轮的消息（最老一批），不做全部压缩，
    保留最近的消息在 L1 中继续使用。

    压缩流程：
      1. 检查轮数是否达到阈值
      2. 取最老 l1_compress_batch_turns × 2 条消息（每轮 user+assistant）
      3. 格式化为对话文本 → LLM 生成四维 L2 摘要
      4. 解析 JSON → 写入 episodic_memory（附带 emotion JSON）
      5. 删除已压缩的原始消息

    Args:
        device_id:  设备标识
        session_id: 会话标识

    Returns:
        True 表示成功压缩了一批消息，False 表示未达到阈值或压缩失败。
    """
    turns = store.count_turns(session_id)
    if turns < settings.working_memory_turns:
        return False

    batch = settings.l1_compress_batch_turns
    msg_limit = batch * 2  # 每轮 2 条消息
    oldest = store.get_oldest_messages(session_id, msg_limit)
    if len(oldest) < 4:  # 至少够 2 轮才有压缩意义
        return False

    # 构建 LLM 摘要请求
    transcript = _format_transcript(oldest)
    prompt = f"""{_L2_SUMMARY_INSTRUCTION}

对话：
{transcript}"""
    raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.2)

    # 安全解析 JSON（容错：提取第一个 {...} 块）
    # LLM 偶尔会在 JSON 前后加解释文字，用 re 提取 {} 可绕过这种噪声
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return False
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return False

    summary = str(data.get("summary", "")).strip()
    if not summary:  # LLM 没生成摘要 → 不压缩（不删除原始消息，下次再试）
        return False

    topics = str(data.get("topics", ""))
    loops = data.get("open_loops", [])
    open_loops = json.dumps(loops, ensure_ascii=False) if loops else ""
    emotion = json.dumps(data.get("emotion") or {}, ensure_ascii=False)
    importance = int(data.get("importance") or 3)
    importance = max(1, min(5, importance))
    # people 字段：JSON 数组字符串
    raw_people = data.get("people", [])
    people = json.dumps(raw_people, ensure_ascii=False) if isinstance(raw_people, list) and raw_people else ""

    # 写入 L2 情景记忆（附带情感、重要性、涉及人物等结构化元数据）
    person_id = store.get_session_active_person_id(session_id) or ""
    episodic_memory.save_summary(
        device_id, person_id, session_id, summary, topics, open_loops,
        emotion=emotion, importance=importance, people=people,
    )
    # 删除已压缩的原始 L1 消息（释放 messages 表存储空间）
    # 压缩是"移出"操作：L1 → L2，原消息不再需要
    store.delete_messages_by_ids([m["id"] for m in oldest])
    logger.debug("L1→L2 compressed %d messages for session=%s", len(oldest), session_id)
    return True


def consolidate_session(device_id: str, session_id: str) -> str:
    """会话结束时的收尾操作：剩余 L1 全量压缩 + 画像转正尝试 + 清空 messages。

    这是会话生命周期的终结点，执行以下操作：
      1. 访客检查：非实名用户直接 finalize 并返回空摘要
      2. 循环压缩：将所有剩余 L1 消息逐批压入 L2
      3. 尾批压缩：取最后 40 条消息生成最终 L2 摘要（避免碎片化）
      4. 画像转正：尝试将临时 draft 画像确认为正式画像
      5. 清空 messages 表（释放 L1 存储）

    Args:
        device_id:  设备标识
        session_id: 会话标识

    Returns:
        生成的 L2 摘要文本（用于后续 L0 提取）；无摘要时返回空字符串。
    """
    from app.memory.identity import is_verified_person_id

    person_id = store.get_session_active_person_id(session_id) or ""
    # 访客模式：不压缩，直接结束
    if not is_verified_person_id(person_id):
        store.finalize_session(session_id)
        return ""

    # 阶段 1：循环压缩所有超量 L1 消息到 L2
    # 为什么要循环：会话可能有几百条消息，单次 compress_l1_to_l2
    # 只压缩最老的 batch 条，需要多次迭代才能将所有超量消息写入 L2。
    if person_id:
        while store.count_turns(session_id) >= settings.working_memory_turns:
            if not compress_l1_to_l2(device_id, session_id):
                break

    # 阶段 2：尾批最终摘要——取最后最多 40 条消息生成最终 L2 摘要。
    # 即使 L1 总量不足一次压缩阈值（<20 轮），会话结束时也要生成摘要，
    # 否则这些对话内容会随着 finalize 被丢弃，丧失中间记忆。
    messages = store.get_session_messages(session_id)
    l2_summary = ""

    if person_id and len(messages) >= 1:
        transcript = _format_transcript(messages[-40:])
        prompt = f"""{_L2_SUMMARY_INSTRUCTION}

{transcript}"""
        raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                l2_summary = str(data.get("summary", "")).strip()
                if l2_summary:
                    topics = str(data.get("topics", ""))
                    loops = data.get("open_loops", [])
                    open_loops = json.dumps(loops, ensure_ascii=False) if loops else ""
                    emotion = json.dumps(data.get("emotion") or {}, ensure_ascii=False)
                    episodic_memory.save_summary(
                        device_id, person_id, session_id, l2_summary, topics, open_loops, emotion=emotion
                    )
            except json.JSONDecodeError:
                pass

    # 阶段 3：画像转正确认（仅在会话结束时触发）
    # 如果画像是 draft/provisional 状态且会话中有实质内容（如关系声明、
    # 人物事实等），则尝试转正。为什么在会话结束时而非每轮都触发：
    # 画像转正是"审核"操作，应该累积足够证据后再判断，避免碎片化决策。
    if person_id:
        raw = store.get_person_profile(person_id)
        if raw and raw.get("provisional"):
            from app.memory.profile import try_promote_provisional_profile

            try_promote_provisional_profile(device_id, person_id, raw)

    # 清空会话消息，标记会话结束
    store.finalize_session(session_id)
    return l2_summary


def rollup_expired_l2(device_id: str | None = None) -> int:
    """定时任务：将过期 L2 摘要（>7 天）归档为 L3 语料块。

    按 person_id 分组处理，每组调用 rollup_l2_rows_to_corpus 批量写入 L3。
    归档成功后调用 archive_episodic 标记原 L2 记录为已归档（不再参与检索）。

    Args:
        device_id: 可选，指定设备（None 表示所有设备）

    Returns:
        成功归档的 L2 记录条数。
    """
    from app.memory.l3 import rollup_l2_rows_to_corpus

    rows = store.list_expired_episodic(device_id)
    if not rows:
        return 0

    # 按 person_id 分组，避免跨用户数据混写
    # L3 语料是按 person_id 分区的，同一人的多天摘要可以合并成一个叙述块
    by_person: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pid = str(row.get("person_id") or "").strip()
        by_person[pid].append(row)

    rolled = 0
    for person_id, items in by_person.items():
        dev_id = str(items[0].get("device_id") or "")
        try:
            # rollup_l2_rows_to_corpus 将多条 L2 摘要 LLM 加工成连贯的 L3 叙述块
            rollup_l2_rows_to_corpus(dev_id, person_id, items)
            # 归档成功后标记原 L2 记录已归档（不再参与 L2 检索，但保留审计记录）
            store.archive_episodic([int(r["id"]) for r in items])
            rolled += len(items)
        except Exception as exc:
            # 单组失败不影响其他组：按 person 隔离异常
            logger.warning("L2→Corpus rollup failed person=%s: %s", person_id, exc)
    return rolled


def ingest_remember_to_l3(device_id: str, session_id: str, user_msg: str, assistant_msg: str) -> None:
    """用户明确说"记住"时：将对话原文直接写入 L3 长期记忆。

    触发条件：用户消息中包含"记住/别忘了/以后要/帮我记"等关键词。
    访客模式不写入（仅已实名用户可存储长期记忆）。

    写入格式：
      用户：{用户原话}
      助手：{助手回复}

    Args:
        device_id:     设备标识
        session_id:    会话标识
        user_msg:      用户原始消息
        assistant_msg: 助手回复内容
    """
    # 仅匹配明确记忆意图的消息
    if not re.search(r"记住|别忘了|以后要|帮我记", user_msg):
        return

    from app.memory.l3 import ingest_l3_text
    from app.memory.identity import is_verified_person_id

    pid = store.get_session_active_person_id(session_id) or ""
    if not is_verified_person_id(pid):
        return

    # 将对话上下文一并存储，保持语境完整
    corpus = f"用户：{user_msg}\n助手：{assistant_msg}"
    ingest_l3_text(
        device_id,
        pid,
        corpus,
        source="user_remember_intent",
        source_session=session_id,
        category="remember",
    )


def extract_facts(device_id: str, session_id: str, user_msg: str, assistant_msg: str) -> None:
    """兼容别名：作用等同于 ingest_remember_to_l3。

    供老代码调用路径兼容。
    """
    ingest_remember_to_l3(device_id, session_id, user_msg, assistant_msg)


def maybe_compress_l1(device_id: str, session_id: str, person_id: str = "") -> bool:
    """每轮对话后检查：L1 超过阈值时压缩最老批次到 L2。

    优化策略：
      - 阈值 + 50% 余量才触发，避免每轮都尝试压缩
      - 例如 threshold=30 时，36 轮才触发，压缩后降至 ~18 轮
      - 这样每 ~18 轮只需要一次 LLM 调用，而非每轮

    Args:
        device_id:  设备标识
        session_id: 会话标识
        person_id:  用户 ID

    Returns:
        True 表示至少执行了一次压缩。
    """
    from app.memory.identity import is_verified_person_id

    pid = str(person_id or store.get_session_active_person_id(session_id) or "").strip()
    if not is_verified_person_id(pid):
        return False
    threshold = settings.working_memory_turns
    # 超过阈值 20% 才触发，给压缩留出余量
    # 例如 threshold=30 → 36 轮触发
    trigger = threshold + max(6, threshold // 5)
    if store.count_turns(session_id) < trigger:
        return False
    did = False
    while store.count_turns(session_id) >= threshold:
        if not compress_l1_to_l2(device_id, session_id):
            break
        did = True
    return did
