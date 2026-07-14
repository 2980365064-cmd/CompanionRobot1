"""
统一写入仲裁器（Consolidator）—— 每轮对话后统一裁决本轮记忆沉淀。

============================================================================
设计目标：
  当前 _post_process() 中 6 个写入操作各自独立触发，缺少统一的"本轮
  应该沉淀什么"的仲裁。Consolidator 将写入决策集中到一个流程中。

关键改进：
  1. 分类本轮对话内容（闲聊/事实/情感/纠错/记住指令）
  2. 按分类决定哪些存储层需要写入，哪些跳过
  3. 避免重复写入（同轮核心事实捕获 + 长期记忆提取可能重复）
  4. 确保优先级：纠错 > 记住 > 事实 > 情感 > 闲聊跳过

用法：
  consolidator = MemoryConsolidator()
  result = consolidator.process_turn(
      device_id, session_id, user_msg, assistant_msg, memory, person_id,
  )
  # result 包含本轮写入统计
============================================================================
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.memory.identity import is_verified_person_id
from app.memory.memory_pipeline import finalize_session_memory, write_explicit_memory_request, maybe_compact_working_context
from app.memory.relationship_state import relationship_manager
from app.memory.open_loops import open_loop_manager
from app.memory.emotional_events import emotional_extractor
from app.memory.unified_store import unified_memory_store
from app.memory.schema import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryVisibility,
)
from app.session import store

logger = logging.getLogger(__name__)


# ============================
# 对话分类
# ============================


# 显式"记住"指令
_REMEMBER_INTENT = re.compile(
    r"(?:记住|别忘了|记下来|记到|写进|记牢)\s*(?::|：|)"
    r"|记得\s*(?::|：)",
)
# 纠错信号
_CORRECTION_SIGNALS = [
    "不对", "不是的", "没有", "没这回事", "你记错了", "你搞错了",
    "弄错了", "错了", "不是这样", "错了错了", "说错了",
]
# 纯社交寒暄（不应沉淀）
_CASUAL_NOISE = re.compile(
    r"^(?:在吗|在不在|你好|您好|嗨|哈喽|hello|hi|"
    r"早安|晚安|吃了吗|干嘛呢|忙吗|谢谢|多谢|好的|好哒|ok|OK|"
    r"嗯+|[哈呵]+|哦+|[啊呀]+)[。！？…~\s]*$"
    r"|^(?:没事|算了|随便|哈哈哈+|笑死|好的|好|嗯)[。！？…~\s]*$"
)
# 情感表达信号（可能值得沉淀）
_EMOTIONAL_EVENT = re.compile(
    r"烦|难过|开心|生气|害怕|激动|紧张|压力|累|困|"
    r"哭了|吵架|闹别扭|想你了|想我|想你|"
    r"面试|考试|体检|住院|手术|结果|offer|录取"
)
# 新事实声明的信号
_FACT_DECLARATION = re.compile(
    r"(?:我|我们|你|他|她|它)"
    r"(?:是|在|住|有|会|喜欢|爱|讨厌|怕|过敏|"
    r"不能|可以|想|要|曾经|以前|小时候|最近|"
    r"住在|来自|毕业于|工作在)"
    r"|(?:叫|名叫|名字)"
)


@dataclass
class TurnClassification:
    """本轮对话的分类结果。"""
    is_small_talk: bool = False          # 纯闲聊→跳过所有写入
    is_correction: bool = False          # 纠错→优先修正旧记忆
    is_remember_intent: bool = False     # 显式"记住"→写入长期记忆
    is_factual: bool = False             # 事实声明→核心记忆/长期记忆
    is_emotional: bool = False           # 情感事件→长期记忆
    is_third_party: bool = False         # 第三方人物→Entity Registry
    confidence: float = 1.0              # 分类置信度
    reason: str = ""                     # 分类原因


@dataclass
class ConsolidationResult:
    """本轮写入结果统计。"""
    classification: TurnClassification = field(default_factory=TurnClassification)
    did_compact_working_context: bool = False  # 是否触发了工作上下文→近期记忆压缩
    core_facts_saved_count: int = 0     # 核心事实写入条数
    long_term_facts_saved_count: int = 0  # 长期记忆写入条数
    corrections_applied: dict = field(default_factory=dict)  # 纠错统计
    contacts_updated: int = 0           # 第三方画像更新数
    episodes_created: int = 0           # 新摘要条数
    skipped: bool = False               # 是否完全跳过
    errors: list[str] = field(default_factory=list)
    # ── 阶段 2 风险收口：unified 写入统计 ────────────────────────────
    unified_items_written: int = 0       # 通过 unified_memory_store.write_item 写入条数
    # ── 阶段 3.0：控制台日志扩展字段 ────────────────────────────────
    open_loops_created: list[str] = field(default_factory=list)    # 新增 Open Loop 标题列表
    open_loops_resolved: list[str] = field(default_factory=list)   # 已解决 Open Loop 标题列表
    relationship_before: dict = field(default_factory=dict)        # 关系状态更新前
    relationship_after: dict = field(default_factory=dict)         # 关系状态更新后
    emotional_events_detected: list[str] = field(default_factory=list)  # 本轮检测到的情感事件标题
    quality_decision: str = ""                                      # 记忆质量门控判定结果


def classify_turn(user_msg: str, assistant_msg: str, memory: dict) -> TurnClassification:
    """分类本轮对话内容。

    Args:
        user_msg:      用户本轮消息
        assistant_msg: 助手本轮回复
        memory:        当前召回的记忆 dict

    Returns:
        TurnClassification 实例。
    """
    cls = TurnClassification()
    msg = user_msg.strip()

    # 1. 纯寒暄噪音 → 跳过所有写入
    if not msg or _CASUAL_NOISE.match(msg):
        cls.is_small_talk = True
        cls.reason = "pure_social_noise"
        return cls

    # 2. 纠错信号
    for signal in _CORRECTION_SIGNALS:
        if signal in msg:
            cls.is_correction = True
            cls.reason = f"correction_signal: {signal}"
            break

    # 3. 显式"记住"指令
    if _REMEMBER_INTENT.search(msg):
        cls.is_remember_intent = True
        cls.confidence = 0.95
        cls.reason = "explicit_remember"

    # 4. 事实声明
    if len(msg) >= 8 and _FACT_DECLARATION.search(msg):
        cls.is_factual = True
        if not cls.reason:
            cls.reason = "factual_declaration"

    # 5. 情感事件
    if _EMOTIONAL_EVENT.search(msg):
        cls.is_emotional = True
        if not cls.reason:
            cls.reason = "emotional_event"

    # 6. 第三方人物：只让明确关系或具名事实进入画像管线。
    #    单纯“你认识某某吗”只能触发检索，绝不能造成持久化写入。
    from app.memory.contacts import has_contact_admission_signal
    if has_contact_admission_signal(msg):
        cls.is_third_party = True
        if not cls.reason:
            cls.reason = "third_party_admission_signal"

    # 默认原因
    if not cls.reason:
        cls.reason = "normal_conversation"

    return cls


# ============================
# MemoryConsolidator
# ============================


class MemoryConsolidator:
    """统一写入仲裁器 —— 每轮对话后统一裁决记忆沉淀。

    替代 _post_process() 中各自为政的 6 个独立写入操作。
    每轮只做一次分类 + 一次路由，避免重复裁决。

    写入路径（按优先级）：
      1. 纠错 → memory_correction（需先修正旧记忆）
      2. 记住 → 长期记忆显式入库
      3. 事实 → 核心事实 + 长期记忆
      4. 情感 → 情景摘要 + 长期记忆
      5. 第三方 → Contact Profile
      6. 闲聊 → 只更新工作上下文，不做任何持久化
    """

    def _build_items_from_turn(
        self,
        device_id: str,
        session_id: str,
        person_id: str,
        user_msg: str,
        assistant_msg: str,
        cls: TurnClassification,
        turn_emotional_event: Any | None,
    ) -> list[MemoryItem]:
        """从本轮对话构造 MemoryItem 列表，供统一写入。

        根据对话分类生成不同类型的 MemoryItem：
          - 显式"记住" → PREFERENCE/FACT，USER_DECLARED，confidence=0.95
          - 普通事实   → FACT，USER_DECLARED，confidence=0.75
          - 高重要性情感事件 → EMOTION/MILESTONE，CONVERSATION_SUMMARY
          - 纠错       → CORRECTION，USER_DECLARED，confidence=1.0
        """
        items: list[MemoryItem] = []
        msg = user_msg.strip()
        if not msg:
            return items

        # ── 显式记住指令 ──
        if cls.is_remember_intent:
            import re as _re2
            body = _re2.sub(r'^(?:记住|记得|别忘了|记下来|记到|写进|记牢)\s*(?::|：|\s)*', '', msg).strip()
            if body:
                items.append(MemoryItem(
                    kind=MemoryKind.PREFERENCE,
                    source=MemorySource.USER_DECLARED,
                    confidence=0.95,
                    emotional_weight=3,
                    visibility=MemoryVisibility.ALWAYS,
                    content=body,
                    context={"session_id": session_id},
                ))
                items.append(MemoryItem(
                    kind=MemoryKind.FACT,
                    source=MemorySource.USER_DECLARED,
                    confidence=0.95,
                    emotional_weight=3,
                    visibility=MemoryVisibility.RECALL_ONLY,
                    content=body,
                    context={"session_id": session_id},
                ))

        # ── 纠错 ──
        if cls.is_correction:
            items.append(MemoryItem(
                kind=MemoryKind.CORRECTION,
                source=MemorySource.USER_DECLARED,
                confidence=1.0,
                emotional_weight=4,
                visibility=MemoryVisibility.ALWAYS,
                content=msg,
                context={"session_id": session_id, "assistant_reply": assistant_msg[:200]},
            ))

        # ── 高重要性情感事件 ──
        if cls.is_emotional and turn_emotional_event is not None:
            importance = getattr(turn_emotional_event, "importance", 3) or 3
            if importance >= 4:
                kind = MemoryKind.MILESTONE if importance >= 5 else MemoryKind.EMOTION
                title = str(getattr(turn_emotional_event, "title", "") or "")
                summary = str(getattr(turn_emotional_event, "summary", "") or "")
                mood = str(getattr(turn_emotional_event, "mood", "") or "")
                items.append(MemoryItem(
                    kind=kind,
                    source=MemorySource.CONVERSATION_SUMMARY,
                    confidence=0.8,
                    emotional_weight=importance,
                    visibility=MemoryVisibility.RECALL_ONLY,
                    content=f"{title}：{summary[:200]}" if title else summary[:200],
                    context={
                        "mood": mood,
                        "session_id": session_id,
                    },
                ))

        # ── 事实声明（非记住/非纠错/非闲聊） ──
        if cls.is_factual and not cls.is_remember_intent and not cls.is_correction:
            items.append(MemoryItem(
                kind=MemoryKind.FACT,
                source=MemorySource.USER_DECLARED,
                confidence=0.75,
                emotional_weight=2,
                visibility=MemoryVisibility.RECALL_ONLY,
                content=msg,
                context={"session_id": session_id},
            ))

        return items

    def process_turn(
        self,
        device_id: str,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        memory: dict,
        person_id: str | None,
    ) -> ConsolidationResult:
        """处理一轮对话后的记忆沉淀。

        Args:
            参数同 _post_process()。

        Returns:
            ConsolidationResult 包含本轮写入统计。
        """
        result = ConsolidationResult()

        # 分类本轮对话
        cls = classify_turn(user_msg, assistant_msg, memory)
        result.classification = cls

        # 访客模式：跳过所有持久化（仅工作上下文更新，已在 handle_chat 中完成）
        if not person_id or not is_verified_person_id(person_id):
            result.skipped = True
            return result

        # 纯闲聊 → 跳过持久化
        if cls.is_small_talk:
            result.skipped = True
            return result

        pid = str(person_id)

        # ══════════ 情感事件检测（提前调用，后续关系状态和持久化复用）══
        turn_emotional_event = None
        try:
            from app.memory.emotional_events import emotional_extractor as _ee
            turn_emotional_event = _ee.extract_from_turn(user_msg, assistant_msg)
            if turn_emotional_event:
                result.emotional_events_detected.append(turn_emotional_event.title)
                # 高重要性情感事件不再直接写入长期记忆，由 unified write 统一覆盖
        except Exception as exc:
            result.errors.append(f"emotional_event extraction failed: {exc}")
            logger.warning("Consolidator: %s", result.errors[-1])

        # ══════════ 关系状态更新 ══════════════════════════════════
        # 每轮更新关系状态，复用 turn_emotional_event 传入真实情绪
        try:
            state = relationship_manager.load(pid)
            state = relationship_manager.expire_old_state(state)
            result.relationship_before = state.to_dict()  # 捕获更新前状态

            mood = turn_emotional_event.mood if turn_emotional_event else ""
            intensity = turn_emotional_event.intensity if turn_emotional_event else 0.0

            state = relationship_manager.update_from_turn(
                state, user_msg, assistant_msg,
                mood=mood, intensity=intensity,
            )
            relationship_manager.save(state)
            result.relationship_after = state.to_dict()   # 捕获更新后状态
        except Exception as exc:
            result.errors.append(f"relationship_state failed: {exc}")
            logger.warning("Consolidator: %s", result.errors[-1])

        # ══════════ Open Loop 检测/创建/解决 ═════════════════════════
        try:
            # 创建：检测用户是否提到需要跟进的事
            create_candidates = open_loop_manager.detect_create(user_msg)
            for title, weight in create_candidates:
                ok = open_loop_manager.create_or_update(
                    pid, title, weight, session_id=session_id,
                )
                if ok:
                    result.open_loops_created.append(title)

            # 解决：检测用户是否提到已经搞定的事
            resolve_keywords = open_loop_manager.detect_resolve(user_msg, assistant_msg)
            for keyword in resolve_keywords:
                n = open_loop_manager.resolve(pid, keyword)
                if n > 0:
                    result.open_loops_resolved.append(keyword)
                    logger.info("Resolved %d open loop(s) for %s (keyword=%s)", n, pid, keyword)
        except Exception as exc:
            result.errors.append(f"open_loop failed: {exc}")
            logger.warning("Consolidator: %s", result.errors[-1])

        # ══════════ 质量门控判定 ══════════════════════════════════════
        # 给 monitor 提供本轮记忆质量判决摘要
        if cls.is_correction:
            result.quality_decision = "correction_flow"
        elif cls.is_small_talk:
            result.quality_decision = "skip_noise"
        elif cls.is_remember_intent:
            result.quality_decision = "store_user_request"
        elif cls.is_factual and cls.is_emotional:
            result.quality_decision = "store_factual_and_emotional"
        elif cls.is_factual:
            result.quality_decision = "store_factual"
        elif cls.is_emotional:
            result.quality_decision = "store_emotion"
        elif cls.is_third_party:
            result.quality_decision = "store_contact"
        else:
            result.quality_decision = "normal_dialogue"

        # ========== 路径 1：纠错（优先执行——先修正旧记忆再沉淀 correction item）==========
        if cls.is_correction:
            try:
                from app.memory.correction import try_apply_memory_corrections

                corr = try_apply_memory_corrections(
                    device_id, pid, session_id,
                    user_msg, assistant_msg, memory,
                )
                if corr:
                    stats = corr.get("stats") or {}
                    result.corrections_applied = {
                        "deleted_facts": stats.get("deleted_facts", 0),
                        "deleted_chunks": stats.get("deleted_chunks", 0),
                        "patched_chunks": stats.get("patched_chunks", 0),
                        "added_facts": stats.get("added_facts", 0),
                        "deleted_core_facts": stats.get("deleted_core_facts", 0),
                    }
                    logger.info(
                        "Consolidator: correction applied "
                        f"(del_fact={result.corrections_applied['deleted_facts']}, "
                        f"add_fact={result.corrections_applied['added_facts']})"
                    )
            except Exception as exc:
                result.errors.append(f"correction failed: {exc}")
                logger.warning("Consolidator: %s", result.errors[-1])

        # ══════════ 统一 MemoryItem 写入（纠错后执行）════════════════════════════
        try:
            built_items = self._build_items_from_turn(
                device_id, session_id, pid, user_msg, assistant_msg,
                cls, turn_emotional_event,
            )
            for item in built_items:
                write_id = unified_memory_store.write_item(
                    device_id, pid, item, source_session=session_id,
                )
                if write_id:
                    result.unified_items_written += 1
                    logger.debug("Consolidator: unified write %s → %s", item.kind.value, write_id)
        except Exception as exc:
            result.errors.append(f"unified_write failed: {exc}")
            logger.warning("Consolidator: %s", result.errors[-1])

        # ========== 路径 2：工作上下文→近期记忆压缩 ==========
        try:
            compressed = maybe_compact_working_context(device_id, session_id, pid)
            if compressed:
                result.did_compact_working_context = True
        except Exception as exc:
            result.errors.append(f"working_context_compaction failed: {exc}")

        # ========== 路径 3：核心记忆 + 长期记忆捕获 ==========
        if cls.is_remember_intent:
            # 记住指令：核心事实捕获仍保留（可解析额外结构化字段）；
            # capture_user_stated_facts 由 unified write 覆盖，跳过避免重复
            try:
                from app.memory.core_facts import capture_core_fact_from_message
                core_saved = capture_core_fact_from_message(device_id, pid, user_msg)
                result.core_facts_saved_count = len(core_saved)
            except Exception as exc:
                result.errors.append(f"core_fact_capture failed: {exc}")
        elif cls.is_factual or cls.is_emotional or not cls.is_small_talk:
            try:
                from app.memory.core_facts import capture_core_fact_from_message
                core_saved = capture_core_fact_from_message(device_id, pid, user_msg)
                result.core_facts_saved_count = len(core_saved)
            except Exception as exc:
                result.errors.append(f"core_fact_capture failed: {exc}")

            try:
                from app.memory.guard import capture_user_stated_facts
                capture_user_stated_facts(device_id, pid, session_id, user_msg)
            except Exception as exc:
                result.errors.append(f"user_stated_memory_capture failed: {exc}")

        # ========== 路径 4：长期记忆提取 ==========
        if not cls.is_small_talk and not cls.is_remember_intent:
            try:
                write_explicit_memory_request(device_id, session_id, user_msg, assistant_msg)
            except Exception as exc:
                result.errors.append(f"memory_extract failed: {exc}")

        # ========== 路径 5：第三方人物画像 ==========
        if cls.is_third_party:
            try:
                from app.memory.contacts import process_third_party_from_turn

                events = process_third_party_from_turn(
                    device_id, pid, user_msg, assistant_msg,
                    owner_profile=memory.get("_person_profile"),
                )
                result.contacts_updated = len(events)
            except Exception as exc:
                result.errors.append(f"contacts failed: {exc}")

        # ========== 路径 6：会话收尾时核心事实提取标记 ==========
        # （触发由 handle_session_end 负责，不在每轮处理）

        return result

    async def process_turn_async(
        self,
        device_id: str,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        memory: dict,
        person_id: str | None,
    ) -> ConsolidationResult:
        """异步版 process_turn（将同步调用包装到线程池）。

        ConsolidationResult 通过 asyncio.to_thread 执行同步存储操作，
        避免阻塞事件循环。
        """
        import asyncio

        return await asyncio.to_thread(
            self.process_turn,
            device_id, session_id,
            user_msg, assistant_msg,
            memory, person_id,
        )


# 模块级单例
consolidator = MemoryConsolidator()
