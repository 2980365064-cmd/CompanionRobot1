"""
记忆编排器（Memory Orchestrator）—— 将底层记忆转化为产品级 MemoryPackV2。

============================================================================
设计目标：
  消除 核心事实/工作上下文/近期记忆/长期记忆/Profile/Emotion/Contacts 等工程分层在 agent prompt
  中的暴露，将多层记忆统一为 MemoryPackV2（基于 MemoryItem 语义）。

  返回的 MemoryPackV2 包含：
  - items:           MemoryItem 列表（统一语义）
  - relationship:    关系状态
  - current_mood/topic: 当前场景
  - missing_memory:  缺失记忆指引
  - history:         近期对话历史（替代旧的 working）
  - diagnostics:     产品语义诊断字段
============================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.memory.router import memory_router
from app.memory.emotion import emotion_trajectory
from app.memory.open_loops import open_loop_manager
from app.memory.relationship_state import relationship_manager
from app.memory.schema import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryVisibility,
    MemoryPackV2,
    RelationshipState as SchemaRelationshipState,
    EmotionalEvent,
)
from app.memory.emotional_events import emotional_extractor

logger = logging.getLogger(__name__)


# ============================
# Memory Orchestrator
# ============================


class MemoryOrchestrator:
    """记忆编排器 —— 协调底层各记忆层，输出 MemoryPackV2。

    职责：
    1. 调用 memory_router.recall() 获取原始工程数据
    2. 调用 emotion_trajectory() 获取情感轨迹
    3. 将核心事实/近期记忆/长期记忆等统一转换为 MemoryPackV2
    4. 对 prompt 屏蔽所有工程分层名
    """

    # 负面情绪标签集
    _NEGATIVE_MOODS = {"低落", "焦虑", "难过", "生气", "烦躁", "疲惫", "害怕", "伤心"}

    def _classify_emotional_climate(self, emo_traj: list[dict]) -> str:
        """基于最近情感轨迹判断近期情感气氛。"""
        if not emo_traj:
            return ""
        recent_moods = [e.get("mood", "") for e in emo_traj[:3]]
        neg_count = sum(1 for m in recent_moods if m in self._NEGATIVE_MOODS)
        if neg_count >= 2:
            return "近期情绪持续偏负面，需要多倾听共情"
        if neg_count == 1:
            return "上次对话情绪偏低，需要留意"
        return "整体感觉还可以"

    def _build_v2_pack(
        self,
        memory: dict,
        emo_traj: list[dict],
        person_id: str | None,
        interlocutor_mode: str,
        identity_hint: str,
        query: str,
    ) -> MemoryPackV2:
        """从语义 memory dict 构建 MemoryPackV2。

        仅读取
        memory["items"]/memory["history"]/memory["diagnostics"]。
        """
        # ── 1. 直接从语义 items 列表构建（已含 core/recent/long_term/related） ──
        items: list[MemoryItem] = list(memory.get("items") or [])

        # ── 2. 状态寄存器：关系状态 + 待跟进事项 ──
        tone = interlocutor_mode or "girlfriend"
        guest_mode = bool(memory.get("guest_mode"))
        if guest_mode:
            schema_rs = SchemaRelationshipState(person_id=person_id or "", mode=tone)
        else:
            items.append(MemoryItem(
                kind=MemoryKind.RELATIONSHIP, source=MemorySource.SYSTEM,
                confidence=1.0, emotional_weight=3,
                content=f"和她之间的关系是{tone}",
            ))
            try:
                schema_rs = relationship_manager.expire_old_state(
                    relationship_manager.load(person_id or "")
                )
                if not schema_rs.mode:
                    schema_rs.mode = tone
            except Exception:
                schema_rs = SchemaRelationshipState(
                    person_id=person_id or "",
                    mode=tone,
                    recent_mood=self._classify_emotional_climate(emo_traj),
                    recent_attitude="正常",
                    relationship_temperature=0.7 if tone == "girlfriend" else 0.3,
                )
            if not schema_rs.recent_mood:
                schema_rs.recent_mood = self._classify_emotional_climate(emo_traj)
            try:
                schema_rs.open_loops = open_loop_manager.list_relevant(person_id or "", query)
            except Exception:
                schema_rs.open_loops = []

            # ── 3. 情感事件 → MemoryItem ──
            try:
                emotional_events = emotional_extractor.extract_all_from_recent_memory(
                    person_id or "", limit=20,
                )
                for ev in emotional_events:
                    kind = MemoryKind.MILESTONE if ev.importance >= 5 else MemoryKind.EMOTION
                    items.append(MemoryItem(
                        kind=kind, source=MemorySource.CONVERSATION_SUMMARY,
                        confidence=0.8, emotional_weight=ev.importance,
                        visibility=MemoryVisibility.RECALL_ONLY,
                        content=f"{ev.title}：{ev.summary[:100]}",
                        context={
                            "mood": ev.mood, "intensity": ev.intensity,
                            "people": ev.people, "date": ev.date,
                        },
                    ))
            except Exception as exc:
                logger.exception("情感事件抽取失败（不影响主流程）: %s", exc)

        # ── 4. 缺失记忆判定（evidence-aware）──
        memory_miss_val = memory.get("memory_miss", False)
        diag = memory.get("diagnostics") or {}

        evidence_count = diag.get("evidence_count", len(items))
        evidence_weak = diag.get("evidence_weak", False)
        evidence_sources = diag.get("evidence_sources", [])
        needs_memory = (
            diag.get("retrieval_plan", {}).get("needs_memory", bool(items))
            if isinstance(diag.get("retrieval_plan"), dict)
            else bool(items)
        )

        miss_lv = 0
        if memory_miss_val:
            # router 判定为完全未命中
            miss_lv = 2
        elif not items and needs_memory and evidence_count == 0:
            # 无 items 且无 evidence 但查询需要记忆 → 完全缺失
            miss_lv = 2
        elif evidence_weak and needs_memory:
            # 有少量 evidence 但标记为弱信号 → 部分不确定
            miss_lv = 1
        elif items and needs_memory:
            # 有有效 evidence → 正常（不强制说"不确定"）
            miss_lv = 0
        # else: 闲聊/寒暄/no items and no needs → miss_lv=0

        missing_memory = {"should_admit_unknown": False, "reason": ""}
        if miss_lv == 2:
            missing_memory = {
                "should_admit_unknown": True,
                "reason": "完全未命中记忆 —— 诚实说不太记得，追问对方补充",
            }
        elif miss_lv == 1:
            missing_memory = {
                "should_admit_unknown": False,
                "reason": "证据较弱 —— 有部分线索但不完全确定，坦诚表达不确定即可",
            }

        # ── 5. 当前场景 ──
        history = memory.get("history", [])
        current_topic = ""
        user_msgs = [
            m.get("content", "") for m in (history or [])[-4:]
            if m.get("role") == "user"
        ]
        for msg in reversed(user_msgs):
            if len(msg.strip()) >= 4:
                current_topic = msg.strip()[:80]
                break

        current_mood = ""
        if emo_traj:
            latest = emo_traj[0]
            mood = str(latest.get("mood", "")).strip()
            if mood:
                current_mood = f"{mood}（{latest.get('intensity', 0)}）"

        # ── 6. diagnostics（产品语义字段） ──
        diagnostics = {
            "person_id": person_id,
            "interlocutor_mode": interlocutor_mode,
            "identity_hint": identity_hint,
            "memory_miss": miss_lv,
            "has_recent": diag.get("has_recent", False),
            "has_long_term": diag.get("has_long_term", False),
            "core_memory_count": diag.get("core_memory_count", 0),
            "recent": diag.get("recent") or [],
            "long_term": diag.get("long_term") or [],
            "related": diag.get("related") or [],
            "month_key": diag.get("month_key", ""),
            "retrieval_plan": diag.get("retrieval_plan", {}),
            "evidence_count": evidence_count,
            "evidence_weak": evidence_weak,
            "evidence_sources": evidence_sources,
        }
        # 透传 recall_mode（如 fast）
        if diag.get("recall_mode"):
            diagnostics["recall_mode"] = diag["recall_mode"]

        # ── 7. 构建 MemoryPackV2 ──
        pack = MemoryPackV2(
            items=items,
            relationship=schema_rs,
            current_mood=current_mood,
            current_topic=current_topic,
            guest_mode=bool(memory.get("guest_mode")),
            missing_memory=missing_memory,
            history=list(history),
            diagnostics=diagnostics,
        )
        return pack

    def recall(
        self,
        device_id: str,
        session_id: str,
        query: str,
        *,
        person_id: str | None = None,
    ) -> MemoryPackV2:
        """执行完整记忆编排，返回 MemoryPackV2。

        Args:
            参数同 memory_router.recall()。

        Returns:
            MemoryPackV2 实例，封装产品级记忆数据。
        """
        # Step 1: 调用底层 router 获取原始工程数据
        memory = memory_router.recall(
            device_id, session_id, query, person_id=person_id,
        )

        # Step 2: 获取情感轨迹
        pid = str(person_id or memory.get("person_id") or "").strip()
        emo_traj = (
            emotion_trajectory(device_id, pid)
            if pid and not memory.get("guest_mode")
            else []
        )

        # Step 3: 构建 MemoryPackV2
        interlocutor_mode = str(memory.get("interlocutor_mode") or "girlfriend")
        identity_hint = str(memory.get("identity_hint") or "")
        pack = self._build_v2_pack(
            memory, emo_traj, pid, interlocutor_mode, identity_hint, query,
        )
        return pack

    def recall_fast(
        self,
        device_id: str,
        session_id: str,
        query: str,
        *,
        person_id: str | None = None,
    ) -> MemoryPackV2:
        """快速记忆编排：跳过情感事件抽取、关系图和关联扩展。

        用于语音场景低延迟首响。MemoryPackV2 中的 related items
        在 fast 路径中为空，emotional_events 也为空。
        """
        memory = memory_router.recall_fast(
            device_id, session_id, query, person_id=person_id,
        )
        pid = str(person_id or memory.get("person_id") or "").strip()
        interlocutor_mode = str(memory.get("interlocutor_mode") or "girlfriend")
        identity_hint = str(memory.get("identity_hint") or "")

        # fast 路径情感轨迹为空，不触发情感事件遍历
        pack = self._build_v2_pack(
            memory, [], pid, interlocutor_mode, identity_hint, query,
        )
        # 标记 fast 路径
        pack.diagnostics["recall_mode"] = "fast"
        return pack


# 模块级单例，供 agent 模块调用
orchestrator = MemoryOrchestrator()
