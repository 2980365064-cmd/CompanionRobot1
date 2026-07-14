"""
关系状态持久化管理器 —— 管理每轮对话后的关系状态更新与存储。

============================================================================
设计目标：

  情感陪伴型机器人最重要的不是知道很多事实，而是知道：
  - 最近你们关系怎么样
  - 她最近情绪如何
  - 她最近是不是压力大
  - 她最近对你依赖、调侃、冷淡，还是在求助

  本模块负责：
    1. 读取当前用户的关系状态
    2. 每轮根据情绪、态度、open loop 更新状态
    3. 给 Orchestrator 提供稳定的 RelationshipState

  核心原则：
    - 所有状态有有效期，不把短期情绪当成长期人格
    - 负面情绪连续出现 2 次 → recent_mood 标记为需要留意
    - 用户表达依赖/求助 → recent_attitude=依赖/求助
    - 用户纠错或不满 → 关系温度短期下降，但不进入永久画像
    - 用户开心/亲密互动 → 关系温度回升
============================================================================
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.memory.schema import RelationshipState, EmotionalEvent
from app.session import store

logger = logging.getLogger(__name__)


# ── 配置 ───────────────────────────────────────────────────────────────────
# 状态过期时间
_MOOD_EXPIRE_HOURS = 48    # 情绪标签 48 小时过期
_ATTITUDE_EXPIRE_HOURS = 24  # 态度标签 24 小时过期
_TEMP_RECOVER_DELTA = 0.1   # 单次正面互动关系温度回升量
_TEMP_DROP_DELTA = 0.15     # 单次负面互动关系温度下降量
_TEMP_MIN = 0.1             # 最低关系温度
_TEMP_MAX = 1.0             # 最高关系温度
_CARE_MAX = 5               # 关心点最多保留 5 条
_AVOID_MAX = 3              # 避雷点最多保留 3 条

# 负面情绪词
_NEGATIVE_MOODS = {"低落", "焦虑", "难过", "生气", "烦躁", "疲惫", "害怕", "伤心", "郁闷", "沮丧", "紧张", "不安"}

# 正面情绪词
_POSITIVE_MOODS = {"开心", "高兴", "快乐", "幸福", "甜蜜", "激动", "期待", "轻松", "放松", "温暖", "安心"}

# 态度关键词
_ATTITUDE_DEPENDENT = {"想你了", "想你", "想我", "需要你", "你在就好了", "好想你", "抱抱", "陪陪我"}
_ATTITUDE_HELPLESS = {"帮帮我", "怎么办", "不知道该怎么办", "好烦", "好难", "救命"}
_ATTITUDE_CORRECTION = {"不对", "不是", "错了", "说错了", "你记错了", "你搞错了", "弄错了", "你这样不对"}
_ATTITUDE_INTIMATE = {"爱你", "最喜欢你", "你最好", "你最好了", "你真好", "亲亲", "么么"}
_ATTITUDE_COLD = {"随便", "无所谓", "不想说", "别烦我", "不想理你", "算了吧"}


class RelationshipStateManager:
    """关系状态管理器 —— 持久化读写 + 每轮更新。

    使用方法：
        manager = RelationshipStateManager()
        state = manager.load(person_id)        # 读取当前状态
        manager.update(state, user_msg, assistant_msg)  # 更新状态
        manager.save(state)                     # 持久化存储
    """

    # ── 读取 ──────────────────────────────────────────────────────────

    def load(self, person_id: str) -> RelationshipState:
        """加载用户的关系状态（从 DB 或返回默认）。"""
        if not person_id:
            return RelationshipState()

        raw = store.get_relationship_state(person_id)
        if raw:
            try:
                return RelationshipState.from_dict(json.loads(raw))
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("RelationshipState parse failed for %s: %s", person_id, exc)

        return RelationshipState(person_id=person_id)

    # ── 存储 ──────────────────────────────────────────────────────────

    def save(self, state: RelationshipState) -> None:
        """持久化关系状态。"""
        if not state.person_id:
            return
        state.last_updated_at = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(state.to_dict(), ensure_ascii=False)
        store.save_relationship_state(state.person_id, payload)

    # ── 更新 ──────────────────────────────────────────────────────────

    def update_from_turn(
        self,
        state: RelationshipState,
        user_msg: str,
        assistant_msg: str,
        mood: str = "",
        intensity: float = 0.0,
    ) -> RelationshipState:
        """每轮对话后更新关系状态。

        Args:
            state:         当前关系状态
            user_msg:      用户本轮消息
            assistant_msg: 机器人本轮回复
            mood:          本轮情绪标签（已由 emotion 模块提取）
            intensity:     情绪强度

        Returns:
            更新后的 RelationshipState。
        """
        if not state or not state.person_id:
            return state

        msg = user_msg.strip().lower()

        # ── 1. 更新情绪趋势 ─────────────────────────────────────
        if mood and mood in _NEGATIVE_MOODS:
            state.recent_mood = f"需要留意的负面情绪（{mood}）"
        elif mood and mood in _POSITIVE_MOODS:
            state.recent_mood = f"情绪不错（{mood}）"
        # 如果没有明确情绪标签但文本含负面词
        elif any(w in msg for w in ["烦", "累", "困", "难过", "生气", "焦虑", "压力"]):
            state.recent_mood = "近期情绪偏负面，需要多倾听共情"

        # ── 2. 更新态度 ─────────────────────────────────────────
        if any(s in msg for s in _ATTITUDE_CORRECTION):
            state.recent_attitude = "纠错/不满"
            state.relationship_temperature = max(
                _TEMP_MIN, state.relationship_temperature - _TEMP_DROP_DELTA
            )
        elif any(s in msg for s in _ATTITUDE_COLD):
            state.recent_attitude = "冷淡"
            state.relationship_temperature = max(
                _TEMP_MIN, state.relationship_temperature - _TEMP_DROP_DELTA
            )
        elif any(s in msg for s in _ATTITUDE_INTIMATE):
            state.recent_attitude = "亲密"
            state.relationship_temperature = min(
                _TEMP_MAX, state.relationship_temperature + _TEMP_RECOVER_DELTA
            )
        elif any(s in msg for s in _ATTITUDE_DEPENDENT):
            state.recent_attitude = "依赖/需要陪伴"
            state.relationship_temperature = min(
                _TEMP_MAX, state.relationship_temperature + _TEMP_RECOVER_DELTA
            )
        elif any(s in msg for s in _ATTITUDE_HELPLESS):
            state.recent_attitude = "求助"
        elif mood == "开心" or mood == "高兴":
            state.recent_attitude = "正常/积极"

        # ── 3. 更新关心点 ─────────────────────────────────────────
        # 从对话中提取需要关心的事项，添加到 care_points
        care_candidates = self._extract_care_points(user_msg)
        for point in care_candidates:
            if point not in state.care_points:
                state.care_points.append(point)
        state.care_points = state.care_points[-_CARE_MAX:]

        # ── 4. 更新避雷点 ─────────────────────────────────────────
        # 如果用户纠错或表达了不喜欢，记录避雷话题
        if state.recent_attitude in ("纠错/不满", "冷淡"):
            avoid_topic = self._extract_avoid_topic(user_msg)
            if avoid_topic and avoid_topic not in state.avoid_topics_recent:
                state.avoid_topics_recent.append(avoid_topic)
        state.avoid_topics_recent = state.avoid_topics_recent[-_AVOID_MAX:]

        return state

    def expire_old_state(self, state: RelationshipState) -> RelationshipState:
        """清理过期状态（不把短期情绪固化）。

        规则：
        - mood 标签超过 48 小时未更新 → 清空
        - attitude 标签超过 24 小时未更新 → 清空
        - 睡眠/休息后关系温度自然回升 0.05
        """
        if not state.last_updated_at:
            return state

        try:
            last = datetime.fromisoformat(state.last_updated_at)
            now = datetime.now(timezone.utc)
            hours = (now - last).total_seconds() / 3600

            if hours > _MOOD_EXPIRE_HOURS:
                if state.recent_mood:
                    logger.debug("Expired mood for %s (%dh+)", state.person_id, _MOOD_EXPIRE_HOURS)
                    state.recent_mood = ""

            if hours > _ATTITUDE_EXPIRE_HOURS:
                if state.recent_attitude:
                    logger.debug("Expired attitude for %s (%dh+)", state.person_id, _ATTITUDE_EXPIRE_HOURS)
                    state.recent_attitude = ""

            # 长时间无互动，关系温度自然回升（时间可以消弭小摩擦）
            if hours > 12 and state.relationship_temperature < 0.5:
                state.relationship_temperature = min(
                    0.5, state.relationship_temperature + 0.05
                )

        except (ValueError, TypeError):
            pass

        return state

    # ── 辅助方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_care_points(user_msg: str) -> list[str]:
        """从用户消息中提取需要关心的点。

        匹配模式：
        - "面试/考试/体检/住院/手术/结果" 等关键词
        - "要/准备/打算/计划" + 具体事项
        """
        points: list[str] = []
        msg = user_msg.strip()

        # 关键词触发
        care_keywords = {
            "面试": "面试进展",
            "考试": "考试情况",
            "体检": "体检结果",
            "住院": "身体情况",
            "手术": "手术情况",
            "offer": "offer 进展",
            "录取": "录取结果",
            "结果": "结果进展",
            "实习": "实习情况",
            "搬家": "搬家进展",
            "出差": "出差情况",
        }
        for keyword, point in care_keywords.items():
            if keyword in msg and point not in points:
                points.append(point)

        return points[:3]

    @staticmethod
    def _extract_avoid_topic(user_msg: str) -> str:
        """从纠错/冷淡消息中提取应避雷的话题。

        简单策略：取消息中最长的非停用词短语（假设为主话题）。
        """
        msg = user_msg.strip()
        if not msg:
            return ""
        # 去除纠错前缀
        for prefix in ("不对", "不是", "错了", "弄错了", "你记错了", "算了", "不想说", "别烦"):
            idx = msg.find(prefix)
            if idx >= 0:
                topic = msg[idx + len(prefix):].strip()
                if topic:
                    return topic[:30]
        return msg[:30]


# 模块级单例
relationship_manager = RelationshipStateManager()
