"""
情感事件抽取器 —— 从近期记忆摘要和对话中提取高重要性情感事件。

============================================================================
设计目标：

  当前问题：
  - 重要情感事件（面试焦虑、吵架、纪念日）混在近期记忆摘要里
  - 没有独立索引，只能靠文字检索，无法按情绪/重要性筛选
  - 无法回答"最近我是不是一直很烦"这类情感回顾问题

  本模块将重要情感事件从近期记忆中独立出来，每条事件有：
  - 独立的标题、摘要、情绪标签、强度
  - 关联的人物和话题
  - 关系影响分析（需要安抚/需要跟进等）

  抽取条件：
    1. 近期记忆中 importance >= 4
    2. 对话中出现高情绪强度关键词
    3. 用户明确说"这件事很重要"
    4. 特定的里程碑事件（面试、考试、吵架、旅行等）
============================================================================
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from app.memory.schema import EmotionalEvent
from app.session import store

logger = logging.getLogger(__name__)


# ── 高情绪关键词匹配 ───────────────────────────────────────────────────
# 用于从对话文本中检测潜在情感事件

_EMOTIONAL_EVENT_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    # 里程碑事件（importance 5）
    ("告白", re.compile(r"喜欢[你我他]|在一起|我爱你|做[我你]女朋友|男朋友"), 5),
    ("纪念日", re.compile(r"纪念日|\d+周年|在一起[^的]*年"), 5),
    ("求婚", re.compile(r"求婚|订婚|结婚|嫁[给我你]"), 5),

    # 重要事件（importance 4）
    ("吵架", re.compile(r"吵架|闹别扭|冷战|生气[了]?[！!]|不理[你我]"), 4),
    ("面试", re.compile(r"面试|面[试过]了|拿到offer|offer到了"), 4),
    ("考试", re.compile(r"考试|考[完试了]|成绩|分数|录取"), 4),
    ("生病", re.compile(r"生病|住院|手术|体检|发烧|去医院"), 4),
    ("旅行", re.compile(r"旅行|旅游|出去玩|度假|去[北京上海深圳广州杭州成都].*玩"), 4),
    ("搬家", re.compile(r"搬家|换房子|找房子|租房|搬[走家]"), 4),
    ("工作变动", re.compile(r"辞职|换工作|跳槽|裁员|被辞|离职|入职"), 4),
    ("离别", re.compile(r"分开|分别|离别|送[别走]|舍不得|想[你我他]了"), 4),

    # 情绪事件（importance 3-4）
    ("焦虑", re.compile(r"好焦虑|压力好大|好紧张|睡不着|失眠|担心"), 4),
    ("开心", re.compile(r"好开心|太开心[了]?|幸福|快乐|高兴坏了|激动"), 3),
    ("难过", re.compile(r"好难过|伤心|哭了|哭[了]?[。！]|难过[了]?[。！]"), 4),
    ("失落", re.compile(r"失落|沮丧|郁闷|emo|不开心|没意思"), 3),

    # 关系事件（importance 4）
    ("误解", re.compile(r"误会|误解|错怪|冤枉|不信任"), 4),
    ("感动", re.compile(r"好感动|感动[坏死了]?|[你他]真好|最[好棒]了"), 3),
]

# 情感事件关键词（用于从近期记忆摘要中识别情感事件）
_RECENT_MEMORY_EMOTIONAL_KEYWORDS = {
    "焦虑", "紧张", "开心", "难过", "生气", "害怕", "激动",
    "面试", "考试", "吵架", "旅行", "生病", "搬家",
    "纪念日", "告白", "分手", "吵架", "冷战", "出差",
}

# 关系影响分析关键词
_RELATIONSHIP_IMPACT_PATTERNS = {
    "需要安抚": ["焦虑", "难过", "伤心", "哭了", "压力", "紧张", "不安"],
    "需要祝贺": ["开心", "激动", "高兴", "幸福", "通过", "录取", "考上"],
    "需要关心": ["生病", "住院", "手术", "体检", "搬家", "失眠"],
    "需要跟进": ["面试", "考试", "offer", "结果", "通知", "安排"],
    "需要修复": ["吵架", "冷战", "误会", "误解", "生气", "闹别扭"],
}


class EmotionalEventExtractor:
    """情感事件抽取器。"""

    # ── 从近期记忆摘要抽取 ──────────────────────────────────────────

    def extract_from_recent_memory(self, recent_row: dict) -> EmotionalEvent | None:
        """从单条近期记忆摘要行中抽取情感事件。

        Args:
            recent_row: 近期记忆行

        Returns:
            如果检测到重要情感事件返回 EmotionalEvent，否则返回 None。
        """
        importance = int(recent_row.get("importance", 3) or 3)
        if importance < 4:
            return None

        summary = str(recent_row.get("summary", "")).strip()
        if not summary:
            return None

        emotion_raw = str(recent_row.get("emotion", "") or "")
        mood = ""
        intensity = 0.0
        if emotion_raw:
            try:
                emo = json.loads(emotion_raw)
                if isinstance(emo, dict):
                    mood = str(emo.get("mood", ""))
                    intensity = float(emo.get("intensity", 0))
            except (json.JSONDecodeError, TypeError):
                pass

        people_raw = str(recent_row.get("people", "[]") or "[]")
        try:
            people = json.loads(people_raw) if people_raw.startswith("[") else []
        except (json.JSONDecodeError, TypeError):
            people = []

        # 识别事件标题
        title = self._extract_title(summary, mood)

        # 分析关系影响
        impact = self._analyze_relationship_impact(summary, mood)

        return EmotionalEvent(
            person_id=str(recent_row.get("person_id", "")),
            date=str(recent_row.get("created_at", ""))[:10],
            title=title,
            summary=summary[:200],
            mood=mood,
            intensity=intensity,
            people=people,
            importance=importance,
            relationship_impact=impact,
            follow_up_needed=("需要跟进" in impact or "需要关心" in impact),
            source_recent_id=recent_row.get("id"),
        )

    def extract_all_from_recent_memory(self, person_id: str, limit: int = 30) -> list[EmotionalEvent]:
        """从用户的近期记忆摘要中批量抽取情感事件。

        Args:
            person_id: 用户 ID
            limit:     最多检查的近期记忆条数

        Returns:
            情感事件列表，按重要性降序。
        """
        events: list[EmotionalEvent] = []

        # 获取活跃近期记忆
        from app.session import store
        # 用 list_important_episodes 获取高重要性事件
        important_rows = store.list_important_episodes(person_id, min_importance=4, limit=limit)
        for row in important_rows:
            event = self.extract_from_recent_memory(row)
            if event:
                events.append(event)

        # 也检查近期记忆（低重要性但高情绪强度的）
        recent_rows = store.list_active_recent_memory("", person_id, limit=limit)
        for row in recent_rows:
            imp = int(row.get("importance", 3) or 3)
            if imp >= 4:
                continue  # 已在上面处理过
            emotion_raw = str(row.get("emotion", "") or "")
            if emotion_raw and ("焦虑" in emotion_raw or "紧张" in emotion_raw or "开心" in emotion_raw):
                event = self.extract_from_recent_memory(row)
                if event:
                    events.append(event)

        # 按重要性 + 时间排序
        events.sort(key=lambda e: (e.importance * 2 + (1 if e.follow_up_needed else 0)), reverse=True)
        return events[:10]

    # ── 从对话轮次抽取 ──────────────────────────────────────────────

    def extract_from_turn(self, user_msg: str, assistant_msg: str) -> EmotionalEvent | None:
        """从单轮对话中检测情感事件。

        Args:
            user_msg:      用户消息
            assistant_msg: 机器人回复

        Returns:
            如果检测到情感事件返回 EmotionalEvent，否则返回 None。
        """
        for title, pattern, importance in _EMOTIONAL_EVENT_PATTERNS:
            if pattern.search(user_msg) or pattern.search(assistant_msg):
                mood = ""
                if title in ("焦虑", "压力"):
                    mood = "焦虑"
                elif title in ("开心", "感动"):
                    mood = "开心"
                elif title in ("难过", "失落"):
                    mood = "难过"
                elif title in ("吵架",):
                    mood = "生气"

                impact = self._analyze_relationship_impact(user_msg, mood)

                return EmotionalEvent(
                    person_id="",
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    title=title,
                    summary=self._summarize_turn(user_msg, assistant_msg)[:200],
                    mood=mood,
                    intensity=4.0 / 5.0 if importance >= 4 else 0.6,
                    importance=importance,
                    relationship_impact=impact,
                    follow_up_needed=("需要跟进" in impact),
                )

        return None

    # ── 格式化 ──────────────────────────────────────────────────────

    def format_for_prompt(self, events: list[EmotionalEvent], max_events: int = 5) -> str:
        """将情感事件列表格式化为人类化的 prompt 块。

        Args:
            events:     情感事件列表
            max_events: 最多显示条数

        Returns:
            格式化文本，无事件时返回空字符串。
        """
        if not events:
            return ""

        lines: list[str] = ["## 你们之间的重要事情"]
        for ev in events[:max_events]:
            date_tag = f"（{ev.date}）" if ev.date else ""
            mood_tag = f" · {ev.mood}" if ev.mood else ""
            imp_tag = " ❤️" if ev.importance >= 5 else ""
            lines.append(f"- {ev.title}：{ev.summary[:100]}{date_tag}{mood_tag}{imp_tag}")

        return "\n".join(lines)

    # ── 内部方法 ────────────────────────────────────────────────────

    @staticmethod
    def _extract_title(summary: str, mood: str) -> str:
        """从摘要中提取事件标题。"""
        # 尝试匹配已知模式
        for keyword in _RECENT_MEMORY_EMOTIONAL_KEYWORDS:
            if keyword in summary[:50]:
                if keyword in ("焦虑", "难过", "生气", "开心", "紧张"):
                    return f"情绪波动（{keyword}）"
                return keyword

        if mood:
            return f"情感体验（{mood}）"

        return "重要事件"

    @staticmethod
    def _analyze_relationship_impact(text: str, mood: str) -> str:
        """分析事件对关系的影响。"""
        impacts: list[str] = []
        for impact, keywords in _RELATIONSHIP_IMPACT_PATTERNS.items():
            for kw in keywords:
                if kw in text or kw in mood:
                    impacts.append(impact)
                    break

        return "；".join(impacts[:3]) if impacts else ""

    @staticmethod
    def _summarize_turn(user_msg: str, assistant_msg: str) -> str:
        """从对话轮次生成简短摘要。"""
        # 取用户消息的主要内容
        msg = user_msg.strip()
        # 去除纯助词
        msg = re.sub(r"^(嗯|哦|好|嗯嗯|好的|是啊|对呀)\s*", "", msg)
        if len(msg) > 60:
            msg = msg[:58] + "…"
        return msg


# 模块级单例
emotional_extractor = EmotionalEventExtractor()
