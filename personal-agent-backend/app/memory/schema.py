"""
记忆统一语义层 —— 定义所有记忆操作的统一数据结构和类型。

============================================================================
设计目标：

  本模块定义一套统一的数据结构，让所有记忆操作都基于相同语义类型：
  - MemoryItem：可注入 prompt 的最基本记忆单元（带 kind/source/confidence）
  - MemoryKind：记忆语义类型（事实/情绪/关系/偏好/禁忌/待办等）
  - MemoryPackV2：基于 MemoryItem 的智能记忆包，按场景选择注入内容

  后续所有模块（orchestrator/relationship_state/open_loops/emotional_events）
  都基于此 schema，不做重复定义。
============================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# 枚举类型
# ══════════════════════════════════════════════════════════════════════════════


class MemoryKind(str, Enum):
    """记忆语义类型 —— 区分"这是什么类型的记忆"。

    每个 MemoryItem 必须指定一种语义类型，便于按场景选择。
    类型粒度的原则：不多到无法推理，不少到无法区分。
    """
    FACT = "fact"               # 事实（"她住在北京"）
    EPISODE = "episode"         # 事件/经历（"上次去杭州的事"）
    EMOTION = "emotion"         # 情绪状态（"她最近很焦虑"）
    RELATIONSHIP = "relationship"  # 关系状态（"你们是恋人关系"）
    PREFERENCE = "preference"   # 偏好（"她喜欢喝奶茶"）
    TABOO = "taboo"             # 禁忌（"别叫大炮"）
    OPEN_LOOP = "open_loop"     # 待跟进（"等面试结果"）
    ENTITY = "entity"           # 实体信息（"唐凯是她的初中同学"）
    WIKI = "wiki"               # 外部语料/知识页（persona corpus / monthly / people）
    CORRECTION = "correction"   # 纠错记录（"她不在上海，在北京"）
    MILESTONE = "milestone"     # 里程碑（"第一次说喜欢你"）

    @classmethod
    def from_core_fact_category(cls, category: str) -> "MemoryKind":
        """将核心事实类别映射为 MemoryKind。"""
        mapping = {
            "identity": MemoryKind.RELATIONSHIP,
            "taboo": MemoryKind.TABOO,
            "preference": MemoryKind.PREFERENCE,
            "milestone": MemoryKind.MILESTONE,
            "key_people": MemoryKind.ENTITY,
        }
        return mapping.get(category, MemoryKind.FACT)


class MemorySource(str, Enum):
    """记忆来源 —— 标注"这条记忆从哪来的"。

    用于：
    - 置信度判断（user_declared > conversation_summary > inferred）
    - 回溯审计
    - 自动修正时删除特定来源的记忆
    """
    USER_DECLARED = "user_declared"            # 用户明确告知
    CONVERSATION_SUMMARY = "conversation_summary"  # 近期记忆摘要提取
    WIKI = "wiki"                              # persona 语料导入
    MANUAL = "manual"                          # 手动管理/API 操作
    CORRECTION = "correction"                  # 纠错修正
    INFERRED = "inferred"                      # 系统推断（低置信）
    SYSTEM = "system"                          # 系统默认


class MemoryVisibility(str, Enum):
    """记忆可见度 —— 控制"什么情况下这条记忆可以被召回"。

    用于隐私保护和语境控制：
    - always：始终可注入（如关系身份事实）
    - recall_only：仅主动召回时可用（不自动注入）
    - private：仅私密语境可用（如亲密内容）
    - intimate：仅亲密关系语境可用（如特别私密的信息）
    """
    ALWAYS = "always"
    RECALL_ONLY = "recall_only"
    PRIVATE = "private"
    INTIMATE = "intimate"


# ══════════════════════════════════════════════════════════════════════════════
# 核心数据结构
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryItem:
    """统一记忆单元 —— 所有记忆操作的核心数据类型。

    设计原则：
    1. 每个 MemoryItem 只有一种语义类型（kind），不混杂。
    2. 每条 MemoryItem 必须有来源（source）和置信度（confidence）。
    3. 情感重要性（emotional_weight）独立于语义类型，允许
       一条事实记忆有高情感权重（如重要纪念日）。
    4. 可见度（visibility）控制隐私和语境边界。

    Args:
        kind:                语义类型
        source:              来源
        confidence:          置信度 0.0-1.0
        emotional_weight:    情感重要性 1-5
        recency_weight:      时效重要性 1-5
        visibility:          可见度
        content:             核心文本内容（不含机器前缀）
        context:             额外上下文（人物、场景、日期等）
        source_id:           来源记录 ID（如核心事实行id、近期记忆episode id）
        tags:                自定义标签
        created_at:          创建时间（ISO UTC）
        expires_at:          过期时间（None=永不过期）
    """
    kind: MemoryKind
    source: MemorySource
    confidence: float = 1.0
    emotional_weight: int = 3
    recency_weight: int = 3
    visibility: MemoryVisibility = MemoryVisibility.ALWAYS

    content: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    source_id: str = ""
    tags: list[str] = field(default_factory=list)

    created_at: str = ""
    expires_at: str | None = None

    def __post_init__(self) -> None:
        """初始化后自动赋值默认时间戳。"""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.emotional_weight = max(1, min(5, self.emotional_weight))
        self.recency_weight = max(1, min(5, self.recency_weight))

    # ── 简化属性 ──────────────────────────────────────────────────────

    @property
    def is_expired(self) -> bool:
        """是否已过期（expires_at 不为 None 且已过当前时间）。"""
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return exp < datetime.now(timezone.utc)
        except (ValueError, TypeError):
            return False

    @property
    def is_high_confidence(self) -> bool:
        """是否高置信（>= 0.8）。"""
        return self.confidence >= 0.8

    @property
    def is_high_emotion(self) -> bool:
        """是否高情感权重（>= 4）。"""
        return self.emotional_weight >= 4

    @property
    def is_milestone(self) -> bool:
        """是否为里程碑事件。"""
        return self.kind == MemoryKind.MILESTONE or (
            self.kind == MemoryKind.EPISODE and self.emotional_weight >= 4
        )

    @property
    def humanized_text(self) -> str:
        """生成人类化可读文本（去除机器前缀）。

        用于 prompt 注入前的最终清洗。
        """
        text = self.content
        # 去除 [人物: xxx] 类机器前缀
        import re
        text = re.sub(r'\[[^\]]+\]\s*', '', text).strip()
        return text

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于存储/传输）。"""
        return {
            "kind": self.kind.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "emotional_weight": self.emotional_weight,
            "recency_weight": self.recency_weight,
            "visibility": self.visibility.value,
            "content": self.content,
            "context": self.context,
            "source_id": self.source_id,
            "tags": self.tags,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        """从字典反序列化。"""
        return cls(
            kind=MemoryKind(data.get("kind", "fact")),
            source=MemorySource(data.get("source", "inferred")),
            confidence=float(data.get("confidence", 1.0)),
            emotional_weight=int(data.get("emotional_weight", 3)),
            recency_weight=int(data.get("recency_weight", 3)),
            visibility=MemoryVisibility(data.get("visibility", "always")),
            content=str(data.get("content", "")),
            context=data.get("context", {}),
            source_id=str(data.get("source_id", "")),
            tags=list(data.get("tags", [])),
            created_at=str(data.get("created_at", "")),
            expires_at=data.get("expires_at"),
        )

    # ── 工厂方法（已清空）─────────────────────────────────────────




@dataclass
class RelationshipState:
    """关系状态 —— 你和对方的稳定关系，每轮小体积注入。

    设计目标：
    1. 不把临时情绪当成长期人格
    2. 所有状态有有效期，自然过期后被替换
    3. 区分"关系温度"（短期可波动）和"关系事实"（稳定不变）

    Fields:
        person_id:           用户 ID
        mode:                关系模式（girlfriend/visitor/friend）
        recent_mood:         最近情绪趋势描述
        recent_attitude:     最近对机器人态度趋势
        relationship_temperature: 关系温度 0.0-1.0
        care_points:         需要关心的重点事项
        avoid_topics_recent: 最近不主动提的话题
        open_loops:          待跟进事项
        last_updated_at:     最后更新时间
    """
    person_id: str = ""
    mode: str = "girlfriend"

    # 近期情感/态度（短期窗口，7-14 天）
    recent_mood: str = ""
    recent_attitude: str = ""
    relationship_temperature: float = 0.5

    # 关心点/避雷（由 Consolidator 每轮更新）
    care_points: list[str] = field(default_factory=list)
    avoid_topics_recent: list[str] = field(default_factory=list)

    # 待跟进事项（由 Open Loop 模块管理）
    open_loops: list[dict] = field(default_factory=list)

    last_updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "mode": self.mode,
            "recent_mood": self.recent_mood,
            "recent_attitude": self.recent_attitude,
            "relationship_temperature": self.relationship_temperature,
            "care_points": self.care_points,
            "avoid_topics_recent": self.avoid_topics_recent,
            "open_loops": self.open_loops,
            "last_updated_at": self.last_updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RelationshipState":
        return cls(
            person_id=str(data.get("person_id", "")),
            mode=str(data.get("mode", "girlfriend")),
            recent_mood=str(data.get("recent_mood", "")),
            recent_attitude=str(data.get("recent_attitude", "")),
            relationship_temperature=float(data.get("relationship_temperature", 0.5)),
            care_points=list(data.get("care_points", [])),
            avoid_topics_recent=list(data.get("avoid_topics_recent", [])),
            open_loops=list(data.get("open_loops", [])),
            last_updated_at=str(data.get("last_updated_at", "")),
        )


@dataclass
class EmotionalEvent:
    """情感事件 —— 独立于近期记忆摘要的高重要性事件。

    与近期记忆摘要的区别：
    - 近期记忆摘要每条代表一次会话的概括
    - EmotionalEvent 每条代表一个高情感重要性的事件
    - 一条近期摘要可能拆出多个 EmotionalEvent
    - EmotionalEvent 可跨会话关联（如某个话题反复出现）

    Fields:
        person_id:        用户 ID
        date:             事件日期
        title:            事件标题
        summary:          简短描述
        mood:             情绪标签（焦虑/开心/难过/生气等）
        intensity:        情绪强度 0.0-1.0
        people:           涉及人物
        topics:           相关话题
        importance:       重要性 1-5
        relationship_impact: 关系影响描述
        follow_up_needed: 是否需要后续跟进
        source_recent_id:     来源近期记忆记录 ID
        source_context: 来源工作上下文
    """
    person_id: str = ""
    date: str = ""
    title: str = ""
    summary: str = ""
    mood: str = ""
    intensity: float = 0.0
    people: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    importance: int = 4
    relationship_impact: str = ""
    follow_up_needed: bool = False
    source_recent_id: int | None = None
    source_context: str = ""


@dataclass
class OpenLoop:
    """结构化待跟进事项。

    与旧版 open loop 的区别：
    - 旧版：从近期摘要句子里硬抽字符串，没有状态管理
    - 新版：有独立 ID、状态机（open→done→stale→cancelled）、
            冷却时间、来源追溯

    Fields:
        person_id:          用户 ID
        title:              标题
        status:             状态（open/done/stale/cancelled）
        due_hint:           时间提示（"下周"、"周三前"）
        emotional_weight:   情感重要性 1-5
        created_at:         创建时间
        last_mentioned_at:  最后提及时间
        cooldown_until:     冷却结束前（小时级：不重复跟进）
        source_session_id:  来源会话
        resolved_evidence:  解决证据（关闭时记录）
    """
    person_id: str = ""
    title: str = ""
    status: str = "open"  # open / done / stale / cancelled
    due_hint: str = ""
    emotional_weight: int = 3
    created_at: str = ""
    last_mentioned_at: str = ""
    cooldown_until: str = ""
    source_session_id: str = ""
    resolved_evidence: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# MemoryPackV2 —— 基于 MemoryItem 的新一代记忆包
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryPackV2:
    """统一的产品记忆包（V2）—— 基于 MemoryItem 的统一语义。

    所有记忆统一用 MemoryItem 列表存储，按 kind/source/confidence 筛选。

    设计原则：
    1. 只存 MemoryItem，不做自定义字段和 MemoryItem 共存
    2. 注入 prompt 前按场景筛选（filter_by_kind/filter_by_confidence）
    3. history 和 diagnostics 替代旧的 working/matches/工程分层字段
    """
    # 核心：所有记忆统一放在 MemoryItem 列表中
    items: list[MemoryItem] = field(default_factory=list)

    # 关系状态（独立结构，不跟 MemoryItem 混在一起）
    relationship: RelationshipState = field(default_factory=RelationshipState)

    # 当前场景快照
    current_mood: str = ""
    current_topic: str = ""
    guest_mode: bool = False

    # 缺失记忆判定
    missing_memory: dict = field(default_factory=lambda: {
        "should_admit_unknown": False,
        "reason": "",
    })

    # ── 新增：产品语义字段（替代旧的 working/matches/_raw） ─────────────
    history: list[dict] = field(default_factory=list)
    """近期对话历史（替代旧的 `working` 字段），每轮一条 `{"role":..., "content":...}`。"""

    diagnostics: dict = field(default_factory=dict)
    """记忆召回诊断信息。

    产品语义字段示例：
    {
        "recent": [...]              # 匹配的近期记忆摘要
        "long_term": [...]             # 匹配的长期记忆块
        "related": [...]               # 关联记忆
        "core_memory_count": 10        # 核心记忆条数
        "has_recent": true/false     # 是否有情景记忆命中
        "has_long_term": true/false    # 是否有长期记忆命中
        "person_id": "..."             # 当前人物 ID
        "interlocutor_mode": "girlfriend"  # 对话角色模式
        "identity_hint": ""            # 身份提示文本
        "memory_miss": 0/1/2           # 记忆未命中级别
    }
    """

    # ── 筛选方法 ──────────────────────────────────────────────────────

    def items_by_kind(self, *kinds: MemoryKind) -> list[MemoryItem]:
        """按语义类型筛选 MemoryItem。"""
        return [m for m in self.items if m.kind in kinds]

    def items_by_confidence(self, min_confidence: float = 0.7) -> list[MemoryItem]:
        """按置信度筛选。"""
        return [m for m in self.items if m.confidence >= min_confidence]

    def items_by_visibility(self, *visibilities: MemoryVisibility) -> list[MemoryItem]:
        """按可见度筛选。"""
        return [m for m in self.items if m.visibility in visibilities]

    def items_for_prompt(self, max_items: int = 15) -> list[MemoryItem]:
        """按优先级排序，取最适合注入 prompt 的 MemoryItem。

        排序规则：
        1. 高置信（>= 0.9）始终保留
        2. 重要事件（emotional_weight >= 4）次之
        3. 时效性高的（recency_weight >= 4）次之
        4. 其他按置信度降序
        5. private/intimate 默认不包含（需显式启用）

        Returns:
            排序后的 MemoryItem 列表，最多 max_items 条。
        """
        # 默认只取 always + recall_only
        candidates = [
            m for m in self.items
            if m.visibility in (MemoryVisibility.ALWAYS, MemoryVisibility.RECALL_ONLY)
            and not m.is_expired
        ]
        month_key = str(self.diagnostics.get("month_key", "") or "").strip()

        def is_target_month_recall(m: MemoryItem) -> bool:
            if not month_key or m.visibility != MemoryVisibility.RECALL_ONLY:
                return False
            haystack = " ".join([
                str(m.content or ""),
                str(m.source_id or ""),
                str(m.context or ""),
            ])
            return month_key in haystack

        def sort_key(m: MemoryItem) -> tuple:
            if is_target_month_recall(m):
                priority_group = 0
            elif m.visibility == MemoryVisibility.ALWAYS:
                priority_group = 1
            else:
                priority_group = 2
            return (
                priority_group,
                -m.confidence if m.is_high_confidence else 0,
                -m.emotional_weight,
                -m.recency_weight,
                -m.confidence,
            )

        candidates.sort(key=sort_key)
        return candidates[:max_items]

    def format_prompt_block(self, max_tokens_hint: int = 600) -> str:
        """生成注入 prompt 的人类化文本块。

        输出格式（不含任何工程词）：
          ## 你和她的关系状态
          ...

          ## 她现在的状态
          ...

          ## 你该记得的相关事
          ...

          ## 这次不要乱说的边界
          ...

          ## 关于不太确定的记忆
          ...

        Args:
            max_tokens_hint: 最大文本长度提示（近似字符数）
        """
        lines: list[str] = []

        # ── 关系状态 ──
        rs_lines: list[str] = []
        if self.relationship.mode:
            rs_lines.append(f"- 你和对方的关系模式是 {self.relationship.mode}")
        if self.relationship.recent_mood:
            rs_lines.append(f"- 她最近的情绪：{self.relationship.recent_mood}")
        if self.relationship.recent_attitude:
            rs_lines.append(f"- 她对你的态度：{self.relationship.recent_attitude}")
        if self.relationship.relationship_temperature:
            temp_desc = "很亲密" if self.relationship.relationship_temperature >= 0.8 else \
                        "还不错" if self.relationship.relationship_temperature >= 0.5 else \
                        "有点疏远" if self.relationship.relationship_temperature >= 0.3 else "不太好"
            rs_lines.append(f"- 你们现在的亲密程度：{temp_desc}")
        for point in self.relationship.care_points[:3]:
            rs_lines.append(f"- 最近可以留意：{point}")
        for loop in self.relationship.open_loops[:3]:
            title = str(loop.get("title", "") if isinstance(loop, dict) else loop).strip()
            if title:
                rs_lines.append(f"- 还惦记着：{title[:80]}")
        if rs_lines:
            lines.append("## 你和她的关系状态")
            lines.extend(rs_lines)
            lines.append("")

        # ── 当前状态 ──
        current_lines: list[str] = []
        if self.current_mood:
            current_lines.append(f"- 她现在的情绪：{self.current_mood}")
        if self.current_topic:
            current_lines.append(f"- 你们正在聊：{self.current_topic}")
        if current_lines:
            lines.append("## 她现在的状态")
            lines.extend(current_lines)
            lines.append("")

        # ── 相关记忆（从 MemoryItem 生成） ──
        memory_items = self.items_for_prompt(max_items=10)
        memory_lines: list[str] = []
        char_budget = max_tokens_hint
        for item in memory_items:
            text = item.humanized_text
            if not text:
                continue
            # 低置信记忆标记
            prefix = "-（不太确定）" if not item.is_high_confidence else "-"
            line = f"{prefix} {text[:200]}"
            if len(line) > char_budget:
                break
            memory_lines.append(line)
            char_budget -= len(line)

        if memory_lines:
            lines.append("## 你该记得的相关事")
            lines.extend(memory_lines)
            lines.append("")

        # ── 边界 ──
        boundary_items = self.items_by_kind(MemoryKind.TABOO)
        avoid_topics = [str(x).strip() for x in self.relationship.avoid_topics_recent if str(x).strip()]
        if boundary_items or avoid_topics:
            lines.append("## 这次不要乱说的边界")
            for item in boundary_items:
                lines.append(f"- 不要提：{item.humanized_text[:100]}")
            for topic in avoid_topics[:3]:
                lines.append(f"- 最近别主动碰：{topic[:80]}")
            lines.append("")

        # ── 缺失记忆指引 ──
        if self.missing_memory.get("should_admit_unknown"):
            lines.append("## 关于不太确定的记忆")
            reason = self.missing_memory.get("reason", "")
            if "完全未命中" in reason:
                lines.append("- 对方提到的事你完全没印象，诚实说不太记得就好，可以追问但别编")
            else:
                lines.append("- 你隐约记得一些，但不太确定的地方坦诚说")
            lines.append("")
        elif self.missing_memory.get("reason") and "证据较弱" in str(self.missing_memory.get("reason", "")):
            lines.append("## 关于不太确定的记忆")
            lines.append('- 你有一些零散的线索但不够完整，可以说"隐约记得"但别太笃定')
            lines.append("")

        return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════


def classify_emotional_intensity(text: str) -> int:
    """从文本判断情感强度（1-5）。

    基于表情符号和情感关键词的频率/强度评估。
    用于给 MemoryItem 的 emotional_weight 提供参考。
    """
    import re

    score = 3  # 默认中等

    # 高强度信号（+2）
    high_patterns = [
        r"太[好难开心难过痛苦绝望崩溃]",
        r"[崩溃绝望伤心欲绝撕心裂肺痛不欲生欣喜若狂]",
        r"！！！+",
        r"[😭😱🤯💔💀]",
    ]
    for p in high_patterns:
        if re.search(p, text):
            score += 2
            break

    # 中强度信号（+1）
    mid_patterns = [
        r"[烦难过伤心生气愤怒开心高兴紧张焦虑害怕激动]",
        r"有点[烦难恼火]",
        r"😂😡😤🥺😢😊🥰😍",
    ]
    for p in mid_patterns:
        if re.search(p, text):
            score += 1
            break

    # 削减因素（-1）
    low_patterns = [
        r"还好吧",
        r"没事",
        r"还行",
        r"一般",
        r"随便",
        r"无所谓",
    ]
    for p in low_patterns:
        if re.search(p, text):
            score -= 1
            break

    return max(1, min(5, score))
