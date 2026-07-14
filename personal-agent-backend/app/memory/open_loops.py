"""
结构化待跟进事项管理器 —— Open Loop 的检测、创建、更新、解决。

============================================================================
设计目标：

  旧版 open loop 的问题：
  - 从 近期记忆 句子里硬抽字符串，没有状态管理
  - 不知道哪些已经解决了
  - 不知道哪些已经问过了（没有冷却机制）
  - 主动关心时缺乏优先级排序

  本模块解决上述所有问题，提供完整的 Open Loop 生命周期管理：

  检测条件：
    "等结果"、"还没定"、"明天要"、"到时候告诉你" → 创建 open loop
    "定了"、"结束了"、"过了"、"不用了" → 关闭 open loop

  使用规则：
    - 沉默破冰或主动关心时优先使用 open loop
    - 不要每轮都问，设置 12-24 小时冷却时间
    - 按情感权重排序，高权重优先

  状态机：open → done / stale / cancelled
============================================================================
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any

from app.session import store

logger = logging.getLogger(__name__)


# ── 配置 ───────────────────────────────────────────────────────────────────

# Open Loop 冷却时间（小时）
_COOLDOWN_HOURS = 12
# 高权重事项冷却时间（小时）
_COOLDOWN_HIGH_WEIGHT_HOURS = 24
# 最多保留的 open loop 数
_MAX_OPEN_LOOPS = 20
# 默认情感权重
_DEFAULT_WEIGHT = 3
# 自动 stale 天数（超过此天数未提及视为 stale）
_STALE_DAYS = 14

# Open Loop 创建触发模式
_CREATE_PATTERNS = [
    (re.compile(r"(?:等|等待|等一个|等到了|在等)\s*(?:.*?)(?:结果|通知|消息|回复|电话|offer|录取书)"), 4),
    (re.compile(r"(?:还没|还没定|没定下来|不确定|不知道\s*(?:能不能|行不行|会不会))"), 3),
    (re.compile(r"(?:要|准备|打算|计划|即将)\s*(?:.*?)(?:面试|考试|体检|住院|手术|出差|旅行|搬家|实习)"), 4),
    (re.compile(r"(?:明天|下周|下个月|过几天)\s*(?:要|有|去|做|参加)"), 3),
    (re.compile(r"(?:答应|约了|约好|说好了)\s*(?:了|的)?\s*(?:.*?)(?:去|来|吃|看|见|玩)"), 3),
    (re.compile(r"(?:如果|要是)\s*(?:.*?)(?:过了|通过了|录取了|成功了)"), 3),
    (re.compile(r"(?:告诉我|跟你说|跟你说一下|说一声)\s*(?:.*?)(?:结果|消息)"), 4),
]

# Open Loop 解决触发模式
_RESOLVE_PATTERNS = [
    re.compile(r"(?:定了|确定了|定下来了|办好了|搞定了|解决了|弄好了|安排好了)"),
    re.compile(r"(?:结束了|过了|通过了|录取了|签了|拿到了|收到了)"),
    re.compile(r"(?:不用了|不需要了|算了|取消了|没了|黄了|吹了)"),
    re.compile(r"(?:没过|没通过|没录取|没拿到|没收到|拒绝了|被拒)"),
]

# 解决锚定关键词（用于匹配应该关闭哪个 open loop）
_ANCHOR_KEYWORDS = {
    "面试": "面试",
    "考试": "考试",
    "offer": "offer",
    "结果": "结果",
    "录取": "录取",
    "实习": "实习",
    "搬家": "搬家",
    "体检": "体检",
    "手术": "手术",
    "出差": "出差",
}


class OpenLoopManager:
    """结构化待跟进事项管理器。"""

    # ── 检测 ──────────────────────────────────────────────────────────

    def detect_create(self, user_msg: str) -> list[tuple[str, int]]:
        """检测用户消息中是否包含需要创建 open loop 的内容。

        Args:
            user_msg: 用户消息

        Returns:
            [(title, emotional_weight), ...] 列表，空列表表示无需创建。
        """
        results: list[tuple[str, int]] = []
        for pattern, weight in _CREATE_PATTERNS:
            m = pattern.search(user_msg)
            if m:
                # 提取标题
                title = self._extract_title(user_msg, m.group(0))
                results.append((title, weight))

        return results[:3]  # 最多 3 条

    def detect_resolve(self, user_msg: str, assistant_msg: str = "") -> list[str]:
        """检测用户消息是否包含解决信号。

        Args:
            user_msg:      用户消息
            assistant_msg: 机器人回复

        Returns:
            要关闭的 open loop 的关键词列表。
        """
        resolved: list[str] = []
        for pattern in _RESOLVE_PATTERNS:
            if pattern.search(user_msg) or pattern.search(assistant_msg):
                # 提取要关闭的具体事项关键词
                for keyword, anchor in _ANCHOR_KEYWORDS.items():
                    if keyword in user_msg or keyword in assistant_msg:
                        resolved.append(anchor)
                        break
                if not resolved:
                    # 没有明确锚定 → 取第一个 open loop 标题
                    resolved.append("general")
        return resolved

    # ── 创建/更新 ─────────────────────────────────────────────────────

    def create_or_update(self, person_id: str, title: str, weight: int = 3, session_id: str = "") -> bool:
        """创建或更新一条 open loop。

        如果已存在相同标题的 open loop（status='open'），
        只更新最后提及时间，不重复创建。

        Args:
            person_id:  用户 ID
            title:      事项标题
            weight:     情感权重 1-5
            session_id: 来源会话 ID

        Returns:
            是否成功。
        """
        if not person_id or not title:
            return False

        # 检查是否已存在（去重）
        existing = store.list_open_loops(person_id)
        for loop in existing:
            if self._titles_match(loop.get("title", ""), title):
                # 已存在相同的 open loop，更新时间
                store.update_open_loop_mentioned(loop["id"])
                return True

        # 检查是否超过上限
        count = store.count_open_loops(person_id)
        if count >= _MAX_OPEN_LOOPS:
            logger.info("Open loop limit reached for %s (%d)", person_id, count)
            return False

        loop_id = store.create_open_loop(
            person_id, title,
            emotional_weight=weight,
            source_session_id=session_id,
        )
        if loop_id:
            logger.info("Created open loop #%s for %s: %s", loop_id, person_id, title[:40])
            return True
        return False

    def resolve(self, person_id: str, keyword: str) -> int:
        """根据关键词关闭 open loop。

        Args:
            person_id: 用户 ID
            keyword:   关键词（如"面试"、"搬家"）

        Returns:
            关闭的 open loop 数量。
        """
        if not person_id:
            return 0

        existing = store.list_open_loops(person_id)
        resolved_count = 0
        for loop in existing:
            title = loop.get("title", "")
            if self._titles_match(title, keyword):
                store.resolve_open_loop(loop["id"], evidence=f"resolved by keyword: {keyword}")
                resolved_count += 1
                logger.info("Resolved open loop #%s for %s: %s", loop["id"], person_id, title[:40])

        return resolved_count

    # ── 查询 ──────────────────────────────────────────────────────────

    def list_relevant(self, person_id: str, query: str = "") -> list[dict]:
        """列出当前待跟进事项，按情感权重排序。

        Args:
            person_id: 用户 ID
            query:     查询文本（用于过滤相关事项）

        Returns:
            待跟进事项列表，最多 5 条。
        """
        loops = store.list_open_loops(person_id)
        if not loops:
            return []

        # 按情感权重降序排列
        loops.sort(key=lambda x: (x.get("emotional_weight", 3) or 3) * -1)

        # 如果有关键词，优先返回匹配的
        if query:
            matched = [l for l in loops if self._titles_match(l.get("title", ""), query)]
            if matched:
                return matched[:3]

        return loops[:5]

    def format_prompt_block(self, person_id: str, query: str = "") -> str:
        """生成人类化的 open loop prompt 块。

        Args:
            person_id: 用户 ID
            query:     当前查询（用于筛选相关事项）

        Returns:
            prompt 块文本，无可跟进事项时返回空字符串。
        """
        loops = self.list_relevant(person_id, query)
        if not loops:
            return ""

        lines: list[str] = []
        now = datetime.now(timezone.utc)

        for loop in loops[:3]:
            title = loop.get("title", "")
            weight = loop.get("emotional_weight", 3)
            last_mentioned = loop.get("last_mentioned_at", "")
            cooldown_until = loop.get("cooldown_until", "")

            # 冷却检查：如果还在冷却期，跳过
            if cooldown_until:
                try:
                    cd = datetime.fromisoformat(cooldown_until)
                    if cd > now:
                        continue
                except (ValueError, TypeError):
                    pass

            lines.append(f"- {title}" + (" ❤️" if weight >= 4 else ""))

        if not lines:
            return ""

        return "\n".join(["## 你惦记着的事"] + lines)

    # ── 内部方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_title(user_msg: str, matched_text: str) -> str:
        """从匹配文本中提取标题。

        策略：
        1. 取匹配文本中最长的 4-30 字短语
        2. 去除"等/要/准备/如果"等前缀
        """
        # 取用户消息中匹配部分
        start = max(0, user_msg.find(matched_text[:8]) - 10)
        end = min(len(user_msg), start + 40)
        title = user_msg[start:end].strip()

        # 清理前缀
        prefixes = ["等", "等待", "等一个", "要", "准备", "打算", "计划", "如果", "要是", "明天", "下周"]
        for p in prefixes:
            if title.startswith(p):
                title = title[len(p):].strip()

        # 限制长度
        if len(title) > 30:
            title = title[:28] + "…"

        return title

    @staticmethod
    def _titles_match(title1: str, title2: str) -> bool:
        """判断两条 open loop 标题是否相似（用于去重）。"""
        t1 = title1.strip().lower()
        t2 = title2.strip().lower()
        if t1 == t2:
            return True
        if len(t1) >= 6 and len(t2) >= 6:
            if t1 in t2 or t2 in t1:
                return True
        # 共享关键子串
        for keyword in _ANCHOR_KEYWORDS:
            if keyword in t1 and keyword in t2:
                return True
        return False


# 模块级单例
open_loop_manager = OpenLoopManager()
