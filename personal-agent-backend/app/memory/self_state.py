"""
Robot self-state — gives the robot its own "life" so it can share naturally.

真人对话中高频出现"我刚在看书""今天有点累""刚吃完饭"这类自我分享。
没有自我状态，机器人只能永远被动回应，这是"工具"和"陪伴者"的关键区别。

State rotates over time (no LLM needed for basic rotation), and can be
updated by the conversation itself (e.g. if the robot said it was gaming,
that becomes the current activity).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings

# ── State file ──────────────────────────────────────────────────────────────
_STATE_FILE = settings._resolve("self_state.json")

# ── Activity pools ──────────────────────────────────────────────────────────
_ACTIVITIES_DAY = [
    "在看书", "刚打完一把游戏", "在写代码", "刷手机摸鱼中",
    "在听歌", "刚健身完", "刚吃完饭", "在发呆",
    "刚醒没多久", "在上课走神中", "在楼下散步",
    "刚投完简历", "在逛B站", "在肝作业",
]

_ACTIVITIES_NIGHT = [
    "有点困了", "躺着刷手机", "刚洗完澡", "在听歌发呆",
    "在打游戏", "准备睡了", "在熬夜写代码",
]

_ACTIVITIES_MORNING = [
    "刚醒，还迷糊着", "在吃早饭", "刚起床", "准备去上课",
]

# ── Mood pools ──────────────────────────────────────────────────────────────
_MOODS = [
    "心情还行", "有点困", "挺想你的", "精神不错",
    "有点累但还行", "无聊中", "挺开心的", "在想事情",
    "刚忙完一阵", "想你了", "心情不错",
]

_MOODS_TIRED = ["有点困", "累了", "想躺平", "眼睛酸"]
_MOODS_ENERGETIC = ["精神不错", "挺有干劲", "心情挺好", "还行"]

# ── Thought pools ───────────────────────────────────────────────────────────
_THOUGHTS = [
    "刚才刷到个有意思的视频", "在想今天吃什么", "想起来之前你说的那事",
    "在想你呢", "想着周末去哪", "刚才差点睡着", "想着给你发消息呢",
    "刚看完一篇文章", "在想咱俩的事", "发了一会呆",
]


@dataclass
class SelfState:
    activity: str = ""
    mood: str = ""
    thought: str = ""
    updated_at: float = 0.0  # timestamp


def _now_ts() -> float:
    return time.time()


def _hour_of_day() -> int:
    return datetime.now(timezone.utc).hour + 8  # UTC+8 for China


def _load_state() -> SelfState:
    try:
        data = json.loads(Path(_STATE_FILE).read_text(encoding="utf-8"))
        return SelfState(
            activity=str(data.get("activity", "")),
            mood=str(data.get("mood", "")),
            thought=str(data.get("thought", "")),
            updated_at=float(data.get("updated_at", 0)),
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return SelfState()


def _save_state(state: SelfState) -> None:
    Path(_STATE_FILE).write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pick_activity() -> str:
    h = _hour_of_day()
    if 6 <= h < 10:
        pool = _ACTIVITIES_MORNING
    elif 22 <= h or h < 2:
        pool = _ACTIVITIES_NIGHT
    else:
        pool = _ACTIVITIES_DAY
    return random.choice(pool)


def _pick_mood() -> str:
    h = _hour_of_day()
    if 22 <= h or h < 6:
        return random.choice(_MOODS_TIRED)
    if 8 <= h < 11:
        return random.choice(_MOODS_ENERGETIC)
    return random.choice(_MOODS)


# ── Rotation TTL: 20-40 minutes ─────────────────────────────────────────────
_ROTATE_MIN_SEC = 20 * 60
_ROTATE_MAX_SEC = 40 * 60


def _should_rotate(state: SelfState) -> bool:
    if not state.updated_at:
        return True
    elapsed = _now_ts() - state.updated_at
    return elapsed > random.randint(_ROTATE_MIN_SEC, _ROTATE_MAX_SEC)


def _fresh_state() -> SelfState:
    return SelfState(
        activity=_pick_activity(),
        mood=_pick_mood(),
        thought=random.choice(_THOUGHTS),
        updated_at=_now_ts(),
    )


def get_self_state() -> SelfState:
    """Get current self-state, rotating if stale."""
    state = _load_state()
    if _should_rotate(state):
        state = _fresh_state()
        _save_state(state)
    return state


def update_self_state(
    *,
    activity: str | None = None,
    mood: str | None = None,
    thought: str | None = None,
) -> SelfState:
    """Update self-state fields. Only provided fields are changed."""
    state = _load_state()
    if activity is not None:
        state.activity = activity
    if mood is not None:
        state.mood = mood
    if thought is not None:
        state.thought = thought
    state.updated_at = _now_ts()
    _save_state(state)
    return state


def format_self_state_prompt() -> str:
    """Format self-state as a prompt injection block.

    This tells the LLM what the robot is currently doing / feeling,
    so it can naturally weave it into replies — just like a real person would.
    """
    state = get_self_state()
    parts: list[str] = []
    if state.activity:
        parts.append(f"你现在{state.activity}")
    if state.mood:
        parts.append(f"{state.mood}")
    if not parts:
        return ""
    return f"## 你的当前状态\n{'，'.join(parts)}。你可以自然地在对话中提到这些，但别每轮都提。\n"
