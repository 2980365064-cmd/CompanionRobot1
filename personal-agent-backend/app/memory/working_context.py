"""
Working Context — 当前会话消息的滑动窗口（自包含模块）。

============================================================================
语义角色：
  Working Context 是"瞬时记忆层"——存放当前会话窗口内的最近 N 轮对话消息。
  数据存储在 messages 表中，按 session_id 隔离。

与其他记忆层的关系：
  - 上下文压缩：当积累满 working_context_turns 轮时触发 LLM 摘要压缩，
    将最老的一批消息压缩为摘要写入近期记忆（Recent Memory），并删除原消息。
  - 访客模式：Working Context 是访客唯一可用的记忆层（不含 embedding 调用）。
  - 已实名模式：Working Context 配合核心事实（Core Facts）、近期记忆（Recent Memory）、
    长期记忆（Long-Term Memory）一起召回。

数据生命周期：
  ┌─────────────────┐  满 N 轮压缩  ┌──────────────┐  过期归档  ┌──────────────────┐
  │ Working Context │ ───────────→ │ Recent Memory│ ─────────→ │ Long-Term Memory │
  │   (messages)    │  删除原消息   │  (summaries) │            │    (corpus)      │
  └─────────────────┘              └──────────────┘            └──────────────────┘
============================================================================
"""

from __future__ import annotations

from app.config import settings
from app.session import store


def get_recent_context(session_id: str) -> list[dict]:
    """获取当前会话的最近 N 轮对话消息。

    N = working_context_turns × 2 条（每轮包含用户消息和助手回复）。
    这些消息将作为本轮 LLM 调用的上下文窗口注入。

    Args:
        session_id: 会话标识

    Returns:
        消息字典列表，按时间正序排列，最多 N 条。每个字典含 role 和 content。
    """
    limit = settings.working_context_turns * 2
    return store.get_recent_messages(session_id, limit)


def append_context_message(session_id: str, role: str, content: str) -> None:
    """向当前会话追加一条消息。

    每轮对话结束后，用户消息和助手回复各调用一次。
    消息直接写入 messages 表，不在内存中维护副本。

    Args:
        session_id: 会话标识
        role:       消息角色（"user" 或 "assistant"）
        content:    消息文本内容
    """
    store.add_message(session_id, role, content)


def count_context_turns(session_id: str) -> int:
    """统计当前会话的已完成对话轮数。

    主要用于判断是否达到上下文压缩阈值：
    当轮数 >= working_context_turns 时触发上下文压缩。

    Returns:
        会话中已完成的对话轮数（每轮 = 用户消息 + 助手回复）。
    """
    return store.count_turns(session_id)
