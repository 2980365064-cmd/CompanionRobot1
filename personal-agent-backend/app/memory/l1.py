"""
L1 工作记忆（Working Memory）—— 当前会话的滑动窗口。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  L1 是"瞬时记忆层"——存放当前会话窗口内的最近 N 轮对话消息。
  数据存储在 messages 表中，按 session_id 隔离。

与其他记忆层的关系：
  - L1 → L2：当 L1 积累满 working_memory_turns 轮时，extractor.compress_l1_to_l2
    触发 LLM 摘要压缩，将最老的一批消息写入 L2（情景记忆），并删除原消息。
  - 访客模式：L1 是访客唯一可用的记忆层（不含 embedding 调用）。
  - 已实名模式：L1 配合 L0（核心事实）、L2（近期摘要）、L3（长期语义）一起召回。

数据生命周期：
  ┌──────────┐  满 N 轮压缩  ┌──────────┐  7天后过期   ┌──────────┐
  │  L1 消息  │ ───────────→ │  L2 摘要  │ ──────────→ │  L3 语料  │
  └──────────┘  删除原消息    └──────────┘  归档入库     └──────────┘
============================================================================
"""

from __future__ import annotations

from app.config import settings
from app.session import store


class WorkingMemory:
    """L1 工作记忆 —— 封装 messages 表的读写操作。

    本质是对当前会话消息的滑动窗口管理。
    每次调用 get_recent 取最近 working_memory_turns × 2 条消息
    （每轮 = 用户 + 助手各一条），作为本轮对话的上下文窗口。

    设计要点：
    - 无状态：本类不持有内存缓存，所有数据读写直接走 session.store（SQLite）。
      这使得多进程/多 worker 部署时不存在 L1 缓存一致性问题。
    - 按需加载：只在 LLM 调用前执行 get_recent；不会在每次 append 后维护内存副本。
    - 滑动窗口淘汰：消息淘汰由 extractor.compress_l1_to_l2 触发——它不是
      简单删除，而是先压缩成 L2 摘要再删除，保证信息不丢失。
    """

    def get_recent(self, session_id: str) -> list[dict]:
        """获取当前会话的最近 N 轮消息。

        N = working_memory_turns × 2 条（每轮包含用户消息和助手回复）。
        这些消息将作为本轮 LLM 调用的上下文窗口注入。

        Args:
            session_id: 会话标识

        Returns:
            消息字典列表，按时间正序排列，最多 N 条。每个字典含 role 和 content。
        """
        # working_memory_turns × 2：每轮对话包含 user 和 assistant 两条消息
        limit = settings.working_memory_turns * 2
        return store.get_recent_messages(session_id, limit)

    def append(self, session_id: str, role: str, content: str) -> None:
        """向当前会话追加一条消息。

        每轮对话结束后，用户消息和助手回复各调用一次。
        消息直接写入 messages 表，不在内存中维护副本。

        Args:
            session_id: 会话标识
            role:       消息角色（"user" 或 "assistant"）
            content:    消息文本内容
        """
        store.add_message(session_id, role, content)

    def count_turns(self, session_id: str) -> int:
        """统计当前会话的对话轮数。

        主要用于 extractor 判断是否达到压缩阈值：
        当 count_turns >= working_memory_turns 时触发 L1→L2 压缩。

        Args:
            session_id: 会话标识

        Returns:
            会话中已完成的对话轮数（每轮 = 用户 + 助手）。
        """
        return store.count_turns(session_id)


# 模块级单例，供 router 和 agent 模块直接引用。
# 单例模式简化了跨模块依赖：其他模块 import working_memory 即可使用，
# 无需依赖注入或工厂模式。WorkingMemory 本身无状态，所以单例是安全的。
working_memory = WorkingMemory()
