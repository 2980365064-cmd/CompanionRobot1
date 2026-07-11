"""
对话编排器 —— 带 fast/slow 路径的对话处理流水线。

角色：
  speech_gateway 和旧 ws_handler 共享的对话处理入口，
  根据 recall_mode 自动选择快慢路径。

快路径（auto 模式）：
  - 默认：仅核心事实+工作上下文+近期记忆，跳过长期检索
  - 仅当 query_needs_memory_answer() 返回 True 时才进入全路径
  - 保证首响速度

全路径（always 模式）：
  - 同现有 handle_chat_stream（全部记忆层+关系图+情感事件）

事件格式（同 handle_chat_stream）：
  ("token", str)    - 增量文本 token
  ("done", tuple)   - (reply, session_id, follow_up)
  ("error", str)    - 错误信息
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app.config import settings
from app.llm import chat_completion_stream_async
from app.memory.guard import (
    query_needs_memory_answer,
    user_message_hints,
    should_suppress_active_topic,
    validate_active_topic,
)
from app.memory.interlocutor import (
    enforce_mode_switch_reply,
    is_mode_switch_message,
    resolve_interlocutor_before_memory,
)
from app.memory.orchestrator import orchestrator
from app.memory.prompt_context import build_prompt_context
from app.persona.card import load_profile_card
from app.memory.working_context import append_context_message
from app.monitor import agent_monitor
from app.session import store
from app.services.incremental_segmenter import IncrementalSegmenter

# ── 从 agent 导入共享函数 ─────────────────────────────────────
from app.agent import (
    build_messages,
    _parse_reply,
    _strip_stage_directions,
    _append_turn,
    _post_process,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentEvent:
    turn_id: str
    segment_id: int
    text: str
    is_final: bool = False


class DialogOrchestrator:
    """对话编排器 —— 处理新音频协议下的对话流水线。

    Usage:
        orch = DialogOrchestrator()
        async for event, data in orch.process(device_id, session_id, message):
            if event == "token": ...
            elif event == "done": ...
    """

    async def process(
        self,
        device_id: str,
        session_id: str,
        message: str,
        *,
        recall_mode: str = "",
        trace=None,
        turn_id: str = "",
    ) -> str:
        """异步生成器：按 recall_mode 选择快慢路径，逐步 yield 事件。

        Args:
            device_id:  设备标识
            session_id: 会话标识
            message:    用户消息（ASR 识别的文本）
            recall_mode: 记忆召回模式
                "auto"   = 快速路径，必要时才查长期记忆
                "always" = 完整路径
                "fast"   = 强制快速（跳过长期记忆）
                空字符串 = 默认 "auto"

        Yields:
            ("token", str) 增量 token
            ("done", (reply, session_id, follow_up)) 完成
            ("error", str) 错误
        """
        mode = (recall_mode or "auto").lower()
        t0 = time.perf_counter()

        # 1. 获取/创建会话
        session_id = await asyncio.to_thread(
            store.get_or_create_session, device_id, session_id or None,
        )
        _t1 = time.perf_counter()
        agent_monitor.set_timing("session", (_t1 - t0) * 1000)
        agent_monitor.start_turn(device_id, message, session_id)

        # 2. 身份门控
        ictx = await asyncio.to_thread(
            resolve_interlocutor_before_memory, device_id, session_id, message,
        )
        person_id = ictx.person_id
        person_profile = ictx.person_profile
        identity_hint = ictx.hint
        identity_event = ictx.monitor_event
        _t2 = time.perf_counter()
        agent_monitor.set_timing("identity", (_t2 - _t1) * 1000)

        # 3. 记忆召回（按模式选择）
        needs_memory = query_needs_memory_answer(message)
        use_fast_recall = (mode == "fast") or (mode == "auto" and not needs_memory)

        if use_fast_recall:
            memory_pack, profile = await asyncio.gather(
                asyncio.to_thread(
                    orchestrator.recall_fast,
                    device_id, session_id, message, person_id=person_id,
                ),
                asyncio.to_thread(load_profile_card, ictx.interlocutor_mode, message),
            )
            agent_monitor.event(
                f"对话编排器: fast_recall (mode={mode}, needs_memory={needs_memory})"
            )
        else:
            memory_pack, profile = await asyncio.gather(
                asyncio.to_thread(
                    orchestrator.recall,
                    device_id, session_id, message, person_id=person_id,
                ),
                asyncio.to_thread(load_profile_card, ictx.interlocutor_mode, message),
            )
            agent_monitor.event(
                f"对话编排器: full_recall (mode={mode}, needs_memory={needs_memory})"
            )

        memory = build_prompt_context(memory_pack)
        memory["identity_hint"] = identity_hint
        memory["interlocutor_mode"] = ictx.interlocutor_mode
        memory["person_id"] = person_id
        _t3 = time.perf_counter()
        agent_monitor.set_timing("recall", (_t3 - _t2) * 1000)
        if trace:
            trace.mark("memory_ready")

        agent_monitor.identity(person_profile, memory, ictx.interlocutor_mode)
        if identity_event:
            agent_monitor.event(identity_event)
        elif identity_hint:
            agent_monitor.event(identity_hint[:48])

        agent_monitor.memory_pack_v2(memory_pack)
        agent_monitor.memory_pack_summary(memory, memory_pack)

        # 4. 构建 Prompt
        messages = build_messages(
            profile, memory, message, device_id=device_id,
            person_profile=person_profile, memory_pack=memory_pack,
        )
        _t4 = time.perf_counter()
        agent_monitor.set_timing("prompt", (_t4 - _t3) * 1000)
        agent_monitor.prompt_summary(messages)

        temp = settings.chat_temperature
        if user_message_hints(message, memory=memory, person_profile=person_profile, device_id=device_id):
            temp = min(temp, 0.72)

        # 5. 流式 LLM 生成
        reply_parts: list[str] = []
        segmenter = IncrementalSegmenter()
        segment_id = 0
        try:
            async for token in chat_completion_stream_async(messages, temperature=temp):
                if trace and "first_token" not in trace.marks:
                    trace.mark("first_token")
                if token.startswith("\n[ERROR]"):
                    reply_parts.append(token)
                    yield ("token", token)
                    break
                reply_parts.append(token)
                yield ("token", token)
                for text in segmenter.feed(token):
                    if trace and "first_segment_ready" not in trace.marks:
                        trace.mark("first_segment_ready")
                    yield ("segment", SegmentEvent(
                        turn_id=turn_id,
                        segment_id=segment_id,
                        text=text,
                        is_final=False,
                    ))
                    segment_id += 1
        except Exception as exc:
            err_msg = f"调用 DeepSeek 失败：{str(exc)[:120]}"
            reply_parts.append(err_msg)
            yield ("token", err_msg)

        final_segment = segmenter.flush()
        if final_segment:
            if trace and "first_segment_ready" not in trace.marks:
                trace.mark("first_segment_ready")
            yield ("segment", SegmentEvent(
                turn_id=turn_id,
                segment_id=segment_id,
                text=final_segment,
                is_final=True,
            ))

        _t5 = time.perf_counter()
        agent_monitor.set_timing("llm", (_t5 - _t4) * 1000)

        # 6. 解析回复
        reply_raw = "".join(reply_parts).strip()
        reply, active_topic = _parse_reply(reply_raw)
        reply = _strip_stage_directions(reply)

        if active_topic and should_suppress_active_topic(message):
            active_topic = None
        if active_topic:
            active_topic = validate_active_topic(active_topic)
        if len(reply) > settings.max_reply_chars:
            reply = reply[: settings.max_reply_chars]
        reply = enforce_mode_switch_reply(reply, ictx.mode_switch_ack)

        # 7. 写入工作上下文
        await asyncio.to_thread(_append_turn, session_id, message, reply)
        if active_topic:
            await asyncio.to_thread(
                append_context_message, session_id, "assistant", active_topic,
            )

        agent_monitor.end_turn(reply, t0)

        # 8. 后台异步处理
        asyncio.create_task(
            _post_process(device_id, session_id, message, reply, memory, person_id)
        )

        yield ("done", (reply, session_id, active_topic))

    async def process_audio(
        self,
        device_id: str,
        session_id: str,
        text: str,
        *,
        recall_mode: str = "",
        trace=None,
    ) -> str:
        """处理音频输入，同时更新延迟追踪打点，返回回复文本。"""
        if trace:
            trace.mark("llm_start")

        reply = ""
        async for event, data in self.process(
            device_id, session_id, text, recall_mode=recall_mode,
        ):
            if event == "token":
                pass  # 调用方另做流式转发
            elif event == "done":
                reply, session_id, _ = data
            elif event == "error":
                reply = data

        if trace:
            trace.mark("llm_end")
        return reply


# 模块级单例
dialog_orchestrator = DialogOrchestrator()
