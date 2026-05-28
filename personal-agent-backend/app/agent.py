"""Agent orchestration: Profile + memory + LLM."""

from __future__ import annotations

import asyncio
import logging
import time

from app.config import settings
from app.llm import chat_completion
from app.memory.extractor import consolidate_session, extract_facts, maybe_compress_l1
from app.memory.guard import (
    ANTI_HALLUCINATION_RULES,
    capture_user_stated_facts,
    format_stored_facts,
    girlfriend_tone_active,
    user_message_hints,
)
from app.memory.profile import load_profile_card
from app.memory.router import memory_router
from app.memory.working import working_memory
from app.monitor import agent_monitor
from app.session import store

logger = logging.getLogger(__name__)


def build_messages(
    profile: str,
    memory: dict,
    user_message: str,
    *,
    device_id: str,
) -> list[dict]:
    episodic_block = "\n".join(f"- {s}" for s in memory["episodic"]) or "（无）"

    if memory["semantic"]:
        semantic_block = "\n".join(f"- {s}" for s in memory["semantic"])
    elif memory.get("l3_triggered"):
        semantic_block = "（已检索长期记忆库，无匹配；不得编造，须说「不太记得」或请对方说明）"
    else:
        semantic_block = "（本轮未命中长期记忆检索；不得编造任何人名、关系、经历）"

    facts_block = format_stored_facts(device_id)
    hints = user_message_hints(user_message)
    hints_block = f"\n{hints}\n" if hints else ""

    gf_tone = ""
    if girlfriend_tone_active(user_message, memory):
        gf_tone = (
            "- 对方为女友刘远慧（记忆/对话已表明）时：可短句调侃如「咋滴，终于晓得回我了啊」\n"
        )

    system = f"""{profile}

{ANTI_HALLUCINATION_RULES}

## 已入库事实（用户确认或系统写入，优先遵守）
{facts_block}

## 近期会话摘要（L2，7天内）
{episodic_block}

## 长期记忆（L3 向量检索）
{semantic_block}
{hints_block}
## 本轮输出要求（必须执行）
- 你就是叶鹏祥，用第一人称，口语，短句，像微信语音转文字
{gf_tone}- 禁止括号动作/语音描写；禁止「有啥事快说」客服腔
- 微信实测平均 8 字/条：优先 5～20 字，不超过 {settings.max_reply_chars} 字；禁止助手腔、markdown、分点列表
"""
    messages: list[dict] = [{"role": "system", "content": system}]
    for m in memory["working"]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


async def handle_chat(device_id: str, session_id: str, message: str) -> tuple[str, str]:
    t0 = time.perf_counter()
    agent_monitor.chat_user(device_id, message)
    session_id = store.get_or_create_session(device_id, session_id or None)
    memory = memory_router.recall(device_id, session_id, message)
    capture_user_stated_facts(device_id, session_id, message)
    profile = load_profile_card()
    messages = build_messages(profile, memory, message, device_id=device_id)
    temp = settings.chat_temperature
    if user_message_hints(message):
        temp = min(temp, 0.72)
    reply = chat_completion(messages, temperature=temp)
    if len(reply) > settings.max_reply_chars:
        reply = reply[: settings.max_reply_chars]

    working_memory.append(session_id, "user", message)
    working_memory.append(session_id, "assistant", reply)

    agent_monitor.finish_turn(memory, message, reply, t0)
    asyncio.create_task(_post_process(device_id, session_id, message, reply))
    return reply, session_id


async def _post_process(device_id: str, session_id: str, user_msg: str, assistant_msg: str) -> None:
    try:
        compressed = await asyncio.to_thread(maybe_compress_l1, device_id, session_id)
        if compressed:
            agent_monitor.event("L1→L2 会话已压缩入库")
    except Exception as exc:
        agent_monitor.warn(f"L1 压缩失败: {exc}")
    try:
        capture_user_stated_facts(device_id, session_id, user_msg)
    except Exception as exc:
        logger.warning("capture user facts failed: %s", exc)
    try:
        extract_facts(device_id, session_id, user_msg, assistant_msg)
    except Exception:
        pass


async def handle_session_end(device_id: str, session_id: str) -> None:
    if session_id:
        await asyncio.to_thread(consolidate_session, device_id, session_id)
        await asyncio.to_thread(maybe_compress_l1, device_id, session_id)
