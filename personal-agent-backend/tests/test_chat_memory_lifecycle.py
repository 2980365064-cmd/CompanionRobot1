"""聊天、画像和语料链路的无外部服务回归。"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.memory.interlocutor import InterlocutorTurnResult, MODE_GIRLFRIEND
from app.memory.schema import MemoryItem, MemoryKind, MemoryPackV2, MemorySource


class ChatPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_writes_context_and_schedules_consolidation(self):
        from app.agent import handle_chat

        pack = MemoryPackV2(
            items=[MemoryItem(
                kind=MemoryKind.PREFERENCE,
                source=MemorySource.USER_DECLARED,
                content="她不吃香菜",
            )],
            diagnostics={"person_id": "123", "interlocutor_mode": MODE_GIRLFRIEND},
        )
        identity = InterlocutorTurnResult(
            person_id="123",
            person_profile={"person_id": "123", "confirmed": True},
            interlocutor_mode=MODE_GIRLFRIEND,
        )

        with patch("app.agent.store.get_or_create_session", return_value="session"), \
             patch("app.agent.resolve_interlocutor_before_memory", return_value=identity), \
             patch("app.agent.orchestrator.recall", return_value=pack), \
             patch("app.agent.load_profile_card", return_value="你是叶鹏祥。"), \
             patch("app.agent.chat_completion_async", new=AsyncMock(return_value="记得啊，你不吃香菜。")), \
             patch("app.agent._append_turn") as append_turn, \
             patch("app.agent._post_process", new=AsyncMock()) as post_process:
            reply, session_id, topic = await handle_chat("dev", "", "我不吃什么来着？")
            await asyncio.sleep(0)

        self.assertEqual(reply, "记得啊，你不吃香菜。")
        self.assertEqual(session_id, "session")
        self.assertIsNone(topic)
        append_turn.assert_called_once_with("session", "我不吃什么来着？", reply)
        post_process.assert_awaited_once()


def test_profile_update_uses_unified_long_term_memory_rows():
    from app.memory.profile import empty_profile, update_profile

    profile = empty_profile("刘远慧", person_id="123")
    profile["confirmed"] = True
    response = (
        '{"need_update":true,"reason":"新增经历",'
        '"patch":{"personality":["做事认真"],"experiences":["通过科二考试"],"emotional_habit":[]}}'
    )

    with patch("app.memory.profile.store") as store, \
         patch("app.llm.chat_completion_small", return_value=response):
        store.get_person_profile.return_value = profile
        store.list_recent_memory_since.return_value = []
        store.list_person_long_term_memory.return_value = [
            {"text": "她最近通过了科二考试", "category": "wiki"},
        ]

        updated = update_profile("dev", "123")

    assert updated is not None
    assert "通过科二考试" in updated["experiences"]
    store.save_person_profile.assert_called_once_with("dev", updated)
