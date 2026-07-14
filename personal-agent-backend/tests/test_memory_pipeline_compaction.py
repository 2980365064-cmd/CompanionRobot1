"""工作上下文压缩的持久化回归。"""

from unittest.mock import patch

from app.memory.memory_pipeline import _compact_working_context_to_recent_memory


def _messages() -> list[dict]:
    return [
        {"id": "m1", "role": "user", "content": "我下周要考科二，有点紧张"},
        {"id": "m2", "role": "assistant", "content": "你已经练得很稳了"},
        {"id": "m3", "role": "user", "content": "考完想去川西玩"},
        {"id": "m4", "role": "assistant", "content": "那我们一起做攻略"},
    ]


def test_compaction_persists_summary_before_deleting_messages():
    response = (
        '{"summary":"她下周考科二，感到紧张，考完计划去川西旅行。",'
        '"topics":"科二,川西","open_loops":["制定川西攻略"],'
        '"emotion":{"mood":"焦虑","intensity":0.6},"importance":4,"people":[]}'
    )
    events: list[str] = []

    with patch("app.memory.memory_pipeline.store") as store, \
         patch("app.memory.memory_pipeline.chat_completion", return_value=response), \
         patch("app.memory.memory_pipeline.recent_memory.save_recent_summary") as save:
        store.get_oldest_messages.return_value = _messages()
        store.get_session_active_person_id.return_value = "123"
        store.count_turns.return_value = 30
        save.side_effect = lambda *args, **kwargs: events.append("saved")
        store.delete_messages_by_ids.side_effect = lambda *args, **kwargs: events.append("deleted")

        assert _compact_working_context_to_recent_memory("dev", "session") is True

    assert events == ["saved", "deleted"]
    save.assert_called_once()
    store.delete_messages_by_ids.assert_called_once_with(["m1", "m2", "m3", "m4"])


def test_compaction_keeps_messages_when_summary_is_invalid():
    with patch("app.memory.memory_pipeline.store") as store, \
         patch("app.memory.memory_pipeline.chat_completion", return_value="不是 JSON"), \
         patch("app.memory.memory_pipeline.recent_memory.save_recent_summary") as save:
        store.get_oldest_messages.return_value = _messages()
        store.get_session_active_person_id.return_value = "123"
        store.count_turns.return_value = 30

        assert _compact_working_context_to_recent_memory("dev", "session") is False

    save.assert_not_called()
    store.delete_messages_by_ids.assert_not_called()
