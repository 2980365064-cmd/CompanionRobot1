from types import SimpleNamespace

import pytest

from app.services import dialog_orchestrator as module
from app.services.dialog_orchestrator import DialogOrchestrator, SegmentEvent
from app.services.latency_trace import LatencyTrace


@pytest.mark.asyncio
async def test_process_emits_numbered_segments_before_done(monkeypatch):
    monkeypatch.setattr(module.store, "get_or_create_session", lambda *_: "session-1")
    monkeypatch.setattr(
        module,
        "resolve_interlocutor_before_memory",
        lambda *_: SimpleNamespace(
            person_id="person-1",
            person_profile=None,
            hint="",
            monitor_event="",
            interlocutor_mode="known",
            mode_switch_ack=None,
        ),
    )
    memory_pack = SimpleNamespace()
    monkeypatch.setattr(module.orchestrator, "recall_fast", lambda *_args, **_kwargs: memory_pack)
    monkeypatch.setattr(module, "load_profile_card", lambda *_: "profile")
    monkeypatch.setattr(module, "build_prompt_context", lambda *_: {})
    monkeypatch.setattr(module, "build_messages", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "user_message_hints", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(module, "query_needs_memory_answer", lambda *_: False)
    monkeypatch.setattr(module, "_append_turn", lambda *_: None)
    async def fake_post_process(*_args):
        return None

    monkeypatch.setattr(module, "_post_process", fake_post_process)
    monkeypatch.setattr(module.agent_monitor, "identity", lambda *_: None)
    monkeypatch.setattr(module.agent_monitor, "memory_pack_v2", lambda *_: None)
    monkeypatch.setattr(module.agent_monitor, "memory_pack_summary", lambda *_: None)
    monkeypatch.setattr(module.agent_monitor, "prompt_summary", lambda *_: None)
    monkeypatch.setattr(module.agent_monitor, "start_turn", lambda *_: None)
    monkeypatch.setattr(module.agent_monitor, "end_turn", lambda *_: None)
    monkeypatch.setattr(module.agent_monitor, "event", lambda *_: None)
    monkeypatch.setattr(module.agent_monitor, "set_timing", lambda *_: None)

    async def fake_stream(*_args, **_kwargs):
        for token in ("第一句。", "第二句！"):
            yield token

    monkeypatch.setattr(module, "chat_completion_stream_async", fake_stream)
    trace = LatencyTrace(turn_id="turn-1")

    events = [event async for event in DialogOrchestrator().process(
        "device-1", "", "你好", trace=trace, turn_id="turn-1"
    )]

    segments = [data for event, data in events if event == "segment"]
    assert segments == [
        SegmentEvent("turn-1", 0, "第一句。", False),
        SegmentEvent("turn-1", 1, "第二句！", False),
    ]
    assert events[-1][0] == "done"
    assert {"memory_ready", "first_token", "first_segment_ready"} <= trace.marks.keys()
