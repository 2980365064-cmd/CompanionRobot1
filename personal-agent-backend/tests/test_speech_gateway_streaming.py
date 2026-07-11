from types import SimpleNamespace

import pytest

from app.services import speech_gateway as module
from app.services.dialog_orchestrator import SegmentEvent
from app.services.speech_gateway import AudioTurn, DeviceConnection, TurnState


class FakeWebSocket:
    def __init__(self):
        self.json = []
        self.pcm = []

    async def send_json(self, data):
        self.json.append(data)

    async def send_bytes(self, data):
        self.pcm.append(data)


@pytest.mark.asyncio
async def test_audio_end_enqueues_segment_before_llm_done(monkeypatch):
    ws = FakeWebSocket()
    turn = AudioTurn(turn_id="turn-1", pcm_buffer=bytearray(b"x" * 6400))
    turn.trace = module.LatencyTrace("turn-1", "device-1")
    turn.tts_ctx = module.tts_service.build_context()
    conn = DeviceConnection(
        websocket=ws,
        device_id="device-1",
        session_id="session-1",
        state=TurnState.LISTENING,
        current_turn=turn,
    )
    monkeypatch.setattr(module.settings, "asr_enabled", True)

    async def transcribe(_pcm):
        return "你好"

    monkeypatch.setattr(
        module,
        "asr_service",
        SimpleNamespace(configured=True, transcribe=transcribe),
    )
    order = []

    async def process(*_args, **_kwargs):
        yield "token", "第一句。"
        yield "segment", SegmentEvent("turn-1", 0, "第一句。")
        order.append("llm_done")
        yield "done", ("第一句。", "session-1", None)

    monkeypatch.setattr(module.dialog_orchestrator, "process", process)

    class FakePipeline:
        async def enqueue(self, event):
            order.append(f"enqueue:{event.segment_id}")

        async def finish(self):
            order.append("tts_finish")

        def cancel(self):
            order.append("cancel")

    monkeypatch.setattr(
        module.tts_service,
        "build_segment_pipeline",
        lambda **_kwargs: FakePipeline(),
    )

    await module._handle_audio_end(conn, {"type": "audio_end", "vad_end_ms": 1234})

    assert order.index("enqueue:0") < order.index("llm_done")
    assert order[-1] == "tts_finish"
    assert any(frame["type"] == "turn_done" and frame["reason"] == "completed" for frame in ws.json)


@pytest.mark.asyncio
async def test_interrupt_cancels_processing_task_and_reports_original_turn():
    ws = FakeWebSocket()
    cancelled = False

    async def processing():
        nonlocal cancelled
        try:
            await module.asyncio.Event().wait()
        except module.asyncio.CancelledError:
            cancelled = True
            raise

    task = module.asyncio.create_task(processing())
    turn = AudioTurn(turn_id="turn-old", processing_task=task)
    conn = DeviceConnection(
        websocket=ws,
        device_id="device-1",
        state=TurnState.PROCESSING,
        current_turn=turn,
    )

    await module.asyncio.sleep(0)
    await module._handle_interrupt(conn)
    await module.asyncio.sleep(0)

    assert cancelled
    assert ws.json[-1] == {
        "type": "turn_done",
        "turn_id": "turn-old",
        "reason": "interrupted",
    }
