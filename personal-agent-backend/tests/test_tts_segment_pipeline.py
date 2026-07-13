import asyncio

import pytest

from app.services.dialog_orchestrator import SegmentEvent
from app.services.latency_trace import LatencyTrace
from app.services.tts_stream import TTSContext, TTSService


@pytest.mark.asyncio
async def test_segments_synthesize_concurrently_but_send_in_order(monkeypatch):
    service = TTSService()
    release_first = asyncio.Event()
    synthesized = []

    async def synthesize(text, config=None):
        synthesized.append(text)
        if text == "第一句。":
            await release_first.wait()
            return b"first"
        release_first.set()
        return b"second"

    monkeypatch.setattr(service, "synthesize_bubble", synthesize)
    json_frames = []
    pcm_frames = []
    trace = LatencyTrace("turn-1")
    pipeline = service.build_segment_pipeline(
        send_json=lambda data: _append(json_frames, data),
        send_bytes=lambda data: _append(pcm_frames, data),
        ctx=TTSContext(),
        turn_id="turn-1",
        session_id="session-1",
        trace=trace,
        retry_first_segment=0,
        pace_audio=False,
    )

    await pipeline.enqueue(SegmentEvent("turn-1", 0, "第一句。"))
    await pipeline.enqueue(SegmentEvent("turn-1", 1, "第二句。", True))
    await pipeline.finish()

    assert synthesized == ["第一句。", "第二句。"]
    assert pcm_frames == [b"first", b"second"]
    starts = [frame for frame in json_frames if frame["type"] == "tts_start"]
    assert [frame["segment_id"] for frame in starts] == [0, 1]
    assert "first_pcm_sent" in trace.marks


@pytest.mark.asyncio
async def test_disabled_tts_sends_segment_text_without_synthesizing(monkeypatch):
    service = TTSService()
    synthesis_calls = 0

    async def unexpected_synthesis(*_args, **_kwargs):
        nonlocal synthesis_calls
        synthesis_calls += 1
        return b"unexpected"

    monkeypatch.setattr(service, "synthesize_bubble", unexpected_synthesis)
    json_frames = []
    pipeline = service.build_segment_pipeline(
        send_json=lambda data: _append(json_frames, data),
        send_bytes=lambda _data: _append([], _data),
        ctx=TTSContext(),
        turn_id="turn-1",
        use_tts=False,
    )

    await pipeline.enqueue(SegmentEvent("turn-1", 0, "纯文本。", True))
    await pipeline.finish()
    await asyncio.sleep(0)

    assert [frame["type"] for frame in json_frames] == ["reply_segment"]
    assert synthesis_calls == 0


async def _append(target, value):
    target.append(value)
    return True
