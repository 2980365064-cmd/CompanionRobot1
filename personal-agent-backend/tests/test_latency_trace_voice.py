from app.services.latency_trace import LatencyStats, LatencyTrace


def test_report_uses_first_pcm_as_first_response(monkeypatch):
    moments = iter((10.0, 10.4, 10.5, 10.9, 11.1, 11.6, 12.0))
    monkeypatch.setattr("app.services.latency_trace.time.monotonic", lambda: next(moments))
    trace = LatencyTrace("turn-1")
    for mark in (
        "vad_end", "asr_final", "memory_ready", "first_token",
        "first_segment_ready", "tts_request_start", "first_pcm_sent",
    ):
        trace.mark(mark)

    intervals = trace.report()["intervals_s"]

    assert intervals["vad_end_to_asr_final"] == 0.4
    assert intervals["asr_final_to_memory_ready"] == 0.1
    assert intervals["memory_ready_to_first_token"] == 0.4
    assert intervals["first_segment_ready_to_first_pcm_sent"] == 0.9
    assert intervals["first_response_latency"] == 2.0


def test_stats_reports_percentiles_for_first_pcm_latency():
    stats = LatencyStats()
    for value in (1.0, 1.5, 2.0):
        trace = LatencyTrace("turn")
        trace.marks = {"vad_end": 10.0, "first_pcm_sent": 10.0 + value}
        stats.record(trace)

    assert stats.summary()["first_response_latency"] == {
        "p50": 1.5,
        "p90": 2.0,
        "avg": 1.5,
        "count": 3,
    }
