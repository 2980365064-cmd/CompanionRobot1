"""
语音管线延迟追踪 —— 统一记录每轮对话各阶段耗时。

设计目标：
  每轮对话打点记录 vad_end → asr_final → first_token → first_tts → turn_done
  各阶段时间戳，输出可观测的 p50/p90 统计，帮助定位首响瓶颈。

使用方式：
  trace = LatencyTrace(turn_id="t1", device_id="esp32_01")
  trace.mark("vad_end")
  ... ASR ...
  trace.mark("asr_final")
  ...
  report = trace.report()
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LatencyTrace:
    """单轮对话延迟追踪。

    Attributes:
        turn_id:    轮次标识（由 speech_gateway 生成）
        device_id:  设备标识
        marks:      打点记录 {阶段名: monotonic_timestamp}
        metadata:   扩展元数据（可选，如 ASR 文本、错误信息）
    """

    turn_id: str
    device_id: str = ""
    marks: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def mark(self, phase: str) -> None:
        """记录当前时刻为指定阶段。"""
        self.marks[phase] = time.monotonic()
        logger.debug("[Latency] %s %s %.3f", self.turn_id[:8], phase, self.marks[phase])

    def elapsed(self, start: str, end: str) -> Optional[float]:
        """返回两个打点之间的耗时（秒），任一缺失则返回 None。"""
        if start in self.marks and end in self.marks:
            return self.marks[end] - self.marks[start]
        return None

    def total(self) -> Optional[float]:
        """返回 turn_done - audio_end 的整轮耗时。"""
        return self.elapsed("audio_end", "turn_done")

    def e2e(self) -> Optional[float]:
        """端到端：audio_end → turn_done（含 LLM+TTS）。"""
        return self.total()

    def report(self) -> dict:
        """生成延迟报告，包含关键路径耗时。"""
        intervals: dict[str, float] = {}
        key_pairs = [
            ("vad_end", "asr_final"),
            ("asr_final", "memory_ready"),
            ("memory_ready", "first_token"),
            ("first_token", "first_segment_ready"),
            ("first_segment_ready", "tts_request_start"),
            ("first_segment_ready", "first_pcm_sent"),
            ("tts_request_start", "first_pcm_sent"),
            ("audio_end", "turn_done"),
        ]
        for start, end in key_pairs:
            val = self.elapsed(start, end)
            if val is not None:
                intervals[f"{start}_to_{end}"] = round(val, 3)

        # 首响延迟 = vad_end → first_pcm_sent（用户说完到第一块音频发出）
        first_response = self.elapsed("vad_end", "first_pcm_sent")
        if first_response is not None:
            intervals["first_response_latency"] = round(first_response, 3)

        return {
            "turn_id": self.turn_id,
            "device_id": self.device_id,
            "intervals_s": intervals,
            "marks": {k: round(v, 3) for k, v in sorted(self.marks.items())},
            "metadata": self.metadata,
        }

    def log(self) -> None:
        """输出延迟报告到日志。"""
        report = self.report()
        intervals = report.get("intervals_s", {})
        summary = ", ".join(f"{k}={v}s" for k, v in intervals.items())
        logger.info(
            "[Latency] turn=%s device=%s %s",
            self.turn_id[:8], self.device_id, summary,
        )


class LatencyStats:
    """延迟统计收集器 —— 累计多轮延迟，计算 p50/p90。

    每个 LatencyTrace 完成后通过 .record(trace) 提交，
    定期调用 .summary() 获取统计。

    Attributes:
        _records: 按阶段名存储的延迟列表
        _max_records: 最大记录数（防止内存泄漏）
    """

    def __init__(self, max_records: int = 1000) -> None:
        self._records: dict[str, list[float]] = {}
        self._max_records = max_records

    def record(self, trace: LatencyTrace) -> None:
        """记录一轮延迟数据。"""
        for key, value in trace.report().get("intervals_s", {}).items():
            if key not in self._records:
                self._records[key] = []
            self._records[key].append(value)
            # 截断到 max_records
            if len(self._records[key]) > self._max_records:
                self._records[key] = self._records[key][-self._max_records:]

    def summary(self) -> dict[str, dict[str, float]]:
        """返回各指标的 p50/p90/均值。"""
        result: dict[str, dict[str, float]] = {}
        for key, values in self._records.items():
            if not values:
                continue
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            result[key] = {
                "p50": round(sorted_vals[n // 2], 3),
                "p90": round(sorted_vals[int(n * 0.9)], 3),
                "avg": round(sum(values) / n, 3),
                "count": n,
            }
        return result


# 模块级单例
latency_stats = LatencyStats()
