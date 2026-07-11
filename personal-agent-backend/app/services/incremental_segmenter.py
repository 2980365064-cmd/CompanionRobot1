"""Incrementally turn streamed LLM tokens into speech-safe segments."""

from __future__ import annotations

import re


_STAGE_DIRECTION = re.compile(r"[（(][^（）()]{1,40}[）)]")
_STRONG_ENDINGS = "。！？!?…\n"
_WEAK_ENDINGS = "，、；：,;:"
_TINY_SPEECH = re.compile(r"^[嗯啊哦唔诶哎哈]+[。！？!?…]?$|^[。！？!?…]+$")


class IncrementalSegmenter:
    """Buffer streamed tokens and emit complete, speakable text segments."""

    def __init__(self, *, soft_limit: int = 24, hard_limit: int = 36) -> None:
        if soft_limit <= 0 or hard_limit < soft_limit:
            raise ValueError("segment limits must satisfy 0 < soft_limit <= hard_limit")
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self._buffer = ""
        self._pending_tiny = ""
        self._control_started = False

    def feed(self, token: str) -> list[str]:
        if not token or self._control_started:
            return []

        clean = _STAGE_DIRECTION.sub("", token)
        if "|||" in clean:
            clean, _ = clean.split("|||", 1)
            self._control_started = True
        self._buffer += clean
        return self._drain()

    def flush(self) -> str | None:
        tail = self._buffer.strip()
        self._buffer = ""
        if self._pending_tiny:
            tail = self._pending_tiny + tail
            self._pending_tiny = ""
        return tail or None

    def _drain(self) -> list[str]:
        emitted: list[str] = []
        while self._buffer:
            boundary = self._strong_boundary()
            if boundary is None and len(self._buffer) >= self.soft_limit:
                boundary = self._weak_boundary()
            if boundary is None and len(self._buffer) >= self.hard_limit:
                boundary = self.hard_limit
            if boundary is None:
                break

            segment = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:]
            if not segment:
                continue
            if _TINY_SPEECH.fullmatch(segment):
                self._pending_tiny += segment
                continue
            if self._pending_tiny:
                segment = self._pending_tiny + segment
                self._pending_tiny = ""
            emitted.append(segment)
        return emitted

    def _strong_boundary(self) -> int | None:
        for index, char in enumerate(self._buffer):
            if char in _STRONG_ENDINGS:
                return index + 1
        return None

    def _weak_boundary(self) -> int | None:
        search_end = min(len(self._buffer), self.hard_limit)
        for index in range(search_end - 1, -1, -1):
            if self._buffer[index] in _WEAK_ENDINGS:
                return index + 1
        return None
