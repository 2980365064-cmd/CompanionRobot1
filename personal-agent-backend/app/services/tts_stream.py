"""
TTS 服务 —— 语音合成与流式播放管理器。

从 ws_handler.py 中提取的 TTS 逻辑，封装为独立服务：
  - Baidu TTS 合成（复用 app/tts.py 底层）
  - 回复文本按句切分为气泡（bubble）
  - 流式分片发送 PCM 音频
  - 并发预合成（下一个气泡提前合成）
  - 打断（interrupt）支持

使用方式：
  tts = TTSService()
  await tts.stream_bubble(websocket, text, tts_config, session_id)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from app.config import settings
from app.services.tts_client import TTSConfig, TTSException, tts_fetch_pcm

logger = logging.getLogger(__name__)

# 气泡拆分正则：句末标点（句号、感叹号、问号、省略号、换行）
_BUBBLE_SPLIT = re.compile(r"(?<=[。！？…!?\n])")

# PCM 分片大小：~100ms 音频（16kHz × 16bit × 0.1s = 3200 字节）
_PCM_CHUNK_SIZE = 3200

# 片间等待时间：95ms 防止 ESP32 音频缓冲区溢出
_INTER_CHUNK_SLEEP = 0.095

# 文本气泡间等待时间（无 TTS 回退）
_TEXT_BUBBLE_DELAY_MIN = 0.8
_TEXT_BUBBLE_DELAY_MAX = 1.5


@dataclass
class TTSContext:
    """单轮 TTS 上下文，跟踪飞行中的合成任务和打断信号。

    Attributes:
        tts_tasks:   当前活跃的 TTS 合成 Task 列表
        cancelled:   是否已被打断
        _tts_config: 缓存的 TTS 配置
    """

    tts_tasks: list[asyncio.Task] = field(default_factory=list)
    cancelled: bool = False
    _tts_config: Optional[TTSConfig] = None

    def cancel(self) -> None:
        """打断当前正在播放的 TTS。"""
        self.cancelled = True
        for task in self.tts_tasks:
            if not task.done():
                task.cancel()
        self.tts_tasks.clear()


@dataclass
class _QueuedSegment:
    event: Any
    task: asyncio.Task | None


class SegmentTTSPipeline:
    """Synthesize segments eagerly while sending their audio in order."""

    def __init__(
        self,
        service: "TTSService",
        *,
        send_json: Callable,
        send_bytes: Callable,
        ctx: TTSContext,
        turn_id: str,
        session_id: str = "",
        trace=None,
        retry_first_segment: int = 1,
        pace_audio: bool = True,
        use_tts: bool = True,
        is_active: Callable[[], bool] | None = None,
    ) -> None:
        self.service = service
        self.send_json = send_json
        self.send_bytes = send_bytes
        self.ctx = ctx
        self.turn_id = turn_id
        self.session_id = session_id
        self.trace = trace
        self.retry_first_segment = retry_first_segment
        self.pace_audio = pace_audio
        self.use_tts = use_tts
        self.is_active = is_active or (lambda: True)
        self._queue: asyncio.Queue[_QueuedSegment | None] = asyncio.Queue()
        self._worker = asyncio.create_task(self._run())
        self._tts_disabled = False

    async def enqueue(self, event: Any) -> None:
        if self.ctx.cancelled or event.turn_id != self.turn_id:
            return
        await self.send_json({
            "type": "reply_segment",
            "turn_id": self.turn_id,
            "segment_id": event.segment_id,
            "text": event.text,
            "is_final": event.is_final,
            "session_id": self.session_id,
        })
        task = None
        if self.use_tts and not self._tts_disabled:
            task = asyncio.create_task(
                self.service.synthesize_bubble(event.text, self.ctx._tts_config)
            )
            self.ctx.tts_tasks.append(task)
        await self._queue.put(_QueuedSegment(event, task))

    async def finish(self) -> None:
        await self._queue.put(None)
        await self._worker

    def cancel(self) -> None:
        self.ctx.cancel()
        if not self._worker.done():
            self._worker.cancel()

    async def _run(self) -> None:
        while True:
            queued = await self._queue.get()
            if queued is None:
                return
            if self.ctx.cancelled or not self.is_active():
                return
            if not self.use_tts or self._tts_disabled:
                continue
            pcm = await self._resolve_pcm(queued)
            if not pcm:
                continue
            if self.ctx.cancelled or not self.is_active():
                return

            event = queued.event
            tts_id = uuid.uuid4().hex[:12]
            await self.send_json({
                "type": "tts_start",
                "turn_id": self.turn_id,
                "segment_id": event.segment_id,
                "tts_id": tts_id,
                "audio_format": "pcm",
                "sample_rate": 16000,
                "bits_per_sample": 16,
                "channels": 1,
                "session_id": self.session_id,
            })
            for offset in range(0, len(pcm), _PCM_CHUNK_SIZE):
                if self.ctx.cancelled or not self.is_active():
                    return
                sent = await self.send_bytes(pcm[offset:offset + _PCM_CHUNK_SIZE])
                if sent is False:
                    self.cancel()
                    return
                if self.trace and "first_pcm_sent" not in self.trace.marks:
                    self.trace.mark("first_pcm_sent")
                if self.pace_audio:
                    await asyncio.sleep(_INTER_CHUNK_SLEEP)
            await self.send_json({
                "type": "tts_end",
                "turn_id": self.turn_id,
                "segment_id": event.segment_id,
                "tts_id": tts_id,
                "session_id": self.session_id,
            })

    async def _resolve_pcm(self, queued: _QueuedSegment) -> bytes | None:
        attempts = 1 + (self.retry_first_segment if queued.event.segment_id == 0 else 0)
        for attempt in range(attempts):
            try:
                if self.trace and "tts_request_start" not in self.trace.marks:
                    self.trace.mark("tts_request_start")
                if attempt == 0:
                    if queued.task is None:
                        return None
                    return await queued.task
                return await self.service.synthesize_bubble(
                    queued.event.text, self.ctx._tts_config,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "TTS segment failed turn=%s segment=%s attempt=%s: %s",
                    self.turn_id, queued.event.segment_id, attempt + 1, exc,
                )
        if queued.event.segment_id == 0:
            self._tts_disabled = True
        return None


def split_into_bubbles(text: str) -> list[str]:
    """将回复文本按句末标点拆分为多个气泡"""
    parts = _BUBBLE_SPLIT.split(text)
    bubbles = [p.strip() for p in parts if p.strip()]
    return bubbles if bubbles else [text]


def _get_tts_config() -> TTSConfig:
    """从 settings 加载 TTSConfig（缓存单例）。"""
    return TTSConfig(
        api_key=settings.tts_api_key,
        speaker_voice=settings.tts_speaker_voice,
        speed=settings.tts_speed,
        pitch=settings.tts_pitch,
        volume=settings.tts_volume,
        audio_format=settings.tts_audio_format,
        baidu_api_key=settings.baidu_api_key,
        clone_voice_id=settings.tts_clone_voice_id,
    )


def tts_enabled() -> bool:
    """TTS 功能是否可用。"""
    return bool(settings.tts_api_key)


class TTSService:
    """语音合成服务 —— 文本→PCM 分片流式发送。

    职责：
      1. 气泡拆分
      2. 单个气泡 TTS 合成 + 分片流式发送
      3. TTS 缓存的预热（预留）
      4. 打断支持
    """

    def __init__(self) -> None:
        # LRU 缓存：文本 → PCM 字节（预留，后续实现）
        self._cache: dict[str, bytes] = {}
        self._cache_max = 50

    def build_context(self) -> TTSContext:
        """创建当前轮次的 TTS 上下文。"""
        ctx = TTSContext()
        ctx._tts_config = _get_tts_config()
        return ctx

    def build_segment_pipeline(self, **kwargs) -> SegmentTTSPipeline:
        return SegmentTTSPipeline(self, **kwargs)

    async def synthesize_bubble(self, text: str, config: Optional[TTSConfig] = None) -> bytes:
        """合成单个气泡文本为完整 PCM。

        Args:
            text:   需要合成的文本
            config: TTS 配置（留空使用默认配置）

        Returns:
            完整 PCM 字节（16kHz 16bit 单声道）
        """
        cfg = config or _get_tts_config()

        # 检查缓存
        cache_key = f"{text}|{cfg.speaker_voice}|{cfg.speed}|{cfg.pitch}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        pcm = await tts_fetch_pcm(text, cfg)

        # 写入缓存（只在缓存未满时）
        if len(self._cache) < self._cache_max:
            self._cache[cache_key] = pcm
        # 可以在这里引入 TTS PCM 长度下限检查，太短的播报时延不明显

        return pcm

    async def stream_synthesize(
        self, text: str, config: Optional[TTSConfig] = None,
    ) -> AsyncIterator[bytes]:
        """合成文本并逐分片 yield PCM。

        Args:
            text:   需要合成的文本
            config: TTS 配置

        Yields:
            每个 yield 一个 PCM 分片（~3200 字节）
        """
        pcm = await self.synthesize_bubble(text, config)
        for offset in range(0, len(pcm), _PCM_CHUNK_SIZE):
            yield pcm[offset:offset + _PCM_CHUNK_SIZE]

    async def send_bubble_tts(
        self,
        send_json: Callable,
        send_bytes: Callable,
        bubble_text: str,
        ctx: TTSContext,
        session_id: str = "",
    ) -> None:
        """合成并流式发送一个气泡的 TTS 音频。

        Args:
            send_json:   发送 JSON 帧的异步函数（如 websocket.send_json）
            send_bytes:  发送二进制帧的异步函数（如 websocket.send_bytes）
            bubble_text: 气泡文本
            ctx:         当前 TTS 上下文（含打断信号）
            session_id:  会话 ID（可选）
        """
        if ctx.cancelled:
            return

        tts_id = uuid.uuid4().hex[:12]

        # 1. 发送 tts_start（音频元数据）
        await send_json({
            "type": "tts_start",
            "tts_id": tts_id,
            "sample_rate": 16000,
            "bits_per_sample": 16,
            "channels": 1,
            "session_id": session_id,
        })

        # 2. 合成 + 分片发送
        tts_task = asyncio.create_task(
            self._stream_pcm_task(send_bytes, bubble_text, ctx, tts_id)
        )
        ctx.tts_tasks.append(tts_task)

        try:
            await tts_task
        except asyncio.CancelledError:
            logger.info("TTS 气泡被取消: %s", bubble_text[:20])
            return
        except TTSException as exc:
            logger.warning("TTS 合成失败 '%s': %s", bubble_text[:20], exc)
        except Exception:
            logger.exception("TTS 流式发送异常: '%s'", bubble_text[:20])
        finally:
            if not ctx.cancelled:
                await send_json({
                    "type": "tts_end",
                    "tts_id": tts_id,
                    "session_id": session_id,
                })

    async def _stream_pcm_task(
        self,
        send_bytes: Callable,
        text: str,
        ctx: TTSContext,
        tts_id: str,
    ) -> None:
        """后台 TTS 合成 + 分片发送任务。"""
        pcm = await self.synthesize_bubble(text, ctx._tts_config)
        if ctx.cancelled:
            return

        for offset in range(0, len(pcm), _PCM_CHUNK_SIZE):
            if ctx.cancelled:
                return
            await send_bytes(pcm[offset:offset + _PCM_CHUNK_SIZE])
            await asyncio.sleep(_INTER_CHUNK_SLEEP)

    async def stream_reply(
        self,
        send_json: Callable,
        send_bytes: Callable,
        reply_text: str,
        ctx: TTSContext,
        session_id: str = "",
        *,
        use_tts: bool = True,
    ) -> None:
        """流式发送完整回复（多个气泡 + TTS）。

        Args:
            send_json:   JSON 发送函数
            send_bytes:  二进制发送函数
            reply_text:  完整回复文本
            ctx:         当前 TTS 上下文
            session_id:  会话 ID
            use_tts:     是否使用 TTS（False = 纯文本）
        """
        bubbles = split_into_bubbles(reply_text)
        for i, bubble_text in enumerate(bubbles):
            if ctx.cancelled:
                return

            # 发送 reply 文本
            await send_json({
                "type": "reply",
                "text": bubble_text,
                "session_id": session_id,
            })
            if ctx.cancelled:
                return

            if use_tts:
                await self.send_bubble_tts(
                    send_json, send_bytes, bubble_text, ctx, session_id,
                )
            else:
                # 纯文本气泡：气泡间有间隔（模拟自然停顿）
                if i > 0:
                    await asyncio.sleep(
                        _TEXT_BUBBLE_DELAY_MIN
                        + (hash(bubble_text) % 100) / 100 * (_TEXT_BUBBLE_DELAY_MAX - _TEXT_BUBBLE_DELAY_MIN)
                    )


# 模块级单例
tts_service = TTSService()
