"""Baidu TTS client — text → raw PCM via REST API.

PCM format: 16-bit signed, mono, 16000 Hz, little-endian.
Uses aue=4 to get raw PCM without WAV header, so WebSocket binary
frames can be written directly to the ESP32 I2S output.

When clone_voice_id is configured, uses Baidu voice-clone API (WAV mode,
resampled to 16kHz) instead of the legacy text2audio endpoint.
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
import numpy as np

logger = logging.getLogger(__name__)

BAIDU_TTS_URL = "https://tsn.baidu.com/text2audio"
BAIDU_CLONE_TTS_URL = (
    "https://aip.baidubce.com/rest/2.0/speech/publiccloudspeech/v1/voice/clone/tts"
)

TARGET_SAMPLE_RATE = 16000

# Persistent HTTP client — avoids TCP+TLS handshake on every TTS call (~200-400ms saved)
_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=8),
        )
    return _http_client


class TTSException(Exception):
    """Baidu TTS API returned an error (JSON body instead of audio)."""


@dataclass
class TTSConfig:
    api_key: str = ""
    speaker_voice: str = "4100"
    speed: int = 6
    pitch: int = 5
    volume: int = 8
    audio_format: str = "pcm"  # "pcm" → aue=4; "wav" → aue=6

    # 声音复刻
    baidu_api_key: str = ""
    clone_voice_id: str = ""

    @property
    def aue(self) -> int:
        return 4 if self.audio_format == "pcm" else 6

    @property
    def use_clone(self) -> bool:
        return bool(self.baidu_api_key and self.clone_voice_id)


def _resample_pcm(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample 16-bit mono PCM from src_rate to dst_rate using linear interpolation."""
    if src_rate == dst_rate:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n_in = len(samples)
    n_out = int(n_in * dst_rate / src_rate)

    # Linear interpolation: map each output index to source position
    src_positions = np.arange(n_out) * src_rate / dst_rate
    idx_lo = np.floor(src_positions).astype(np.int32)
    idx_hi = np.minimum(idx_lo + 1, n_in - 1)
    frac = src_positions - idx_lo

    out = samples[idx_lo] * (1 - frac) + samples[idx_hi] * frac
    return out.astype(np.int16).tobytes()


def _parse_wav(data: bytes) -> tuple[int, bytes]:
    """Parse WAV bytes → (sample_rate, raw_pcm_bytes)."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return sr, pcm


async def tts_fetch_pcm(text: str, config: TTSConfig) -> bytes:
    """Convert text to raw PCM bytes via Baidu TTS REST API.

    Automatically chooses voice-clone or legacy endpoint based on config.
    Returns raw 16-bit mono 16kHz PCM — no WAV header.
    Raises TTSException on API error.
    """
    if config.use_clone:
        return await _tts_clone(text, config)
    return await _tts_legacy(text, config)


async def _tts_legacy(text: str, config: TTSConfig) -> bytes:
    """旧版 text2audio 接口。"""
    if not config.api_key:
        raise TTSException("TTS API key not configured")

    body = urlencode({
        "tex": text,
        "tok": config.api_key,
        "cuid": "sparkbot-backend",
        "ctp": 1,
        "lan": "zh",
        "spd": config.speed,
        "pit": config.pitch,
        "vol": config.volume,
        "per": config.speaker_voice,
        "aue": config.aue,
    })

    client = _get_client()
    resp = await client.post(
        BAIDU_TTS_URL,
        content=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    content_type = resp.headers.get("content-type", "")
    if "audio" not in content_type:
        err_text = resp.text[:300]
        raise TTSException(f"TTS API error: {err_text}")

    return resp.content


async def _tts_clone(text: str, config: TTSConfig) -> bytes:
    """新版声音复刻 TTS 接口。aue=6 (WAV) → 剥离头 → 重采样到16kHz。"""
    body = urlencode({
        "tex": text,
        "tok": config.baidu_api_key,
        "cuid": "sparkbot-backend",
        "ctp": 1,
        "lan": "zh",
        "per": config.clone_voice_id,
        "spd": config.speed,
        "pit": config.pitch,
        "vol": config.volume,
        "aue": 6,  # WAV — aue=4 PCM is broken (returns silence) in clone API
    })

    client = _get_client()
    resp = await client.post(
        BAIDU_CLONE_TTS_URL,
        content=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    content_type = resp.headers.get("content-type", "")
    if "audio" not in content_type:
        err_text = resp.text[:300]
        raise TTSException(f"Clone TTS API error: {err_text}")

    sr, pcm = _parse_wav(resp.content)
    if sr != TARGET_SAMPLE_RATE:
        pcm = _resample_pcm(pcm, sr, TARGET_SAMPLE_RATE)

    return pcm
