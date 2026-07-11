"""
语音通信网关 —— v2 WebSocket 音频协议（/ws/v2/audio）。

角色：
  ESP32 通过 WebSocket 上传 PCM 音频分片，后端统一编排 ASR→对话→TTS 管线。

协议概览：
  ┌─────────────┬──────────┬──────────────────────────────────────┐
  │ 方向        │ 消息类型  │ 说明                                 │
  ├─────────────┼──────────┼──────────────────────────────────────┤
  │ 上行 JSON   │ hello    │ 握手认证 + 绑定 device_id + session  │
  │ 上行 JSON   │ ping     │ 心跳                                 │
  │ 上行 JSON   │ audio_start │ 开始上传音频，含 turn_id          │
  │ 上行 BINARY │ PCM chunk │ 16kHz 16bit 单声道 PCM 分片          │
  │ 上行 JSON   │ audio_end   │ 音频上传完毕，触发 ASR+对话       │
  │ 上行 JSON   │ interrupt   │ 打断当前 TTS/处理过程              │
  │ 下行 JSON   │ hello_ack   │ 握手成功，返回 session_id          │
  │ 下行 JSON   │ pong        │ 心跳回复                           │
  │ 下行 JSON   │ asr_final   │ ASR 识别文本                       │
  │ 下行 JSON   │ reply_token │ LLM 增量 token                     │
  │ 下行 JSON   │ reply       │ 完整气泡文本                       │
  │ 下行 JSON   │ tts_start   │ TTS 元数据（采样率等）              │
  │ 下行 BINARY │ PCM chunk   │ TTS 生成的 PCM 音频分片             │
  │ 下行 JSON   │ tts_end     │ 当前气泡 TTS 完成                  │
  │ 下行 JSON   │ turn_done   │ 整轮对话完成                       │
  │ 下行 JSON   │ follow_up   │ 主动话题                           │
  │ 下行 JSON   │ error       │ 错误信息（含 code）                │
  └─────────────┴──────────┴──────────────────────────────────────┘

状态机：
  INIT → (hello) → READY
  READY → (audio_start) → LISTENING
  LISTENING → (binary) → LISTENING (accumulate PCM)
  LISTENING → (audio_end) → PROCESSING
  PROCESSING → (ASR done + LLM first token) → SPEAKING
  SPEAKING → (TTS done) → READY
  任意状态 → (interrupt) → READY
  SPEAKING → (audio_start) → LISTENING (打断 + 新轮)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.services.asr import ASRError, ASRTimeout, asr_service
from app.config import settings
from app.services.dialog_orchestrator import dialog_orchestrator
from app.services.latency_trace import LatencyTrace, latency_stats
from app.memory.interlocutor import is_mode_switch_message
from app.monitor import agent_monitor
from app.session import store
from app.services.tts_stream import TTSService, TTSContext, tts_service

logger = logging.getLogger(__name__)


class TurnState(Enum):
    """一轮对话的状态。"""
    INIT = auto()          # 初始，未握手
    READY = auto()         # 就绪，等待音频输入
    LISTENING = auto()     # 接收音频分片中
    PROCESSING = auto()    # ASR + LLM 处理中
    SPEAKING = auto()      # TTS 播放中
    INTERRUPTED = auto()   # 已被打断（瞬态，会自动转 READY）


@dataclass
class AudioTurn:
    """单轮语音对话的上下文。

    Attributes:
        turn_id:     轮次 ID
        pcm_buffer:  累积的 PCM 音频数据（来自 ESP32 上行）
        trace:       延迟追踪实例
        tts_ctx:     TTS 上下文（含打断信号）
    """

    turn_id: str = ""
    pcm_buffer: bytearray = field(default_factory=bytearray)
    trace: Optional[LatencyTrace] = None
    tts_ctx: Optional[TTSContext] = None
    processing_task: Optional[asyncio.Task] = None

    def clear(self) -> None:
        """清空本轮状态，为下一轮准备。"""
        self.pcm_buffer.clear()
        self.trace = None
        self.tts_ctx = None
        self.processing_task = None


@dataclass
class DeviceConnection:
    """单设备 v2 WebSocket 连接状态。"""

    websocket: WebSocket
    device_id: str
    session_id: str = ""
    person_id: str = ""
    state: TurnState = TurnState.INIT
    current_turn: AudioTurn = field(default_factory=AudioTurn)
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # 中断事件：其他协程通过设置它来通知处理协程停止
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)

    def update_active(self) -> None:
        self.last_active = datetime.now(timezone.utc)


# 全局连接注册表
connections: dict[str, DeviceConnection] = {}


# ── 辅助函数 ──────────────────────────────────────────────────

def _next_turn_id() -> str:
    """生成全局唯一的 turn_id。"""
    return uuid.uuid4().hex[:12]


async def _safe_send_json(ws: WebSocket, data: dict) -> bool:
    """安全发送 JSON 帧，失败返回 False。"""
    try:
        await ws.send_json(data)
        return True
    except Exception:
        return False


async def _safe_send_bytes(ws: WebSocket, data: bytes) -> bool:
    """安全发送二进制帧，失败返回 False。"""
    try:
        await ws.send_bytes(data)
        return True
    except Exception:
        return False


# ── v2 音频端点 ────────────────────────────────────────────────

async def ws_audio_endpoint(websocket: WebSocket) -> None:
    """v2 音频 WebSocket 主入口。

    处理流程：
      1. 接受连接
      2. 等待 hello 握手（否则断开）
      3. 循环处理 audio_start/binary/audio_end/interrupt/ping
      4. 断开时清理连接
    """
    await websocket.accept()
    conn: Optional[DeviceConnection] = None
    logger.info("v2 音频 WS 已接受，等待 hello…")

    try:
        while True:
            # ── 接收消息 ─────────────────────────────────
            raw = await websocket.receive_text()

            if conn is None:
                # 第一条消息必须是 hello
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await _safe_send_json(websocket, {
                        "type": "error", "code": "invalid_json",
                        "message": "首条消息必须是 JSON hello",
                    })
                    await websocket.close()
                    return

                if data.get("type") != "hello":
                    await _safe_send_json(websocket, {
                        "type": "error", "code": "not_handshaken",
                        "message": "请先发送 hello",
                    })
                    await websocket.close()
                    return

                # ── 处理 hello ──
                token = data.get("token", "")
                device_id = data.get("device_id", "default_audio")
                client_session_id = data.get("session_id", "")

                if token != settings.api_token:
                    await _safe_send_json(websocket, {
                        "type": "error", "code": "unauthorized",
                        "message": "token 无效",
                    })
                    await websocket.close()
                    return

                session_id = store.get_or_create_session(
                    device_id, client_session_id or None,
                )
                conn = DeviceConnection(
                    websocket=websocket,
                    device_id=device_id,
                    session_id=session_id,
                    state=TurnState.READY,
                )
                connections[device_id] = conn
                await _safe_send_json(websocket, {
                    "type": "hello_ack", "session_id": session_id,
                })
                agent_monitor.event(
                    f"v2 音频已连接 device={device_id} session={session_id[:12]}…"
                )
                logger.info(
                    "v2 audio hello_ack: device=%s session=%s",
                    device_id, session_id,
                )
                continue

            # ── 已握手：处理普通消息 ──
            conn.update_active()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # 如果是 LISTENING 状态的二进制帧，走二进制处理
                # 但这里我们收到的是 text frame，所以是无效 JSON
                await _safe_send_json(websocket, {
                    "type": "error", "code": "invalid_json",
                    "message": "JSON 解析失败",
                })
                continue

            msg_type = data.get("type", "")

            # ── ping ──
            if msg_type == "ping":
                await _safe_send_json(websocket, {"type": "pong"})
                continue

            # ── interrupt（任意状态下可打） ──
            if msg_type == "interrupt":
                await _handle_interrupt(conn)
                continue

            # ── audio_start ──
            if msg_type == "audio_start":
                await _handle_audio_start(conn, data)
                continue

            # ── audio_end ──
            if msg_type == "audio_end":
                conn.current_turn.processing_task = asyncio.create_task(
                    _handle_audio_end(conn, data)
                )
                continue

            # 未知类型
            await _safe_send_json(websocket, {
                "type": "error", "code": "unknown_type",
                "message": f"未知消息类型: {msg_type}",
            })

    except WebSocketDisconnect:
        logger.info(
            "v2 音频 WS 断开: device=%s",
            conn.device_id if conn else "(no hello)",
        )
    except Exception as exc:
        logger.exception(
            "v2 音频 WS handler 异常: device=%s %s",
            conn.device_id if conn else "(no hello)", exc,
        )
    finally:
        if conn and connections.get(conn.device_id) and connections[conn.device_id].websocket is websocket:
            connections.pop(conn.device_id, None)


# ── 二进制帧接收 ──────────────────────────────────────────────

async def _receive_binary_loop(conn: DeviceConnection) -> None:
    """在 LISTENING 状态下持续接收二进制 PCM 分片。

    使用 Starlette 底层 receive() 同时监听 text 和 binary 帧。
    """
    ws = conn.websocket
    turn = conn.current_turn

    while conn.state == TurnState.LISTENING:
        try:
            # receive() 返回底层 ASGI 事件，同时支持 text/bytes/close
            event = await asyncio.wait_for(ws.receive(), timeout=0.3)
        except asyncio.TimeoutError:
            if conn.interrupt_event.is_set():
                return
            continue

        if conn.interrupt_event.is_set():
            return

        event_type = event.get("type", "")

        # WebSocket 文本帧
        if event_type == "websocket.receive":
            text = event.get("text")
            if text is not None:
                try:
                    ctrl = json.loads(str(text))
                except json.JSONDecodeError:
                    continue
                ctype = ctrl.get("type", "")
                if ctype == "audio_end":
                    turn.processing_task = asyncio.create_task(
                        _handle_audio_end(conn, ctrl)
                    )
                    return
                elif ctype == "interrupt":
                    await _handle_interrupt(conn)
                    return
                elif ctype == "ping":
                    await _safe_send_json(ws, {"type": "pong"})
                continue

            # 二进制帧 = PCM 分片
            pcm = event.get("bytes")
            if pcm is not None:
                turn.pcm_buffer.extend(bytes(pcm))
                continue

        # WebSocket 关闭帧
        elif event_type == "websocket.close":
            return

        # 其他类型
        return


# ── 消息处理函数 ──────────────────────────────────────────────

async def _handle_audio_start(conn: DeviceConnection, data: dict) -> None:
    """处理 audio_start：开始一轮新对话。"""
    ws = conn.websocket

    # 如果正在 SPEAKING，先打断
    if conn.state in (TurnState.SPEAKING, TurnState.PROCESSING):
        conn.interrupt_event.set()
        if conn.current_turn.tts_ctx:
            conn.current_turn.tts_ctx.cancel()
        if conn.current_turn.processing_task and not conn.current_turn.processing_task.done():
            conn.current_turn.processing_task.cancel()
            try:
                await conn.current_turn.processing_task
            except asyncio.CancelledError:
                pass
        conn.interrupt_event.clear()

    # 创建新轮
    turn_id = data.get("turn_id", _next_turn_id())
    turn = AudioTurn(turn_id=turn_id)
    turn.trace = LatencyTrace(turn_id=turn_id, device_id=conn.device_id)
    turn.tts_ctx = tts_service.build_context()
    turn.trace.mark("audio_start")

    conn.current_turn = turn
    conn.state = TurnState.LISTENING
    conn.interrupt_event.clear()

    logger.debug("audio_start: device=%s turn=%s", conn.device_id, turn_id)

    # 进入二进制接收循环
    await _receive_binary_loop(conn)


async def _handle_audio_end(conn: DeviceConnection, data: dict) -> None:
    """处理 audio_end：音频上传完毕，触发 ASR + 对话管线。"""
    if conn.state not in (TurnState.LISTENING, TurnState.PROCESSING):
        return

    ws = conn.websocket
    turn = conn.current_turn
    turn.trace.mark("audio_end")
    turn.trace.mark("vad_end")
    if data.get("vad_end_ms") is not None:
        turn.trace.metadata["firmware_vad_end_ms"] = str(data["vad_end_ms"])

    conn.state = TurnState.PROCESSING

    pcm_data = bytes(turn.pcm_buffer)
    pcm_len = len(pcm_data)
    logger.info(
        "audio_end: device=%s turn=%s pcm_len=%d",
        conn.device_id, turn.turn_id, pcm_len,
    )

    # 短语音过滤（< 0.2 秒 = 可能误触或呼吸声）
    MIN_PCM_BYTES = int(16000 * 2 * 0.2)  # 16kHz * 2bytes * 0.2s = 6400
    if pcm_len < MIN_PCM_BYTES:
        logger.info("语音过短（%d bytes < %d），跳过处理", pcm_len, MIN_PCM_BYTES)
        await _safe_send_json(ws, {
            "type": "error", "code": "audio_too_short",
            "message": "语音太短",
        })
        conn.state = TurnState.READY
        turn.clear()
        return

    # ── 1. ASR ──
    text = ""
    asr_configured = asr_service.configured
    if asr_configured and settings.asr_enabled:
        try:
            turn.trace.mark("asr_start")
            text = await asr_service.transcribe(pcm_data)
            turn.trace.mark("asr_final")
            turn.trace.metadata["asr_text"] = text
        except (ASRError, ASRTimeout) as exc:
            logger.warning("ASR 失败，降级返回空文本: %s", exc)
            await _safe_send_json(ws, {
                "type": "error", "code": "asr_error",
                "message": f"ASR 识别失败: {str(exc)[:80]}",
            })
            # 降级：仍然尝试处理，用空文本
            text = ""
        except Exception as exc:
            logger.exception("ASR 异常: %s", exc)
            text = ""
    else:
        # ASR 未配置 → 模拟模式或来自文本源的输入
        text = data.get("asr_text_override", "")

    if conn.interrupt_event.is_set():
        conn.state = TurnState.READY
        conn.interrupt_event.clear()
        turn.clear()
        return

    # 发送 ASR 结果
    if text:
        await _safe_send_json(ws, {
            "type": "asr_final", "text": text,
            "turn_id": turn.turn_id,
        })
        agent_monitor.event(f"ASR: 「{text[:40]}」")
    else:
        # ASR 无结果 → 可能为静音，跳过 LLM
        if not text:
            logger.info("ASR 无识别结果（空文本），跳过对话")
            await _safe_send_json(ws, {
                "type": "turn_done", "turn_id": turn.turn_id,
            })
            conn.state = TurnState.READY
            turn.clear()
            return

    if conn.interrupt_event.is_set():
        conn.state = TurnState.READY
        conn.interrupt_event.clear()
        turn.clear()
        return

    # ── 2. 对话 ──
    turn.trace.mark("llm_start")
    pipeline = tts_service.build_segment_pipeline(
        send_json=lambda d: _safe_send_json(ws, d),
        send_bytes=lambda b: _safe_send_bytes(ws, b),
        ctx=turn.tts_ctx,
        turn_id=turn.turn_id,
        session_id=conn.session_id,
        trace=turn.trace,
        use_tts=bool(settings.tts_api_key),
        is_active=lambda: (
            conn.current_turn is turn
            and conn.current_turn.turn_id == turn.turn_id
            and not conn.interrupt_event.is_set()
        ),
    )
    try:
        async for event, event_data in dialog_orchestrator.process(
            conn.device_id, conn.session_id, text,
            trace=turn.trace,
            turn_id=turn.turn_id,
        ):
            if conn.interrupt_event.is_set():
                break

            if event == "token":
                if "first_token" not in turn.trace.marks:
                    turn.trace.mark("first_token")
                await _safe_send_json(ws, {
                    "type": "reply_token", "text": event_data,
                    "turn_id": turn.turn_id,
                })
            elif event == "segment":
                conn.state = TurnState.SPEAKING
                await pipeline.enqueue(event_data)
            elif event == "done":
                reply, session_id, follow_up = event_data
                conn.session_id = session_id

                # 主动话题（仅文本，不 TTS）
                if follow_up and not conn.interrupt_event.is_set():
                    await _safe_send_json(ws, {
                        "type": "follow_up", "text": follow_up,
                        "turn_id": turn.turn_id,
                    })

            elif event == "error":
                await _safe_send_json(ws, {
                    "type": "error", "code": "llm_error",
                    "message": event_data[:200],
                    "turn_id": turn.turn_id,
                })
    except Exception as exc:
        logger.exception("对话编排异常: %s", exc)
        await _safe_send_json(ws, {
            "type": "error", "code": "internal_error",
            "message": str(exc)[:200],
        })
    finally:
        if conn.interrupt_event.is_set() or conn.current_turn is not turn:
            pipeline.cancel()
        else:
            await pipeline.finish()

    # ── 3. turn_done ──
    turn.trace.mark("turn_done")
    if not conn.interrupt_event.is_set():
        await _safe_send_json(ws, {
            "type": "turn_done", "turn_id": turn.turn_id,
            "reason": "completed",
        })
        # 模式切换口令不开启沉默监听
        if not is_mode_switch_message(text):
            pass  # 沉默话题生成：v2 暂不实现沉默话题

    # 记录延迟数据
    if turn.trace:
        turn.trace.log()
        latency_stats.record(turn.trace)

    conn.state = TurnState.READY
    turn.clear()


async def _send_reply_with_tts(conn: DeviceConnection, reply: str) -> None:
    """发送回复 + TTS 音频（逐个气泡）。"""
    ws = conn.websocket
    turn = conn.current_turn
    tts_ctx = turn.tts_ctx

    if not reply.strip():
        return

    use_tts = settings.tts_api_key and not tts_ctx.cancelled

    conn.state = TurnState.SPEAKING
    await tts_service.stream_reply(
        send_json=lambda d: _safe_send_json(ws, d),
        send_bytes=lambda b: _safe_send_bytes(ws, b),
        reply_text=reply,
        ctx=tts_ctx,
        session_id=conn.session_id,
        use_tts=use_tts,
    )
    if turn.trace and not turn.trace.marks.get("first_tts"):
        turn.trace.mark("first_tts")

    # 如果未被打断，回到 READY
    if not tts_ctx.cancelled:
        conn.state = TurnState.READY


async def _handle_interrupt(conn: DeviceConnection) -> None:
    """处理打断信号：取消当前所有飞行中操作。"""
    turn = conn.current_turn
    turn_id = turn.turn_id or _next_turn_id()
    if conn.state in (TurnState.SPEAKING, TurnState.PROCESSING):
        conn.interrupt_event.set()
        if turn.tts_ctx:
            turn.tts_ctx.cancel()
        if turn.processing_task and not turn.processing_task.done():
            turn.processing_task.cancel()
            try:
                await turn.processing_task
            except asyncio.CancelledError:
                pass
        logger.info(
            "interrupt: device=%s turn=%s",
            conn.device_id, turn_id,
        )
    conn.state = TurnState.READY
    turn.clear()
    conn.interrupt_event.clear()

    await _safe_send_json(conn.websocket, {
        "type": "turn_done",
        "turn_id": turn_id,
        "reason": "interrupted",
    })
