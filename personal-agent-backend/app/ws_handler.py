"""WebSocket 对话协议处理 —— hello 握手 → chat 对话 → session_end 收尾。

本模块的角色：
  陪伴机器人的 WebSocket 长连接入口。与 HTTP /v1/chat 接口功能等效，
  但通过 WebSocket 支持：
  - 持续连接无需重复握手
  - ping/pong 心跳保活
  - 空闲超时自动结束会话（idle_session_sweeper）

消息协议（JSON 格式）：

  hello:        握手认证 + 绑定 device_id
    → 请求:  {"type":"hello", "token":"...", "device_id":"...", "session_id":"..."}
    ← 响应:  {"type":"hello_ack", "session_id":"..."}

  chat:         发送用户消息，返回流式 TTS 回复
    → 请求:  {"type":"chat", "message":"你好"}
    ← 响应序列（每个 bubble）:
       1. {"type":"reply_start", "session_id":"..."}              (text frame, 本轮回复开始)
       2. {"type":"reply_token", "text":"...", "session_id":"..."} (text frame, LLM 增量 token)
       3. {"type":"reply", "text":"...", "session_id":"..."}     (text frame, 气泡完整句，兼容)
       4. {"type":"tts_start", "tts_id":"uuid",                    (text frame)
           "sample_rate":16000, "bits_per_sample":16, "channels":1}
       5. [binary frame] PCM chunk ... [binary frame] PCM chunk    (opcode 0x02)
       6. {"type":"tts_end", "tts_id":"uuid"}                       (text frame)
       ... 多 bubble 时重复以上 ...
       7. {"type":"chat_done", "session_id":"..."}                  (text frame)

  abort:        打断当前 TTS 播放
    → 请求:  {"type":"abort"}
    （后端取消飞行中的 TTS 任务，丢弃剩余 bubbles）

  session_end:  主动结束会话（L1→L2 收尾）
    → 请求:  {"type":"session_end"}
    ← 响应:  {"type":"session_end_ack", "session_id":"..."}

  new_session:  强制新会话（先结束旧会话，再创建新的）
    → 请求:  {"type":"new_session"}
    ← 响应:  {"type":"hello_ack", "session_id":"..."}

  ping:         心跳保活
    → 请求:  {"type":"ping"}
    ← 响应:  {"type":"pong"}

错误响应格式：
  {"type":"error", "code":"unauthorized", "message":"..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from fastapi import WebSocket, WebSocketDisconnect

from app.agent import generate_memory_topic, handle_chat_stream, handle_session_end
from app.memory.interlocutor import is_mode_switch_message
from app.config import settings
from app.monitor import agent_monitor
from app.tts import TTSConfig, TTSException, tts_fetch_pcm

logger = logging.getLogger(__name__)

# 气泡拆分：按句末标点切分
_BUBBLE_SPLIT = re.compile(r"(?<=[。！？…!?\n])")


@dataclass
class ActiveTTSContext:
    """Tracks in-flight TTS so abort can cancel it."""
    tts_task: asyncio.Task | None = None
    cancelled: bool = False

# TTS config singleton — loaded once from settings
_tts_config: TTSConfig | None = None


def _get_tts_config() -> TTSConfig:
    global _tts_config
    if _tts_config is None:
        _tts_config = TTSConfig(
            api_key=settings.tts_api_key,
            speaker_voice=settings.tts_speaker_voice,
            speed=settings.tts_speed,
            pitch=settings.tts_pitch,
            volume=settings.tts_volume,
            audio_format=settings.tts_audio_format,
            baidu_api_key=settings.baidu_api_key,
            clone_voice_id=settings.tts_clone_voice_id,
        )
    return _tts_config


def _tts_enabled() -> bool:
    return bool(settings.tts_api_key)


@dataclass
class DeviceConnection:
    """单设备 WebSocket 连接状态。

    字段说明:
        websocket:    FastAPI WebSocket 对象
        device_id:    设备标识（来自 hello 消息，用于区分不同终端）
        session_id:   当前会话 ID（由后端分配，可通过 hello_ack 返回给客户端）
        last_active:  最后活跃时间（用于 idle_session_sweeper 检测空闲连接）

    每个 device_id 同一时间只有一个活跃连接（存储在 connections 字典中）。
    新连接会覆盖旧连接（旧的 WebSocket 会自动断开）。
    """

    websocket: WebSocket
    device_id: str
    session_id: str = ""
    person_id: str = ""
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# 全局连接注册表：key=device_id, value=DeviceConnection
# 同一设备建立新连接时会覆盖旧条目（旧 WebSocket 由 FastAPI 自动断开）
# 为什么用 device_id 做 key 而非 session_id：
#   一个设备可能跨多个 session，但同一时间只有一个活跃连接；
#   用 device_id 可以将新连接与旧连接关联起来，方便复用 session。
connections: dict[str, DeviceConnection] = {}


def _split_into_bubbles(text: str) -> list[str]:
    """将回复文本按句末标点拆分为多个气泡。"""
    parts = _BUBBLE_SPLIT.split(text)
    bubbles = [p.strip() for p in parts if p.strip()]
    return bubbles if bubbles else [text]


async def _stream_tts_chunks(
    websocket: WebSocket, text: str, config: TTSConfig
) -> None:
    """Fetch PCM from Baidu TTS and send as binary WebSocket frames."""
    pcm = await tts_fetch_pcm(text, config)
    # Send in ~100ms chunks (3200 bytes = 16000*2*0.1)
    # Larger chunks = fewer WebSocket sends and sleep calls = less overhead
    chunk_size = 3200
    for offset in range(0, len(pcm), chunk_size):
        await websocket.send_bytes(pcm[offset:offset + chunk_size])
        await asyncio.sleep(0.095)  # pace to real-time so ESP32 queue doesn't overflow


async def _send_bubble_tts(
    websocket: WebSocket,
    bubble_text: str,
    config: TTSConfig,
    active_tts: ActiveTTSContext,
    session_id: str,
) -> None:
    """Send tts_start, PCM binary frames, and tts_end for one bubble."""
    tts_id = uuid.uuid4().hex[:12]
    await websocket.send_json({
        "type": "tts_start",
        "tts_id": tts_id,
        "sample_rate": 16000,
        "bits_per_sample": 16,
        "channels": 1,
        "session_id": session_id,
    })
    tts_task = asyncio.create_task(
        _stream_tts_chunks(websocket, bubble_text, config)
    )
    active_tts.tts_task = tts_task
    try:
        await tts_task
    except asyncio.CancelledError:
        pass
    except TTSException as exc:
        logger.warning("TTS error for bubble '%s': %s", bubble_text[:20], exc)
    except Exception:
        logger.exception("TTS stream crashed for bubble '%s'", bubble_text[:20])
    finally:
        if not active_tts.cancelled:
            await websocket.send_json({"type": "tts_end", "tts_id": tts_id, "session_id": session_id})


async def _send_bubble_with_tts(
    websocket: WebSocket,
    bubble_text: str,
    bubble_index: int,
    tts_cfg: TTSConfig,
    active_tts: ActiveTTSContext,
    session_id: str,
) -> bool:
    """Send one reply bubble with TTS audio. Returns False if abort cancelled it."""
    # 1. Reply text
    await websocket.send_json({
        "type": "reply", "text": bubble_text, "session_id": session_id,
    })
    if active_tts.cancelled:
        return False

    # 2. TTS audio
    await _send_bubble_tts(websocket, bubble_text, tts_cfg, active_tts, session_id)
    return not active_tts.cancelled


async def _send_textonly_bubble(
    websocket: WebSocket,
    bubble_text: str,
    bubble_index: int,
    session_id: str,
) -> None:
    """Send a reply bubble with inter-bubble delay (backward compat, no TTS)."""
    if bubble_index > 0:
        await asyncio.sleep(0.8 + random.random() * 0.7)
    await websocket.send_json({
        "type": "reply", "text": bubble_text, "session_id": session_id,
    })


async def ws_chat_endpoint(websocket: WebSocket) -> None:
    """WebSocket 主循环：接受连接后处理 hello/chat/session_end/ping/new_session 消息。

    参数:
        websocket: FastAPI 注入的 WebSocket 连接对象

    处理流程：
      1. 接受 WebSocket 连接
      2. 等待 hello 消息完成认证和 session 绑定（在此之前拒绝其他消息）
      3. 循环处理 chat/session_end/ping/new_session 消息
      4. 连接断开时自动从 connections 中移除

    超时处理：
      chat 消息有 45 秒超时（handle_chat 包含 LLM 调用，可能较慢），
      超时返回 "llm_timeout" 错误而非断开连接。
    """
    await websocket.accept()
    conn: DeviceConnection | None = None
    # silence_timeout: None = 无限等待（初始/非chat消息后）；数字 = 沉默 N 秒后主动找话题
    silence_timeout: float | None = None
    # 沉默超时区间：默认固定 4 秒，避免机械固定的触发节奏
    _SILENCE_TIMEOUT_MIN = 4.0
    _SILENCE_TIMEOUT_MAX = 4.0
    silence_stage: int = 0
    last_sent_topic_at: float = 0.0
    silence_topic_sent_this_round: bool = False  # 每轮最多一次
    silence_lottery_attempts: int = 0  # 每轮抽奖次数，最多3次
    # 跨轮话题去重：存最近 N 条已发送话题的规范化文本，不随用户发言清空
    _sent_topic_norms: list[str] = []

    def _next_silence_timeout() -> float:
        """返回固定 4 秒超时。"""
        return _SILENCE_TIMEOUT_MIN

    from app.session import store as session_store

    def _topic_is_dup(topic: str) -> bool:
        """检查话题是否与近期已发送话题语义重复（规范化后精确/子串比对）。"""
        norm = "".join(ch for ch in topic if ch.isalpha() or ch == "？")
        if len(norm) < 3:
            return False
        for prev in _sent_topic_norms[-10:]:
            if norm in prev or prev in norm:
                return True
        return False

    def _mark_topic_sent(topic: str) -> None:
        nonlocal _sent_topic_norms
        norm = "".join(ch for ch in topic if ch.isalpha() or ch == "？")
        if norm:
            _sent_topic_norms.append(norm)
            if len(_sent_topic_norms) > 20:
                _sent_topic_norms = _sent_topic_norms[-10:]

    def _reset_silence_state() -> None:
        """用户有活动时重置沉默退避阶段，回到首轮 3s 等待。话题去重列表不清空。"""
        nonlocal silence_stage, silence_topic_sent_this_round, silence_lottery_attempts
        silence_stage = 0
        silence_topic_sent_this_round = False
        silence_lottery_attempts = 0

    agent_monitor.event("WS 等待 hello…")
    logger.info("WebSocket accepted, waiting for hello")
    try:
        while True:
            # ---- 接收消息（带可选的沉默超时） ----
            if silence_timeout is not None:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=silence_timeout)
                    # 用户在沉默等待期间发了消息 → 解析处理
                except asyncio.TimeoutError:
                    # 沉默超时 → 20% 概率从记忆生成话题（每轮最多3次抽奖）
                    silence_timeout = _next_silence_timeout()
                    if silence_topic_sent_this_round:
                        continue  # 本轮已生成过一次，跳过
                    silence_lottery_attempts += 1
                    roll = random.random()
                    hit = roll <= 0.20
                    logger.info(f"沉默话题抽奖[{silence_lottery_attempts}/3] roll={roll:.2f} {'HIT' if hit else 'MISS'}")
                    if not hit:
                        if silence_lottery_attempts >= 3:
                            silence_topic_sent_this_round = True  # 3次不中，本轮不再抽
                            logger.info(f"沉默话题本轮放弃: 3次抽奖均未命中")
                        continue
                    now = time.monotonic()
                    if conn and conn.session_id and (now - last_sent_topic_at) >= 30.0:
                        pid = session_store.get_session_active_person_id(conn.session_id) or ""
                        try:
                            topic = await generate_memory_topic(conn.device_id, pid, session_id=conn.session_id)
                            if topic and not _topic_is_dup(topic):
                                _mark_topic_sent(topic)
                                last_sent_topic_at = now
                                silence_topic_sent_this_round = True
                                await websocket.send_json(
                                    {"type": "follow_up", "text": topic, "session_id": conn.session_id}
                                )
                        except Exception:
                            logger.warning("沉默话题生成异常", exc_info=True)
                    continue  # 继续等待
            else:
                raw = await websocket.receive_text()

            # 用户有任何消息则重置沉默退避状态
            _reset_silence_state()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse failed for incoming WS frame (%d bytes): %.160s", len(raw), raw)
                continue
            msg_type = data.get("type", "")

            # ---- hello: 握手认证 + 绑定会话 ----
            if msg_type == "hello":
                token = data.get("token", "")
                device_id = data.get("device_id", "default")
                client_session_id = data.get("session_id", "")
                logger.info("hello received: device_id=%s token=%s session_id=%s",
                            device_id, token[:8] if token else "(empty)", client_session_id or "(empty)")
                if token != settings.api_token:
                    agent_monitor.warn(
                        f"WS 认证失败 device={device_id} "
                        f"token={token[:8] if token else '(empty)'}…"
                    )
                    logger.warning("hello token mismatch: client=%s server=%s",
                                   token[:8] if token else "(empty)",
                                   settings.api_token[:8] if settings.api_token else "(empty)")
                    await websocket.send_json({"type": "error", "code": "unauthorized", "message": "invalid token"})
                    await websocket.close()
                    return
                try:
                    conn = DeviceConnection(websocket=websocket, device_id=device_id, session_id=client_session_id)
                    connections[device_id] = conn
                    conn.session_id = session_store.get_or_create_session(device_id, client_session_id or None)
                    await websocket.send_json({"type": "hello_ack", "session_id": conn.session_id})
                    agent_monitor.event(
                        f"WS 已连接 device={device_id} session={conn.session_id[:12]}…"
                    )
                    logger.info("hello_ack sent: device_id=%s session_id=%s", device_id, conn.session_id)
                    silence_timeout = None
                except Exception as exc:
                    logger.exception("hello processing crashed for device_id=%s: %s", device_id, exc)
                    try:
                        await websocket.send_json({"type": "error", "code": "internal_error", "message": str(exc)[:200]})
                    except Exception:
                        pass
                    await websocket.close()
                    return
                continue

            # 未握手先发其他消息：拒绝
            if conn is None:
                await websocket.send_json({"type": "error", "code": "not_handshaken", "message": "send hello first"})
                continue

            # ---- ping: 心跳保活 ----
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                conn.last_active = datetime.now(timezone.utc)
                continue

            # ---- abort: 打断当前 TTS ----
            if msg_type == "abort":
                conn.last_active = datetime.now(timezone.utc)
                logger.info("abort received from device=%s (handled in chat flow if active)", conn.device_id)
                await websocket.send_json({
                    "type": "chat_done",
                    "session_id": conn.session_id or "",
                })
                continue

            # ---- chat: 对话消息 ----
            if msg_type == "chat":
                message = data.get("message", "").strip()
                if not message:
                    await websocket.send_json({"type": "error", "code": "empty_message", "message": "empty message"})
                    continue
                try:
                    tts_cfg = _get_tts_config() if _tts_enabled() else None
                    active_tts = ActiveTTSContext()

                    # 流式生成：边收 token 边累积发送气泡
                    reply_parts: list[str] = []
                    sent_bubbles: int = 0
                    final_data: tuple | None = None
                    abort_seen: bool = False

                    chat_done = asyncio.Event()

                    async def _abort_watcher():
                        """Monitor for abort messages during chat processing."""
                        nonlocal abort_seen
                        while not abort_seen and not chat_done.is_set():
                            try:
                                raw = await asyncio.wait_for(
                                    websocket.receive_text(), timeout=0.5
                                )
                                data = json.loads(raw)
                                if data.get("type") == "abort":
                                    abort_seen = True
                                    active_tts.cancelled = True
                                    if active_tts.tts_task:
                                        active_tts.tts_task.cancel()
                                    logger.info("TTS aborted for device=%s", conn.device_id)
                                    return
                            except asyncio.TimeoutError:
                                continue
                            except Exception:
                                pass

                    async def _stream_handle():
                        nonlocal final_data, sent_bubbles, abort_seen
                        try:
                            await websocket.send_json({
                                "type": "reply_start",
                                "session_id": conn.session_id or "",
                            })
                            async for event, data in handle_chat_stream(conn.device_id, conn.session_id, message):
                                if abort_seen or active_tts.cancelled:
                                    return
                                if event == "token":
                                    reply_parts.append(data)
                                    await websocket.send_json({
                                        "type": "reply_token",
                                        "text": data,
                                        "session_id": conn.session_id or "",
                                    })
                                    current = "".join(reply_parts)
                                    bubbles = _split_into_bubbles(current)
                                    while sent_bubbles < len(bubbles) - 1:
                                        bubble_text = bubbles[sent_bubbles]
                                        if tts_cfg:
                                            alive = await _send_bubble_with_tts(
                                                websocket, bubble_text, sent_bubbles,
                                                tts_cfg, active_tts, conn.session_id or "",
                                            )
                                            if not alive:
                                                abort_seen = True
                                                return
                                        else:
                                            await _send_textonly_bubble(
                                                websocket, bubble_text, sent_bubbles,
                                                conn.session_id or "",
                                            )
                                        sent_bubbles += 1
                                elif event == "done":
                                    final_data = data
                                elif event == "error":
                                    await websocket.send_json(
                                        {"type": "error", "code": "llm_error", "message": data[:200]}
                                    )
                                    await websocket.send_json({
                                        "type": "chat_done",
                                        "session_id": conn.session_id or "",
                                    })
                                    return
                        finally:
                            chat_done.set()

                    abort_task = asyncio.create_task(_abort_watcher())
                    try:
                        await asyncio.wait_for(_stream_handle(), timeout=45.0)
                    except asyncio.TimeoutError:
                        abort_seen = True
                        active_tts.cancelled = True
                        if active_tts.tts_task:
                            active_tts.tts_task.cancel()
                        await websocket.send_json({
                            "type": "error",
                            "code": "llm_timeout",
                            "message": "响应超时",
                        })
                        await websocket.send_json({
                            "type": "chat_done",
                            "session_id": conn.session_id or "",
                        })
                        abort_task.cancel()
                        try:
                            await abort_task
                        except asyncio.CancelledError:
                            pass
                        _reset_silence_state()
                        silence_timeout = _next_silence_timeout()
                        continue
                    finally:
                        chat_done.set()
                        abort_task.cancel()
                        try:
                            await abort_task
                        except asyncio.CancelledError:
                            pass

                    if abort_seen or active_tts.cancelled:
                        await websocket.send_json({
                            "type": "chat_done",
                            "session_id": conn.session_id or "",
                        })
                        _reset_silence_state()
                        silence_timeout = _next_silence_timeout()
                        continue

                    if final_data is None:
                        await websocket.send_json({
                            "type": "error",
                            "code": "llm_timeout",
                            "message": "响应超时",
                        })
                        await websocket.send_json({
                            "type": "chat_done",
                            "session_id": conn.session_id or "",
                        })
                        continue

                    reply, session_id, follow_up = final_data
                    conn.session_id = session_id
                    conn.last_active = datetime.now(timezone.utc)

                    # 发送剩余未发送的气泡
                    bubbles = _split_into_bubbles(reply)
                    while sent_bubbles < len(bubbles):
                        bubble_text = bubbles[sent_bubbles]
                        if tts_cfg:
                            alive = await _send_bubble_with_tts(
                                websocket, bubble_text, sent_bubbles,
                                tts_cfg, active_tts, conn.session_id or "",
                            )
                            if not alive:
                                break
                        else:
                            await _send_textonly_bubble(
                                websocket, bubble_text, sent_bubbles,
                                conn.session_id or "",
                            )
                        sent_bubbles += 1

                    # 主动话题（text-only，不 TTS）
                    if follow_up and not active_tts.cancelled:
                        if tts_cfg:
                            await websocket.send_json({
                                "type": "follow_up", "text": follow_up,
                                "session_id": session_id,
                            })
                        else:
                            await asyncio.sleep(0.5 + random.random() * 0.5)
                            await websocket.send_json({
                                "type": "follow_up", "text": follow_up,
                                "session_id": session_id,
                            })

                    # chat_done: 本轮对话结束
                    if not active_tts.cancelled:
                        await websocket.send_json({
                            "type": "chat_done", "session_id": session_id,
                        })

                    # 模式切换口令不开启沉默监听
                    if is_mode_switch_message(message):
                        silence_timeout = None
                    else:
                        silence_timeout = _next_silence_timeout()
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "error", "code": "llm_timeout", "message": "响应超时"})
                except Exception as exc:
                    agent_monitor.warn(f"WS chat 异常 device={conn.device_id}: {exc}")
                    logger.exception("chat handler crashed: %s", exc)
                    await websocket.send_json(
                        {"type": "error", "code": "internal_error", "message": str(exc)[:200]}
                    )
                continue

            # ---- 非 chat 消息：关闭沉默超时 ----
            silence_timeout = None

            # ---- session_end: 主动结束会话 ----
            if msg_type == "session_end":
                conn.session_id = await handle_session_end(conn.device_id, conn.session_id) or ""
                await websocket.send_json({"type": "session_end_ack", "session_id": conn.session_id})
                continue

            # ---- new_session: 强制新会话 ----
            if msg_type == "new_session":
                if conn.session_id:
                    conn.session_id = await handle_session_end(conn.device_id, conn.session_id) or ""
                else:
                    conn.session_id = session_store.get_or_create_session(conn.device_id, None)
                await websocket.send_json({"type": "hello_ack", "session_id": conn.session_id})
                continue

            # 未知消息类型
            await websocket.send_json({"type": "error", "code": "unknown_type", "message": f"unknown type: {msg_type}"})

    except WebSocketDisconnect:
        # 客户端主动断开：静默处理，不做额外清理
        logger.info("WebSocket disconnected: device_id=%s", conn.device_id if conn else "(no hello)")
    except Exception as exc:
        logger.exception("WebSocket handler crashed: device_id=%s, error=%s",
                         conn.device_id if conn else "(no hello)", exc)
    finally:
        # 从全局连接表中移除（仅当该 WebSocket 对象未被新连接覆盖时）
        if conn and connections.get(conn.device_id) and connections[conn.device_id].websocket is websocket:
            connections.pop(conn.device_id, None)


async def idle_session_sweeper() -> None:
    """后台定时任务：每分钟扫描空闲连接和过期 HTTP 会话，自动触发 session_end。

    工作原理：
      1. 遍历活跃的 WebSocket 连接，对超过 session_idle_minutes 无活动的连接调用 handle_session_end
      2. 对 HTTP 创建的会话，通过 store.list_idle_active_sessions() 查询过期会话，
         过滤掉仍在活跃 WS 连接中的，对其余执行 consolidate_session（L1→L2 压缩）
      3. 每 tick 最多处理 20 个过期 HTTP 会话，避免一次性负载过大

    为什么需要这个机制：
      - 用户可能直接关闭浏览器/微信而不发送 session_end 消息
      - L1 消息积累太多会占用内存，需要定期清理
      - 实名用户的 L1 需要压入 L2 才能形成长期记忆
    """
    from app.memory.extractor import consolidate_session
    from app.session import store

    max_per_tick = 20  # 每 tick 最多处理的 HTTP 会话数，防止瞬时负载过高

    while True:
        await asyncio.sleep(60)  # 每分钟扫描一次
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=settings.session_idle_minutes)
        cutoff_iso = cutoff.isoformat()

        # 第一步：清理空闲的 WebSocket 连接
        for device_id, conn in list(connections.items()):
            if conn.last_active < cutoff and conn.session_id:
                try:
                    conn.session_id = await handle_session_end(conn.device_id, conn.session_id) or ""
                except Exception as exc:
                    agent_monitor.warn(f"空闲 WS 会话整理失败: {exc}")

        # 第二步：清理空闲的 HTTP 会话（非 WebSocket）
        # HTTP /v1/chat 接口不建立持久连接，会话只能靠这个 sweeper 清理。
        # 收集当前活跃 WS 的 session_id，避免重复处理（第一步已照料 WS 会话）。
        ws_session_ids = {c.session_id for c in connections.values() if c.session_id}
        idle_rows = store.list_idle_active_sessions(cutoff_iso)[:max_per_tick]
        closed = 0
        for row in idle_rows:
            sid = str(row["id"])
            # 跳过仍在 WebSocket 活跃的会话（已在第一步处理）
            if sid in ws_session_ids:
                continue
            try:
                # consolidate_session 包含 L1→L2 压缩和画像转正，是同步操作；
                # 通过 asyncio.to_thread 放到线程池，避免阻塞事件循环。
                await asyncio.to_thread(consolidate_session, str(row["device_id"]), sid)
                closed += 1
            except Exception as exc:
                agent_monitor.warn(f"HTTP 空闲会话整理失败 {sid[:8]}: {exc}")
        if closed:
            agent_monitor.event(f"空闲整理 · {closed} 个历史会话已压缩进 L2")
