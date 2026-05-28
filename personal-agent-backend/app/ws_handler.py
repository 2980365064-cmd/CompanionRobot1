"""WebSocket protocol handler."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from fastapi import WebSocket, WebSocketDisconnect

from app.agent import handle_chat, handle_session_end
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DeviceConnection:
    websocket: WebSocket
    device_id: str
    session_id: str = ""
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


connections: dict[str, DeviceConnection] = {}


async def ws_chat_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    conn: DeviceConnection | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "")

            if msg_type == "hello":
                token = data.get("token", "")
                if token != settings.api_token:
                    await websocket.send_json({"type": "error", "code": "unauthorized", "message": "invalid token"})
                    await websocket.close()
                    return
                device_id = data.get("device_id", "default")
                session_id = data.get("session_id", "")
                conn = DeviceConnection(websocket=websocket, device_id=device_id, session_id=session_id)
                connections[device_id] = conn
                from app.session import store

                conn.session_id = store.get_or_create_session(device_id, session_id or None)
                await websocket.send_json({"type": "hello_ack", "session_id": conn.session_id})
                continue

            if conn is None:
                await websocket.send_json({"type": "error", "code": "not_handshaken", "message": "send hello first"})
                continue

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                conn.last_active = datetime.now(timezone.utc)
                continue

            if msg_type == "chat":
                message = data.get("message", "").strip()
                if not message:
                    await websocket.send_json({"type": "error", "code": "empty_message", "message": "empty message"})
                    continue
                try:
                    reply, session_id = await asyncio.wait_for(
                        handle_chat(conn.device_id, conn.session_id, message),
                        timeout=45.0,
                    )
                    conn.session_id = session_id
                    conn.last_active = datetime.now(timezone.utc)
                    await websocket.send_json({"type": "reply", "text": reply, "session_id": session_id})
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "error", "code": "llm_timeout", "message": "响应超时"})
                continue

            if msg_type == "session_end":
                await handle_session_end(conn.device_id, conn.session_id)
                from app.session import store
                from uuid import uuid4

                conn.session_id = str(uuid4())
                store.get_or_create_session(conn.device_id, None)
                await websocket.send_json({"type": "session_end_ack", "session_id": conn.session_id})
                continue

            if msg_type == "new_session":
                from app.session import store
                from uuid import uuid4

                if conn.session_id:
                    await handle_session_end(conn.device_id, conn.session_id)
                conn.session_id = store.get_or_create_session(conn.device_id, None)
                await websocket.send_json({"type": "hello_ack", "session_id": conn.session_id})
                continue

            await websocket.send_json({"type": "error", "code": "unknown_type", "message": f"unknown type: {msg_type}"})

    except WebSocketDisconnect:
        pass
    finally:
        if conn and connections.get(conn.device_id) and connections[conn.device_id].websocket is websocket:
            connections.pop(conn.device_id, None)


async def idle_session_sweeper() -> None:
    """Close idle sessions and consolidate to L2."""
    while True:
        await asyncio.sleep(60)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=settings.session_idle_minutes)
        # Simple sweep: consolidate connections idle too long
        for device_id, conn in list(connections.items()):
            if conn.last_active < cutoff and conn.session_id:
                try:
                    await handle_session_end(conn.device_id, conn.session_id)
                    from app.session import store

                    conn.session_id = store.get_or_create_session(conn.device_id, None)
                except Exception as exc:
                    from app.monitor import agent_monitor

                    agent_monitor.warn(f"空闲会话整理失败: {exc}")
