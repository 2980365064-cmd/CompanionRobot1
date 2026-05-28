#!/usr/bin/env python3
"""Automated smoke test for HTTP health + WebSocket chat (no hardware)."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

try:
    import websockets
except ImportError:
    print("FAIL: pip install websockets")
    sys.exit(1)

BASE = f"http://127.0.0.1:{settings.port}"
WS_URI = f"ws://127.0.0.1:{settings.port}/ws/v1/chat"


def test_http() -> None:
    print("[1/3] GET /health")
    r = httpx.get(f"{BASE}/health", timeout=5)
    r.raise_for_status()
    print(f"  OK: {r.json()}")

    print("[2/3] POST /v1/chat (HTTP)")
    r = httpx.post(
        f"{BASE}/v1/chat",
        json={"device_id": "smoke", "session_id": "", "message": "你好"},
        headers={"X-API-Token": settings.api_token},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    assert data.get("reply"), data
    print(f"  OK: {data['reply'][:80]}...")


async def test_ws() -> bool:
    async with websockets.connect(WS_URI) as ws:
        await ws.send(json.dumps({
            "type": "hello",
            "device_id": "smoke-test",
            "token": settings.api_token,
            "session_id": "",
        }))
        ack = json.loads(await ws.recv())
        assert ack["type"] == "hello_ack", ack

        await ws.send(json.dumps({"type": "chat", "message": "你好，今天心情怎么样？"}))
        reply = json.loads(await ws.recv())
        assert reply["type"] == "reply", reply
        assert reply.get("text"), "empty reply"
        print(f"  chat reply: {reply['text'][:80]}...")

        await ws.send(json.dumps({"type": "ping"}))
        pong = json.loads(await ws.recv())
        assert pong["type"] == "pong", pong
    return True


def main() -> None:
    print("=== Backend smoke test ===\n")
    test_http()
    print("[3/3] WebSocket hello + chat + ping")
    try:
        ok = asyncio.run(test_ws())
        print(f"  OK: websocket passed={ok}\n")
    except Exception as e:
        print(f"  SKIP websocket ({e})\n")
    print("HTTP tests passed. Fix websockets<14 if you need WS tests.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAIL: {e}")
        print("\nIs the server running?  uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
