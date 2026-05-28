#!/usr/bin/env python3
"""HTTP chat test — simplest way to test backend without WebSocket or hardware."""

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

BASE = f"http://127.0.0.1:{settings.port}"


def main() -> None:
    headers = {"X-API-Token": settings.api_token}
    print(f"POST {BASE}/v1/chat\n")

    session_id = ""
    prompts = [
        "你好，今天心情怎么样？",
        "周末一般喜欢做什么？",
    ]
    if len(sys.argv) > 1:
        prompts = sys.argv[1:]

    for msg in prompts:
        payload = {
            "device_id": "test-http",
            "session_id": session_id,
            "message": msg,
        }
        r = httpx.post(f"{BASE}/v1/chat", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        session_id = data.get("session_id", session_id)
        print(f"You> {msg}")
        print(f"Bot> {data['reply']}\n")

    print(f"session_id={session_id}")


if __name__ == "__main__":
    main()
