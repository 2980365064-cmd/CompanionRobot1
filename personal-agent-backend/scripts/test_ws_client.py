#!/usr/bin/env python3
"""Interactive WebSocket client — simulates the robot without hardware."""

import asyncio
import json
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Install: pip install websockets")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

DEFAULT_URI = "ws://127.0.0.1:8000/ws/v1/chat"


async def run_client(uri: str, token: str, device_id: str) -> None:
    print(f"Connecting to {uri} ...")
    async with websockets.connect(uri) as ws:
        hello = {
            "type": "hello",
            "device_id": device_id,
            "token": token,
            "session_id": "",
        }
        await ws.send(json.dumps(hello, ensure_ascii=False))
        ack = json.loads(await ws.recv())
        print(f"<< {json.dumps(ack, ensure_ascii=False)}")
        if ack.get("type") != "hello_ack":
            print("Handshake failed.")
            return

        session_id = ack.get("session_id", "")
        print(f"\nSession: {session_id}")
        print("Type a message (empty line or 'quit' to exit, 'new' for new session):\n")

        while True:
            try:
                user_input = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input or user_input.lower() in {"quit", "exit", "q"}:
                await ws.send(json.dumps({"type": "session_end"}))
                try:
                    end_ack = await asyncio.wait_for(ws.recv(), timeout=5)
                    print(f"<< {end_ack}")
                except asyncio.TimeoutError:
                    pass
                break

            if user_input.lower() == "new":
                await ws.send(json.dumps({"type": "new_session"}))
                ack = json.loads(await ws.recv())
                session_id = ack.get("session_id", session_id)
                print(f"<< new session: {session_id}")
                continue

            await ws.send(json.dumps({"type": "chat", "message": user_input}, ensure_ascii=False))
            reply = json.loads(await ws.recv())
            print(f"Bot> {reply.get('text', reply)}\n")
            if reply.get("session_id"):
                session_id = reply["session_id"]


def main() -> None:
    uri = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URI
    token = sys.argv[2] if len(sys.argv) > 2 else settings.api_token
    device_id = sys.argv[3] if len(sys.argv) > 3 else "test-robot"
    asyncio.run(run_client(uri, token, device_id))


if __name__ == "__main__":
    main()
