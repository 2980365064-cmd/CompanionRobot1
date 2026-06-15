#!/usr/bin/env python3
"""WebSocket 对话客户端 —— 模拟硬件机器人的交互方式。

用途：通过 WebSocket 协议与陪伴机器人后端建立长连接，
模拟真实机器人硬件的聊天交互流程，用于开发和测试。

支持的命令：
  - 直接输入文本 → 发送 chat 消息，获取 AI 回复
  - new           → 创建新的对话会话
  - quit/exit/q   → 发送 session_end 并退出
  - 空行          → 同上，安全退出

WebSocket 协议交互流程：
  1. 发送 hello（握手认证：device_id + token）
  2. 收到 hello_ack（握手确认，含 session_id）
  3. 循环发送 chat 消息，收到 reply 回复
  4. 可选发送 session_end 结束会话
  5. 连接关闭

与 HTTP 接口的区别：
  - WS 是长连接，适合嵌入式设备持续对话
  - 支持 session 管理（new_session / session_end）
  - 每个消息有类型标识（hello/chat/reply/ping/pong）
  - 实际机器人硬件通过 ESP32 等设备使用 WS 协议

典型用法：
    python scripts/test_ws_client.py                                # 默认 localhost:8000
    python scripts/test_ws_client.py ws://myserver:8000/ws/v1/chat  # 指定服务器
    python scripts/test_ws_client.py ws://localhost:8000/ws/v1/chat mytoken mydevice

前置条件：
    - 后端服务已启动
    - pip install websockets
"""

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
    """运行 WebSocket 客户端，进入交互式对话循环。

    Args:
        uri: WebSocket 服务器地址
        token: API 认证 token
        device_id: 设备标识（模拟不同硬件设备）
    """
    print(f"Connecting to {uri} ...")
    async with websockets.connect(uri) as ws:
        # ---------- 第 1 步：发送握手认证 ----------
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

        # ---------- 第 2 步：交互式对话循环 ----------
        while True:
            try:
                user_input = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            # 退出命令：发送 session_end 通知服务器归档当前会话
            if not user_input or user_input.lower() in {"quit", "exit", "q"}:
                await ws.send(json.dumps({"type": "session_end"}))
                try:
                    end_ack = await asyncio.wait_for(ws.recv(), timeout=5)
                    print(f"<< {end_ack}")
                except asyncio.TimeoutError:
                    pass
                break

            # 创建新会话命令
            if user_input.lower() == "new":
                await ws.send(json.dumps({"type": "new_session"}))
                ack = json.loads(await ws.recv())
                session_id = ack.get("session_id", session_id)
                print(f"<< new session: {session_id}")
                continue

            # 发送对话消息
            # ensure_ascii=False 确保中文不被转义为 \\uXXXX
            await ws.send(json.dumps({"type": "chat", "message": user_input}, ensure_ascii=False))
            reply = json.loads(await ws.recv())
            print(f"Bot> {reply.get('text', reply)}\n")
            # 更新 session_id（服务器可能在回复中返回新的 session_id）
            if reply.get("session_id"):
                session_id = reply["session_id"]


def main() -> None:
    """解析命令行参数并启动客户端。"""
    uri = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URI
    token = sys.argv[2] if len(sys.argv) > 2 else settings.api_token
    device_id = sys.argv[3] if len(sys.argv) > 3 else "test-robot"
    asyncio.run(run_client(uri, token, device_id))


if __name__ == "__main__":
    main()
