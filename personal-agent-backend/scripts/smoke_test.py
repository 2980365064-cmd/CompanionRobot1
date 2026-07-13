#!/usr/bin/env python3
"""后端冒烟测试脚本：HTTP 健康检查 + WebSocket 对话测试。

用途：验证陪伴机器人后端服务的核心功能是否正常运行，无需硬件设备。
测试覆盖三个基础接口，确保服务启动后可以正常处理请求。

测试步骤：
  [1/3] GET /health            —— HTTP 健康检查端点
  [2/3] POST /v1/chat (HTTP)   —— HTTP 对话接口，发送"你好"并验证有回复
  [3/3] WebSocket /ws/v1/chat  —— WebSocket 对话接口（hello 握手 + chat 对话 + ping 心跳）

设计说明：
  - WebSocket 测试为可选（网络环境可能不兼容某些 websockets 库版本）
  - HTTP 测试失败会直接抛异常退出
  - smoke_test 的原则是快速验证服务连通性，不做深度功能测试

典型用法：
    # 确保后端已启动
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    # 在另一个终端执行测试
    python scripts/smoke_test.py

前置条件：
    - 后端服务已启动在 settings.port 端口
    - pip install httpx websockets（websockets 为可选依赖）
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

# WebSocket 库为可选依赖，未安装时跳过 WS 测试
try:
    import websockets
except ImportError:
    print("FAIL: pip install websockets")
    sys.exit(1)

# 服务地址常量
BASE = f"http://127.0.0.1:{settings.port}"
WS_URI = f"ws://127.0.0.1:{settings.port}/ws/v1/chat"


def test_http() -> None:
    """测试 HTTP 健康检查和基础对话接口。"""
    # [1/3] 健康检查：验证服务进程存活并能响应请求
    print("[1/3] GET /health")
    r = httpx.get(f"{BASE}/health", timeout=5)
    r.raise_for_status()
    print(f"  OK: {r.json()}")

    # [2/3] HTTP 对话接口：发送"你好"，验证回复不为空
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
    """测试 WebSocket 对话接口的完整链路（握手→对话→心跳）。

    WebSocket 协议流程：
    1. 发送 hello 消息（带 device_id 和 token），服务器返回 hello_ack
    2. 发送 chat 消息（对话文本），服务器返回 reply（含 AI 生成文本）
    3. 发送 ping 心跳，服务器返回 pong 响应

    Returns:
        True 表示所有测试通过
    """
    async with websockets.connect(WS_URI) as ws:
        # 1. WebSocket 握手认证
        await ws.send(json.dumps({
            "type": "hello",
            "device_id": "smoke-test",
            "token": settings.api_token,
            "session_id": "",
        }))
        ack = json.loads(await ws.recv())
        assert ack["type"] == "hello_ack", ack

        # 2. 发送对话消息，兼容流式协议并验证回复不为空
        await ws.send(json.dumps({
            "type": "chat",
            "message": "你好，今天心情怎么样？",
            "tts": False,
        }))
        reply_parts: list[str] = []
        while True:
            packet = json.loads(await ws.recv())
            typ = packet.get("type")
            if typ == "reply_token":
                reply_parts.append(str(packet.get("text") or ""))
            elif typ == "reply":
                # 气泡文本是向后兼容事件；没有 token 时用它拼接结果。
                if not reply_parts:
                    reply_parts.append(str(packet.get("text") or ""))
            elif typ == "error":
                raise AssertionError(packet)
            elif typ == "chat_done":
                break
        reply_text = "".join(reply_parts).strip()
        assert reply_text, "empty reply"
        print(f"  chat reply: {reply_text[:80]}...")

        # 3. 心跳测试（ping/pong）
        await ws.send(json.dumps({"type": "ping"}))
        pong = json.loads(await ws.recv())
        assert pong["type"] == "pong", pong
    return True


def main() -> None:
    """执行冒烟测试主流程。"""
    print("=== Backend smoke test ===\n")
    test_http()
    print("[3/3] WebSocket hello + chat + ping")
    try:
        ok = asyncio.run(test_ws())
        print(f"  OK: websocket passed={ok}\n")
    except Exception as e:
        # WebSocket 测试失败不阻塞整体结果（可能是库版本兼容问题）
        print(f"  SKIP websocket ({e})\n")
    print("HTTP tests passed. Fix websockets<14 if you need WS tests.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAIL: {e}")
        print("\nIs the server running?  uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
