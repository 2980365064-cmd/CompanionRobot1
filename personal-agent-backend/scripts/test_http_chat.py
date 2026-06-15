#!/usr/bin/env python3
"""HTTP 对话测试脚本 —— 最简单的后端验证方式。

用途：通过 HTTP POST 请求测试陪伴机器人的对话接口，无需 WebSocket 或硬件设备。
适合开发时快速验证 LLM 配置、人设注入和记忆召回是否正常。

与 test_ws_client.py 的区别：
  - 本脚本用 HTTP REST 接口（POST /v1/chat），更简单、更稳定
  - test_ws_client.py 用 WebSocket，支持流式输出和更丰富的交互

测试流程：
  1. 向 /v1/chat 发送消息（携带 device_id、session_id、API token）
  2. 从响应中提取 session_id 用于保持多轮对话上下文
  3. 从响应中提取 AI 回复文本并打印

支持自定义输入参数：
    python scripts/test_http_chat.py                    # 使用内置问题
    python scripts/test_http_chat.py "今天心情如何？"    # 自定义单轮问题
    python scripts/test_http_chat.py "问题1" "问题2"     # 自定义多轮对话

前置条件：
    - 后端服务已启动
    - LLM_API_KEY 已配置
"""

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

BASE = f"http://127.0.0.1:{settings.port}"


def main() -> None:
    """发送 HTTP 对话请求并打印回复。"""
    headers = {"X-API-Token": settings.api_token}
    print(f"POST {BASE}/v1/chat\n")

    # 初始化 session_id 为空，后端会自动创建新会话
    session_id = ""

    # 默认测试问题：一个情感问候 + 一个日常习惯
    prompts = [
        "你好，今天心情怎么样？",
        "周末一般喜欢做什么？",
    ]
    # 允许命令行参数覆盖默认问题
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
        # 保持 session_id 以维持多轮对话上下文
        session_id = data.get("session_id", session_id)
        print(f"You> {msg}")
        print(f"Bot> {data['reply']}\n")

    print(f"session_id={session_id}")


if __name__ == "__main__":
    main()
