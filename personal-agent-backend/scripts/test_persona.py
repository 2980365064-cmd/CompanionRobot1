#!/usr/bin/env python3
"""Persona 冒烟测试脚本。

用途：在不需要硬件设备的本地环境下，对陪伴机器人的人设/人格系统进行快速验证。
通过发送几个典型对话问题，观察机器人的回复是否符合预期的人设和口吻。

测试内容：
1. 加载 Profile Card（人设卡）—— 验证 persona.md 和 style 范例是否正确加载
2. 发送情感问候类问题 —— 观察回复是否温暖、符合角色设定
3. 发送日常习惯类问题 —— 观察是否从记忆中回忆起相关信息
4. 发送偏好记忆类问题 —— 观察 L3 长期记忆召回是否工作正常

测试流程：
  - 每个问题依次发送，携带之前的多轮对话上下文
  - 通过 memory_router.recall 检索相关记忆
  - 调用 chat_completion 生成回复
  - 将对话存入 session store，模拟完整对话过程

典型用法：
    python scripts/test_persona.py

前置条件：
    - LLM_API_KEY 已配置
    - persona/corpus/ 语料已入库（python scripts/ingest.py）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import build_messages
from app.llm import chat_completion
from app.persona.card import load_profile_card
from app.memory.router import memory_router
from app.session import store


def main() -> None:
    """执行 persona 冒烟测试，用几个典型问题验证人设系统。"""
    device_id = "test-device"
    # 创建测试 session，模拟一次完整的对话过程
    session_id = store.get_or_create_session(device_id, None)

    # 测试问题集：覆盖情感问候、日常习惯、偏好记忆三个维度
    questions = [
        "今天心情怎么样？",       # 情感问候 —— 验证温暖、陪伴的语气
        "周末一般干嘛？",          # 日常习惯 —— 验证 L3 语料检索 + persona 中写明的日常
        "你还记得我喜欢喝什么咖啡？", # 偏好记忆 —— 验证 L3 长期记忆能否正确召回
    ]

    # 加载机器人的人格卡（每轮对话固定注入 system prompt）
    persona = load_profile_card()
    print("=== Persona test ===\n")

    for q in questions:
        # 从多层记忆系统中检索与当前问题相关的内容
        memory = memory_router.recall(device_id, session_id, q, person_id="")
        # 构建 LLM 消息列表（system prompt = persona + memory 上下文）
        messages = build_messages(persona, memory, q, device_id=device_id)
        # 调用大模型生成回复
        reply = chat_completion(messages)

        # 将本轮对话写入 session store，供后续轮次作为短期记忆引用
        store.add_message(session_id, "user", q)
        store.add_message(session_id, "assistant", reply)

        # 打印问答对，方便人工评估回复质量
        print(f"Q: {q}")
        print(f"A: {reply}\n")


if __name__ == "__main__":
    main()
