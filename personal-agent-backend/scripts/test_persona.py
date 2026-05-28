#!/usr/bin/env python3
"""Local persona smoke test without hardware."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import build_messages
from app.llm import chat_completion
from app.memory.profile import load_profile_card
from app.memory.router import memory_router
from app.session import store


def main() -> None:
    device_id = "test-device"
    session_id = store.get_or_create_session(device_id, None)
    questions = [
        "今天心情怎么样？",
        "周末一般干嘛？",
        "你还记得我喜欢喝什么咖啡？",
    ]
    persona = load_profile_card()
    print("=== Persona test ===\n")
    for q in questions:
        memory = memory_router.recall(device_id, session_id, q)
        messages = build_messages(persona, memory, q, device_id=device_id)
        reply = chat_completion(messages)
        store.add_message(session_id, "user", q)
        store.add_message(session_id, "assistant", reply)
        print(f"Q: {q}")
        print(f"A: {reply}\n")


if __name__ == "__main__":
    main()
