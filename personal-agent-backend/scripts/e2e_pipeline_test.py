#!/usr/bin/env python3
"""全链路冒烟：persona 加载 → interlocutor → prompt 组装 → handle_chat（需 LLM）。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import build_messages, handle_chat
from app.config import settings
from app.memory.interlocutor import (
    MODE_GIRLFRIEND,
    MODE_VISITOR,
    resolve_interlocutor_before_memory,
)
from app.memory.router import memory_router
from app.persona.card import intimate_topic_relevant, load_profile_card
from app.session import store

PASS = 0
FAIL = 0


def ok(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def test_profile_loading() -> None:
    section("1. Profile Card 加载")
    gf_full = load_profile_card(MODE_GIRLFRIEND, "你喜欢SM吗")
    gf_casual = load_profile_card(MODE_GIRLFRIEND, "今天吃了啥")
    vis = load_profile_card(MODE_VISITOR, "你好")
    ok("女友+私密话题含性癖段", "淫乱" in gf_full)
    ok("女友+日常不含性癖段", "淫乱" not in gf_casual)
    ok("访客不含性癖段", "淫乱" not in vis)
    ok("intimate_topic_relevant(SM)", intimate_topic_relevant("你喜欢SM吗"))
    ok("intimate_topic_relevant(日常)", not intimate_topic_relevant("今天吃了啥"))


def test_interlocutor(device_id: str, session_id: str) -> None:
    section("2. Interlocutor 模式切换")
    r1 = resolve_interlocutor_before_memory(device_id, session_id, "访客模式")
    ok("访客切换 ack", r1.mode_switch_ack == "访客模式开启")
    ok("访客 mode", r1.interlocutor_mode == MODE_VISITOR)
    r2 = resolve_interlocutor_before_memory(device_id, session_id, "你好呀")
    ok("访客保持", r2.interlocutor_mode == MODE_VISITOR)
    ok("访客 hint 含大大咧咧", "大大咧咧" in (r2.hint or ""))
    r3 = resolve_interlocutor_before_memory(device_id, session_id, "女友模式")
    ok("女友切换 ack", r3.mode_switch_ack == "女友模式开启")
    ok("女友 mode", r3.interlocutor_mode == MODE_GIRLFRIEND)
    r4 = resolve_interlocutor_before_memory(device_id, session_id, "想你了")
    ok("女友 hint 含刘远慧", "刘远慧" in (r4.hint or ""))


def test_build_messages(device_id: str, session_id: str) -> None:
    section("3. Prompt 组装")
    ictx = resolve_interlocutor_before_memory(device_id, session_id, "想你了")
    profile = load_profile_card(ictx.interlocutor_mode, "想你了")
    memory = memory_router.recall(
        device_id, session_id, "想你了", person_id=ictx.person_id,
    )
    memory["interlocutor_mode"] = ictx.interlocutor_mode
    memory["identity_hint"] = ictx.hint
    msgs = build_messages(
        profile, memory, "想你了", device_id=device_id, person_profile=ictx.person_profile,
    )
    system = msgs[0]["content"]
    ok("system 含场景路由", "场景路由" in system or "路由" in system)
    ok("system 不含性癖(日常)", "淫乱" not in system)
    ok("system 含对话角色", "对话角色" in system or "女友" in system)

    ictx_v = resolve_interlocutor_before_memory(device_id, session_id, "访客模式")
    profile_v = load_profile_card(ictx_v.interlocutor_mode, "你好")
    memory_v = memory_router.recall(
        device_id, session_id, "你好", person_id=ictx_v.person_id,
    )
    memory_v["interlocutor_mode"] = MODE_VISITOR
    memory_v["identity_hint"] = ictx_v.hint
    msgs_v = build_messages(
        profile_v, memory_v, "你好", device_id=device_id, person_profile=ictx_v.person_profile,
    )
    sys_v = msgs_v[0]["content"]
    ok("访客 system 不含性癖", "淫乱" not in sys_v)
    ok("访客 system 含访客模式", "访客" in sys_v)


async def test_handle_chat(device_id: str) -> None:
    section("4. handle_chat（调用 LLM）")
    session_id = store.get_or_create_session(device_id, None)
    cases = [
        ("想你了", MODE_GIRLFRIEND, None),
        ("访客模式", MODE_VISITOR, "访客模式开启"),
        ("你好，我是你朋友", MODE_VISITOR, None),
        ("女友模式", MODE_GIRLFRIEND, "女友模式开启"),
    ]
    for msg, _exp_mode, exp_ack in cases:
        try:
            reply, sid, _topic = await handle_chat(device_id, session_id, msg)
            session_id = sid
            mode = store.get_session_interlocutor_mode(session_id) or MODE_GIRLFRIEND
            ok(f"LLM 回复非空: {msg[:12]}", bool(reply.strip()), reply[:60])
            if exp_ack:
                ok(f"含确认语 {exp_ack}", exp_ack in reply, reply[:80])
            if mode == MODE_VISITOR and "访客" not in msg:
                ok("访客回复不含性癖词", "淫乱" not in reply and "深喉" not in reply)
        except Exception as exc:
            ok(f"handle_chat: {msg[:12]}", False, str(exc)[:120])


async def test_ws(uri: str, token: str) -> None:
    section("5. WebSocket 链路")
    try:
        import websockets
    except ImportError:
        ok("websockets 已安装", False, "pip install websockets")
        return

    try:
        async with websockets.connect(uri, open_timeout=8) as ws:
            await ws.send(json.dumps({
                "type": "hello",
                "device_id": "e2e-ws",
                "token": token,
                "session_id": "",
            }, ensure_ascii=False))
            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            ok("hello_ack", ack.get("type") == "hello_ack", str(ack.get("session_id", ""))[:8])
            await ws.send(json.dumps({"type": "chat", "message": "在吗"}, ensure_ascii=False))
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
            ok("chat reply", reply.get("type") == "reply" and bool(reply.get("text")), reply.get("text", "")[:60])
            await ws.send(json.dumps({"type": "ping"}))
            pong = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            ok("ping/pong", pong.get("type") == "pong")
    except Exception as exc:
        ok("WebSocket 连通", False, str(exc)[:160])


async def main() -> None:
    print("EmoRobot pipeline test")
    print(f"  port={settings.port}  persona={settings.resolved_persona_path()}")
    device_id = "e2e-pipeline"
    session_id = store.get_or_create_session(device_id, None)

    test_profile_loading()
    test_interlocutor(device_id, session_id)
    test_build_messages(device_id, session_id)
    await test_handle_chat(device_id)

    uri = f"ws://127.0.0.1:{settings.port}/ws/v1/chat"
    await test_ws(uri, settings.api_token)

    section("Summary")
    print(f"  PASS={PASS}  FAIL={FAIL}")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
