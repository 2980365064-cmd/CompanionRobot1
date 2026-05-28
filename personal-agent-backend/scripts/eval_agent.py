#!/usr/bin/env python3
"""Evaluation script for memory and persona quality."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import handle_chat, handle_session_end
from app.session import store


async def run_eval() -> None:
    device_id = "eval-device"
    print("=== Eval: L1 same-session context ===")
    sid = store.get_or_create_session(device_id, None)
    r1, sid = await handle_chat(device_id, sid, "我下周二要去上海出差。")
    print("A1:", r1)
    r2, sid = await handle_chat(device_id, sid, "我下周二去哪？")
    print("A2:", r2)
    ok_l1 = any(k in r2 for k in ("上海", "出差"))
    print("L1 pass:" if ok_l1 else "L1 fail:", ok_l1)

    print("\n=== Eval: L2/L3 cross-session ===")
    await handle_session_end(device_id, sid)
    sid2 = store.get_or_create_session(device_id, None)
    r3, sid2 = await handle_chat(device_id, sid2, "还记得我下周二去哪来着？")
    print("A3:", r3)
    ok_long = any(k in r3 for k in ("上海", "出差", "不确定", "印象"))
    print("Long-term pass:" if ok_long else "Long-term fail:", ok_long)

    print("\n=== Eval: L2 after session end ===")
    r4, sid2 = await handle_chat(device_id, sid2, "对了，我不吃香菜。")
    print("A4:", r4)
    await handle_session_end(device_id, sid2)
    sid3 = store.get_or_create_session(device_id, None)
    r5, _ = await handle_chat(device_id, sid3, "我有什么忌口？")
    print("A5:", r5)
    ok_fact = "香菜" in r5
    print("L2 pass:" if ok_fact else "L2 fail:", ok_fact)


if __name__ == "__main__":
    asyncio.run(run_eval())
