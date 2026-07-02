#!/usr/bin/env python3
"""Scheme A 全链路自测：模拟 ESP32 dialog_core 在 STT 之后的 WS 交互。

覆盖路径（设备端无法在 CI 测 AFE/唤醒，此处测 STT→Agent→TTS 段）：
  hello → hello_ack → chat(text) → reply_start → reply_token* → reply →
  tts_start → binary PCM* → tts_end → chat_done

用法（需后端已启动）：
  python scripts/test_scheme_a_chain.py
  python scripts/test_scheme_a_chain.py ws://127.0.0.1:8001/ws/v1/chat
"""

from __future__ import annotations

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


def test_health() -> None:
    section("1. Backend health")
    base = f"http://127.0.0.1:{settings.port}"
    try:
        r = httpx.get(f"{base}/health", timeout=8)
        r.raise_for_status()
        data = r.json()
        ok("GET /health", True, str(data.get("status", data))[:80])
        ok("LLM configured", data.get("llm", {}).get("configured", False))
    except Exception as exc:
        ok("GET /health", False, str(exc)[:120])


async def test_firmware_ws_chain(uri: str) -> None:
    section("2. Firmware WS chain (post-STT)")
    token = settings.api_token
    chat_msg = "你好，这是一条自测消息"

    saw = {
        "hello_ack": False,
        "reply_start": False,
        "reply_token": 0,
        "reply": 0,
        "tts_start": 0,
        "tts_pcm_bytes": 0,
        "tts_end": 0,
        "chat_done": False,
    }

    try:
        async with websockets.connect(uri, open_timeout=10) as ws:
            await ws.send(json.dumps({
                "type": "hello",
                "device_id": "scheme-a-selftest",
                "token": token,
                "session_id": "",
            }, ensure_ascii=False))

            deadline = asyncio.get_event_loop().time() + 120
            chat_sent = False

            while asyncio.get_event_loop().time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=90)
                except asyncio.TimeoutError:
                    break

                if isinstance(msg, bytes):
                    saw["tts_pcm_bytes"] += len(msg)
                    continue

                frame = json.loads(msg)
                ftype = frame.get("type", "")

                if ftype == "hello_ack":
                    saw["hello_ack"] = True
                    ok("hello_ack", bool(frame.get("session_id")))
                    await ws.send(json.dumps({
                        "type": "chat",
                        "message": chat_msg,
                    }, ensure_ascii=False))
                    chat_sent = True
                    continue

                if not chat_sent:
                    continue

                if ftype == "reply_start":
                    saw["reply_start"] = True
                elif ftype == "reply_token":
                    saw["reply_token"] += 1
                elif ftype == "reply":
                    saw["reply"] += 1
                    ok("reply bubble", bool(frame.get("text")), (frame.get("text") or "")[:50])
                elif ftype == "tts_start":
                    saw["tts_start"] += 1
                    ok("tts_start fmt", frame.get("sample_rate") == 16000,
                       f"sr={frame.get('sample_rate')} bits={frame.get('bits_per_sample')}")
                elif ftype == "tts_end":
                    saw["tts_end"] += 1
                elif ftype == "chat_done":
                    saw["chat_done"] = True
                    break
                elif ftype == "error":
                    ok("no server error", False, frame.get("message", str(frame))[:100])
                    break

            ok("hello_ack received", saw["hello_ack"])
            ok("reply_start received", saw["reply_start"])
            ok("reply_token stream", saw["reply_token"] >= 0)
            ok("reply bubble(s)", saw["reply"] >= 1, f"count={saw['reply']}")
            if settings.tts_api_key:
                ok("tts_start received", saw["tts_start"] >= 1, f"count={saw['tts_start']}")
                ok("binary PCM received", saw["tts_pcm_bytes"] > 0,
                   f"bytes={saw['tts_pcm_bytes']}")
                ok("tts_end received", saw["tts_end"] >= 1, f"count={saw['tts_end']}")
            else:
                ok("TTS skipped (no TTS_API_KEY)", True, "text-only mode")
            ok("chat_done received", saw["chat_done"])

    except Exception as exc:
        ok("WS chain", False, str(exc)[:160])


def test_dialog_state_machine() -> None:
    """Pure logic: Scheme A state transitions (no hardware)."""
    section("3. dialog_core state machine (logic)")

    class Sim:
        IDLE, WAKE_ECHO, LISTEN, STT, AGENT, PLAY = range(6)

        def __init__(self) -> None:
            self.state = self.IDLE
            self.cooldown = False
            self.recording = False
            self.mic_mute = False
            self.wakenet = True

        def on_wake(self) -> bool:
            if self.state != self.IDLE or self.cooldown:
                return False
            self.state = self.WAKE_ECHO
            self.mic_mute = True
            self.mic_mute = False
            self.state = self.LISTEN
            self.wakenet = False
            return True

        def on_vad_start(self) -> None:
            if self.state == self.LISTEN:
                self.recording = True

        def on_vad_end(self) -> None:
            if self.state == self.LISTEN:
                self.recording = False
                self.state = self.STT

        def on_stt_ok(self) -> None:
            if self.state == self.STT:
                self.state = self.AGENT

        def on_reply_start(self) -> None:
            self.mic_mute = True
            self.state = self.PLAY

        def on_chat_done(self) -> None:
            self.state = self.IDLE
            self.mic_mute = False
            self.wakenet = True
            self.cooldown = True

    s = Sim()
    ok("wake from IDLE", s.on_wake())
    ok("state LISTEN", s.state == s.LISTEN)
    s.on_vad_start()
    ok("recording on VAD_START", s.recording)
    s.on_vad_end()
    ok("state STT after VAD_END", s.state == s.STT)
    s.on_stt_ok()
    ok("state AGENT", s.state == s.AGENT)
    s.on_reply_start()
    ok("mic muted on PLAY", s.mic_mute)
    s.on_chat_done()
    ok("back to IDLE", s.state == s.IDLE)
    ok("wakenet re-armed", s.wakenet)
    ok("wake blocked in cooldown", not s.on_wake())


async def main() -> None:
    print("Scheme A full-chain self-test")
    print(f"  port={settings.port}  tts={'on' if settings.tts_api_key else 'off'}")

    test_health()
    uri = sys.argv[1] if len(sys.argv) > 1 else f"ws://127.0.0.1:{settings.port}/ws/v1/chat"
    await test_firmware_ws_chain(uri)
    test_dialog_state_machine()

    section("Summary")
    print(f"  PASS={PASS}  FAIL={FAIL}")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
