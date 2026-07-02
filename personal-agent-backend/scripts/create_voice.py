"""Create Baidu voice clone — uses Authorization header (no Secret Key needed)."""

import sys
import json
import base64
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

env = {}
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

API_KEY = env.get("BAIDU_API_KEY", "")
AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else None
VOICE_NAME = sys.argv[2] if len(sys.argv) > 2 else "my_voice"

if not API_KEY:
    print("ERROR: BAIDU_API_KEY not found in .env")
    sys.exit(1)

if not AUDIO_PATH:
    print("Usage: python scripts/create_voice.py <audio_path> [voice_name]")
    sys.exit(1)

audio = Path(AUDIO_PATH)
if not audio.exists():
    print(f"ERROR: audio file not found: {AUDIO_PATH}")
    sys.exit(1)

audio_bytes = audio.read_bytes()
print(f"Audio: {audio.name} ({len(audio_bytes) / 1024:.1f} KB)")

# Encode audio as base64
b64_audio = base64.b64encode(audio_bytes).decode()
print(f"Base64 length: {len(b64_audio)} chars")
print("Creating voice clone...")

url = "https://aip.baidubce.com/rest/2.0/speech/publiccloudspeech/v1/voice/clone/create"

resp = requests.post(
    url,
    headers={
        "Authorization": API_KEY,
        "Content-Type": "application/json",
    },
    json={
        "voice_name": VOICE_NAME,
        "voice_desc": "我的声音复刻",
        "lang": "zh",
        "audio_file": b64_audio,
    },
    timeout=30,
)

result = resp.json()
print(json.dumps(result, ensure_ascii=False, indent=2))

if result.get("status") == 0 or "voice_id" in str(result):
    vid = (
        result.get("voice_id")
        or result.get("data", {}).get("voice_id", "")
    )
    print(f"\nSUCCESS! voice_id = {vid}")
    print(f"Add to .env: TTS_CLONE_VOICE_ID={vid}")
else:
    print(f"\nFAILED: status={result.get('status')}, message={result.get('message')}")
