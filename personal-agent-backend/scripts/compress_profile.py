#!/usr/bin/env python3
"""将 persona.md + style 写入 profile_card.md（每轮对话实际注入的人设）。

默认：persona.md 全文 + 口吻范例（含性癖好等全部章节，不裁剪）。

必须在 personal-agent-backend 目录运行：
    cd personal-agent-backend
    python scripts/compress_profile.py

若 python 无反应，改用完整路径：
    D:\\TOOL\\Python\\python.exe scripts/compress_profile.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Windows 终端 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _die(msg: str, code: int = 1) -> None:
    print(msg, flush=True)
    sys.exit(code)


def main() -> None:
    if not (_BACKEND / "app" / "main.py").is_file():
        _die(
            "[!] 当前不在 personal-agent-backend 目录。\n"
            "    请先执行: cd personal-agent-backend\n"
            "    再运行: python scripts/compress_profile.py"
        )

    parser = argparse.ArgumentParser(description="生成 persona/config/profile_card.md")
    parser.add_argument("--llm", action="store_true", help="LLM 压缩（易丢内容，不推荐）")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="结构化精简版（可能丢章节；默认用全文模式）",
    )
    args = parser.parse_args()

    from app.config import settings
    from app.persona.card import load_persona_raw, write_profile_card

    persona_path = settings.resolved_persona_path()
    fallback = settings.resolved_persona_dir() / "config" / "persona.md"
    style_dir = settings.resolved_style_dir()
    out_path = settings.resolved_profile_card_path()

    print("=== compress_profile ===", flush=True)
    print(f"cwd:      {_BACKEND}", flush=True)
    print(f"persona:  {persona_path}  exists={persona_path.is_file()}", flush=True)
    if not persona_path.is_file() and fallback.is_file():
        print(f"fallback: {fallback}  (将使用此文件)", flush=True)
    print(f"style:    {style_dir}", flush=True)
    print(f"output:   {out_path}", flush=True)

    persona_text = load_persona_raw()
    if "叶鹏祥" not in persona_text:
        _die(
            "[!] persona 源未加载成功（未找到「叶鹏祥」）。\n"
            f"    检查 .env 中 PERSONA_PATH，应为: ../persona/config/persona.md\n"
            f"    当前读取结果开头: {persona_text[:80]!r}"
        )

    mode = "LLM" if args.llm else ("compact" if args.compact else "full")
    print(f"mode:     {mode}", flush=True)

    old_mtime = out_path.stat().st_mtime if out_path.is_file() else 0
    path = write_profile_card(use_llm=args.llm, full=not args.compact and not args.llm)
    text = path.read_text(encoding="utf-8")
    new_mtime = path.stat().st_mtime

    print(f"\nOK 写入 {len(text)} 字 → {path}", flush=True)
    print(f"    更新时间: {datetime.fromtimestamp(new_mtime)}", flush=True)
    if old_mtime and abs(new_mtime - old_mtime) < 0.5:
        print("    （文件时间几乎未变：若 persona 没改，内容可能相同）", flush=True)

    checks = [
        ("叶鹏祥", "身份"),
        ("性癖好", "性癖好章节"),
        ("刘远慧", "女友"),
        ("语音播报", "语音规则"),
    ]
    print("\n章节检查:", flush=True)
    for token, label in checks:
        mark = "✓" if token in text else "✗ 缺失"
        print(f"  {mark}  {label}", flush=True)

    print("\n--- 开头预览 ---", flush=True)
    print(text[:600] + ("…" if len(text) > 600 else ""), flush=True)
    print("--- 完成；请重启 uvicorn 后生效 ---", flush=True)


if __name__ == "__main__":
    main()
