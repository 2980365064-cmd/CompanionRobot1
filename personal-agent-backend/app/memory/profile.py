"""Profile card (config + style, not in vector DB)."""

from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.llm import chat_completion


def load_persona_raw() -> str:
    path = settings.resolved_persona_path()
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "你是一个温暖、简洁的陪伴伙伴。"


def load_style_examples() -> str:
    style_dir = settings.resolved_style_dir()
    parts: list[str] = []
    for name in ("examples.md", "examples_intimate.md"):
        path = style_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def load_profile_card() -> str:
    """Always include persona.md rules; card/style are supplementary."""
    return load_persona_fallback_bundle()


def load_persona_fallback_bundle() -> str:
    parts = [load_persona_raw()]
    style = load_style_examples()
    if style:
        parts.append(style)
    return "\n\n".join(parts)


def compress_profile_to_card(*, max_chars: int = 600) -> str:
    persona = load_persona_raw()
    style = load_style_examples()
    prompt = f"""将以下「身份/背景/禁忌」与「口吻范例」压缩为一份 Profile Card（中文），供每轮对话固定注入。

要求：
- 总长度不超过 {max_chars} 字
- 保留：身份、关键人际关系称谓、说话习惯、禁忌、3～5 条最短口吻范例（问→答各一句）
- 删除：重复、空模板、markdown 标题符号可简化
- 不要编造原文没有的信息

【身份与规则】
{persona}

【口吻范例】
{style or "（无）"}

直接输出 Profile Card 正文，不要解释。"""
    return chat_completion([{"role": "user", "content": prompt}], temperature=0.3).strip()


def write_profile_card(text: str | None = None) -> Path:
    card = (text or compress_profile_to_card()).strip()
    path = settings.resolved_profile_card_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card + "\n", encoding="utf-8")
    return path
