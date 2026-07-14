"""智能体人格工作台 —— 将 persona.md / profile_card.md 解析为结构化区块。

区块定义（按 Markdown 标题级别划分）：
  - identity:      身份（最高优先级）
  - speech_habits: 说话习惯（长度节奏、口头禅、对女友怎么聊）
  - scene_routing: 场景路由
  - prohibitions:  禁止
  - memory_rules:  记忆怎么用
  - private:       私密设定
  - background:    背景经历
  - style_principles: 口吻原则（仅 profile_card.md 有）

GET  /v1/admin/agent/persona-blocks  → 读取当前区块
PATCH /v1/admin/agent/persona-blocks  → 保存指定区块
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.config import settings
from app.persona.card import clear_profile_cache

# 区块定义：名称、标题、说明
BLOCK_DEFINITIONS: list[dict[str, str]] = [
    {"name": "identity", "title": "身份边界", "description": "你是谁、身份边界、对女友/熟人的基本态度"},
    {"name": "speech_habits", "title": "说话习惯", "description": "长度节奏、口头禅、对女友聊天方式"},
    {"name": "scene_routing", "title": "场景路由", "description": "不同场景选择什么风格回复"},
    {"name": "prohibitions", "title": "禁止事项", "description": "一出现就失败的禁忌"},
    {"name": "memory_rules", "title": "记忆使用规则", "description": "如何防编造、称呼规则、纠正处理"},
    {"name": "private", "title": "私密设定", "description": "仅女友模式下的私密内容（谨慎编辑）"},
    {"name": "background", "title": "背景经历", "description": "出生、成长、学习、工作经历"},
    {"name": "style_principles", "title": "口吻原则", "description": "风格比例、词池骨架（仅 Profile Card）"},
]


def _read_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text("utf-8").strip()


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", "utf-8")


def _parse_sections(text: str) -> list[dict[str, str]]:
    """将 Markdown 按标题级别（## 或 #）分割为区块。"""
    sections: list[dict[str, str]] = []
    current_title = ""
    current_content: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^#(#?)\s+(.+)$", line)
        if m:
            # 保存上一个区块
            if current_title:
                sections.append({
                    "heading": current_title,
                    "content": "\n".join(current_content).strip(),
                })
            current_title = line
            current_content = []
        else:
            current_content.append(line)

    if current_title:
        sections.append({
            "heading": current_title,
            "content": "\n".join(current_content).strip(),
        })
    return sections


def _heading_to_block_name(heading: str) -> str | None:
    """将标题映射为区块名称。"""
    mapping: dict[str, str] = {
        "身份": "identity",
        "身份（最高优先级）": "identity",
        "说话习惯": "speech_habits",
        "场景路由": "scene_routing",
        "场景路由（本轮怎么选风格）": "scene_routing",
        "禁止": "prohibitions",
        "禁止（一出现就失败）": "prohibitions",
        "记忆怎么用": "memory_rules",
        "记忆怎么用（防编造）": "memory_rules",
        "私密设定": "private",
        "私密设定（仅女友模式 · 话题相关时注入）": "private",
        "背景经历": "background",
        "口吻原则": "style_principles",
    }
    # 去掉 # 号
    clean = re.sub(r"^#+\s*", "", heading).strip()
    # 尝试精确匹配
    if clean in mapping:
        return mapping[clean]
    # 模糊匹配
    for key, name in mapping.items():
        if key in clean:
            return name
    return None


def _block_to_heading(block_name: str) -> str:
    """区块名称转 Markdown 标题。"""
    headings = {
        "identity": "# 身份（最高优先级）",
        "speech_habits": "# 说话习惯",
        "scene_routing": "# 场景路由（本轮怎么选风格）",
        "prohibitions": "# 禁止（一出现就失败）",
        "memory_rules": "# 记忆怎么用（防编造）",
        "private": "# 私密设定（仅女友模式 · 话题相关时注入）",
        "background": "# 背景经历",
        "style_principles": "## 口吻原则",
    }
    return headings.get(block_name, f"# {block_name}")


def load_persona_blocks() -> dict[str, Any]:
    """读取 persona.md 和 profile_card.md，解析为结构化区块。"""
    persona_path = settings.resolved_persona_path()
    profile_path = settings.resolved_profile_card_path()

    persona_text = _read_file(persona_path)
    profile_text = _read_file(profile_path)

    persona_sections = _parse_sections(persona_text)
    profile_sections = _parse_sections(profile_text)

    blocks: dict[str, Any] = {}
    for block_def in BLOCK_DEFINITIONS:
        name = block_def["name"]
        # persona.md 中找
        persona_content = ""
        for section in persona_sections:
            if _heading_to_block_name(section["heading"]) == name:
                persona_content = section["content"]
                break
        # profile_card.md 中找
        profile_content = ""
        for section in profile_sections:
            if _heading_to_block_name(section["heading"]) == name:
                profile_content = section["content"]
                break
        blocks[name] = {
            "title": block_def["title"],
            "description": block_def["description"],
            "persona_content": persona_content,
            "profile_card_content": profile_content,
        }

    return {
        "blocks": blocks,
        "persona_file": {
            "path": str(persona_path),
            "exists": persona_path.exists(),
        },
        "profile_card_file": {
            "path": str(profile_path),
            "exists": profile_path.exists(),
        },
    }


def _rebuild_markdown(blocks: dict[str, str], source_heading: str) -> str:
    """将区块内容拼回 Markdown。"""
    parts: list[str] = []
    for name, content in blocks.items():
        heading = _block_to_heading(name)
        # 跳过源标记为不同的（只处理同一文件的区块）
        if not isinstance(content, str):
            continue
        if content.strip():
            parts.append(heading)
            parts.append("")
            parts.append(content.strip())
            parts.append("")
    return "\n".join(parts).strip()


def _rebuild_persona(blocks: dict) -> str:
    """拼回完整的 persona.md（不含 style_principles 区块）。"""
    b = {}
    for name in ("identity", "speech_habits", "scene_routing", "prohibitions", "memory_rules", "private", "background"):
        if name in blocks:
            content = blocks[name]
            if isinstance(content, dict):
                content = content.get("persona_content", blocks[name])
            if isinstance(content, str):
                b[name] = content
    return _rebuild_markdown(b, "persona")


def _rebuild_profile_card(blocks: dict) -> str:
    """拼回完整的 profile_card.md（包含所有可读区块）。"""
    b = {}
    for name in ("identity", "speech_habits", "scene_routing", "prohibitions", "memory_rules", "private", "background", "style_principles"):
        if name in blocks:
            content = blocks[name]
            if isinstance(content, dict):
                content = content.get("profile_card_content", blocks[name])
            if isinstance(content, str):
                b[name] = content
    return _rebuild_markdown(b, "profile_card")


def save_persona_blocks(blocks_update: dict[str, str], sync_both: bool = True) -> dict:
    """保存指定区块内容到文件。

    blocks_update: {block_name: content_text}
    sync_both: 是否同时同步到 persona.md 和 profile_card.md
    """
    current = load_persona_blocks()
    current_blocks = current.get("blocks", {})

    # 更新指定的区块
    for name, content in blocks_update.items():
        if name in current_blocks:
            if isinstance(current_blocks[name], dict):
                current_blocks[name]["persona_content"] = content
                if sync_both:
                    current_blocks[name]["profile_card_content"] = content
            else:
                current_blocks[name] = content
        else:
            # 未知区块，加进去
            current_blocks[name] = {
                "title": name,
                "description": "",
                "persona_content": content,
                "profile_card_content": content if sync_both else "",
            }

    # 重写文件
    persona_path = settings.resolved_persona_path()
    profile_path = settings.resolved_profile_card_path()

    persona_text = _rebuild_persona(current_blocks)
    if persona_text:
        _write_file(persona_path, persona_text)

    profile_text = _rebuild_profile_card(current_blocks)
    if profile_text:
        _write_file(profile_path, profile_text)

    # 清缓存
    clear_profile_cache()

    return load_persona_blocks()
