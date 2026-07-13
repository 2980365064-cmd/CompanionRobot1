"""Profile Card 加载与生成模块。

职责：为每一轮对话提供固定注入的机器人人设/口吻文本，不依赖向量数据库。
这是一个陪伴型情感机器人的"人格卡"系统，确保机器人在任何对话中都能保持一致的
身份、语气和关系认知。

数据来源与优先级：
  profile_card.md（压缩缓存）> persona.md + style/examples.md

与其他模块的概念区分：
  - 本模块 = 机器人「扮演者」（叶鹏祥）的全局设定，注入到每轮对话的 system prompt
  - person_profile.py = 对话对象（用户/亲友）的结构化档案，按 person_id 索引
  - person_profiles 表 = 数据库中存储的多人物画像，与全局人设是不同维度

缓存策略：
  使用进程内内存缓存 profile_card.md 的内容，以文件 mtime 作为缓存键，
  文件更新时自动失效，避免每次对话都从磁盘读取。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from app.config import settings
from app.llm import chat_completion

# 文件路径 → (修改时间, 内容) 的进程内缓存
# 键为 profile_card.md 的绝对路径字符串，避免多次读取同一文件
_profile_cache: dict[str, tuple[float, str]] = {}

# persona.md 中不进每轮 prompt 的章节（留空 = 全部保留）
_PERSONA_SKIP_SECTIONS: frozenset[str] = frozenset()

# 访客模式须剔除的 persona / profile_card 章节（标题含以下关键词即视为私密）
_INTIMATE_SECTION_KEYWORDS: frozenset[str] = frozenset(
    {"私密设定", "性癖好", "性癖"}
)

# 女友模式下，用户消息命中以下模式才注入「私密设定」段
_INTIMATE_TOPIC_RE = re.compile(
    r"(SM|捆绑|调教|训狗|深喉|口交|做爱|上床|开房|射|淫|骚|裸|"
    r"丝袜|渔网|腿环|情趣|亲密|屁股|尿|鞭打|轮奸|幻想|腿控|妆容|美瞳)",
    re.IGNORECASE,
)


def intimate_topic_relevant(message: str | None) -> bool:
    """用户本轮是否在聊私密/性相关话题。"""
    msg = (message or "").strip()
    if not msg:
        return False
    return bool(_INTIMATE_TOPIC_RE.search(msg))

# ── 兜底 bundle 缓存 ───────────────────────────────────────────────────────
# 当 profile_card.md 不存在时，load_persona_fallback_bundle() 每次读取
# persona.md + style/*.md 文件。用文件 mtime 组合做缓存键。
_fallback_cache: tuple[float, str] | None = None


def _fallback_mtime() -> float:
    """计算 persona.md + style/ 文件的最新 mtime。"""
    mt = settings.resolved_persona_path().stat().st_mtime if settings.resolved_persona_path().exists() else 0
    style_dir = settings.resolved_style_dir()
    if style_dir.is_dir():
        for f in style_dir.glob("*.md"):
            mt = max(mt, f.stat().st_mtime)
    return mt


def load_persona_raw() -> str:
    """读取机器人核心身份/规则文本（persona.md）。

    该文件包含机器人的基础身份设定、行为禁忌、回复规则等。
    如果文件不存在，返回默认的温暖陪伴型机器人兜底文本。

    Returns:
        persona.md 的完整文本内容，或默认兜底人设字符串
    """
    path = settings.resolved_persona_path()
    if path.exists():
        return path.read_text(encoding="utf-8")
    # 兼容错误配置：../persona/persona.md → ../persona/config/persona.md
    fallback = settings.resolved_persona_dir() / "config" / "persona.md"
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")
    return "你是一个温暖、简洁的陪伴伙伴。"


def load_style_examples() -> str:
    """读取口吻范例文件（style/ 目录下的 examples.md 等）。

    口吻范例是 Q→A 格式的对话参考，教机器人如何用特定风格和语气回复。
    支持多个范例文件：日常口吻（examples.md）和亲密口吻（examples_intimate.md）。

    范例文件中的内容会直接拼接到 system prompt 中，作为 few-shot 提示，
    让 LLM 模仿特定说话风格（简短口语、不使用括号旁白等）。

    Returns:
        拼接后的所有口吻范例文本，文件间用两个换行分隔；无文件时返回空字符串
    """
    style_dir = settings.resolved_style_dir()
    parts: list[str] = []
    for name in ("examples.md",):
        path = style_dir / name
        if path.exists():
            # 读取并去除首尾空白，避免多余空行影响拼接效果
            text = path.read_text(encoding="utf-8").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _is_intimate_section_title(title: str) -> bool:
    t = title.strip().lstrip("【").rstrip("】")
    return any(kw in t for kw in _INTIMATE_SECTION_KEYWORDS)


def _strip_intimate_sections(text: str) -> str:
    """从 profile_card / persona 正文中移除私密章节（访客模式）。"""
    lines = text.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("【") and stripped.endswith("】"):
            if _is_intimate_section_title(stripped):
                skipping = True
                continue
            skipping = False
        elif stripped.startswith("#"):
            if _is_intimate_section_title(stripped.lstrip("#").strip()):
                skipping = True
                continue
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out).strip()


def load_profile_card(
    interlocutor_mode: str | None = None,
    user_message: str | None = None,
) -> str:
    """加载每轮对话固定注入的 Profile Card（人格卡）。

    优先级：
    1. 如果存在 profile_card.md（LLM 预压缩的人格缓存文件），直接使用
    2. 否则，现场拼接 persona.md + style/ 范例作为兜底

    - ``visitor``：剔除「私密设定」等章节
    - ``girlfriend``：默认也剔除私密段，仅当 user_message 命中私密话题时保留

    缓存机制：
    - 以文件的绝对路径 + mtime（修改时间）作为缓存键
    - 只要文件未修改，后续调用直接返回缓存内容，避免磁盘 IO
    - 文件修改后自动失效，下次调用重新读取

    Returns:
        用于注入 system prompt 的完整人格文本
    """
    card_path = settings.resolved_profile_card_path()
    if card_path.is_file():
        key = str(card_path.resolve())
        mtime = card_path.stat().st_mtime
        cached = _profile_cache.get(key)
        if cached and cached[0] == mtime:
            text = cached[1]
        else:
            text = card_path.read_text(encoding="utf-8").strip()
            if text:
                _profile_cache[key] = (mtime, text)
            else:
                text = load_persona_fallback_bundle()
    else:
        text = load_persona_fallback_bundle()

    if interlocutor_mode == "visitor":
        text = _strip_intimate_sections(text)
    elif interlocutor_mode == "girlfriend" and not intimate_topic_relevant(user_message):
        text = _strip_intimate_sections(text)
    return text


def load_persona_fallback_bundle() -> str:
    """当 profile_card.md 不存在时的兜底方案（带文件 mtime 缓存）。

    将 persona.md（身份规则）和 style/ 范例拼接成一段完整文本。
    这是原始、未经 LLM 压缩的版本，会消耗更多 token，
    因此正常流程中应优先使用 compress_profile_to_card 生成缓存文件。

    Returns:
        persona + style 拼接文本，用两个换行分隔
    """
    global _fallback_cache
    mtime = _fallback_mtime()
    if _fallback_cache and _fallback_cache[0] == mtime:
        return _fallback_cache[1]
    parts = [load_persona_raw()]
    style = load_style_examples()
    if style:
        parts.append(style)
    text = "\n\n".join(parts)
    _fallback_cache = (mtime, text)
    return text


def clear_profile_cache() -> None:
    """手动清空所有 profile 缓存。

    当外部工具（如 compress_profile.py）更新文件后，
    调用此函数强制下次重新从磁盘读取。
    """
    global _fallback_cache
    _profile_cache.clear()
    _fallback_cache = None


def _strip_md(text: str) -> str:
    """去掉常见 markdown 标记，保留可读正文。"""
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    return t.strip()


def _parse_persona_sections(persona: str) -> dict[str, str]:
    """按 # / ## 标题切分 persona.md。"""
    sections: dict[str, list[str]] = {}
    current = "_preamble"
    sections[current] = []
    for line in persona.splitlines():
        m2 = re.match(r"^##\s+(.+)$", line.strip())
        m1 = re.match(r"^#\s+(.+)$", line.strip())
        if m2:
            current = m2.group(1).strip()
            sections.setdefault(current, [])
        elif m1:
            current = m1.group(1).strip()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def parse_style_qa_pairs(style: str, *, limit: int = 8) -> list[tuple[str, str]]:
    """从 style/examples.md 解析「问/答」对。"""
    pairs: list[tuple[str, str]] = []
    q: str | None = None
    for line in style.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        mq = re.match(r"^问[：:]\s*(.+)$", line)
        ma = re.match(r"^答[：:]\s*(.+)$", line)
        if mq:
            q = mq.group(1).strip()
        elif ma and q:
            pairs.append((q, ma.group(1).strip()))
            q = None
            if len(pairs) >= limit:
                break
    return pairs


def _persona_to_card_text(persona: str) -> str:
    """将 persona.md 转为 Profile Card 正文（保留全部章节，仅简化 markdown 标题）。"""
    out: list[str] = []
    for line in persona.splitlines():
        raw = line.strip()
        if not raw:
            if out and out[-1] != "":
                out.append("")
            continue
        m2 = re.match(r"^##\s+(.+)$", raw)
        m1 = re.match(r"^#\s+(.+)$", raw)
        if m2:
            title = m2.group(1).strip()
            if title in _PERSONA_SKIP_SECTIONS:
                continue
            out.append(f"【{title}】")
            continue
        if m1:
            title = m1.group(1).strip()
            if title in _PERSONA_SKIP_SECTIONS:
                continue
            out.append(f"【{title}】")
            continue
        if raw.startswith("|") and raw.endswith("|"):
            continue
        if re.match(r"^[-|]+$", raw):
            continue
        out.append(_strip_md(raw))
    return "\n".join(out).strip()


def _style_principles_to_card(style: str) -> str:
    """将 style/examples.md 转为 Profile Card 附录（不含问→答对）。"""
    if not style.strip():
        return ""
    out: list[str] = ["【口吻原则】"]
    for line in style.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if re.match(r"^问[：:]", raw) or re.match(r"^答[：:]", raw):
            continue
        out.append(_strip_md(raw))
    return "\n".join(out).strip() if len(out) > 1 else ""


def build_profile_card_full(*, example_limit: int = 20) -> str:
    """从 persona.md 全文 + style 原则生成 Profile Card（尽量完整，不丢章节）。"""
    del example_limit
    persona = load_persona_raw()
    style = load_style_examples()
    parts: list[str] = [
        "Profile Card（persona.md + style 原则；改源文件后运行 compress_profile.py）",
        "",
        _persona_to_card_text(persona),
    ]
    style_card = _style_principles_to_card(style)
    if style_card:
        parts.extend(["", style_card])
    return "\n".join(parts).strip()


def build_profile_card_deterministic(*, example_limit: int = 8) -> str:
    """从 persona.md 与 style 结构化抽取 Profile Card。"""
    persona = load_persona_raw()
    style = load_style_examples()
    sections = _parse_persona_sections(persona)

    parts: list[str] = [
        "Profile Card（由 persona.md + style 生成；改源文件后运行 compress_profile.py）",
        "",
    ]

    identity = sections.get("身份（最高优先级）", sections.get("_preamble", ""))
    if identity:
        parts.append("【身份】")
        for line in identity.splitlines():
            line = _strip_md(line.strip())
            if line.startswith("- "):
                line = line[2:]
            if line:
                parts.append(line)
        parts.append("")

    bg = sections.get("人生经历", "")
    if bg:
        bg_line = _strip_md(bg.replace("\n", " ").strip())
        if bg_line.startswith("- "):
            bg_line = bg_line[2:]
        parts.extend(["【背景】", bg_line, ""])

    intimate = sections.get("性癖好（只能跟刘远慧聊）", sections.get("性癖好", ""))
    if intimate:
        parts.append("【性癖好（只能跟刘远慧聊）】")
        for line in intimate.splitlines():
            line = _strip_md(line.strip())
            if line.startswith("- "):
                parts.append(line)
        parts.append("")

    speech_blocks = (
        "长度与节奏（按微信 7.1 万条实测）",
        "口头禅（按场景自然用，不要每句都堆）",
        "四川口语感",
        "绝对禁止（一出现就失败）",
    )
    speech_lines: list[str] = []
    for block_name in speech_blocks:
        block = sections.get(block_name, "")
        if not block:
            continue
        for line in block.splitlines():
            line = _strip_md(line.strip())
            if not line or line.startswith("|") or re.match(r"^[-|]+$", line):
                continue
            if line.startswith("- "):
                speech_lines.append(line)
            elif not line.startswith("##"):
                speech_lines.append(f"- {line}")
    if speech_lines:
        parts.append("【说话习惯】")
        parts.extend(speech_lines)
        parts.append("")

    memory = sections.get("记忆怎么用（防编造）", "")
    if memory:
        parts.append("【记忆与禁忌】")
        for line in memory.splitlines():
            line = _strip_md(line.strip())
            if line.startswith("- "):
                parts.append(line)
        parts.append("")

    gf = sections.get("和女朋友私下聊（刘远慧找你）", "")
    if gf:
        parts.append("【女友刘远慧】")
        for line in gf.splitlines():
            line = _strip_md(line.strip())
            if line.startswith("- "):
                parts.append(line)
        parts.append("")

    voice = sections.get("语音播报", "")
    if voice:
        parts.append("【语音播报】")
        for line in voice.splitlines():
            line = _strip_md(line.strip())
            if line.startswith("- "):
                parts.append(line)
        parts.append("")

    pairs = parse_style_qa_pairs(style, limit=example_limit)
    if pairs:
        parts.append("【口吻范例（问→答）】")
        for q, a in pairs:
            parts.append(f"- {q} → {a}")

    return "\n".join(parts).strip()


def compress_profile_to_card(*, max_chars: int = 1200) -> str:
    """调用 LLM 将 persona+style 压缩为精简的 Profile Card 正文。

    压缩目标：
    - 保留核心身份、关键人际关系称谓、说话习惯、行为禁忌
    - 保留 3-5 条最短的口吻范例（问→答各一句）
    - 删除重复内容、空模板、markdown 标题符号
    - 不编造原文中没有的信息
    - 总长度控制在 max_chars 字以内

    这是 profile_card.md 的生成方法，通常由 scripts/compress_profile.py 调用。
    压缩后的文本更适合作为每轮对话的固定前缀注入，节省 token。

    Args:
        max_chars: 压缩后最大字数，默认 600 字

    Returns:
        LLM 压缩后的精简人格文本
    """
    persona = load_persona_raw()
    style = load_style_examples()
    prompt = f"""将以下「身份/背景/禁忌」与「口吻范例」压缩为一份 Profile Card（中文），供每轮对话固定注入。

硬性要求（违反即失败）：
- 第一行必须写明：你就是叶鹏祥本人在聊天（第一人称），不是 AI/助手/客服
- 必须保留女友姓名：刘远慧（称呼仅限：大炮、刘远慧、秋雨）
- 必须保留说话长度：平均约 8 字/条，回复优先 5～20 字，不超过 40 字
- 口吻范例必须严格按原文问→答配对，禁止张冠李戴、禁止编造新对话
- 禁止把身份概括成「温暖陪伴伙伴」「简洁助手」等泛化描述
- 总长度不超过 {max_chars} 字

【身份与规则】
{persona}

【口吻范例】
{style or "（无）"}

直接输出 Profile Card 正文，不要解释。"""
    return chat_completion([{"role": "user", "content": prompt}], temperature=0.2).strip()


def write_profile_card(
    text: str | None = None,
    *,
    use_llm: bool = False,
    full: bool = True,
) -> Path:
    """写入 profile_card.md 并清空缓存。

    Args:
        text: 可选的手动指定人格卡文本；为空时按参数选择生成方式
        use_llm: True 时用 LLM 压缩（易丢内容，不推荐）
        full: True 时写入 persona.md 全文 + 口吻范例（默认，最完整）

    Returns:
        写入的 profile_card.md 文件路径
    """
    if text is not None:
        card = text.strip()
    elif use_llm:
        card = compress_profile_to_card().strip()
    elif full:
        card = build_profile_card_full().strip()
    else:
        card = build_profile_card_deterministic().strip()
    path = settings.resolved_profile_card_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写入时末尾加换行，保证文件符合 Unix 文本规范
    path.write_text(card + "\n", encoding="utf-8")
    clear_profile_cache()
    return path
