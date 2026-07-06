"""persona/corpus/ 语料入库模块 —— 支持 frontmatter 解析、章节级分块、上下文注入。

职责：扫描 persona/corpus/ 目录下的 Markdown 语料文件，经过 frontmatter 解析、
剔除对话范例、去噪、章节级分块、上下文前缀注入后写入 L3 长期记忆存储。

数据流：
  persona/corpus/*.md
  → 读取 → frontmatter 解析 → 剔除对话范例 → 降噪
  → ## 章节分块 → 上下文前缀注入 → 字符分块 → L3 存储

阶段 2 增强（2026-07）：
  - 解析 YAML frontmatter（type/people/time/topics/privacy/status/confidence）
  - 按 ## 标题划分语义块，保留结构完整性
  - 每个块注入 [人物: xxx][时间: YYYY-MM][主题: xxx][类型: xxx] 上下文前缀
  - 将 type 写入 L3 category 字段，支持后续检索按类型偏向
  - 保持向后兼容：无 frontmatter 的旧文件按原有方式处理

注意：
  - 本模块只做分块入库，不再抽取 Facts（Facts 抽取已迁移到独立模块）
  - 入库后的语料可通过 semantic_memory 模块进行语义/关键词检索
  - 启动时的语料同步由 startup_ingest_corpus 控制，避免重复入库
"""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path
from typing import Any

from app.config import settings
from app.memory.l3 import (
    batch_extract_facts_from_persona_chunks,
    clear_persona_derived_memory,
    semantic_memory,
)
from app.memory.person_resolver import invalidate_wiki_cache, sync_wiki_people_to_contacts

logger = logging.getLogger(__name__)

# ---------- 过滤规则 ----------

# 入库时跳过的文件名（示例文件、说明文档等非语料文件）
_SKIP_NAMES = frozenset({
    "README.md",
    "intimate.md.example",
    "sample_corpus.md",
})

# ---------- 正则预编译（性能优化：模块加载时编译，避免每次调用重复编译）----------

# 匹配 URL（http/https/www 开头），用于去除链接噪声
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
# 匹配 HTML 实体字符（如 &#x20;），微信导出文本中常见
_ENTITY_RE = re.compile(r"&#x[0-9a-fA-F]+;")
# 匹配纯符号噪声行（如 "------"、"======" 等分隔线）
_NOISE_LINE_RE = re.compile(r"^\s*[\W_]{3,}\s*$")
# 匹配低信息量行（纯数字、连续感叹词、纯标点等无意义内容）
_LOW_INFO_RE = re.compile(r"^(666+|哈{3,}|嗯{3,}|[?？!！\.。]{2,})$")
# YAML frontmatter 分隔符（--- 开头和结尾）
_FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


# ---------- Frontmatter 解析 ----------


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """解析 YAML 风格 frontmatter，返回 (metadata_dict, body_text)。

    支持的字段：
      type / people（列表或单值） / time / topics（列表或单值）
      privacy / status / confidence / source / last_verified

    不依赖外部 YAML 库，只解析约定格式。无 frontmatter 时返回空 dict。

    Args:
        text: 含或不含 frontmatter 的原始文件内容

    Returns:
        (metadata, body_text) — metadata 为 dict，
        body 为去掉 frontmatter 后的 markdown 正文
    """
    m = _FM_PATTERN.match(text)
    if not m:
        return {}, text

    raw = m.group(1)
    body = text[m.end():].strip()
    meta: dict[str, Any] = {}

    for line in raw.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not key or not val:
            continue
        meta[key] = _parse_fm_value(val)

    return meta, body


def _parse_fm_value(val: str) -> Any:
    """解析 frontmatter 单行值：支持列表、数字、布尔、字符串。"""
    # 列表： [a, b, c]
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
    # 布尔值
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    # 数字
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    # 字符串，去引号
    return val.strip("'\"")


# ---------- 章节级分块与上下文注入 ----------


def build_context_prefix(meta: dict) -> str:
    """从 frontmatter 构建可检索的上下文前缀字符串。

    格式：[人物: xxx][时间: YYYY-MM][主题: xxx, yyy][类型: xxx][状态: xxx]

    这个前缀会嵌入到每个 chunk 的文本中，同时被 FTS5 索引和向量化，
    确保关键词检索（如"刘远慧"）和语义检索都能命中结构元信息。
    对于无 frontmatter 的旧文件，返回空字符串（保持向后兼容）。

    Args:
        meta: parse_frontmatter 返回的元数据字典

    Returns:
        空格分隔的上下文标签字符串，或空字符串
    """
    tags: list[str] = []
    if meta.get("people"):
        people = meta["people"]
        if isinstance(people, list):
            people = "、".join(people)
        tags.append(f"[人物: {people}]")
    if meta.get("time"):
        tags.append(f"[时间: {meta['time']}]")
    if meta.get("topics"):
        topics = meta["topics"]
        if isinstance(topics, list):
            topics = "、".join(topics)
        tags.append(f"[主题: {topics}]")
    if meta.get("type"):
        tags.append(f"[类型: {meta['type']}]")
    if meta.get("status"):
        tags.append(f"[状态: {meta['status']}]")
    return " ".join(tags)


def split_by_sections(text: str) -> list[str]:
    """按 ## 标题将正文划分为语义块，保留标题行。

    每个二级标题及其后续内容形成一个独立块。块间无重叠。
    对于只有 # 标题而缺少 ## 结构的短文件（如旧式无结构 Markdown），
    返回整个文本作为单一块。

    分区逻辑：
      - 以新行后的 "## " 为分割点
      - 文件头部（# 标题后、第一个 ## 前）的有内容段落单独成块
      - 纯标题（如 "# 某页"）且无副内容的头部块会被跳过

    Args:
        text: 去噪后的 markdown 正文（已剔除 frontmatter）

    Returns:
        按 ## 标题划分的文本块列表，每块包含其标题行
    """
    if not text.strip():
        return []

    # 以 ## 为分界拆分，保留标题行在块内
    parts = re.split(r"\n(?=## )", text)

    result: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 跳过纯标题无内容块（如单独 "# xxx" 后紧接 ##）
        _body = part.lstrip("#").strip()
        if not _body or len(_body) < 5:
            continue
        result.append(part)

    return result


def _chunk_section(
    section: str,
    context_prefix: str,
    chunk_size: int = 320,
    overlap: int = 64,
) -> list[str]:
    """将单个章节块注入上下文前缀，必要时子分块。

    如果 prefixed_text <= chunk_size，直接返回单块。
    否则用滑动窗口（chunk_size / overlap）子分块。

    注入时机：去噪之后，因此 context_prefix 中的 [标签] 不会被 denoise_text 滤掉。

    Args:
        section: 单个 ## 章节的文本（含标题行）
        context_prefix: 上下文前缀字符串
        chunk_size: 子分块字符上限
        overlap: 子分块重叠字符数

    Returns:
        注入前缀后的文本块列表
    """
    text = re.sub(r"\s+", " ", section.strip())
    if not text:
        return []

    full = f"{context_prefix}\n{text}" if context_prefix else text

    if len(full) <= chunk_size:
        return [full]

    # 子分块：从 section 开始（不重复前缀）
    subs = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size - len(context_prefix) - 1)
        sub = text[start:end].strip()
        if sub:
            subs.append(f"{context_prefix}\n{sub}" if context_prefix else sub)
        if end >= len(text):
            break
        start = end - overlap
    return subs


# ---------- 旧式语料清洗（保持向后兼容）----------


def strip_dialogue_examples(text: str) -> str:
    """剔除语料中的 Q→A 对话范例段落。

    原因：口吻范例应放在 persona/style/ 目录下，通过 Profile Card 注入；
    persona/corpus/ 中的语料入库到 L3 后用于检索召回，不应包含对话范例，
    否则会导致"用户问了一个相似问题，机器人检索到了范例中的回答"这种错误行为。

    剔除逻辑：
    - 从"典型对话"/"口吻范例"/"对话范例"的 markdown 标题开始，到下一个标题结束
    - 同时剔除独立的"问："/"答："行（无标题包裹的扁平范例）

    Args:
        text: 原始语料文本

    Returns:
        剔除对话范例段落后的清洁文本
    """
    lines = text.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        # 遇到范例段落的标题，开始跳过
        if re.match(r"^#+\s*(典型对话|口吻范例|对话范例)", line):
            skip = True
            continue
        # 遇到下一个 markdown 标题，结束跳过
        if skip and re.match(r"^#+\s+", line):
            skip = False
        if skip:
            continue
        # 跳过无标题包裹的扁平 Q→A 行（如单独出现的"问：..."）
        if re.match(r"^问[:：]", line.strip()) or re.match(r"^答[:：]", line.strip()):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _is_noise_file(path: Path) -> bool:
    """判断文件是否命中配置中的噪声文件跳过规则。

    从 settings.l3_noise_file_patterns 读取以逗号分隔的 glob 模式，
    匹配文件名（支持通配符如 *.log、*.tmp 等）。

    Args:
        path: 待检查的文件路径

    Returns:
        True 表示该文件应被跳过，不参与入库
    """
    if not settings.l3_denoise_enabled:
        return False
    patterns = [p.strip() for p in settings.l3_noise_file_patterns.split(",") if p.strip()]
    name = path.name
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def denoise_text(text: str) -> str:
    """行级文本降噪处理。

    对每行文本执行以下清洗操作：
    - 移除 HTML 实体字符（&#x20; 等）
    - 移除 URL 链接
    - 规范化空白字符（多个空格合并为一个）
    - 过滤纯符号噪声行
    - 过滤低信息量行（纯感叹词、纯标点）
    - 过滤方括号包裹行（纯标记行如 [图片]）
    - 过滤过短行（< 3 字符，无检索价值）

    这些操作对于微信导出的聊天记录文本尤其重要，原始导出文件中
    含有大量多媒体占位符和格式噪声。

    Args:
        text: 待清洗的原始文本

    Returns:
        清洗后的文本，每行一个有效句子，用换行分隔
    """
    lines = text.splitlines()
    kept: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        line = _ENTITY_RE.sub(" ", line)
        line = _URL_RE.sub(" ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if _NOISE_LINE_RE.match(line):
            continue
        if _LOW_INFO_RE.match(line):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        if len(line) < 3:
            continue
        kept.append(line)
    return "\n".join(kept)


def chunk_text(text: str, chunk_size: int = 320, overlap: int = 64) -> list[str]:
    """将文本按固定窗口大小切分为重叠块（保留为旧文件 fallback）。

    采用滑动窗口策略，相邻块之间有 overlap 字符的重叠区域。
    对于已通过 split_by_sections + _chunk_section 处理的块不再使用此函数，
    仅对无 ## 结构的纯文本文件保留。

    Args:
        text: 待切分的文本
        chunk_size: 每块的字符数上限，默认 320
        overlap: 相邻块的重叠字符数，默认 64（约 20% 重叠率）

    Returns:
        切分后的文本块列表；空文本返回空列表
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def ingest_directory(
    corpus_dir: Path | None = None,
    *,
    reset: bool = False,
    extract_facts: bool | None = None,
) -> dict:
    """扫描 persona/corpus/ 目录，将所有 md/txt 语料入库到 L3 长期记忆。

    增强的入库流程（阶段 2）：
    1. 遍历 corpus_dir 下所有 .md/.txt 文件
    2. 跳过 skip 名单中的文件（README、示例等）
    3. 对每个文件：
       a. 解析 YAML frontmatter（提取 type/people/time/topics/privacy/status/confidence）
       b. 剔除对话范例
       c. 行级降噪
       d. 按 ## 标题划分语义块
       e. 每块注入 [人物: xxx][时间: YYYY-MM][类型: xxx] 上下文前缀
       f. 必要时子分块
    4. 批量写入 L3 存储（自动生成 embedding + type→category + confidence）
    5. 可选：从语料块中提取结构化 Facts

    Args:
        corpus_dir: 语料目录路径，默认从 settings 读取
        reset: 是否先清空旧语料再入库（True 用于换 embedding 模型后重建）
        extract_facts: 是否提取 Facts；None 时使用配置项 persona_ingest_extract_facts

    Returns:
        入库结果汇总字典，包含字段：
        - files: 已入库的文件相对路径列表
        - corpus_chunks: 入库的总块数
        - fact_stats: Facts 提取统计（chunks/facts/skipped）
    """
    corpus_dir = corpus_dir or settings.resolved_corpus_dir()
    chunks: list[dict] = []
    idx = 0
    ingested: list[str] = []
    for path in sorted(corpus_dir.glob("**/*")):
        if not path.is_file():
            continue
        # 仅处理 Markdown 和纯文本格式的语料
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        if path.name in _SKIP_NAMES or path.name.endswith(".example"):
            continue
        if _is_noise_file(path):
            continue
        # 跳过 archive/ 目录下的旧语料，避免重复入库
        rel = path.relative_to(corpus_dir).as_posix()
        if rel.startswith("archive/"):
            continue

        # === 阶段 2 增强：frontmatter 解析 + 章节分块 + 上下文注入 ===
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        fm_meta, body = parse_frontmatter(raw)
        body = strip_dialogue_examples(body)
        body = denoise_text(body)
        if not body:
            continue

        context_prefix = build_context_prefix(fm_meta)
        sections = split_by_sections(body)

        if sections:
            # 新流程：按 ## 章节分块
            for section in sections:
                parts = _chunk_section(section, context_prefix)
                for part in parts:
                    chunks.append({
                        "id": f"doc-{idx}",
                        "text": part,
                        "meta": {
                            "source": rel,
                            "category": str(fm_meta.get("type", "")),
                            "confidence": float(fm_meta.get("confidence", 0.0)),
                        },
                    })
                    idx += 1
        elif context_prefix:
            # 有 frontmatter 但无 ## 结构（如极短文）：注入前缀后全量
            full = f"{context_prefix}\n{body}"
            chunks.append({
                "id": f"doc-{idx}",
                "text": full,
                "meta": {
                    "source": rel,
                    "category": str(fm_meta.get("type", "")),
                    "confidence": float(fm_meta.get("confidence", 0.0)),
                },
            })
            idx += 1
        else:
            # 无 frontmatter 也无 ## 结构：纯文本 fallback（保持向后兼容）
            for part in chunk_text(body):
                chunks.append({"id": f"doc-{idx}", "text": part, "meta": {"source": rel}})
                idx += 1

        ingested.append(rel)

    fact_stats = {"chunks": 0, "facts": 0, "skipped": 0}
    do_extract = (
        extract_facts
        if extract_facts is not None
        else bool(getattr(settings, "persona_ingest_extract_facts", False))
    )

    if chunks:
        # 将语料块写入 L3 存储（含向量化和索引构建）
        semantic_memory.ingest_chunks(chunks, reset=reset)
        if do_extract:
            # 可选：从语料块中提取结构化事实，用于关系网络构建
            fact_stats = batch_extract_facts_from_persona_chunks(chunks)

    # Wiki 人物页同步到 contact：入库后，将 persona/corpus/people/*.md 的人物生成为第三方画像
    wiki_synced = 0
    try:
        sync_results = sync_wiki_people_to_contacts()
        invalidate_wiki_cache()
        synced = [r for r in sync_results if r["action"] in ("新建", "已存在，补全")]
        wiki_synced = len(synced)
        if synced:
            log_detail = "; ".join(f'{r["name"]}({r["action"]})' for r in synced[:5])
            logger.info("wiki->contact sync: %d synced (%s)", wiki_synced, log_detail)
    except Exception as exc:
        logger.warning("wiki->contact sync failed: %s", exc)

    return {
        "files": ingested,
        "corpus_chunks": len(chunks),
        "fact_stats": fact_stats,
        "wiki_synced": wiki_synced,
    }


def startup_ingest_corpus() -> dict:
    """启动时自动语料同步。

    设计意图：确保应用启动后 L3 长期记忆中有可供检索的语料数据，
    但不重复入库已有的数据。

    逻辑：
    - 如果 persona_ingest_on_startup 配置为 False，跳过
    - 如果 L3 中已有语料且未设置 reset，跳过（避免重复入库）
    - 如果设置了 persona_ingest_reset_on_startup=True，则清空后重新入库
    - 启动时只做 Corpus 入库，不做 Facts 提取（提取在 scripts/ingest.py 中手动触发）

    Returns:
        执行结果字典，包含 skipped（是否跳过）和相关统计
    """
    if not getattr(settings, "persona_ingest_on_startup", True):
        return {"skipped": True, "reason": "disabled"}

    existing = semantic_memory.corpus.count()
    reset = bool(getattr(settings, "persona_ingest_reset_on_startup", False))
    # 已有语料且不需要重置时跳过，避免启动变慢
    if existing > 0 and not reset:
        # 尽管 L3 已存在，每次启动仍同步 Wiki 人物页到 contact（处理新增人物页）
        wiki_synced = 0
        try:
            sync_results = sync_wiki_people_to_contacts()
            invalidate_wiki_cache()
            wiki_synced = len([r for r in sync_results if r["action"] in ("新建", "已存在，补全")])
        except Exception as exc:
            logger.warning("wiki->contact sync failed: %s", exc)
        return {
            "skipped": True,
            "reason": "corpus_exists",
            "corpus_chunks": existing,
            "files": [],
            "fact_stats": {"chunks": 0, "facts": 0, "skipped": 0},
            "wiki_synced": wiki_synced,
        }

    cleared: dict = {}
    if reset:
        semantic_memory.reset_corpus()
        cleared = clear_persona_derived_memory()

    result = ingest_directory(extract_facts=False, reset=reset)
    result["cleared"] = cleared
    return result
