"""persona/corpus/ 语料入库模块 —— 两阶段幂等同步 pipeline。

职责：扫描 persona/corpus/ 目录下的 Markdown 语料文件，经过 frontmatter 解析、
剔除对话范例、去噪、章节级分块、上下文前缀注入后，以幂等方式同步到 memory_items。

数据流：
  persona/corpus/*.md
  → build_corpus_chunk_specs()   [阶段 1：纯文件扫描，产出稳定 source_id]
  → sync_corpus_chunk_specs()    [阶段 2：幂等同步到 DB， upsert 新块、删除过期块]
  → startup_ingest_corpus() √ audit  [启动时审计性跳过]

新规范（2026-07 收口版）：
  - 所有 corpus 行：kind='wiki', source='wiki', source_table='corpus'
  - source_id = '<relative_path>#s<section_index>p<part_index>'（稳定主键）
    例如：monthly/liu_yuanhui/2025-04.md#s1p0  表示第1张第0块
  - source_path 写入 context_json.source_path
  - month_key 从 files/frontmatter/time 解析，写入 context_json.month_key
  - sync 逻辑确保 DB 中的 corpus 子集 ≡ 文件系统切块结果
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import settings
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
    persona/corpus/ 中的语料入库到长期记忆后用于检索召回，不应包含对话范例，
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

    从 settings.long_term_memory_noise_file_patterns 读取以逗号分隔的 glob 模式，
    匹配文件名（支持通配符如 *.log、*.tmp 等）。

    Args:
        path: 待检查的文件路径

    Returns:
        True 表示该文件应被跳过，不参与入库
    """
    if not settings.long_term_memory_denoise_enabled:
        return False
    patterns = [p.strip() for p in settings.long_term_memory_noise_file_patterns.split(",") if p.strip()]
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


# ══════════════════════════════════════════════════════════════════════════════
# 阶段 1：build_corpus_chunk_specs — 纯文件扫描，产出稳定 source_id
# ══════════════════════════════════════════════════════════════════════════════


def _guess_month_key(source_path: str, body: str, fm_meta: dict) -> str:
    """从 frontmatter time / 文件路径 / 正文中推测 YYYY-MM。

    Returns:
        "YYYY-MM" 格式的字符串，无匹配时返回空字符串。
    """
    import re as _re

    # 1. frontmatter time 字段
    raw_time = str(fm_meta.get("time", "") or "").strip()
    m = _re.match(r"(\d{4})\s*[-/年]\s*(\d{1,2})(?:月|$)", raw_time)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"

    # 2. 文件路径中的月份模式：monthly/xxx/2025-04.md, 2025年4月.md 等
    m2 = _re.search(r"(\d{4})\s*[-/年]?\s*(\d{1,2})\s*月?", source_path)
    if m2:
        return f"{m2.group(1)}-{int(m2.group(2)):02d}"

    # 3. 正文中的 ## YYYY-MM 标题
    m3 = _re.search(r"##\s*(\d{4})-(\d{2})", body)
    if m3:
        return f"{m3.group(1)}-{m3.group(2)}"

    return ""


def _is_corpus_file(path: Path, corpus_dir: Path) -> bool:
    """判断文件是否应纳入语料扫描范围。"""
    if not path.is_file():
        return False
    if path.suffix.lower() not in {".md", ".txt"}:
        return False
    if path.name in _SKIP_NAMES or path.name.endswith(".example"):
        return False
    if _is_noise_file(path):
        return False
    rel = path.relative_to(corpus_dir).as_posix()
    if rel.startswith("archive/"):
        return False
    return True


def build_corpus_chunk_specs(
    corpus_dir: Path | None = None,
) -> list[dict]:
    """阶段 1：纯文件扫描，构建 corpus chunk 规格列表。

    每个 spec 包含稳定的 source_id（<relative_path>#<chunk_index>），
    用于后续 sync 阶段的幂等 upsert。

    Args:
        corpus_dir: 语料目录路径，默认从 settings 读取

    Returns:
        chunk_spec 列表，每项为 dict：
        - source_path: str — 文件相对路径
        - source_id: str — '<source_path>#s<section>p<part>'
        - kind: str — 固定 'wiki'
        - source: str — 固定 'wiki'
        - text: str — 注入上下文前缀后的完整文本块
        - month_key: str — 推测的月份标识（空串表示无法推测）
        - category: str — 类型标签（从 frontmatter type 或猜测）
        - confidence: float — 置信度
        - meta: dict — 元数据（含 source_table/source_id/source_path/month_key 等）
    """
    corpus_dir = corpus_dir or settings.resolved_corpus_dir()
    specs: list[dict] = []

    for path in sorted(corpus_dir.glob("**/*")):
        if not _is_corpus_file(path, corpus_dir):
            continue

        rel = path.relative_to(corpus_dir).as_posix()
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        fm_meta, body = parse_frontmatter(raw)
        body = strip_dialogue_examples(body)
        body = denoise_text(body)
        if not body:
            continue

        context_prefix = build_context_prefix(fm_meta)
        month_key = _guess_month_key(rel, body, fm_meta)
        category = str(fm_meta.get("type", ""))
        confidence = float(fm_meta.get("confidence", 0.0))
        sections = split_by_sections(body)

        if sections:
            for ci, section in enumerate(sections):
                parts = _chunk_section(section, context_prefix)
                for pi, part in enumerate(parts):
                    source_id = f"{rel}#s{ci}p{pi}"
                    specs.append({
                        "source_path": rel,
                        "source_id": source_id,
                        "kind": "wiki",
                        "source": "wiki",
                        "text": part,
                        "month_key": month_key,
                        "category": category or _guess_category(rel, fm_meta),
                        "confidence": confidence,
                        "meta": {
                            "source_table": "corpus",
                            "source_id": source_id,
                            "source_path": rel,
                            "month_key": month_key,
                            "category": category or _guess_category(rel, fm_meta),
                            "confidence": confidence,
                        },
                    })
        elif context_prefix:
            full = f"{context_prefix}\n{body}"
            source_id = f"{rel}#s0p0"
            specs.append({
                "source_path": rel,
                "source_id": source_id,
                "kind": "wiki",
                "source": "wiki",
                "text": full,
                "month_key": month_key,
                "category": category or _guess_category(rel, fm_meta),
                "confidence": confidence,
                "meta": {
                    "source_table": "corpus",
                    "source_id": source_id,
                    "source_path": rel,
                    "month_key": month_key,
                    "category": category or _guess_category(rel, fm_meta),
                    "confidence": confidence,
                },
            })
        else:
            for ci, part in enumerate(chunk_text(body)):
                source_id = f"{rel}#s0p{ci}"
                specs.append({
                    "source_path": rel,
                    "source_id": source_id,
                    "kind": "wiki",
                    "source": "wiki",
                    "text": part,
                    "month_key": month_key,
                    "category": category,
                    "confidence": confidence,
                    "meta": {
                        "source_table": "corpus",
                        "source_id": source_id,
                        "source_path": rel,
                        "month_key": month_key,
                        "category": category,
                        "confidence": confidence,
                    },
                })

    return specs


def _guess_category(rel_path: str, fm_meta: dict) -> str:
    """从文件路径或 frontmatter 推测类别。"""
    cat = str(fm_meta.get("type", "") or "").strip()
    if cat:
        return cat
    if "people/" in rel_path or rel_path.startswith("people/"):
        return "person"
    if "monthly/" in rel_path:
        return "monthly"
    if "profile/" in rel_path or rel_path.startswith("profile/"):
        return "profile"
    return "general"


# ══════════════════════════════════════════════════════════════════════════════
# 阶段 2：sync_corpus_chunk_specs — 幂等同步到 memory_items
# ══════════════════════════════════════════════════════════════════════════════


def sync_corpus_chunk_specs(
    specs: list[dict],
    *,
    reset: bool = False,
) -> dict:
    """阶段 2：将 chunk specs 幂等同步到 memory_items（source_table='corpus'）。

    同步规则（一次性保证源文件状态 ≡ DB 状态）：
      1. 对 specs 中的每个 source_id 执行 upsert
      2. 删除 DB 中存在但 specs 中不存在的 source_id（过期块）
      3. 当 reset=True 时，先全量删除再全量写入

    Args:
        specs: build_corpus_chunk_specs 返回的 spec 列表
        reset: 是否先全量清空再写入

    Returns:
        {"written": int, "errors": int, "stale_deleted": int, "total_specs": int}
    """
    from app.llm import embed_texts
    from app.session import store

    if not specs:
        if reset:
            store.reset_corpus_items()
            return {"written": 0, "errors": 0, "stale_deleted": 0, "total_specs": 0}
        return {"written": 0, "errors": 0, "stale_deleted": 0, "total_specs": 0}

    # ── reset 模式：先全量删除 ──
    if reset:
        store.reset_corpus_items()

    # ── 批量计算 embedding ──
    texts = [s["text"] for s in specs]
    embs = embed_texts(texts) or []
    if len(embs) != len(texts):
        embs = embs + ([[]] * (len(texts) - len(embs)))

    # ── 查询 DB 中现有的 corpus source_id 集合 ──
    existing_ids = set(store.list_corpus_source_ids())
    expected_ids = {s["source_id"] for s in specs if s.get("source_id")}

    # ── 逐个 upsert ──
    written = 0
    errors = 0
    for i, spec in enumerate(specs):
        meta = spec.get("meta", {})
        source_id = spec.get("source_id", "")
        if not source_id:
            continue
        emb = embs[i] if i < len(embs) else []
        context = {
            "source_path": spec.get("source_path", ""),
        }
        if spec.get("month_key"):
            context["month_key"] = spec["month_key"]
        if meta.get("category"):
            context["category"] = meta["category"]

        try:
            store.write_memory_item(
                person_id="",
                device_id="",
                kind=spec.get("kind", "wiki"),
                source=spec.get("source", "wiki"),
                source_table="corpus",
                source_id=source_id,
                visibility="recall_only",
                content=spec.get("text", ""),
                confidence=meta.get("confidence", 0.8),
                context_json=json.dumps(context, ensure_ascii=False),
                tags_json="[]",
                embedding_json=json.dumps(emb) if emb else "[]",
            )
            written += 1
        except Exception as exc:
            logger.error("corpus sync failed for source_id=%s: %s", source_id, exc)
            errors += 1

    # ── 删除过期 source_id（reset 模式下无需再删） ──
    stale_deleted = 0
    if not reset:
        stale_ids = existing_ids - expected_ids
        for sid in stale_ids:
            try:
                if store.delete_corpus_item_by_source_id(sid):
                    stale_deleted += 1
            except Exception as exc:
                logger.warning("corpus stale delete failed for %s: %s", sid, exc)

    # ── 去重：清理 upsert 后可能残留的重复行（如旧数据遗留） ──
    dedup_removed = 0
    if not reset:
        try:
            dedup_removed = store.dedup_corpus_source_ids()
            if dedup_removed:
                logger.info("corpus sync dedup: removed %d duplicate rows", dedup_removed)
        except Exception as exc:
            logger.warning("corpus dedup failed: %s", exc)

    # ── 最终库状态统计 ──
    try:
        final_corpus_rows = store.count_corpus_memory_items()
        final_corpus_ids = len(store.list_corpus_source_id_counts())
    except Exception:
        final_corpus_rows = written - errors
        final_corpus_ids = written - errors

    return {
        "written": written,
        "errors": errors,
        "stale_deleted": stale_deleted,
        "total_specs": len(specs),
        "dedup_removed": dedup_removed,
        "final_corpus_rows": final_corpus_rows,
        "final_corpus_ids": final_corpus_ids,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 审计：audit_corpus_sync_state — 只读对比文件系统与 DB
# ══════════════════════════════════════════════════════════════════════════════


def audit_corpus_sync_state(
    corpus_dir: Path | None = None,
) -> dict:
    """只读审计：对比文件系统预期 source_id 与数据库中实际 source_id。

    除检测缺失/多余外，新增对数据库物理重复行的检测。
    不修改任何数据。

    Returns:
        {
            "is_complete": bool,
            "missing_source_ids": list[str],       — DB 中缺失的 source_id
            "stale_source_ids": list[str],           — DB 中多余的 source_id
            "duplicate_source_ids": list[str],       — 物理重复的 source_id（有多行）
            "expected_chunk_count": int,             — 逻辑期望唯一块数
            "actual_chunk_count": int,               — 去重后的唯一 source_id 数
            "actual_row_count": int,                 — 物理行数
            "source_files": list[str],               — 扫描到的源文件列表
        }

    完整性判定：
        missing_source_ids == []
        stale_source_ids == []
        duplicate_source_ids == []
        actual_row_count == expected_chunk_count
    """
    from app.session import store

    corpus_dir = corpus_dir or settings.resolved_corpus_dir()
    if not corpus_dir.exists():
        return {
            "is_complete": False,
            "missing_source_ids": [],
            "stale_source_ids": [],
            "duplicate_source_ids": [],
            "expected_chunk_count": 0,
            "actual_chunk_count": 0,
            "actual_row_count": 0,
            "source_files": [],
        }

    # 文件系统端
    expected_specs = build_corpus_chunk_specs(corpus_dir)
    expected_ids = {s["source_id"] for s in expected_specs if s.get("source_id")}
    source_files = sorted({s["source_path"] for s in expected_specs if s.get("source_path")})

    # DB 端：带重复检测
    id_counts = store.list_corpus_source_id_counts()
    actual_ids_set = set(id_counts.keys())
    actual_row_count = sum(id_counts.values()) if id_counts else 0
    duplicate_ids = sorted(
        sid for sid, cnt in id_counts.items() if cnt > 1
    )

    missing = sorted(expected_ids - actual_ids_set)
    stale = sorted(actual_ids_set - expected_ids)

    is_complete = (
        len(missing) == 0
        and len(stale) == 0
        and len(duplicate_ids) == 0
        and actual_row_count == len(expected_ids)
    )

    return {
        "is_complete": is_complete,
        "missing_source_ids": missing,
        "stale_source_ids": stale,
        "duplicate_source_ids": duplicate_ids,
        "expected_chunk_count": len(expected_ids),
        "actual_chunk_count": len(actual_ids_set),
        "actual_row_count": actual_row_count,
        "source_files": source_files,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 对外接口：ingest_directory / startup_ingest_corpus
# ══════════════════════════════════════════════════════════════════════════════


def ingest_directory(
    corpus_dir: Path | None = None,
    *,
    reset: bool = False,
    extract_facts: bool | None = None,
) -> dict:
    """扫描 persona/corpus/ 目录，以幂等方式同步全部语料到 memory_items。

    内部调用 build_corpus_chunk_specs() + sync_corpus_chunk_specs()。

    Args:
        corpus_dir: 语料目录路径，默认从 settings 读取
        reset: 是否先清空旧 corpus 再全量重建
        extract_facts: 不再使用（仅保持签名兼容）

    Returns:
        入库结果字典：
        - files: 已处理的源文件相对路径列表
        - corpus_chunks: 写入的总块数
        - fact_stats: （保持兼容）{"chunks":0,"facts":0,"skipped":0}
        - sync_stats: sync_corpus_chunk_specs 返回的详细统计
        - wiki_synced: wiki→contact 同步的条目数
    """
    del extract_facts  # 不再使用

    specs = build_corpus_chunk_specs(corpus_dir)
    sync_stats = sync_corpus_chunk_specs(specs, reset=reset)

    ingested_files = sorted(
        {s["source_path"] for s in specs if s.get("source_path")}
    )

    # Wiki 人物页同步到 contact
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
        "files": ingested_files,
        "corpus_chunks": sync_stats.get("written", 0),
        "fact_stats": {"chunks": 0, "facts": 0, "skipped": 0},
        "sync_stats": sync_stats,
        "wiki_synced": wiki_synced,
    }


def startup_ingest_corpus() -> dict:
    """启动时自动语料同步（审计型跳过）。

    逻辑：
    - persona_ingest_on_startup=false → 跳过
    - persona_ingest_reset_on_startup=true → 强制全量重建
    - 否则执行 audit_corpus_sync_state()：
      - is_complete → 跳过（打印审计结果）
      - 不完整 → 自动增量同步

    Returns:
        执行结果字典，包含 skipped（是否跳过）和相关统计
    """
    if not getattr(settings, "persona_ingest_on_startup", True):
        return {"skipped": True, "reason": "disabled"}

    reset = bool(getattr(settings, "persona_ingest_reset_on_startup", False))

    if reset:
        logger.info("startup_ingest_corpus: reset=True, 全量重建")
        cleared: dict = {}
        try:
            from app.memory.long_term_memory import clear_derived_memory
            cleared = clear_derived_memory()
        except Exception:
            pass
        result = ingest_directory(reset=True)
        result["cleared"] = cleared
        return result

    # ── 审计模式：检查 corpus 完整性 ──
    audit = audit_corpus_sync_state()

    if audit["is_complete"]:
        logger.info(
            "corpus 已同步（%d 块，%d 个源文件），跳过启动入库",
            audit["actual_chunk_count"],
            len(audit["source_files"]),
        )
        # 仍同步 Wiki 人物页到 contact
        wiki_synced = 0
        try:
            sync_results = sync_wiki_people_to_contacts()
            invalidate_wiki_cache()
            wiki_synced = len(
                [r for r in sync_results if r["action"] in ("新建", "已存在，补全")]
            )
        except Exception:
            pass
        return {
            "skipped": True,
            "reason": "corpus_complete",
            "audit": audit,
            "corpus_chunks": audit["actual_chunk_count"],
            "files": [],
            "fact_stats": {"chunks": 0, "facts": 0, "skipped": 0},
            "wiki_synced": wiki_synced,
        }

    # ── 不完整：自动增量同步 ──
    missing_count = len(audit["missing_source_ids"])
    stale_count = len(audit["stale_source_ids"])
    logger.warning(
        "corpus 不完整：缺失 %d 个 source_id，多余 %d 个 source_id，触发增量同步",
        missing_count,
        stale_count,
    )

    result = ingest_directory()
    result["audit"] = audit
    result["audit_fixed"] = True
    return result
