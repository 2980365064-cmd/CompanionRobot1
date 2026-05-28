"""Corpus ingestion utilities (persona/corpus/ only)."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from app.config import settings
from app.memory.semantic import semantic_memory

_SKIP_NAMES = frozenset({
    "README.md",
    "intimate.md.example",
    "sample_corpus.md",
})
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_ENTITY_RE = re.compile(r"&#x[0-9a-fA-F]+;")
_NOISE_LINE_RE = re.compile(r"^\s*[\W_]{3,}\s*$")
_LOW_INFO_RE = re.compile(r"^(666+|哈{3,}|嗯{3,}|[?？!！\.。]{2,})$")


def strip_dialogue_examples(text: str) -> str:
    """Remove Q→A sections; tone examples belong in persona/style/."""
    lines = text.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        if re.match(r"^#+\s*(典型对话|口吻范例|对话范例)", line):
            skip = True
            continue
        if skip and re.match(r"^#+\s+", line):
            skip = False
        if skip:
            continue
        if re.match(r"^问[:：]", line.strip()) or re.match(r"^答[:：]", line.strip()):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _is_noise_file(path: Path) -> bool:
    if not settings.l3_denoise_enabled:
        return False
    patterns = [p.strip() for p in settings.l3_noise_file_patterns.split(",") if p.strip()]
    name = path.name
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def denoise_text(text: str) -> str:
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


def ingest_directory(corpus_dir: Path | None = None, *, reset: bool = False) -> list[str]:
    """Ingest markdown under persona/corpus/ into L3 vector store."""
    corpus_dir = corpus_dir or settings.resolved_corpus_dir()
    chunks = []
    idx = 0
    ingested: list[str] = []
    for path in sorted(corpus_dir.glob("**/*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        if path.name in _SKIP_NAMES or path.name.endswith(".example"):
            continue
        if _is_noise_file(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        text = strip_dialogue_examples(text)
        text = denoise_text(text)
        if not text:
            continue
        rel = path.relative_to(corpus_dir).as_posix()
        for part in chunk_text(text):
            chunks.append({"id": f"doc-{idx}", "text": part, "meta": {"source": rel}})
            idx += 1
        ingested.append(rel)
    if chunks:
        semantic_memory.ingest_chunks(chunks, reset=reset)
    return ingested


def startup_ingest_corpus() -> list[str]:
    """Re-sync persona/corpus/ into L3 search index (call on app startup)."""
    semantic_memory.reset_corpus()
    files = ingest_directory()
    return files
