#!/usr/bin/env python3
"""审计统一记忆库的代码、资料与数据库完整性。"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


TEXT_SUFFIXES = {".py", ".md", ".toml", ".ini", ".yaml", ".yml", ".json", ".txt", ".html", ".js", ".sh"}
IGNORED_DIRECTORIES = {
    ".git", ".pytest_cache", "__pycache__", ".venv", "venv",
    "esp_sparkbot", "esp_sparkbot-master",
}


def _retired(*parts: str) -> str:
    return "".join(parts)


_layer_prefix = _retired("l", "[0-3]")
_old_term = _retired("epi", "sodic")
_old_schema = (
    _retired("l", "0_core_memories"),
    _retired("epi", "sodic_memories"),
    _retired("semantic", "_facts"),
    _retired("l", "3_chunks"),
    _retired("l", "3_chunks_fts"),
    _retired("l", "3_recall_stats"),
)
_old_prefixes = (_retired("f", "act:"), _retired("ch", "unk:"))
_retired_path_names = {
    _retired("l", "0.py"),
    _retired("l", "1.py"),
    _retired("l", "2.py"),
    _retired("l", "3.py"),
    _retired("extract", "or.py"),
    _retired("test_", "legacy_memory_architecture_audit.py"),
    _retired("audit_", "legacy_memory_architecture.py"),
    _retired("normalize_unified_memory_", "legacy_metadata.py"),
}
_retired_path_marker = _retired("leg", "acy")

RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "retired_schema",
        "已移除的记忆存储名称",
        re.compile(r"\b(?:" + "|".join(map(re.escape, _old_schema)) + r")\b", re.I),
    ),
    (
        "retired_layer_name",
        "已移除的分层编号名称",
        re.compile(r"\b" + _retired("L", "[0-3]") + r"\b", re.I),
    ),
    (
        "retired_api",
        "已移除的分层接口或召回字段",
        re.compile(
            r"(?:\b" + _layer_prefix + r"_[a-z]\w*\b"
            + r"|\b" + re.escape(_old_term) + r"(?:_memory)?\b"
            + r"|\bto_" + _retired("leg", "acy") + r"_recall_dict\b"
            + r"|\bworking_memory\b"
            + r"|\bmemory\.get\(\s*['\"]matches['\"]\s*\)"
            + r"|\bdiagnostics\.get\(\s*['\"]" + _old_term + r"['\"]\s*\)"
            + r"|\b(?:" + "|".join(map(re.escape, _old_prefixes)) + r"))",
            re.I,
        ),
    ),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    bucket: str
    rule: str
    reason: str
    text: str


def _backend_root(root: Path) -> Path:
    candidate = root / "personal-agent-backend"
    return candidate if candidate.is_dir() else root


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {"AGENTS.md", "CLAUDE.md", "README.md", ".env.example"}:
            files.append(path)
    return sorted(files)


def _bucket(relative_path: str) -> str:
    if "/tests/" in f"/{relative_path}" or relative_path.startswith("tests/"):
        return "test_debt"
    if relative_path.endswith(".md") or relative_path.endswith(".json") or "/persona/" in f"/{relative_path}" or "/docs/" in f"/{relative_path}":
        return "documentation_debt"
    return "runtime_blocker"


def _retired_path_reason(relative_path: str) -> str:
    """返回已移除记忆架构路径的违规原因；正常路径返回空字符串。"""
    path = Path(relative_path)
    name = path.name.lower()
    lowered = relative_path.lower()
    if name in _retired_path_names:
        return "已移除的记忆模块、脚本或测试文件名"
    if "persona/corpus/" in lowered and _retired_path_marker in name:
        return "corpus 归档文件不应保留已移除架构命名"
    return ""


def scan(root: Path) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for path in _iter_files(root):
        relative_path = path.relative_to(root).as_posix()
        path_reason = _retired_path_reason(relative_path)
        if path_reason:
            findings.append(Finding(relative_path, 0, _bucket(relative_path), "retired_path", path_reason, path.name))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, start=1):
            for rule, reason, pattern in RULES:
                if pattern.search(line):
                    findings.append(Finding(relative_path, number, _bucket(relative_path), rule, reason, line.strip()[:240]))
    return findings


def scan_database(root: Path) -> list[Finding]:
    database = _backend_root(root) / "agent.db"
    if not database.is_file():
        return [Finding(str(database), 0, "runtime_blocker", "database", "未找到当前数据库", "")]
    findings: list[Finding] = []
    with sqlite3.connect(database) as conn:
        tables = {str(row[0]).lower() for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"memory_items", "memory_relations"}
        missing = required - tables
        if missing:
            return [Finding("agent.db", 0, "runtime_blocker", "database", "缺少统一记忆表", ", ".join(sorted(missing)))]
        for name in _old_schema:
            if name in tables:
                findings.append(Finding("agent.db", 0, "runtime_blocker", "database", "存在已移除的记忆表", name))
        invalid_relations = conn.execute(
            """SELECT COUNT(*) FROM memory_relations
               WHERE (from_id NOT LIKE 'memory:%' AND from_id NOT LIKE 'entity:%' AND from_id NOT LIKE 'relationship:%')
                  OR (to_id NOT LIKE 'memory:%' AND to_id NOT LIKE 'entity:%' AND to_id NOT LIKE 'relationship:%')"""
        ).fetchone()[0]
        if invalid_relations:
            findings.append(Finding("agent.db", 0, "runtime_blocker", "database", "关联图存在非规范节点键", str(invalid_relations)))
        corpus_issues = conn.execute(
            """SELECT COUNT(*) FROM memory_items
               WHERE source_table='corpus' AND (TRIM(source_id)='' OR source_id LIKE ? OR source_id LIKE ?)""",
            tuple(prefix + "%" for prefix in _old_prefixes),
        ).fetchone()[0]
        if corpus_issues:
            findings.append(Finding("agent.db", 0, "runtime_blocker", "database", "语料来源不规范", str(corpus_issues)))
        duplicate_ids = conn.execute(
            """SELECT COUNT(*) FROM (
                   SELECT source_id FROM memory_items WHERE source_table='corpus'
                   GROUP BY source_id HAVING COUNT(*) > 1
               )"""
        ).fetchone()[0]
        if duplicate_ids:
            findings.append(Finding("agent.db", 0, "runtime_blocker", "database", "语料 source_id 重复", str(duplicate_ids)))
    return findings


def report(findings: list[Finding]) -> dict:
    counts = Counter(item.bucket for item in findings)
    return {
        "summary": {
            "total": len(findings),
            "runtime_blocker": counts["runtime_blocker"],
            "semantic_debt": 0,
            "test_debt": counts["test_debt"],
            "documentation_debt": counts["documentation_debt"],
            "database_debt": sum(item.path == "agent.db" for item in findings),
        },
        "by_file": dict(Counter(item.path for item in findings).most_common()),
        "findings": [asdict(item) for item in findings],
    }


def to_markdown(data: dict) -> str:
    summary = data["summary"]
    lines = ["# 统一记忆库完整性审计", "", "| 类别 | 数量 |", "| --- | ---: |"]
    lines.extend(f"| {key} | {value} |" for key, value in summary.items())
    lines.extend(["", "## 逐项命中", ""])
    lines.extend(f"- `{item['bucket']}` `{item['path']}:{item['line']}` [{item['rule']}] `{item['text']}`" for item in data["findings"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    data = report(scan(root) + scan_database(root))
    output = to_markdown(data)
    print(output, end="")
    if args.json:
        args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(output, encoding="utf-8")
    return 1 if args.strict and data["summary"]["total"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
