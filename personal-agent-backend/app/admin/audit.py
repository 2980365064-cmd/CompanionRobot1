"""后台操作审计日志：记录配置修改、persona 保存、记忆编辑、服务启停等。

存储方式：
  使用 JSONL 文件（appends only），按 UTC 日期分片，自动清理 90 天前旧日志。

使用方式：
  log_audit(kind, action, detail, operator="admin")
  每个需要审计的操作在完成后调用一次。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

# 审计日志目录（和 agent.db 同级）
AUDIT_DIR = settings.resolved_db_path().parent / "audit_logs"
_lock = threading.Lock()
_MAX_ENTRIES_PER_PAGE = 200
_RETENTION_DAYS = 90


def _ensure_dir() -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIT_DIR


def _today_file() -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _ensure_dir() / f"audit-{date_str}.jsonl"


def _cleanup_old_logs() -> None:
    """清理超过 RETENTION_DAYS 天的审计日志文件。"""
    cutoff = datetime.now(timezone.utc).timestamp() - _RETENTION_DAYS * 86400
    for f in sorted(_ensure_dir().glob("audit-*.jsonl")):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def log_audit(
    kind: str,
    action: str,
    detail: str = "",
    operator: str = "admin",
    metadata: dict[str, Any] | None = None,
) -> dict:
    """写入一条审计日志。

    参数：
      kind:     操作分类，如 config/persona/memory/task/service/backup/deploy
      action:   具体操作描述，如 "修改配置: LLM_MODEL=deepseek-v3"
      detail:   更详细的上下文（可选）
      operator: 操作者身份，默认 "admin"
      metadata: 附加结构化字段（可选）
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "action": action,
        "detail": detail,
        "operator": operator,
    }
    if metadata:
        entry["metadata"] = metadata

    with _lock:
        try:
            fpath = _today_file()
            with fpath.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            _cleanup_old_logs()
        except OSError:
            pass  # 审计日志写入失败不应影响主流程
    return entry


def query_audit(
    kind: str | None = None,
    operator: str | None = None,
    limit: int = 100,
    offset: int = 0,
    days_back: int = 7,
) -> list[dict]:
    """查询审计日志，按时间倒序返回。

    参数：
      kind:      可筛选操作分类
      operator:  可筛选操作者
      limit:     返回条数上限
      offset:    偏移量
      days_back: 回溯天数
    """
    cutoff = datetime.now(timezone.utc).timestamp() - days_back * 86400
    entries: list[dict] = []

    for f in sorted(_ensure_dir().glob("audit-*.jsonl"), reverse=True):
        try:
            if f.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        with _lock:
            try:
                lines = f.read_text("utf-8").strip().splitlines()
            except OSError:
                continue
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind and entry.get("kind") != kind:
                continue
            if operator and entry.get("operator") != operator:
                continue
            entries.append(entry)
            if len(entries) >= offset + limit:
                break
        if len(entries) >= offset + limit:
            break

    total = len(entries)
    return entries[offset : offset + limit]


def get_audit_stats(days_back: int = 7) -> dict:
    """返回审计统计摘要。"""
    cutoff = datetime.now(timezone.utc).timestamp() - days_back * 86400
    counts: dict[str, int] = {}
    total = 0
    today_count = 0
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for f in sorted(_ensure_dir().glob("audit-*.jsonl"), reverse=True):
        try:
            if f.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        is_today = today_str in f.name
        with _lock:
            try:
                lines = f.read_text("utf-8").strip().splitlines()
            except OSError:
                continue
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = entry.get("kind", "unknown")
            counts[k] = counts.get(k, 0) + 1
            total += 1
            if is_today:
                today_count += 1

    return {"total": total, "today": today_count, "by_kind": counts}
