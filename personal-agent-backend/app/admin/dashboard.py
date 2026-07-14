"""运维仪表盘 —— 统一返回服务健康、风险清单、推荐动作、最近任务与错误。

/GET /v1/admin/dashboard
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from app.admin.audit import get_audit_stats
from app.admin.ops import deep_health, detect_restart_mode
from app.config import settings
from app.admin.persons import list_persons_admin
from app.session import store


def _count_memory_items() -> int:
    """返回统一记忆库条目总数。"""
    from app.session import store
    try:
        stats = store.count_memory_items()
        return stats.get("total", 0)
    except Exception:
        return 0


def _recent_errors(limit: int = 10) -> list[dict]:
    """从最近任务日志中提取有 error 的记录。"""
    # 从 admin_tasks 倒序扫描失败的或带 error 的
    from app.admin.tasks import task_manager
    all_tasks = task_manager.list()
    errors = []
    for t in all_tasks:
        if t.get("status") in ("failed",) or t.get("error"):
            errors.append({
                "task_id": t["id"],
                "task_name": t["name"],
                "title": t["title"],
                "error": (t.get("error") or "")[:200],
                "finished_at": t.get("finished_at", ""),
            })
            if len(errors) >= limit:
                break
    return errors


def _recent_tasks(limit: int = 10) -> list[dict]:
    from app.admin.tasks import task_manager
    return task_manager.list()[:limit]


def _config_risks() -> list[dict]:
    """检查常见配置风险。"""
    risks = []
    db = settings.resolved_db_path()
    if not db.exists():
        risks.append({
            "level": "error",
            "title": "数据库文件不存在",
            "detail": f"DB_PATH={db} 文件不存在，系统将在首次对话时自动创建空库。",
            "action": {"label": "查看配置", "href": "#config"},
            "fixable": True,
        })
    else:
        try:
            size_mb = db.stat().st_size / 1024 / 1024
            if size_mb > 500:
                risks.append({
                    "level": "warning",
                    "title": "数据库文件偏大",
                    "detail": f"当前 {size_mb:.0f} MB，建议备份后执行 VACUUM。",
                    "action": {"label": "查看备份", "href": "#backup"},
                    "fixable": True,
                })
        except OSError:
            pass

    if not settings.llm_api_key:
        risks.append({
            "level": "error",
            "title": "LLM API 未配置",
            "detail": "未设置 LLM_API_KEY，对话将无法生成。请检查 .env 文件。",
            "action": {"label": "修改配置", "href": "#config"},
            "fixable": True,
        })

    if not settings.embed_api_key:
        risks.append({
            "level": "warning",
            "title": "向量 API 未配置（使用本地 fallback）",
            "detail": "未设置 EMBED_API_KEY，将使用本地哈希伪向量。语义检索精度会大幅下降。",
            "action": {"label": "修改配置", "href": "#config"},
            "fixable": True,
        })

    try:
        from app.persona.ingest import audit_corpus_sync_state
        audit = audit_corpus_sync_state()
        if not audit["is_complete"]:
            missing = len(audit["missing_source_ids"])
            stale = len(audit["stale_source_ids"])
            duplicate = len(audit.get("duplicate_source_ids", []))
            details = []
            if missing:
                details.append(f"缺失 {missing} 块")
            if stale:
                details.append(f"多余 {stale} 块")
            if duplicate:
                details.append(f"重复 {duplicate} 个 source_id")
            if not details:
                details.append(f"行数不一致（物理 {audit.get('actual_row_count', '?')} ≠ 期望 {audit.get('expected_chunk_count', '?')}）")
            risks.append({
                "level": "warning",
                "title": "语料未同步",
                "detail": f"corpus 不完整（{'，'.join(details)}）。建议执行入库。",
                "action": {"label": "执行入库", "href": "#tasks"},
                "fixable": True,
            })
    except Exception:
        pass

    return risks


def _memory_risks() -> list[dict]:
    """检查记忆层健康状态。"""
    risks = []
    stats = store.count_memory_stat() if hasattr(store, 'count_memory_stat') else {}
    profiles = stats.get("profiles", 0)
    persons = stats.get("persons", 0)
    if persons and not profiles:
        risks.append({
            "level": "warning",
            "title": "存在已实名用户但无画像",
            "detail": f"共 {persons} 个实名用户，无人像画像。需确认是否正常。",
            "action": {"label": "查看人物", "href": "#persons"},
            "fixable": False,
        })

    mi_active = stats.get("memory_items_active", 0)
    if mi_active > 10000:
        risks.append({
            "level": "info",
            "title": "统一记忆库条目较多",
            "detail": f"当前 {mi_active} 条有效记忆条目，检查归档任务是否正常。",
            "action": {"label": "查看记忆库", "href": "#memory-items"},
            "fixable": True,
        })
    return risks


def _needs_restart() -> bool:
    """检查是否检测到 systemd 且可以重启。"""
    mode = detect_restart_mode()
    return mode.get("can_restart", False)


def _needs_reingest() -> bool:
    """检查语料目录是否有比上次入库更新的文件。"""
    corpus_dir = settings.resolved_corpus_dir()
    if not corpus_dir.exists():
        return False
    corpus_mtime = 0.0
    for f in corpus_dir.rglob("*.md"):
        try:
            mtime = f.stat().st_mtime
            if mtime > corpus_mtime:
                corpus_mtime = mtime
        except OSError:
            pass
    if corpus_mtime == 0.0:
        return False
    # 对比 agent.db 修改时间
    db = settings.resolved_db_path()
    if not db.exists():
        return True
    try:
        db_mtime = db.stat().st_mtime
        # 如果新文件时间比数据库更新，可能需重新入库
        return corpus_mtime > db_mtime + 3600  # 宽容 1 小时
    except OSError:
        return False


def build_dashboard() -> dict:
    """构建运维仪表盘数据。"""
    health = deep_health()
    restart_mode = detect_restart_mode()
    persons_list = list_persons_admin()
    audit_stats = get_audit_stats()
    config_risks = _config_risks()
    memory_risks = _memory_risks()
    all_risks = config_risks + memory_risks

    # 推荐动作
    suggested_actions = []
    has_error = any(r["level"] == "error" for r in all_risks)
    has_warning = any(r["level"] == "warning" for r in all_risks)
    for r in all_risks:
        if r.get("fixable"):
            suggested_actions.append({
                "level": r["level"],
                "title": r["title"],
                "label": r["action"]["label"],
                "href": r["action"]["href"],
            })

    # 系统是否建议重启
    needs_restart = _needs_restart()
    needs_reingest = _needs_reingest()

    # 数据库状态
    db = settings.resolved_db_path()
    db_status = {"path": str(db), "exists": db.exists()}
    if db.exists():
        try:
            db_status["size_mb"] = round(db.stat().st_size / 1024 / 1024, 1)
        except OSError:
            pass
        try:
            conn = sqlite3.connect(str(db))
            db_status["wal_size"] = 0
            wal_path = db.with_suffix(".db-wal")
            if wal_path.exists():
                db_status["wal_size"] = round(wal_path.stat().st_size / 1024, 1)
            conn.execute("PRAGMA wal_checkpoint;").fetchall()
            conn.close()
        except Exception:
            pass

    return {
        "server": health.get("server", {}),
        "llm": health.get("llm", {}),
        "embed": health.get("embed", {}),
        "database": {
            **db_status,
            "ok": health.get("database", {}).get("ok", False),
        },
        "memory": health.get("memory", {}),
        "memory_items_count": _count_memory_items(),
        "sessions": health.get("sessions", {}),
        "persons": {
            "count": len(persons_list),
            "list": persons_list[:20],
        },
        "risks": all_risks,
        "has_errors": has_error,
        "has_warnings": has_warning,
        "suggested_actions": suggested_actions[:8],
        "needs_restart": needs_restart,
        "needs_reingest": needs_reingest,
        "recent_errors": _recent_errors(8),
        "recent_tasks": _recent_tasks(10),
        "restart_mode": restart_mode,
        "audit_stats": audit_stats,
        "search_backend": health.get("search", {}).get("backend", "unknown"),
        "uptime_sec": int(time.time() - (health.get("server", {}).get("started_at", time.time()))),
    }
