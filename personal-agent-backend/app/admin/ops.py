"""后台运维、健康检查、部署状态与记忆调试。"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable

from app.config import settings
from app.llm import embed_provider_name
from app.memory.router import memory_router
from app.session import store

STARTED_AT = time.time()
BACKEND_ROOT = Path(__file__).resolve().parent.parent
SERVICE_UNIT_NAME = "sparkbot-agent.service"
SERVICE_UNIT_FILE = BACKEND_ROOT / "deploy" / SERVICE_UNIT_NAME
SYSTEMD_RUNTIME = Path("/run/systemd/system")
SERVICE_ACTIONS = {"start", "stop", "restart"}


def _git(args: list[str], timeout: int = 8) -> dict:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(BACKEND_ROOT.parent),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {"ok": p.returncode == 0, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc)}


def deep_health() -> dict:
    db = settings.resolved_db_path()
    db_ok = False
    db_detail = "missing"
    if db.exists():
        try:
            conn = sqlite3.connect(str(db))
            conn.execute("select 1").fetchone()
            conn.close()
            db_ok = True
            db_detail = f"{db.stat().st_size} bytes"
        except Exception as exc:
            db_detail = str(exc)
    stats = store.count_memory_stat()
    return {
        "server": {
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "started_at": STARTED_AT,
            "uptime_sec": int(time.time() - STARTED_AT),
            "host": settings.host,
            "port": settings.port,
        },
        "llm": {"ok": bool(settings.llm_api_key), "model": settings.llm_model, "base_url": settings.llm_base_url},
        "embed": {"ok": bool(settings.embed_api_key), "model": settings.embed_model, "provider": embed_provider_name()},
        "database": {"ok": db_ok, "path": str(db), "detail": db_detail},
        "search": {"backend": settings.search_backend, "es_url": settings.es_url},
        "memory": stats,
    }


def deploy_status() -> dict:
    rev = _git(["rev-parse", "--short", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _git(["status", "--short"])
    restart_mode = detect_restart_mode()
    return {
        "backend_root": str(BACKEND_ROOT),
        "git": {
            "commit": rev["stdout"] if rev["ok"] else "",
            "branch": branch["stdout"] if branch["ok"] else "",
            "dirty": dirty["stdout"].splitlines()[:80] if dirty["ok"] and dirty["stdout"] else [],
        },
        "service": {
            "systemd_unit": SERVICE_UNIT_FILE.exists(),
            "unit_name": SERVICE_UNIT_NAME,
            "unit_path": str(SERVICE_UNIT_FILE),
            "restart_mode": restart_mode,
            "pid": os.getpid(),
            "start_command": "python -m app.main / uvicorn app.main:app",
        },
    }


async def run_deploy_update() -> dict:
    pull = await asyncio.to_thread(_git, ["pull", "--ff-only"], 60)
    return {"pull": pull, "needs_restart": True}


def detect_restart_mode(
    *,
    systemd_runtime: Path = SYSTEMD_RUNTIME,
    systemctl_path: str | None = None,
    service_file: Path = SERVICE_UNIT_FILE,
) -> dict:
    """Detect the only restart path we allow from the admin UI."""
    resolved_systemctl = systemctl_path if systemctl_path is not None else shutil.which("systemctl")
    if systemd_runtime.exists() and resolved_systemctl and service_file.exists():
        return {
            "mode": "systemd",
            "can_restart": True,
            "unit": SERVICE_UNIT_NAME,
            "command": ["systemctl", "restart", SERVICE_UNIT_NAME],
            "reason": "检测到 systemd 运行目录和服务单元文件",
        }
    if not systemd_runtime.exists():
        return {
            "mode": "local_dev_safe",
            "can_restart": False,
            "unit": SERVICE_UNIT_NAME,
            "command": [],
            "reason": "当前环境未检测到 systemd，通常是 PyCharm/本地 uvicorn 启动",
        }
    return {
        "mode": "unsupported",
        "can_restart": False,
        "unit": SERVICE_UNIT_NAME,
        "command": [],
        "reason": "未找到 systemctl 或服务单元文件，已拒绝执行重启",
    }


async def _delayed_restart(command: list[str], delay_sec: float) -> None:
    await asyncio.sleep(delay_sec)
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        _audit_restart(
            "systemd restart executed",
            mode="systemd",
            accepted=result.returncode == 0,
            detail=(result.stderr or result.stdout or "").strip(),
        )
    except Exception as exc:
        _audit_restart("systemd restart failed", mode="systemd", accepted=False, detail=str(exc))


async def _delayed_service_action(command: list[str], delay_sec: float) -> None:
    await asyncio.sleep(delay_sec)
    action = command[1] if len(command) > 1 else "unknown"
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        _audit_restart(
            f"systemd {action} executed",
            mode="systemd",
            accepted=result.returncode == 0,
            detail=(result.stderr or result.stdout or "").strip(),
        )
    except Exception as exc:
        _audit_restart(f"systemd {action} failed", mode="systemd", accepted=False, detail=str(exc))


def _audit_restart(title: str, *, mode: str, accepted: bool, detail: str = "") -> None:
    try:
        from app.monitor import agent_monitor

        status = "accepted" if accepted else "rejected"
        suffix = f" · {detail}" if detail else ""
        agent_monitor.warn(f"{title} · mode={mode} · {status}{suffix}")
    except Exception:
        pass


async def request_restart(
    *,
    restart_runner: Callable[[list[str], float], Awaitable[None]] | None = None,
    restart_mode: dict | None = None,
    delay_sec: float = 0.8,
) -> dict:
    return await request_service_action(
        "restart",
        action_runner=restart_runner,
        restart_mode=restart_mode,
        delay_sec=delay_sec,
    )


async def request_service_action(
    action: str,
    *,
    action_runner: Callable[[list[str], float], Awaitable[None]] | None = None,
    restart_mode: dict | None = None,
    delay_sec: float = 0.8,
) -> dict:
    if action not in SERVICE_ACTIONS:
        raise ValueError(f"unsupported service action: {action}")

    mode = restart_mode or detect_restart_mode()
    command = ["systemctl", action, SERVICE_UNIT_NAME] if mode.get("mode") == "systemd" else []
    accepted = bool(mode.get("can_restart")) and mode.get("mode") == "systemd" and command == [
        "systemctl",
        action,
        SERVICE_UNIT_NAME,
    ]

    if accepted:
        if action_runner is None:
            asyncio.create_task(_delayed_service_action(command, delay_sec))
        else:
            await action_runner(command, delay_sec)
        message_map = {
            "start": "已触发服务启动，稍后会自动检查服务状态。",
            "stop": "已触发服务停止，当前页面可能很快断开。",
            "restart": "已触发一键重启，页面可能短暂断开，稍后会自动恢复。",
        }
        message = message_map[action]
    else:
        label = {"start": "启动", "stop": "停止", "restart": "重启"}[action]
        message = f"当前是本地开发/非 systemd 环境，未执行{label}。请在 PyCharm 或进程管理器中手动操作。"

    _audit_restart(
        f"服务{ {'start':'启动','stop':'停止','restart':'重启'}[action] }尝试",
        mode=str(mode.get("mode") or "unknown"),
        accepted=accepted,
        detail=str(mode.get("reason") or ""),
    )

    return {
        "accepted": accepted,
        "action": action,
        "mode": mode.get("mode"),
        "command": command,
        "pid_before": os.getpid(),
        "started_at": time.time(),
        "message": message,
        "reason": mode.get("reason"),
    }


def _compute_readiness_hint(diag: dict) -> str:
    """基于 diagnostics 计算记忆系统状态提示。"""
    read_path = diag.get("read_path", "memory_items")
    if read_path == "memory_items":
        return "已使用统一记忆库，状态正常。"
    return f"当前读路径: {read_path}"


def debug_recall(device_id: str, session_id: str, person_id: str, query: str) -> dict:
    memory = memory_router.recall(device_id or "admin-debug", session_id or "admin-debug", query, person_id=person_id)
    diag = memory.get("diagnostics") or {}
    items = memory.get("items") or []
    history = memory.get("history") or []

    # 从 items 中按语义属性分类
    def _is_always_visible(it) -> bool:
        return getattr(it, "visibility", None) is not None and str(it.visibility.value) == "always"

    def _kind_value(it) -> str:
        k = getattr(it, "kind", None)
        return str(k.value) if k is not None else ""

    core_items = [it for it in items if _is_always_visible(it)]
    recent_items = [it for it in items if _kind_value(it) in ("episode", "emotion") and not _is_always_visible(it)]
    long_term_items = [it for it in items if _kind_value(it) not in ("episode", "emotion", "") and not _is_always_visible(it)]
    related_items = [it for it in items if _kind_value(it) == "fact" and not _is_always_visible(it)]

    # 序列化 MemoryItem 为 dict
    def _item_to_dict(it) -> dict:
        if hasattr(it, "to_dict"):
            return it.to_dict()
        return {"content": str(it.content) if hasattr(it, "content") else str(it)}

    result = {
        "query": query,
        "person_id": memory.get("person_id"),
        "guest_mode": memory.get("guest_mode"),
        "memory_miss": memory.get("memory_miss"),
        "read_mode": "unified_store",
        "core_memory": [_item_to_dict(it) for it in core_items],
        "recent_episodes": [_item_to_dict(it) for it in recent_items],
        "long_term_memory": [_item_to_dict(it) for it in long_term_items],
        "evidence": {
            "recent": diag.get("recent") or [],
            "long_term": diag.get("long_term") or [],
            "related": diag.get("related") or [],
        },
        "prompt_summary": {
            "history": len(history),
            "core_memory": len(core_items),
            "recent_episodes": len(recent_items),
            "long_term_memory": len(long_term_items),
            "related": len(related_items),
        },
        "memory_items": store.count_memory_items(person_id=person_id or ""),
        "read_path": diag.get("read_path", "memory_items"),
        "query_supported": diag.get("query_supported", True),
        "top_memory_sources": diag.get("top_memory_sources", []),
        "top_memory_samples": diag.get("top_memory_samples", []),
        "readiness_hint": _compute_readiness_hint(diag),
    }
    # 如果 diagnostics 中有 shadow_compare 差异数据，一并返回
    mi_compare = diag.get("memory_items_compare")
    if mi_compare:
        result["memory_items_compare"] = mi_compare
    return result
