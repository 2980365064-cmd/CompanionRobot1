"""后台任务中心：统一执行、记录和并发保护。"""

from __future__ import annotations

import asyncio
import contextlib
import io
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable
from uuid import uuid4

from app.monitor import agent_monitor

TaskHandler = Callable[[], Awaitable[dict[str, Any]]]


@dataclass
class AdminTask:
    id: str
    name: str
    title: str
    status: str = "queued"
    started_at: str = ""
    finished_at: str = ""
    result: dict[str, Any] | None = None
    error: str = ""
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "log": self.log[-500:],
        }


class AdminTaskManager:
    def __init__(self) -> None:
        self._tasks: dict[str, AdminTask] = {}
        self._order: list[str] = []
        self._running_names: set[str] = set()

    def start(self, name: str, title: str, handler: TaskHandler) -> dict:
        if name in self._running_names:
            raise ValueError(f"task already running: {name}")
        task = AdminTask(id=uuid4().hex[:12], name=name, title=title)
        self._tasks[task.id] = task
        self._order.insert(0, task.id)
        self._running_names.add(name)
        asyncio.create_task(self._run(task, handler))
        return task.to_dict()

    async def _run(self, task: AdminTask, handler: TaskHandler) -> None:
        task.status = "running"
        task.started_at = datetime.now().isoformat(timespec="seconds")
        agent_monitor.event(f"任务启动 · {task.title}")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                task.result = await handler()
            task.status = "success"
            agent_monitor.event(f"任务完成 · {task.title}")
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            task.log.append(traceback.format_exc())
            agent_monitor.warn(f"任务失败 · {task.title}: {exc}")
        finally:
            captured = buf.getvalue().strip()
            if captured:
                task.log.extend(captured.splitlines())
            task.finished_at = datetime.now().isoformat(timespec="seconds")
            self._running_names.discard(task.name)

    def list(self) -> list[dict]:
        return [self._tasks[tid].to_dict() for tid in self._order[:80]]

    def get(self, task_id: str) -> dict | None:
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None


task_manager = AdminTaskManager()
