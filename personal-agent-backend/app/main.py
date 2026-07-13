"""FastAPI 应用组合根。"""

from __future__ import annotations

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# 支持直接在 IDE 中右键运行 main.py（不需要从项目根目录启动 uvicorn）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.lifecycle import lifespan as backend_lifespan
from app.monitor import agent_monitor
from app.routers import admin, audio_ws, chat_ws, health, pages

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


agent_monitor.configure()


@asynccontextmanager
async def lifespan(service: FastAPI):
    """委托到 lifecycle 模块，保持 main.py 只负责应用组合。"""
    async with backend_lifespan(service):
        yield


def create_app() -> FastAPI:
    """创建 FastAPI 应用并挂载路由。"""
    service = FastAPI(title="SparkBot Personal Agent", lifespan=lifespan)
    if STATIC_DIR.is_dir():
        service.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    service.include_router(pages.router)
    service.include_router(health.router)
    service.include_router(admin.router)
    service.include_router(chat_ws.router)
    service.include_router(audio_ws.router)
    return service


app = create_app()


if __name__ == "__main__":
    import socket

    import uvicorn

    from app.log_config import LOG_CONFIG

    def _ensure_port_free(host: str, port: int) -> None:
        """Bind probe: fail fast with actionable message when port is taken."""
        probe_host = host if host not in ("", "0.0.0.0") else "0.0.0.0"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((probe_host, port))
            except OSError as exc:
                in_use = getattr(exc, "winerror", None) == 10048 or exc.errno in (98, 10048)
                if not in_use:
                    raise
                print(
                    f"\n[错误] 端口 {port} 已被占用，后端未启动（因此不会有对话/记忆日志）。\n"
                    f"  1. 查占用: netstat -ano | findstr :{port}\n"
                    f"  2. 结束旧进程: taskkill /PID <PID> /F\n"
                    f"  3. 再运行: py -3 app/main.py\n",
                    flush=True,
                )
                raise SystemExit(1) from exc

    _ensure_port_free(settings.host, settings.port)
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_config=LOG_CONFIG,
        access_log=False,
    )
