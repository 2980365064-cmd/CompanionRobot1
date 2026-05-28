"""FastAPI application entry."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import handle_chat
from app.config import settings
from app.llm import embed_provider_name
from app.memory.extractor import rollup_expired_l2
from app.memory.semantic import semantic_memory
from app.monitor import agent_monitor
from app.rag import startup_ingest_corpus
from app.ws_handler import idle_session_sweeper, ws_chat_endpoint

agent_monitor.configure()
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class ChatRequest(BaseModel):
    device_id: str = Field(default="default")
    session_id: str = Field(default="")
    message: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str


def _check_token(authorization: str | None = None, token: str | None = None) -> None:
    expected = settings.api_token
    if not expected:
        return
    if token == expected:
        return
    if authorization and authorization == f"Bearer {expected}":
        return
    raise HTTPException(status_code=401, detail="invalid token")


async def l2_rollup_sweeper() -> None:
    """Periodically move expired L2 summaries into L3 (corpus + facts)."""
    while True:
        await asyncio.sleep(3600)
        try:
            n = await asyncio.to_thread(rollup_expired_l2, None)
            if n:
                agent_monitor.event(f"L2→L3 归档 {n} 条")
        except Exception as exc:
            agent_monitor.warn(f"L2 归档失败: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_monitor.configure()
    idle_task = asyncio.create_task(idle_session_sweeper())
    rollup_task = asyncio.create_task(l2_rollup_sweeper())
    llm_ok = "DeepSeek 已就绪" if settings.llm_api_key else "未配置 LLM_API_KEY"
    embed_ok = "向量 API 已就绪" if settings.embed_api_key else "未配置 EMBED_API_KEY（使用本地向量）"
    agent_monitor.banner("服务启动")
    agent_monitor.startup(f"{llm_ok} | {embed_ok}")
    agent_monitor.startup(f"http://127.0.0.1:{settings.port}/  WebSocket /ws/v1/chat")
    try:
        files = await asyncio.to_thread(startup_ingest_corpus)
        from app.embed_meta import check_embed_compat

        ok, _ = check_embed_compat()
        n = semantic_memory.corpus.count()
        if files:
            agent_monitor.startup(f"记忆入库 {n} 块 ← {len(files)} 个文件")
        elif not ok:
            agent_monitor.warn("向量维度不一致，请检查 .env 后重启")
        else:
            agent_monitor.warn(f"语料为空，请在 {settings.resolved_corpus_dir()} 添加 .md")
    except Exception as exc:
        agent_monitor.warn(f"记忆入库失败: {exc}")
    yield
    idle_task.cancel()
    rollup_task.cancel()


app = FastAPI(title="SparkBot Personal Agent", lifespan=lifespan)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def chat_ui():
    """Web chat UI for backend testing."""
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(index)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm": {
            "provider": "deepseek",
            "configured": bool(settings.llm_api_key),
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
        },
        "embed": {
            "provider": embed_provider_name(),
            "configured": bool(settings.embed_api_key),
            "model": settings.embed_model,
            "base_url": settings.embed_base_url,
        },
        "memory": {
            "l1_turns": settings.working_memory_turns,
            "l2_retention_days": settings.l2_retention_days,
            "l3_recall_mode": settings.l3_recall_mode,
        },
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def http_chat(
    body: ChatRequest,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """HTTP chat API — same logic as WebSocket, handy for testing without firmware."""
    _check_token(authorization, x_api_token)
    try:
        reply, session_id = await handle_chat(body.device_id, body.session_id, body.message)
    except Exception as e:
        agent_monitor.warn(f"对话失败: {e}")
        if "dimension" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="向量维度不匹配：请停止 uvicorn 后重新启动，确保 .env 已配置 EMBED_API_KEY，再执行 python scripts/ingest.py --reset",
            ) from e
        raise HTTPException(status_code=500, detail=f"对话失败：{str(e)[:120]}") from e
    return ChatResponse(reply=reply, session_id=session_id)


@app.websocket("/ws/v1/chat")
async def websocket_chat(websocket: WebSocket):
    await ws_chat_endpoint(websocket)


if __name__ == "__main__":
    import uvicorn

    from app.log_config import LOG_CONFIG

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_config=LOG_CONFIG,
        access_log=False,
    )
