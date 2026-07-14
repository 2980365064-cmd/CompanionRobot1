"""文本对话 WebSocket 路由。"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, WebSocket
from pydantic import BaseModel, Field

from app.agent import handle_chat
from app.config import settings
from app.ws_handler import ws_chat_endpoint

router = APIRouter()


class ChatRequest(BaseModel):
    device_id: str = Field(default="http-client")
    session_id: str = Field(default="")
    message: str


def _check_token(authorization: str | None = None, token: str | None = None) -> None:
    expected = settings.api_token
    if not expected:
        return
    if token == expected:
        return
    if authorization and authorization == f"Bearer {expected}":
        return
    raise HTTPException(status_code=401, detail="invalid token")


@router.post("/v1/chat")
async def http_chat(
    body: ChatRequest,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """HTTP 单轮对话入口，复用文本 WebSocket 的核心对话链路。"""
    _check_token(authorization, x_api_token)
    reply, session_id, active_topic = await handle_chat(
        body.device_id, body.session_id, body.message
    )
    return {"reply": reply, "session_id": session_id, "active_topic": active_topic}


@router.websocket("/ws/v1/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket 对话入口，协议详情见 ws_handler.py。"""
    await ws_chat_endpoint(websocket)
