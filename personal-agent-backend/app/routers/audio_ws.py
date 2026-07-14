"""音频对话 WebSocket 路由。"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

from app.services.speech_gateway import ws_audio_endpoint

router = APIRouter()


@router.websocket("/ws/v2/audio")
async def websocket_audio(websocket: WebSocket):
    """WebSocket 音频对话入口，v2 协议。"""
    await ws_audio_endpoint(websocket)
