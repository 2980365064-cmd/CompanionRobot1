"""内置静态页面路由。"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


@router.get("/")
async def index():
    """根路径重定向到后台管理页面。"""
    return RedirectResponse(url="/admin")


@router.get("/chat")
async def chat_ui():
    """聊天测试页面。"""
    chat = STATIC_DIR / "chat.html"
    if not chat.is_file():
        raise HTTPException(status_code=404, detail="static/chat.html not found")
    return FileResponse(chat, headers={"Cache-Control": "no-store"})


@router.get("/admin")
async def admin_ui():
    """后台管理页面。"""
    admin = STATIC_DIR / "admin.html"
    if not admin.is_file():
        raise HTTPException(status_code=404, detail="static/admin.html not found")
    return FileResponse(admin, headers={"Cache-Control": "no-store"})
