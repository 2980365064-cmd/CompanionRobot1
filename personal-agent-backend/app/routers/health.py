"""健康检查路由。"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.llm import embed_provider_name

router = APIRouter()


@router.get("/health")
async def health():
    """返回 LLM、向量和记忆配置摘要。"""
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
            "working_context_turns": settings.working_context_turns,
            "recent_memory_retention_days": settings.recent_memory_retention_days,
            "search_backend": settings.search_backend,
        },
    }
