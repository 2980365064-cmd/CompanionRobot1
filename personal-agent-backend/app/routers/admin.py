"""后台管理 API 路由。"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.admin.audit import get_audit_stats, log_audit, query_audit
from app.admin.backup import create_backup, list_backups, restore_backup
from app.admin.config import list_config, test_config, update_config
from app.admin.dashboard import build_dashboard
from app.admin.files import default_file_store
from app.admin.ops import deep_health, debug_recall, deploy_status, request_restart, request_service_action, run_deploy_update
from app.admin.person_detail import build_person_detail
from app.admin.persona_blocks import load_persona_blocks, save_persona_blocks
from app.admin.self_cognition import default_self_cognition_store
from app.admin.tasks import task_manager
from app.config import settings
from app.admin.contacts import (
    build_graph_admin,
    build_memory_by_person_admin,
    create_contact_admin,
    delete_contact_admin,
    get_contact_admin,
    list_contacts_admin,
    update_contact_admin,
)
from app.memory.core_facts import CORE_FACT_CATEGORY_LABELS
from app.admin.memory import (
    create_core_memory_admin,
    create_long_term_memory_admin,
    create_recent_memory_admin,
    delete_core_memory_admin,
    delete_long_term_memory_admin,
    delete_person_memory_admin,
    delete_recent_memory_admin,
    get_profile_admin,
    list_core_memory_admin,
    list_long_term_memory_admin,
    list_recent_memory_admin,
    update_core_memory_admin,
    update_long_term_memory_admin,
    update_profile_admin,
    update_recent_memory_admin,
)
from app.monitor import agent_monitor, sse_get_history
from app.admin.persons import create_person_admin, delete_person_admin, list_persons_admin, update_person_admin
from app.session import store

router = APIRouter()

class PersonAdminUpdate(BaseModel):
    """后台修改用户的请求体。

    字段:
        new_person_id: 新的用户 ID（可选，提供时执行级联重命名）
        display_name:  新的显示名（可选，用于对话中称呼该用户）
    """
    new_person_id: str | None = Field(default=None)
    display_name: str | None = Field(default=None)


class CoreMemoryCreate(BaseModel):
    category: str
    content: str


class CoreMemoryUpdate(BaseModel):
    category: str | None = Field(default=None)
    content: str | None = Field(default=None)


class RecentMemoryCreate(BaseModel):
    kind: str = "episode"
    content: str
    emotional_weight: int = Field(default=3, ge=1, le=5)
    recency_weight: int = Field(default=3, ge=1, le=5)


class RecentMemoryUpdate(BaseModel):
    kind: str | None = Field(default=None)
    content: str | None = Field(default=None)
    emotional_weight: int | None = Field(default=None, ge=1, le=5)
    recency_weight: int | None = Field(default=None, ge=1, le=5)


class LongTermMemoryCreate(BaseModel):
    kind: str = "fact"
    content: str
    confidence: float = Field(default=1.0, ge=0, le=1)


class LongTermMemoryUpdate(BaseModel):
    kind: str | None = Field(default=None)
    content: str | None = Field(default=None)
    confidence: float | None = Field(default=None, ge=0, le=1)


class ProfileAdminUpdate(BaseModel):
    display_name: str | None = Field(default=None)
    aliases: list[str] | None = Field(default=None)
    relationship: str | None = Field(default=None)
    personality: list[str] | None = Field(default=None)
    experiences: list[str] | None = Field(default=None)
    emotional_habit: list[str] | None = Field(default=None)
    confirmed: bool | None = Field(default=None)
    sync_core_facts: bool = Field(default=True)


class ContactAdminCreate(BaseModel):
    owner_person_id: str
    display_name: str
    relationship: str = ""
    aliases: list[str] = Field(default_factory=list)
    personality: list[str] = Field(default_factory=list)
    experiences: list[str] = Field(default_factory=list)
    emotional_habit: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    confirmed: bool = True


class ContactAdminUpdate(BaseModel):
    owner_person_id: str | None = Field(default=None)
    display_name: str | None = Field(default=None)
    relationship: str | None = Field(default=None)
    aliases: list[str] | None = Field(default=None)
    personality: list[str] | None = Field(default=None)
    experiences: list[str] | None = Field(default=None)
    emotional_habit: list[str] | None = Field(default=None)
    notes: list[str] | None = Field(default=None)
    confirmed: bool | None = Field(default=None)


class ConfigAdminUpdate(BaseModel):
    values: dict = Field(default_factory=dict)


class ConfigTestRequest(BaseModel):
    kind: str = "all"


class AdminFileSave(BaseModel):
    content: str = ""


class AdminBackupCreate(BaseModel):
    kind: str = "all"


class RecallDebugRequest(BaseModel):
    query: str
    person_id: str = ""
    device_id: str = "admin-debug"
    session_id: str = "admin-debug"


class SelfCognitionUpdate(BaseModel):
    persona_text: str | None = Field(default=None)
    profile_card_text: str | None = Field(default=None)


class PersonaBlocksUpdate(BaseModel):
    """人格工作台区块更新请求体。"""
    blocks: dict[str, str] = Field(default_factory=dict)
    sync_both: bool = Field(default=True, description="是否同步更新 profile_card.md")


class AuditLogQuery(BaseModel):
    kind: str | None = Field(default=None)
    operator: str | None = Field(default=None)
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    days_back: int = Field(default=7, ge=1, le=365)


# ============================
# 认证
# ============================

def _check_token(authorization: str | None = None, token: str | None = None) -> None:
    """校验 API Token。

    支持两种传入方式：
      - Bearer Token（Authorization header）
      - X-API-Token（自定义 header）

    若 settings.api_token 为空（未配置），则跳过校验（开发环境兼容）。
    校验失败抛出 HTTPException(401)。
    """
    expected = settings.api_token
    if not expected:
        return  # 未配置 token 时放行（开发环境）
    if token == expected:
        return
    if authorization and authorization == f"Bearer {expected}":
        return
    raise HTTPException(status_code=401, detail="invalid token")

@router.get("/v1/admin/persons")
async def admin_list_persons(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """列出已实名用户（后台身份管理 API）。"""
    _check_token(authorization, x_api_token)
    return {"persons": await asyncio.to_thread(list_persons_admin)}


class PersonAdminCreate(BaseModel):
    """后台新建用户的请求体。"""
    person_id: str
    display_name: str
    device_id: str = ""


@router.post("/v1/admin/persons")
async def admin_create_person(
    body: PersonAdminCreate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """后台新建已实名用户。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(
            create_person_admin, body.person_id, body.display_name, body.device_id or ""
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"身份后台新建 · {result['display_name']} id={result['person_id']}")
    log_audit("memory", f"新建用户: {result['display_name']}", detail=f"person_id={result['person_id']}")
    return result


@router.patch("/v1/admin/persons/{person_id}")
async def admin_update_person(
    person_id: str,
    body: PersonAdminUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """修改 person_id 和/或 display_name（会级联更新全部记忆表）。

    至少需要提供 new_person_id 或 display_name 其中之一。
    person_id 重命名会在一个事务内更新所有关联表。
    """
    _check_token(authorization, x_api_token)
    if body.new_person_id is None and body.display_name is None:
        raise HTTPException(status_code=400, detail="new_person_id or display_name required")
    try:
        result = await asyncio.to_thread(
            update_person_admin,
            person_id,
            new_person_id=body.new_person_id,
            display_name=body.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    new_id = result.get("renamed_from") or ""
    new_name = result.get("new_person_id") or result["person_id"]
    agent_monitor.event(f"身份后台更新 · {new_id} → {result['person_id']}")
    log_audit("memory", f"更新用户: {new_id} → {new_name}", detail=f"display_name={result.get('display_name','')}")
    return result


@router.delete("/v1/admin/persons/{person_id}")
async def admin_delete_person(
    person_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除 person_id 及其全部关联记忆（不可逆操作）。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(delete_person_admin, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"身份已删除 · {result['display_name']} id={result['person_id'][:12]}")
    log_audit("memory", f"删除用户: {result['display_name']}", detail=f"person_id={result['person_id'][:12]}")
    return result


@router.get("/v1/admin/persons/{person_id}/core-memories")
async def admin_list_core_memories(
    person_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """列出指定用户的核心记忆。"""
    _check_token(authorization, x_api_token)
    try:
        items = await asyncio.to_thread(list_core_memory_admin, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"person_id": person_id, "items": items, "categories": CORE_FACT_CATEGORY_LABELS}


@router.post("/v1/admin/persons/{person_id}/core-memories")
async def admin_create_core_memory(
    person_id: str,
    body: CoreMemoryCreate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """新增一条核心记忆（管理员写入，跳过自动门控）。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(
            create_core_memory_admin, person_id, body.category, body.content
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"核心记忆后台新增 · {person_id[:12]} [{body.category}]")
    log_audit("memory", f"新增核心记忆: [{body.category}]", detail=f"person_id={person_id[:12]}")
    return result


@router.patch("/v1/admin/persons/{person_id}/core-memories/{memory_item_id}")
async def admin_update_core_memory(
    person_id: str,
    memory_item_id: str,
    body: CoreMemoryUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """修改核心记忆的类别或内容。"""
    _check_token(authorization, x_api_token)
    if body.category is None and body.content is None:
        raise HTTPException(status_code=400, detail="category or content required")
    try:
        result = await asyncio.to_thread(
            update_core_memory_admin,
            person_id,
            memory_item_id,
            category=body.category,
            content=body.content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"核心记忆后台更新 · {person_id[:12]} id={memory_item_id}")
    log_audit("memory", f"更新核心记忆 id={memory_item_id}", detail=f"person_id={person_id[:12]}")
    return result


@router.delete("/v1/admin/persons/{person_id}/core-memories/{memory_item_id}")
async def admin_delete_core_memory(
    person_id: str,
    memory_item_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除一条核心记忆。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(delete_core_memory_admin, person_id, memory_item_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"核心记忆后台删除 · {person_id[:12]} id={memory_item_id}")
    return result


@router.get("/v1/admin/persons/{person_id}/profile")
async def admin_get_profile(
    person_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """获取指定用户的人物画像。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(get_profile_admin, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@router.delete("/v1/admin/persons/{person_id}/episodes")
async def admin_delete_episodes(
    person_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除指定用户的所有情景记忆（不可逆）。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(delete_person_memory_admin, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"情景记忆后台删除 · {person_id[:12]} · {result.get('memory_items_deleted', 0)} 条")
    return result


@router.put("/v1/admin/persons/{person_id}/profile")
async def admin_update_profile(
    person_id: str,
    body: ProfileAdminUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """更新人物画像字段；默认同步身份/关系到 核心事实。"""
    _check_token(authorization, x_api_token)
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="no fields to update")
    try:
        result = await asyncio.to_thread(update_profile_admin, person_id, **fields)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"画像后台更新 · {result.get('display_name')} id={person_id[:12]}")
    log_audit("memory", f"更新画像: {result.get('display_name', person_id)}", detail=f"person_id={person_id[:12]}")
    return result


@router.get("/v1/admin/contacts")
async def admin_list_contacts(
    owner_person_id: str | None = None,
    q: str | None = None,
    confirmed: bool | None = None,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """列出第三方画像。"""
    _check_token(authorization, x_api_token)
    contacts = await asyncio.to_thread(
        list_contacts_admin,
        owner_person_id=owner_person_id,
        q=q,
        confirmed=confirmed,
    )
    return {"contacts": contacts, "count": len(contacts)}


@router.post("/v1/admin/contacts")
async def admin_create_contact(
    body: ContactAdminCreate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """新增第三方画像。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(create_contact_admin, **body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(
        f"第三方画像后台新增 · {result.get('display_name')} owner={result.get('owner_person_id', '')[:12]}"
    )
    return result


@router.get("/v1/admin/contacts/{person_id}")
async def admin_get_contact(
    person_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """获取第三方画像详情。"""
    _check_token(authorization, x_api_token)
    try:
        return await asyncio.to_thread(get_contact_admin, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/v1/admin/contacts/{person_id}")
async def admin_update_contact(
    person_id: str,
    body: ContactAdminUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """更新第三方画像。"""
    _check_token(authorization, x_api_token)
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="no fields to update")
    try:
        result = await asyncio.to_thread(update_contact_admin, person_id, **fields)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"第三方画像后台更新 · {result.get('display_name')} id={person_id[:12]}")
    return result


@router.delete("/v1/admin/contacts/{person_id}")
async def admin_delete_contact(
    person_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除第三方画像。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(delete_contact_admin, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    name = (result.get("contact") or {}).get("display_name") or person_id
    agent_monitor.event(f"第三方画像后台删除 · {name} id={person_id[:12]}")
    return result


# ============================
# 后台管理 API（新）
# ============================

@router.get("/v1/admin/overview")
async def admin_overview(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """后台概览：服务健康、DB 路径、LLM/Embedding 状态、记忆规模、活跃会话。"""
    _check_token(authorization, x_api_token)
    stats = await asyncio.to_thread(store.count_memory_stat)
    active_sessions = await asyncio.to_thread(store.count_sessions, "active")
    closed_sessions = await asyncio.to_thread(store.count_sessions, "closed")
    persons = await asyncio.to_thread(list_persons_admin)
    return {
        "server": {
            "pid": os.getpid(),
            "db_path": str(settings.resolved_db_path()),
            "port": settings.port,
            "host": settings.host,
        },
        "llm": {
            "configured": bool(settings.llm_api_key),
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
        },
        "embed": {
            "configured": bool(settings.embed_api_key),
            "model": settings.embed_model,
        },
        "memory": stats,
        "sessions": {
            "active": active_sessions,
            "closed": closed_sessions,
            "total": active_sessions + closed_sessions,
        },
        "persons": {
            "count": len(persons),
            "list": persons,
        },
    }


@router.get("/v1/admin/memory/by-person")
async def admin_memory_by_person(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """按实名用户分组返回记忆概览和首屏数据。"""
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(build_memory_by_person_admin)


@router.get("/v1/admin/graph")
async def admin_graph(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """返回关系图谱节点与边。"""
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(build_graph_admin)


@router.get("/v1/admin/sessions")
async def admin_list_sessions(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """列出所有会话。"""
    _check_token(authorization, x_api_token)
    sessions = await asyncio.to_thread(store.list_all_sessions, limit, offset, status)
    return {"sessions": sessions}


@router.get("/v1/admin/sessions/{session_id}/messages")
async def admin_session_messages(
    session_id: str,
    limit: int = 200,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """获取会话的消息记录。"""
    _check_token(authorization, x_api_token)
    session = await asyncio.to_thread(store.get_session_by_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    msgs = await asyncio.to_thread(store.get_session_messages, session_id)
    if limit and limit < len(msgs):
        msgs = msgs[-limit:]
    return {"session": session, "messages": msgs, "count": len(msgs)}


# ── 情景记忆管理 ──

@router.get("/v1/admin/persons/{person_id}/recent-memories")
async def admin_list_recent_memories(
    person_id: str,
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """列出指定用户的情景记忆。"""
    _check_token(authorization, x_api_token)
    try:
        items = await asyncio.to_thread(list_recent_memory_admin, person_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"person_id": person_id, "items": items}


@router.post("/v1/admin/persons/{person_id}/recent-memories")
async def admin_create_recent_memory(
    person_id: str,
    body: RecentMemoryCreate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """新增情景记忆。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(
            create_recent_memory_admin,
            person_id,
            content=body.content,
            kind=body.kind,
            emotional_weight=body.emotional_weight,
            recency_weight=body.recency_weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"情景记忆后台新增 · {person_id[:12]} [{result.get('kind', '')}]")
    log_audit("memory", f"新增情景记忆: [{result.get('kind', '')}]", detail=f"person_id={person_id[:12]}")
    return result


@router.patch("/v1/admin/persons/{person_id}/recent-memories/{memory_item_id}")
async def admin_update_recent_memory(
    person_id: str,
    memory_item_id: str,
    body: RecentMemoryUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """修改情景记忆。"""
    _check_token(authorization, x_api_token)
    if not body.model_dump(exclude_unset=True):
        raise HTTPException(status_code=400, detail="no fields to update")
    try:
        result = await asyncio.to_thread(
            update_recent_memory_admin,
            person_id,
            memory_item_id,
            **body.model_dump(exclude_unset=True),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"情景记忆后台更新 · {person_id[:12]} id={memory_item_id}")
    return result


@router.delete("/v1/admin/persons/{person_id}/recent-memories/{memory_item_id}")
async def admin_delete_recent_memory(
    person_id: str,
    memory_item_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除单条情景记忆。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(delete_recent_memory_admin, person_id, memory_item_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    agent_monitor.event(f"情景记忆后台删除 · {person_id[:12]} id={memory_item_id}")
    return result


@router.get("/v1/admin/persons/{person_id}/episodes/{episode_id}")
async def admin_get_episode(
    person_id: str,
    episode_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """获取单条情景记忆（从统一记忆库读取）。"""
    _check_token(authorization, x_api_token)
    items = await asyncio.to_thread(list_recent_memory_admin, person_id, limit=1000)
    item = next((x for x in items if str(x.get("id")) == str(episode_id)), None)
    if not item:
        raise HTTPException(status_code=404, detail="episode not found")
    return item


@router.patch("/v1/admin/persons/{person_id}/episodes/{episode_id}")
async def admin_update_episode(
    person_id: str,
    episode_id: str,
    body: dict,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """更新情景摘要（通过统一记忆库重写）。"""
    _check_token(authorization, x_api_token)
    try:
        return await asyncio.to_thread(
            update_recent_memory_admin,
            person_id,
            episode_id,
            content=body.get("content") or body.get("summary"),
            kind=body.get("kind"),
            emotional_weight=body.get("emotional_weight"),
            recency_weight=body.get("recency_weight"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/v1/admin/persons/{person_id}/episodes/{episode_id}")
async def admin_delete_episode(
    person_id: str,
    episode_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除单条情景记忆（从统一记忆库删除）。"""
    _check_token(authorization, x_api_token)
    try:
        return await asyncio.to_thread(delete_recent_memory_admin, person_id, episode_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── 长期记忆管理（基于统一记忆库） ──

@router.get("/v1/admin/persons/{person_id}/long-term-memories")
async def admin_list_person_long_term_memories(
    person_id: str,
    limit: int = 100,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """列出指定用户的长期记忆。"""
    _check_token(authorization, x_api_token)
    try:
        items = await asyncio.to_thread(list_long_term_memory_admin, person_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"person_id": person_id, "items": items}


@router.post("/v1/admin/persons/{person_id}/long-term-memories")
async def admin_create_person_long_term_memory(
    person_id: str,
    body: LongTermMemoryCreate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """新增长期记忆。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(
            create_long_term_memory_admin,
            person_id,
            content=body.content,
            kind=body.kind,
            confidence=body.confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"长期记忆后台新增 · {person_id[:12]} [{result.get('kind', '')}]")
    log_audit("memory", f"新增长期记忆: [{result.get('kind', '')}]", detail=f"person_id={person_id[:12]}")
    return result


@router.patch("/v1/admin/persons/{person_id}/long-term-memories/{memory_item_id}")
async def admin_update_person_long_term_memory(
    person_id: str,
    memory_item_id: str,
    body: LongTermMemoryUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """修改长期记忆。"""
    _check_token(authorization, x_api_token)
    if not body.model_dump(exclude_unset=True):
        raise HTTPException(status_code=400, detail="no fields to update")
    try:
        result = await asyncio.to_thread(
            update_long_term_memory_admin,
            person_id,
            memory_item_id,
            **body.model_dump(exclude_unset=True),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    agent_monitor.event(f"长期记忆后台更新 · {person_id[:12]} id={memory_item_id}")
    return result


@router.delete("/v1/admin/persons/{person_id}/long-term-memories/{memory_item_id}")
async def admin_delete_person_long_term_memory(
    person_id: str,
    memory_item_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除单条长期记忆。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(delete_long_term_memory_admin, person_id, memory_item_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    agent_monitor.event(f"长期记忆后台删除 · {person_id[:12]} id={memory_item_id}")
    return result

@router.get("/v1/admin/long-term-memory")
async def admin_list_long_term_memory(
    collection: str | None = None,
    person_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """列出长期记忆（从统一记忆库按 person_id 筛选）。"""
    _check_token(authorization, x_api_token)
    if not person_id:
        return {"items": [], "count": 0, "note": "请指定 person_id 筛选用户记忆"}
    kinds = ["fact", "entity", "wiki", "relationship", "episode"]
    items = await asyncio.to_thread(
        store.search_memory_items,
        person_id, kinds=kinds, visibility="recall_only",
        limit=limit,
    )
    return {"items": items[offset:offset + limit] if offset else items, "count": len(items)}


@router.get("/v1/admin/long-term-memory/{chunk_id}")
async def admin_get_long_term_chunk(
    chunk_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """获取单条长期记忆（从统一记忆库读取）。"""
    _check_token(authorization, x_api_token)
    item = await asyncio.to_thread(store.get_memory_item, chunk_id)
    if item:
        return item
    raise HTTPException(status_code=404, detail="memory item not found")


@router.patch("/v1/admin/long-term-memory/{chunk_id}")
async def admin_update_long_term_chunk(
    chunk_id: str,
    body: dict,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """更新长期记忆（通过统一记忆库重写）。"""
    _check_token(authorization, x_api_token)
    item = await asyncio.to_thread(store.get_memory_item, chunk_id)
    if not item:
        raise HTTPException(status_code=404, detail="memory item not found")
    text = body.get("text") or item.get("content", "")
    kind = body.get("category") or item.get("kind", "fact")
    new_id = await asyncio.to_thread(
        store.write_memory_item,
        person_id=item.get("person_id", ""),
        device_id=item.get("device_id", ""),
        kind=kind,
        source=item.get("source", "admin"),
        visibility=item.get("visibility", "recall_only"),
        content=text,
    )
    await asyncio.to_thread(store.delete_memory_item, chunk_id)
    updated = await asyncio.to_thread(store.get_memory_item, new_id)
    return updated


@router.delete("/v1/admin/long-term-memory/{chunk_id}")
async def admin_delete_long_term_chunk(
    chunk_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除长期记忆（从统一记忆库删除）。"""
    _check_token(authorization, x_api_token)
    ok = await asyncio.to_thread(store.delete_memory_item, chunk_id)
    if not ok:
        raise HTTPException(status_code=404, detail="memory item not found")
    agent_monitor.event(f"记忆后台删除 · id={chunk_id[:16]}")
    return {"deleted": True, "chunk_id": chunk_id}


# ── SSE 控制台日志推送 ──

@router.get("/v1/admin/logs/stream")
async def admin_logs_stream(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """SSE 实时推送控制台日志。"""
    _check_token(authorization, x_api_token)

    async def event_generator():
        last_index = 0
        while True:
            lines, last_index = await asyncio.to_thread(sse_get_history, last_index)
            for line in lines:
                yield f"data: {line}\n\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 运维控制台：配置、文件、健康、备份、部署、调试 ──

@router.get("/v1/admin/config")
async def admin_get_config(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(list_config)


@router.patch("/v1/admin/config")
async def admin_update_config(
    body: ConfigAdminUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    result = await asyncio.to_thread(update_config, body.values)
    updated_keys = result.get("updated") or []
    agent_monitor.event(f"配置后台更新 · {', '.join(updated_keys)}")
    log_audit("config", f"修改配置: {', '.join(updated_keys)}", operator="admin")
    return result


@router.post("/v1/admin/config/test")
async def admin_test_config(
    body: ConfigTestRequest,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(test_config, body.kind)


@router.get("/v1/admin/files")
async def admin_list_files(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return {"files": await asyncio.to_thread(default_file_store().list_files)}


@router.get("/v1/admin/files/{file_path:path}")
async def admin_read_file(
    file_path: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    try:
        return await asyncio.to_thread(default_file_store().read, file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file not found")


@router.put("/v1/admin/files/{file_path:path}")
async def admin_save_file(
    file_path: str,
    body: AdminFileSave,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(default_file_store().write, file_path, body.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    agent_monitor.event(f"文件后台保存 · {file_path}")
    return result


@router.post("/v1/admin/files/{file_path:path}")
async def admin_create_file(
    file_path: str,
    body: AdminFileSave,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    return await admin_save_file(file_path, body, authorization, x_api_token)


@router.delete("/v1/admin/files/{file_path:path}")
async def admin_delete_file(
    file_path: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(default_file_store().delete, file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="file not found")
    agent_monitor.event(f"文件后台删除 · {file_path}")
    return result


@router.get("/v1/admin/agent/self")
async def admin_get_agent_self(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """读取智能体自我认知配置。"""
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(default_self_cognition_store().load)


@router.put("/v1/admin/agent/self")
async def admin_update_agent_self(
    body: SelfCognitionUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """修改智能体自我认知配置。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(
            default_self_cognition_store().save,
            persona_text=body.persona_text,
            profile_card_text=body.profile_card_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event("智能体自我认知已更新")
    log_audit("persona", "更新自我认知文件", detail="persona.md / profile_card.md")
    return result


@router.get("/v1/admin/health/deep")
async def admin_deep_health(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(deep_health)


@router.get("/v1/admin/backups")
async def admin_list_backups(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(list_backups)


@router.post("/v1/admin/backups")
async def admin_create_backup(
    body: AdminBackupCreate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    result = await asyncio.to_thread(create_backup, body.kind)
    agent_monitor.event(f"备份创建 · {body.kind}")
    return result


@router.post("/v1/admin/backups/{backup_id}/restore")
async def admin_restore_backup(
    backup_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(restore_backup, backup_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="backup not found")
    agent_monitor.warn(f"备份恢复 · {backup_id}")
    log_audit("backup", f"恢复备份: {backup_id}", detail="备份恢复操作，涉及数据库替换")
    return result


@router.get("/v1/admin/deploy/status")
async def admin_deploy_status(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(deploy_status)


@router.post("/v1/admin/deploy/update")
async def admin_deploy_update(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    agent_monitor.event("部署更新启动")
    result = await run_deploy_update()
    log_audit("deploy", "部署更新", detail=f"需要重启: {result.get('needs_restart')}")
    return result


@router.post("/v1/admin/service/restart")
async def admin_service_restart(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    agent_monitor.warn("收到服务重启请求")
    result = await request_restart()
    if result.get("accepted"):
        log_audit("service", "服务重启", detail=f"PID {result.get('pid_before')} → 重启中", metadata=result)
    return result


@router.post("/v1/admin/service/start")
async def admin_service_start(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    agent_monitor.warn("收到服务启动请求")
    result = await request_service_action("start")
    if result.get("accepted"):
        log_audit("service", "服务启动", metadata=result)
    return result


@router.post("/v1/admin/service/stop")
async def admin_service_stop(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    agent_monitor.warn("收到服务停止请求")
    result = await request_service_action("stop")
    if result.get("accepted"):
        log_audit("service", "服务停止", metadata=result)
    return result


# ── 运维仪表盘 ──

@router.get("/v1/admin/dashboard")
async def admin_dashboard(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """运维仪表盘：服务健康、风险清单、推荐动作、最近任务与错误。"""
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(build_dashboard)


# ── 人物详情页 ──

@router.get("/v1/admin/persons/{person_id}/detail")
async def admin_person_detail(
    person_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """聚合单个实名用户的全部信息：画像、核心事实/近期记忆/长期记忆、第三方画像、会话摘要。"""
    _check_token(authorization, x_api_token)
    try:
        return await asyncio.to_thread(build_person_detail, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── 智能体人格工作台 ──

@router.get("/v1/admin/agent/persona-blocks")
async def admin_get_persona_blocks(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """读取结构化人格区块：将 persona.md / profile_card.md 按标题拆分。"""
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(load_persona_blocks)


@router.patch("/v1/admin/agent/persona-blocks")
async def admin_update_persona_blocks(
    body: PersonaBlocksUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """保存指定人格区块内容到底层 Markdown 文件。"""
    _check_token(authorization, x_api_token)
    if not body.blocks:
        raise HTTPException(status_code=400, detail="no blocks to update")
    try:
        result = await asyncio.to_thread(
            save_persona_blocks, body.blocks, body.sync_both
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    log_audit("persona", "保存人格区块", f"区块: {', '.join(body.blocks.keys())}")
    return result


# ── 变更审计日志 ──

@router.get("/v1/admin/audit-log")
async def admin_get_audit_log(
    kind: str | None = None,
    operator: str | None = None,
    limit: int = 100,
    offset: int = 0,
    days_back: int = 7,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """查询后台操作审计日志，按时间倒序返回。"""
    _check_token(authorization, x_api_token)
    return {
        "entries": await asyncio.to_thread(
            query_audit, kind=kind, operator=operator,
            limit=limit, offset=offset, days_back=days_back
        ),
        "stats": await asyncio.to_thread(get_audit_stats, days_back),
    }


@router.post("/v1/admin/debug/recall")
async def admin_debug_recall(
    body: RecallDebugRequest,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(
        debug_recall, body.device_id, body.session_id, body.person_id, body.query
    )


# ── 任务中心 ──

async def _run_task_ingest():
    """执行语料入库。"""
    from app.persona.ingest import startup_ingest_corpus
    try:
        result = await asyncio.to_thread(startup_ingest_corpus)
        files = result.get("files") or []
        agent_monitor.event(f"任务完成 · 语料入库 · {len(files)} 文件")
        return {
            "status": "done", "files": len(files),
            "summary": f"语料入库完成：{len(files)} 个文件同步到 长期记忆",
            "next_actions": ["检查 /admin#memory 确认 chunk 数"],
        }
    except Exception as exc:
        agent_monitor.warn(f"语料入库失败: {exc}")
        raise


async def _run_task_ingest_reset():
    """全量重建 persona/corpus 语料索引。"""
    import subprocess
    proc = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "scripts/ingest.py", "--reset"],
        cwd=str(Path(__file__).resolve().parent.parent),
        text=True,
        capture_output=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ingest reset failed")
    agent_monitor.event("任务完成 · 语料全量重建")
    return {
        "status": "done", "stdout": proc.stdout[-4000:],
        "summary": "语料全量重建完成",
        "next_actions": ["重启服务使新语料生效"],
    }


async def _run_task_compress_profile():
    """生成 Profile Card。"""
    import subprocess
    proc = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "scripts/compress_profile.py"],
        cwd=str(Path(__file__).resolve().parent.parent),
        text=True,
        capture_output=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "compress_profile failed")
    agent_monitor.event("任务完成 · Profile Card 生成")
    return {
        "status": "done", "stdout": proc.stdout[-4000:],
        "summary": "Profile Card 已重新生成",
        "next_actions": ["确认 /admin#persona 中 profile_card 内容"],
    }


async def _run_task_cleanup_contacts():
    """清理重复第三方画像。"""
    import subprocess
    proc = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "scripts/cleanup_duplicate_contacts.py", "--apply"],
        cwd=str(Path(__file__).resolve().parent.parent),
        text=True,
        capture_output=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "cleanup contacts failed")
    agent_monitor.event("任务完成 · 第三方画像去重")
    return {
        "status": "done", "stdout": proc.stdout[-4000:],
        "summary": "第三方画像去重完成",
        "next_actions": ["检查 /admin#contacts 确认去重结果"],
    }


async def _run_task_diagnose():
    """执行记忆诊断。"""
    try:
        stats = await asyncio.to_thread(store.count_memory_stat)
        msg = (
            f"核心记忆={stats['core_memories']} 情景摘要(活跃)={stats['episodes_active']} "
            f"情景摘要(总计)={stats['episodes_total']} "
            f"长期记忆={stats['long_term_memory']} 画像={stats['profiles']}"
        )
        agent_monitor.event(f"任务完成 · 记忆诊断 · {msg}")
        return {
            "status": "done", "stats": stats,
            "summary": msg,
        }
    except Exception as exc:
        agent_monitor.warn(f"记忆诊断失败: {exc}")
        raise


async def _run_task_rollup():
    """执行情景摘要归档为长期记忆。"""
    from app.memory.memory_pipeline import archive_expired_recent_memory
    try:
        n = await asyncio.to_thread(archive_expired_recent_memory, None)
        agent_monitor.event(f"任务完成 · 情景摘要归档 · {n} 条")
        return {
            "status": "done", "archived": n,
            "summary": f"情景摘要归档：{n} 条到期摘要已归档到 长期记忆",
        }
    except Exception as exc:
        agent_monitor.warn(f"情景摘要归档失败: {exc}")
        raise


async def _run_task_update_profiles():
    """执行画像履历更新。"""
    from app.memory.profile import update_all_profiles
    try:
        result = await asyncio.to_thread(update_all_profiles)
        updated = int(result.get("updated") or 0)
        total = int(result.get("total") or 0)
        agent_monitor.event(f"任务完成 · 画像更新 · {updated}/{total} 人")
        return {
            "status": "done", "updated": updated, "total": total,
            "summary": f"人物画像批量更新：{updated}/{total} 人已更新",
        }
    except Exception as exc:
        agent_monitor.warn(f"画像更新失败: {exc}")
        raise


TASK_HANDLERS = {
    "ingest": ("语料同步入库", _run_task_ingest),
    "ingest-reset": ("语料全量重建", _run_task_ingest_reset),
    "compress-profile": ("生成 Profile Card", _run_task_compress_profile),
    "diagnose-memory": ("记忆健康诊断", _run_task_diagnose),
    "rollup-recent": ("情景摘要归档", _run_task_rollup),
    "update-profiles": ("人物画像批量更新", _run_task_update_profiles),
    "cleanup-contacts": ("第三方画像去重", _run_task_cleanup_contacts),
}


@router.get("/v1/admin/tasks")
async def admin_list_tasks(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return {
        "available": [
            {"name": name, "title": title}
            for name, (title, _handler) in TASK_HANDLERS.items()
        ],
        "tasks": task_manager.list(),
    }


@router.get("/v1/admin/tasks/{task_id}")
async def admin_get_task(
    task_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    task = task_manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@router.post("/v1/admin/tasks/{task_name}")
async def admin_run_task(
    task_name: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """执行后台管理任务。"""
    _check_token(authorization, x_api_token)
    item = TASK_HANDLERS.get(task_name)
    if not item:
        raise HTTPException(status_code=404, detail=f"unknown task: {task_name}")
    title, handler = item
    try:
        result = task_manager.start(task_name, title, handler)
        log_audit("task", f"启动任务: {title}", detail=f"task_name={task_name}")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
