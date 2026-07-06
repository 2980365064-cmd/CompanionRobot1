"""FastAPI 入口 —— WebSocket 对话 + 后台定时任务 + 应用生命周期管理。

本模块是陪伴机器人的"总控中心"，负责：

一、HTTP 接口（管理用）
  GET  /                         返回 static/index.html 测试页面
  GET  /health                   健康检查（LLM/向量/记忆配置摘要）
  GET  /v1/admin/persons        列出已实名用户
  PATCH /v1/admin/persons/{pid} 修改用户 person_id / display_name
  DELETE /v1/admin/persons/{pid} 删除用户及其全部记忆
  GET/POST/PATCH/DELETE /v1/admin/persons/{pid}/core-memories[/{id}]  核心记忆管理
  GET/PUT /v1/admin/persons/{pid}/profile                             人物画像管理

二、WebSocket 接口（对话）
  /ws/v1/chat                    长连接对话（协议见 ws_handler.py）

三、后台定时任务（在 lifespan 中启动）
  idle_session_sweeper     每分钟扫描：超时 WebSocket 会话自动 session_end
  l2_rollup_sweeper        每小时执行：过期情景摘要归档到长期记忆
  profile_batch_sweeper    每 N 小时执行：画像性格/履历批量归档

四、启动初始化
  startup_ingest_corpus()   同步 persona/corpus/ 到长期记忆
  在后台异步执行，不阻塞 WebSocket 接受连接
"""

from __future__ import annotations

import sys
from pathlib import Path

# 支持直接在 IDE 中右键运行 main.py（不需要从项目根目录启动 uvicorn）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import logging
import os
import sqlite3
from datetime import datetime, timedelta

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, WebSocket
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.admin_backup import create_backup, list_backups, restore_backup
from app.admin_config import list_config, test_config, update_config
from app.admin_files import default_file_store
from app.admin_ops import deep_health, debug_recall, deploy_status, request_restart, request_service_action, run_deploy_update
from app.admin_tasks import task_manager
from app.config import settings
from app.person_admin import create_person_admin, delete_person_admin, list_persons_admin, update_person_admin
from app.contact_admin import (
    build_graph_admin,
    build_memory_by_person_admin,
    create_contact_admin,
    delete_contact_admin,
    get_contact_admin,
    list_contacts_admin,
    update_contact_admin,
)
from app.memory_admin import (
    create_l0_admin,
    delete_l0_admin,
    delete_l2_admin,
    get_profile_admin,
    list_l0_admin,
    update_l0_admin,
    update_profile_admin,
)
from app.memory.l0 import L0_CATEGORY_LABELS
from app.llm import embed_provider_name, warmup_llm_client
from app.memory.extractor import rollup_expired_l2
from app.memory.l3 import semantic_memory
from app.memory.profile import update_all_profiles
from app.memory.interlocutor import get_default_owner_person_id
from app.monitor import agent_monitor, sse_get_history
from app.persona.ingest import startup_ingest_corpus
from app.ws_handler import idle_session_sweeper, ws_chat_endpoint
from app.speech_gateway import ws_audio_endpoint
from app.session import store

# 初始化控制台监控（静音第三方日志，设置 agent 输出通道）
agent_monitor.configure()
logger = logging.getLogger(__name__)

# 前端静态文件目录（index.html 测试页面）
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# ============================
# 请求/响应模型定义
# ============================

class PersonAdminUpdate(BaseModel):
    """后台修改用户的请求体。

    字段:
        new_person_id: 新的用户 ID（可选，提供时执行级联重命名）
        display_name:  新的显示名（可选，用于对话中称呼该用户）
    """
    new_person_id: str | None = Field(default=None)
    display_name: str | None = Field(default=None)


class L0AdminCreate(BaseModel):
    category: str
    content: str


class L0AdminUpdate(BaseModel):
    category: str | None = Field(default=None)
    content: str | None = Field(default=None)


class ProfileAdminUpdate(BaseModel):
    display_name: str | None = Field(default=None)
    aliases: list[str] | None = Field(default=None)
    relationship: str | None = Field(default=None)
    personality: list[str] | None = Field(default=None)
    experiences: list[str] | None = Field(default=None)
    emotional_habit: list[str] | None = Field(default=None)
    confirmed: bool | None = Field(default=None)
    sync_l0: bool = Field(default=True)


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


# ============================
# 后台定时任务
# ============================

async def l2_rollup_sweeper() -> None:
    """情景摘要归档任务：每小时将过期的情景摘要写入长期记忆。

    工作原理：
      调用 rollup_expired_l2 扫描 episodic_memories 表中 archived=0
      且 expires_at 已过期的记录，将其内容写入长期记忆，
      然后标记 archived=1。

    为什么每小时一次：
      情景摘要的默认保留期是 14 天，每小时检查一次足够及时，
      同时不会给系统带来明显负载。
    """
    while True:
        await asyncio.sleep(3600)  # 每小时执行一次
        try:
            n = await asyncio.to_thread(rollup_expired_l2, None)
            if n:
                agent_monitor.event(f"情景摘要归档 {n} 条")
        except Exception as exc:
            agent_monitor.warn(f"情景摘要归档失败: {exc}")


async def profile_batch_sweeper() -> None:
    """画像批量归档任务：每 N 小时从情景摘要和长期记忆更新用户的性格/履历档案。

    工作原理：
      调用 update_all_profiles() 扫描所有已实名用户的画像，
      拉取最近 profile_batch_lookback_hours 内的情景摘要和长期记忆，
      通过 LLM 提取性格描述、生活事件、关系变化等并写入 Profile Card。

    为什么需要定期归档：
      - 性格和履历是累积式的，不会每轮都提取（成本高）
      - 定时批量处理更高效，一次 LLM 调用处理若干小时的增量
      - 不包含 L1（工作记忆是临时的，不应影响性格判断）
        L1 中的内容可能是玩笑、情绪宣泄或上下文相关的临时表达，
        直接用于性格判断会导致画像噪音（例如一句气话被当作性格特征）
    """
    interval_hours = max(1, int(getattr(settings, "profile_batch_interval_hours", 3)))
    interval_sec = interval_hours * 3600
    while True:
        await asyncio.sleep(interval_sec)
        try:
            result = await asyncio.to_thread(update_all_profiles)
            updated = int(result.get("updated") or 0)
            if updated:
                agent_monitor.event(
                    f"Profile 履历归档 · 更新 {updated}/{result.get('total', 0)} 人"
                )
            else:
                logger.debug("profile batch: nothing to update")
        except Exception as exc:
            agent_monitor.warn(f"Profile 履历归档失败: {exc}")


async def _startup_corpus_ingest() -> None:
    """启动时后台语料入库，不阻塞 HTTP/WebSocket 接受连接。

    流程：
      1. 调用 startup_ingest_corpus() 扫描 persona/corpus/ 中的 .md 文件
      2. 切块、向量化、写入长期记忆
      4. 校验向量维度一致性
      5. 通过 agent_monitor 报告入库结果

    跳过条件：
      - persona_ingest_on_startup = false 时跳过
      - 语料已有数据且源文件未变化时跳过（增量模式）
      - persona_ingest_reset_on_startup = true 时全量重建
    """
    try:
        ingest_result = await asyncio.to_thread(startup_ingest_corpus)
        from app.embed_meta import check_embed_compat

        ok, _ = check_embed_compat()
        n = semantic_memory.corpus.count()
        backend = (settings.search_backend or "sqlite").lower()

        if ingest_result.get("skipped"):
            reason = str(ingest_result.get("reason") or "")
            if reason == "corpus_exists":
                # Corpus 已有数据且无变化，跳过重建
                # 增量模式：通过文件哈希对比源文件是否变更，未变更则复用已有向量
                agent_monitor.startup(
                    f"长期记忆[{backend}] 已有 {n} 块，跳过启动入库"
                    "（全量重建: python scripts/ingest.py --reset）"
                )
            elif reason != "disabled":
                agent_monitor.startup(f"长期记忆[{backend}] {n} 块")
            elif not ok:
                # persona_ingest_on_startup=false 但仍需检查向量维度兼容性
                agent_monitor.warn("向量维度不一致，请检查 .env 后重启")
            return

        files = ingest_result.get("files") or []
        if files:
            agent_monitor.startup(
                f"长期记忆[{backend}] {n} 块 ← {len(files)} 文件"
            )
        elif not ok:
            agent_monitor.warn("向量维度不一致，请检查 .env 后重启")
        else:
            agent_monitor.warn(f"语料为空，请在 {settings.resolved_corpus_dir()} 添加 .md")
    except Exception as exc:
        agent_monitor.warn(f"语料入库失败: {exc}")


def _startup_db_diagnostics() -> None:
    """启动时在控制台打印 DB / 主人绑定 / 记忆规模，便于排查 cwd 与空库问题。"""
    db = settings.resolved_db_path()
    agent_monitor.startup(f"PID={os.getpid()}  DB={db}")
    if not db.exists():
        agent_monitor.warn(f"数据库文件不存在，将新建: {db}")
        return
    try:
        conn = sqlite3.connect(str(db))
        l3_n = conn.execute("SELECT COUNT(*) FROM l3_chunks").fetchone()[0]
        l0_n = conn.execute("SELECT COUNT(*) FROM l0_core_memories").fetchone()[0]
        prof_n = conn.execute("SELECT COUNT(*) FROM person_profiles").fetchone()[0]
        conn.close()
        agent_monitor.startup(f"记忆规模 长期记忆={l3_n}条  核心记忆={l0_n}条  画像={prof_n}人")
        if l3_n == 0:
            agent_monitor.warn("长期记忆为空：请确认语料已 ingest，且 DB_PATH 指向含数据的 agent.db")
    except Exception as exc:
        agent_monitor.warn(f"读取数据库统计失败: {exc}")

    owner_cfg = str(settings.default_owner_person_id or "").strip()
    owner_name = str(settings.default_owner_display_name or "").strip()
    owner_resolved = get_default_owner_person_id("default")
    if owner_resolved:
        prof = store.get_person_profile(owner_resolved)
        label = owner_name or owner_resolved
        agent_monitor.startup(
            f"默认主人 person_id={owner_resolved}（{label}）"
            f"{' · 画像已存在' if prof else ' · 画像未入库'}"
        )
    elif owner_cfg or owner_name:
        agent_monitor.warn(
            f"DEFAULT_OWNER 已配置（id={owner_cfg or '—'} name={owner_name or '—'}）"
            " 但未解析到有效 person_id，会话将仍为访客"
        )
    else:
        agent_monitor.startup("未配置 DEFAULT_OWNER_PERSON_ID，新会话需口语实名后才检索 L0/L2/L3")


# ============================
# 应用生命周期
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 应用生命周期管理。

    启动阶段（按顺序）：
      1. 初始化控制台监控（静音第三方日志，仅 agent 通道输出）
      2. 启动 idle_session_sweeper —— 每分钟清理超时的 WebSocket/HTTP 会话
         （长时间无活动会自动触发 session_end，防止 L1 内存泄漏）
      3. 启动 l2_rollup_sweeper —— 每小时将过期 L2 摘要归档到 L3 Corpus
         （这是记忆从短期→长期的桥梁，保证 7 天前的对话仍可检索）
      4. 启动 profile_batch_sweeper —— 每 N 小时从 L2+L3 增量更新用户画像
         （批量处理比每轮提取效率高，且避免 L1 临时对话污染性格判断）
      5. 输出 LLM/向量配置摘要到控制台
      6. 异步启动语料入库 —— 不阻塞应用接受 HTTP/WebSocket 连接
         （入库可能耗时数十秒，用户不应等待它完成才能对话）

    关闭阶段：
      依次取消所有后台任务（asyncio.Task.cancel），实现优雅退出。
      Python 的 cancel 机制会在下一个 await 点抛出 CancelledError，
      各 sweeper 的 while True 循环会因 sleep 中断而退出。
    """
    agent_monitor.configure(force=True)

    # 创建后台定时任务（这些协程与 app 同生命周期，在 yield 前启动）
    # create_task 会将协程注册到当前事件循环，应用运行期间持续执行
    idle_task = asyncio.create_task(idle_session_sweeper())
    rollup_task = asyncio.create_task(l2_rollup_sweeper())
    profile_batch_task = asyncio.create_task(profile_batch_sweeper())

    # 输出启动信息到控制台
    llm_ok = "DeepSeek 已就绪" if settings.llm_api_key else "未配置 LLM_API_KEY"
    embed_ok = "向量 API 已就绪" if settings.embed_api_key else "未配置 EMBED_API_KEY（使用本地向量）"
    agent_monitor.banner("服务启动")
    agent_monitor.startup(f"{llm_ok} | {embed_ok}")
    _startup_db_diagnostics()
    port_display = settings.port
    agent_monitor.startup(f"聊天测试: http://127.0.0.1:{port_display}/chat")
    agent_monitor.startup(f"后台管理: http://127.0.0.1:{port_display}/admin")
    agent_monitor.startup(f"WebSocket: /ws/v1/chat")
    agent_monitor.startup(f"音频 WS:   /ws/v2/audio")

    # 预热 LLM HTTP 连接池，减少首次对话 TCP+TLS 握手延迟
    warmup_llm_client()

    # 语料入库在后台进行，不阻塞应用启动
    # 注意：ingest_task 是可选的（如果 persona_ingest_on_startup=False 则不创建）
    ingest_task: asyncio.Task | None = None
    if getattr(settings, "persona_ingest_on_startup", True):
        agent_monitor.startup("语料同步在后台进行，可先访问页面/对话")
        ingest_task = asyncio.create_task(_startup_corpus_ingest())

    yield  # ← 应用在此处运行，yield 之后的代码是关闭阶段

    # ---- 关闭阶段：优雅退出 ----
    # cancel 顺序：先取消可选的 ingest_task，再取消三个必然后台任务
    if ingest_task:
        ingest_task.cancel()
    idle_task.cancel()
    rollup_task.cancel()
    profile_batch_task.cancel()
    # 注意：cancel() 不会立即终止协程，而是在下一个 await 点抛出 CancelledError
    # 各 sweeper 的 while True 循环中 await asyncio.sleep 是 await 点


# ============================
# FastAPI 应用实例
# ============================

app = FastAPI(title="SparkBot Personal Agent", lifespan=lifespan)

# 挂载静态文件目录（前端测试页面用）
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================
# HTTP 接口
# ============================

@app.get("/")
async def index():
    """根路径重定向到后台管理页面。"""
    return RedirectResponse(url="/admin")


@app.get("/chat")
async def chat_ui():
    """聊天测试页面。"""
    chat = STATIC_DIR / "chat.html"
    if not chat.is_file():
        raise HTTPException(status_code=404, detail="static/chat.html not found")
    return FileResponse(chat, headers={"Cache-Control": "no-store"})


@app.get("/admin")
async def admin_ui():
    """后台管理页面。"""
    admin = STATIC_DIR / "admin.html"
    if not admin.is_file():
        raise HTTPException(status_code=404, detail="static/admin.html not found")
    return FileResponse(admin, headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health():
    """健康检查接口：返回 LLM、向量、记忆参数的配置摘要。

    供运维监控使用，可快速判断：
      - DeepSeek API 是否配置
      - 向量模型是否为本地 fallback
      - 各记忆层参数当前值

    返回字段含义（memory 块）：
      l1_turns:            L1 滑动窗口保留的最大轮次数
      l2_retention_days:   L2 情景摘要保留天数（过期后归档到 L3）
      l3_recall_mode:      L3 召回模式（recent=时间倒序, query=向量语义检索）
      search_backend:      向量搜索后端（chroma/sqlite）
      l0_batch_hour:       [已废弃] 旧版每日批量 L0 升级触发时间
      l0_batch_min_recall: [已废弃] 旧版 L0 批量升级的最小召回轮数阈值
    """
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
            "search_backend": settings.search_backend,
            "l0_batch_hour": settings.l0_batch_hour,
            "l0_batch_min_recall": settings.l0_batch_min_recall,
        },
    }


@app.get("/v1/admin/persons")
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


@app.post("/v1/admin/persons")
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
    return result


@app.patch("/v1/admin/persons/{person_id}")
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
    agent_monitor.event(
        f"身份后台更新 · {result.get('renamed_from') or person_id} → {result['person_id']}"
    )
    return result


@app.delete("/v1/admin/persons/{person_id}")
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
    agent_monitor.event(
        f"身份已删除 · {result['display_name']} id={result['person_id'][:12]}"
    )
    return result


@app.get("/v1/admin/persons/{person_id}/core-memories")
async def admin_list_core_memories(
    person_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """列出指定用户的核心记忆。"""
    _check_token(authorization, x_api_token)
    try:
        items = await asyncio.to_thread(list_l0_admin, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"person_id": person_id, "items": items, "categories": L0_CATEGORY_LABELS}


@app.post("/v1/admin/persons/{person_id}/core-memories")
async def admin_create_core_memory(
    person_id: str,
    body: L0AdminCreate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """新增一条核心记忆（管理员写入，跳过自动门控）。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(
            create_l0_admin, person_id, body.category, body.content
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"核心记忆后台新增 · {person_id[:12]} [{body.category}]")
    return result


@app.patch("/v1/admin/persons/{person_id}/core-memories/{l0_id}")
async def admin_update_core_memory(
    person_id: str,
    l0_id: int,
    body: L0AdminUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """修改核心记忆的类别或内容。"""
    _check_token(authorization, x_api_token)
    if body.category is None and body.content is None:
        raise HTTPException(status_code=400, detail="category or content required")
    try:
        result = await asyncio.to_thread(
            update_l0_admin,
            person_id,
            l0_id,
            category=body.category,
            content=body.content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"核心记忆后台更新 · {person_id[:12]} id={l0_id}")
    return result


@app.delete("/v1/admin/persons/{person_id}/core-memories/{l0_id}")
async def admin_delete_core_memory(
    person_id: str,
    l0_id: int,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除一条核心记忆。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(delete_l0_admin, person_id, l0_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"核心记忆后台删除 · {person_id[:12]} id={l0_id}")
    return result


@app.get("/v1/admin/persons/{person_id}/profile")
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


@app.delete("/v1/admin/persons/{person_id}/episodes")
async def admin_delete_episodes(
    person_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除指定用户的所有情景记忆（不可逆）。"""
    _check_token(authorization, x_api_token)
    try:
        result = await asyncio.to_thread(delete_l2_admin, person_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"情景记忆后台删除 · {person_id[:12]} · {result.get('l2_count', 0)} 条")
    return result


@app.put("/v1/admin/persons/{person_id}/profile")
async def admin_update_profile(
    person_id: str,
    body: ProfileAdminUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """更新人物画像字段；默认同步身份/关系到 L0。"""
    _check_token(authorization, x_api_token)
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="no fields to update")
    try:
        result = await asyncio.to_thread(update_profile_admin, person_id, **fields)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    agent_monitor.event(f"画像后台更新 · {result.get('display_name')} id={person_id[:12]}")
    return result


@app.get("/v1/admin/contacts")
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


@app.post("/v1/admin/contacts")
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


@app.get("/v1/admin/contacts/{person_id}")
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


@app.patch("/v1/admin/contacts/{person_id}")
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


@app.delete("/v1/admin/contacts/{person_id}")
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

@app.get("/v1/admin/overview")
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


@app.get("/v1/admin/memory/by-person")
async def admin_memory_by_person(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """按实名用户分组返回记忆概览和首屏数据。"""
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(build_memory_by_person_admin)


@app.get("/v1/admin/graph")
async def admin_graph(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """返回关系图谱节点与边。"""
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(build_graph_admin)


@app.get("/v1/admin/sessions")
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


@app.get("/v1/admin/sessions/{session_id}/messages")
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

@app.get("/v1/admin/persons/{person_id}/episodes/{episode_id}")
async def admin_get_episode(
    person_id: str,
    episode_id: int,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """获取单条情景记忆。"""
    _check_token(authorization, x_api_token)
    from app.memory_admin import _validate_person
    _validate_person(person_id)
    try:
        item = await asyncio.to_thread(store.get_episodic_by_id, int(episode_id))
    except TypeError:
        raise HTTPException(status_code=400, detail="invalid episode_id")
    if not item or str(item.get("person_id", "")) != person_id:
        raise HTTPException(status_code=404, detail="episode not found")
    return item


@app.patch("/v1/admin/persons/{person_id}/episodes/{episode_id}")
async def admin_update_episode(
    person_id: str,
    episode_id: int,
    body: dict,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """更新情景摘要内容。"""
    _check_token(authorization, x_api_token)
    item = await asyncio.to_thread(store.get_episodic_by_id, int(episode_id))
    if not item or str(item.get("person_id", "")) != person_id:
        raise HTTPException(status_code=404, detail="episode not found")
    ok = await asyncio.to_thread(store.update_episodic_admin, int(episode_id), **body)
    if not ok:
        raise HTTPException(status_code=400, detail="no fields to update")
    updated = await asyncio.to_thread(store.get_episodic_by_id, int(episode_id))
    return updated


@app.delete("/v1/admin/persons/{person_id}/episodes/{episode_id}")
async def admin_delete_episode(
    person_id: str,
    episode_id: int,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除单条情景记忆。"""
    _check_token(authorization, x_api_token)
    item = await asyncio.to_thread(store.get_episodic_by_id, int(episode_id))
    if not item or str(item.get("person_id", "")) != person_id:
        raise HTTPException(status_code=404, detail="episode not found")
    ok = await asyncio.to_thread(store.delete_episodic_by_id, int(episode_id))
    return {"deleted": ok, "id": episode_id}


# ── 长期记忆管理（旧 L3） ──

@app.get("/v1/admin/long-term-memory")
async def admin_list_long_term_memory(
    collection: str | None = None,
    person_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """列出长期记忆向量块。"""
    _check_token(authorization, x_api_token)
    items = await asyncio.to_thread(
        store.l3_list_chunks_detailed, collection, person_id,
        limit=limit, offset=offset,
    )
    return {"items": items, "count": len(items)}


@app.get("/v1/admin/long-term-memory/{chunk_id}")
async def admin_get_long_term_chunk(
    chunk_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """获取单条长期记忆块。"""
    _check_token(authorization, x_api_token)
    chunk = await asyncio.to_thread(store.l3_get_chunk, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="chunk not found")
    return chunk


@app.patch("/v1/admin/long-term-memory/{chunk_id}")
async def admin_update_long_term_chunk(
    chunk_id: str,
    body: dict,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """更新长期记忆块文本或类别。"""
    _check_token(authorization, x_api_token)
    existing = await asyncio.to_thread(store.l3_get_chunk, chunk_id)
    if not existing:
        raise HTTPException(status_code=404, detail="chunk not found")
    text = body.get("text")
    category = body.get("category")
    ok = await asyncio.to_thread(
        store.l3_update_chunk, chunk_id, text=text, category=category
    )
    if not ok:
        raise HTTPException(status_code=400, detail="no changes")
    updated = await asyncio.to_thread(store.l3_get_chunk, chunk_id)
    return updated


@app.delete("/v1/admin/long-term-memory/{chunk_id}")
async def admin_delete_long_term_chunk(
    chunk_id: str,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    """删除长期记忆块。"""
    _check_token(authorization, x_api_token)
    ok = await asyncio.to_thread(store.l3_delete_chunk, chunk_id)
    if not ok:
        raise HTTPException(status_code=404, detail="chunk not found")
    agent_monitor.event(f"长期记忆后台删除 · chunk={chunk_id[:16]}")
    return {"deleted": True, "chunk_id": chunk_id}


# ── SSE 控制台日志推送 ──

@app.get("/v1/admin/logs/stream")
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

@app.get("/v1/admin/config")
async def admin_get_config(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(list_config)


@app.patch("/v1/admin/config")
async def admin_update_config(
    body: ConfigAdminUpdate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    result = await asyncio.to_thread(update_config, body.values)
    agent_monitor.event(f"配置后台更新 · {', '.join(result.get('updated') or [])}")
    return result


@app.post("/v1/admin/config/test")
async def admin_test_config(
    body: ConfigTestRequest,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(test_config, body.kind)


@app.get("/v1/admin/files")
async def admin_list_files(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return {"files": await asyncio.to_thread(default_file_store().list_files)}


@app.get("/v1/admin/files/{file_path:path}")
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


@app.put("/v1/admin/files/{file_path:path}")
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


@app.post("/v1/admin/files/{file_path:path}")
async def admin_create_file(
    file_path: str,
    body: AdminFileSave,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    return await admin_save_file(file_path, body, authorization, x_api_token)


@app.delete("/v1/admin/files/{file_path:path}")
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


@app.get("/v1/admin/health/deep")
async def admin_deep_health(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(deep_health)


@app.get("/v1/admin/backups")
async def admin_list_backups(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(list_backups)


@app.post("/v1/admin/backups")
async def admin_create_backup(
    body: AdminBackupCreate,
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    result = await asyncio.to_thread(create_backup, body.kind)
    agent_monitor.event(f"备份创建 · {body.kind}")
    return result


@app.post("/v1/admin/backups/{backup_id}/restore")
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
    return result


@app.get("/v1/admin/deploy/status")
async def admin_deploy_status(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    return await asyncio.to_thread(deploy_status)


@app.post("/v1/admin/deploy/update")
async def admin_deploy_update(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    agent_monitor.event("部署更新启动")
    return await run_deploy_update()


@app.post("/v1/admin/service/restart")
async def admin_service_restart(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    agent_monitor.warn("收到服务重启请求")
    return await request_restart()


@app.post("/v1/admin/service/start")
async def admin_service_start(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    agent_monitor.warn("收到服务启动请求")
    return await request_service_action("start")


@app.post("/v1/admin/service/stop")
async def admin_service_stop(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    _check_token(authorization, x_api_token)
    agent_monitor.warn("收到服务停止请求")
    return await request_service_action("stop")


@app.post("/v1/admin/debug/recall")
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
        return {"status": "done", "files": len(files)}
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
    return {"status": "done", "stdout": proc.stdout[-4000:]}


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
    return {"status": "done", "stdout": proc.stdout[-4000:]}


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
    return {"status": "done", "stdout": proc.stdout[-4000:]}


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
        return {"status": "done", "stats": stats}
    except Exception as exc:
        agent_monitor.warn(f"记忆诊断失败: {exc}")
        raise


async def _run_task_rollup():
    """执行情景摘要归档为长期记忆。"""
    from app.memory.extractor import rollup_expired_l2
    try:
        n = await asyncio.to_thread(rollup_expired_l2, None)
        agent_monitor.event(f"任务完成 · 情景摘要归档 · {n} 条")
        return {"status": "done", "archived": n}
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
        return {"status": "done", "updated": updated, "total": total}
    except Exception as exc:
        agent_monitor.warn(f"画像更新失败: {exc}")
        raise


TASK_HANDLERS = {
    "ingest": ("语料同步入库", _run_task_ingest),
    "ingest-reset": ("语料全量重建", _run_task_ingest_reset),
    "compress-profile": ("生成 Profile Card", _run_task_compress_profile),
    "diagnose-memory": ("记忆健康诊断", _run_task_diagnose),
    "rollup-l2": ("情景摘要归档", _run_task_rollup),
    "update-profiles": ("人物画像批量更新", _run_task_update_profiles),
    "cleanup-contacts": ("第三方画像去重", _run_task_cleanup_contacts),
}


@app.get("/v1/admin/tasks")
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


@app.get("/v1/admin/tasks/{task_id}")
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


@app.post("/v1/admin/tasks/{task_name}")
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
        return task_manager.start(task_name, title, handler)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.websocket("/ws/v1/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket 对话入口 —— 协议详情见 ws_handler.py。"""
    await ws_chat_endpoint(websocket)


@app.websocket("/ws/v2/audio")
async def websocket_audio(websocket: WebSocket):
    """WebSocket 音频对话入口 —— v2 协议，支持 PCM 上传/打断/流式 TTS。"""
    await ws_audio_endpoint(websocket)


# ============================
# 直接运行入口（开发调试用）
# ============================

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

    # 直接运行 python app/main.py 时启动 uvicorn
    # 与命令行 uvicorn app.main:app 等效，但无需手动指定参数
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,       # 生产模式不启用热重载（避免文件变更时意外重启）
        log_config=LOG_CONFIG,
        access_log=False,   # 静音 HTTP 访问日志（200 OK 行干扰 agent 监控输出）
    )
