"""FastAPI 生命周期与后台任务。"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.llm import warmup_llm_client
from app.memory.memory_pipeline import archive_expired_recent_memory
from app.memory.interlocutor import get_default_owner_person_id
from app.memory.long_term_memory import long_term_memory
from app.memory.profile import update_all_profiles
from app.monitor import agent_monitor
from app.persona.ingest import startup_ingest_corpus
from app.session import store
from app.ws_handler import idle_session_sweeper

logger = logging.getLogger(__name__)

async def recent_memory_rollup_sweeper() -> None:
    """近期记忆归档任务：每小时将过期的近期记忆写入长期记忆。

    工作原理：
      调用 archive_expired_recent_memory 扫描 memory_items 中 kind=episode/emotion
      且 expires_at 已过期的记录，将其内容归档到长期记忆，
      然后标记 archived=1。

    为什么每小时一次：
      近期记忆的默认保留期是 14 天，每小时检查一次足够及时，
      同时不会给系统带来明显负载。
    """
    while True:
        await asyncio.sleep(3600)  # 每小时执行一次
        try:
            n = await asyncio.to_thread(archive_expired_recent_memory, None)
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
      - 不包含 工作上下文（工作记忆是临时的，不应影响性格判断）
        工作上下文 中的内容可能是玩笑、情绪宣泄或上下文相关的临时表达，
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
        n = long_term_memory.count_chunks()
        backend = (settings.search_backend or "sqlite").lower()

        if ingest_result.get("skipped"):
            reason = str(ingest_result.get("reason") or "")
            if reason == "corpus_complete":
                audit = ingest_result.get("audit", {})
                agent_monitor.startup(
                    f"长期记忆  corpus={audit.get('actual_chunk_count', n)}块"
                    f"  |  {audit.get('expected_chunk_count', n)} expected"
                    " 已同步，跳过启动入库"
                )
            elif reason == "corpus_exists":
                agent_monitor.startup(
                    f"长期记忆  {n} 块（corpus，旧计数），跳过启动入库"
                )
            elif reason != "disabled":
                agent_monitor.startup(f"长期记忆  {n} 块")
            elif not ok:
                agent_monitor.warn("向量维度不一致，请检查 .env 后重启")
            return

        files = ingest_result.get("files") or []
        if files:
            agent_monitor.startup(
                f"长期记忆  {n} 块 ← {len(files)} 文件"
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
        prof_n = conn.execute("SELECT COUNT(*) FROM person_profiles").fetchone()[0]
        conn.close()
        # 统一记忆库统计
        from app.session import store
        stats = store.count_memory_stat()
        mi_total = stats.get("memory_items", 0)
        corpus_n = store.count_corpus_memory_items()
        user_n = mi_total - corpus_n
        agent_monitor.startup(
            f"记忆规模  统一记忆库={mi_total}条  "
            f"(corpus={corpus_n}  /  user={user_n})  "
            f"画像={prof_n}人"
        )
        if mi_total == 0:
            agent_monitor.warn("统一记忆库为空：请确认语料已 ingest，且 DB_PATH 指向含数据的 agent.db")
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
        agent_monitor.startup("未配置 DEFAULT_OWNER_PERSON_ID，新会话需口语实名后才检索核心事实/近期/长期记忆")


# ============================
# 应用生命周期
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 应用生命周期管理。

    启动阶段（按顺序）：
      1. 初始化控制台监控（静音第三方日志，仅 agent 通道输出）
      2. 启动 idle_session_sweeper —— 每分钟清理超时的 WebSocket/HTTP 会话
         （长时间无活动会自动触发 session_end，防止工作上下文内存泄漏）
      3. 启动 recent_memory_rollup_sweeper —— 每小时将过期近期记忆归档到长期记忆
         （这是短期记忆→长期记忆的桥梁，保证旧对话仍可检索）
      4. 启动 profile_batch_sweeper —— 每 N 小时从近期+长期记忆增量更新用户画像
         （批量处理比每轮提取效率高，且避免临时对话污染性格判断）
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
    rollup_task = asyncio.create_task(recent_memory_rollup_sweeper())
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
