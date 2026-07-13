"""SQLite 持久层 —— 会话、统一记忆库（memory_items）、画像、关联图、状态寄存器的统一存储。

本模块是陪伴机器人所有数据的持久化入口。记忆系统已统一为三层抽象：
  - Working Context（工作上下文：当前会话消息）
  - Unified Memory Store（统一记忆库：所有长期/中期记忆通过 memory_items 统一表管理）
  - State Registers（状态寄存器：画像、关联图、关系状态、Open Loops）

记忆系统统一为三层抽象：
  - Working Context（工作上下文：当前会话消息）
  - Unified Memory Store（统一记忆库：所有长期/中期记忆通过 memory_items 统一表管理）
  - State Registers（状态寄存器：画像、关联图、关系状态、Open Loops）

旧四层记忆架构（核心事实/工作上下文/近期记忆/长期记忆）已整体迁移到 memory_items，旧表已物理删除。

============================
数据表总览
============================

  sessions              会话表（device_id + session_id + active_person_id）
  messages              消息表（工作上下文：role + content）
  memory_items          统一记忆库表（全部长期/中期/核心记忆）
  memory_items_fts      统一记忆库全文索引（FTS5）
  person_profiles       人物画像表（JSON 格式存储完整 Profile Card）
  memory_relations      记忆关联图表（from_id ↔ to_id 语义关系）
  open_loops            待跟进事项表（结构化 Open Loop）
  relationship_states   关系状态表（温度/情绪/态度）

============================
关键设计
============================

  - 单文件数据库（agent.db），无外部依赖
  - 使用 sqlite3.Row 工厂以字典访问列值
  - 增量 schema 迁移：各 _migrate_* 方法检测列是否存在再 ALTER TABLE
  - 所有时间使用 UTC ISO 格式字符串存储
  - 全局单例 store = SessionStore() 供所有模块引用
  - FTS5 全文索引用于中文关键词检索（unicode61 tokenizer）
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from app.config import settings

logger = logging.getLogger(__name__)


# ============================
# 时间工具
# ============================

def _utc_now() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串。

    为什么用 UTC：
      SQLite 没有时区概念，ISO 格式 UTC 字符串便于排序和比较。
    """
    return datetime.now(timezone.utc).isoformat()


def _expires_at(days: int | None = None) -> str:
    """计算过期时间（默认使用配置的近期记忆保留天数）。

    参数:
        days: 保留天数，None 时使用 settings.recent_memory_retention_days
    返回:
        ISO 格式的过期时间字符串（UTC）
    """
    d = days if days is not None else settings.recent_memory_retention_days
    return (datetime.now(timezone.utc) + timedelta(days=d)).isoformat()


def _decode_text_cell(value: object, *, column: str, rowid: int | None = None) -> str | None:
    """把 SQLite 文本列安全转为 str，遇到损坏 UTF-8 时返回 None。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            where = f" rowid={rowid}" if rowid is not None else ""
            logger.warning("跳过损坏的 SQLite 文本列%s column=%s hex=%s", where, column, value.hex())
            return None
    return str(value)


# ============================
# SessionStore 主类
# ============================

class SessionStore:
    """SQLite 持久层：会话上下文 + 统一记忆库 + 状态寄存器的统一存储。

    本类对外提供三类语义接口：

    1. Working Context（工作上下文）
       append_message / list_recent_messages / compact_working_context / finalize_session

    2. Unified Memory（统一记忆库）
       write_memory_item / upsert_memory_item / list_memory_items
       search_memory_items / count_memory_items / get_memory_item
       archive_memory_item / delete_memory_item

       语义视图（均基于 memory_items 统一表）：
         list_core_facts          — 核心事实（visibility=always）
         search_recent_memory     — 近期记忆（episode/emotion 类）
         search_long_term_memory  — 长期记忆（fact/entity/wiki/relationship 类）

    3. State Registers（状态寄存器）
       get_or_create_session / save_person_profile / get_person_profile
       upsert_memory_relation / get_memory_relations / save_relationship_state
       create_open_loop / resolve_open_loop / list_open_loops

    所有公开方法都是同步的（SQLite 本身不支持异步），
    调用方应通过 asyncio.to_thread 在后台线程中执行以避免阻塞事件循环。
    """

    def __init__(self, db_path: str | None = None) -> None:
        """初始化数据库连接并自动执行 schema 迁移。

        参数:
            db_path: 数据库文件路径，None 时使用 settings.db_path（默认 ./agent.db）
        """
        self.db_path = Path(db_path) if db_path else settings.resolved_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 持久连接 + 线程锁：避免每次操作创建/关闭连接的开销
        self._conn_lock = threading.Lock()
        self._persistent_conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取持久化连接（线程安全，惰性创建）。"""
        if self._persistent_conn is None:
            with self._conn_lock:
                if self._persistent_conn is None:
                    conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=-8000")  # 8MB cache
                    conn.execute("PRAGMA busy_timeout=5000")
                    self._persistent_conn = conn
        return self._persistent_conn

    # ============================
    # 数据库连接管理
    # ============================

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """获取数据库连接上下文管理器（复用持久连接）。

        复用 _get_conn() 返回的持久连接，消除每次操作的 connect/close 开销。
        WAL 模式允许并发读写，commit 时短暂加锁保证原子性。
        """
        conn = self._get_conn()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            with self._conn_lock:
                conn.commit()

    # ============================
    # Schema 迁移
    # ============================

    def _migrate_relationship_state(self, conn: sqlite3.Connection) -> None:
        """创建关系状态存储表（relationship_states）。"""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relationship_states (
                person_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def _migrate_open_loops(self, conn: sqlite3.Connection) -> None:
        """创建结构化待跟进事项表（open_loops）。"""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS open_loops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                due_hint TEXT DEFAULT '',
                emotional_weight INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                last_mentioned_at TEXT DEFAULT '',
                cooldown_until TEXT DEFAULT '',
                source_session_id TEXT DEFAULT '',
                resolved_evidence TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_open_loops_person ON open_loops(person_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_open_loops_status ON open_loops(person_id, status)"
        )

    def _migrate_memory_items(self, conn: sqlite3.Connection) -> None:
        """创建 memory_items 统一记忆表及 FTS5 全文索引。

        memory_items 是统一的记忆存储层，替代旧的 核心事实/近期记忆/长期记忆 多表架构。
        所有记忆类型（核心事实、近期记忆、长期记忆）统一存入此表。
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                device_id TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                visibility TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                emotional_weight INTEGER NOT NULL DEFAULT 3,
                recency_weight INTEGER NOT NULL DEFAULT 3,
                context_json TEXT NOT NULL DEFAULT '{}',
                tags_json TEXT NOT NULL DEFAULT '[]',
                embedding_json TEXT NOT NULL DEFAULT '[]',
                source_table TEXT NOT NULL DEFAULT '',
                source_id TEXT NOT NULL DEFAULT '',
                source_session TEXT NOT NULL DEFAULT '',
                expires_at TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
                    id UNINDEXED,
                    content_fts,
                    tokenize='unicode61'
                )
                """
            )
        except Exception as exc:
            logger.warning("memory_items_fts creation failed (non-fatal): %s", exc)

    def _ensure_indexes(self, conn: sqlite3.Connection) -> None:
        """创建所有必要的索引以优化查询性能。

        索引策略：
          - messages: 按 session_id 查询
          - profiles: 按 device_id 查询
          - relations: 按 from_id / to_id 双向查
          - memory_items: 按 source_table+source_id、person_id+kind+visibility 等
        """
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_person_profiles_device ON person_profiles(device_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_from ON memory_relations(from_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_to ON memory_relations(to_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_items_source"
            " ON memory_items(source_table, source_id)"
            " WHERE source_table != '' AND source_id != ''"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_items_person_kind"
            " ON memory_items(person_id, kind, visibility)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_items_person_updated"
            " ON memory_items(person_id, updated_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_items_hash"
            " ON memory_items(person_id, kind, content_hash)"
        )

    def _migrate_memory_relations(self, conn: sqlite3.Connection) -> None:
        """创建 memory_relations 表（记忆关联图）。

        存储记忆之间的语义关联关系，如：
          - 因果关系（cause_effect）
          - 同一事件（same_event）
          - 人物关联（related_person）
          - 时序关系（before/after）

        UNIQUE(from_id, to_id, relation_type) 防止重复边。
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                UNIQUE(from_id, to_id, relation_type)
            )
            """
        )

    def _migrate_sessions(self, conn: sqlite3.Connection) -> None:
        """迁移 sessions 表，添加身份相关字段。

        active_person_id: 当前会话绑定的用户（访客为 tmp_*）
        guest_turn_count: 访客已对话轮数（用于 N 轮提醒实名）
        identity_pending: 待确认的身份声明（JSON，如用户说了名字但 ID 未验证）
        """
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if cols and "active_person_id" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN active_person_id TEXT")
        if cols and "guest_turn_count" not in cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN guest_turn_count INTEGER NOT NULL DEFAULT 0"
            )
        if cols and "identity_pending" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN identity_pending TEXT")
        if cols and "interlocutor_mode" not in cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN interlocutor_mode TEXT NOT NULL DEFAULT 'girlfriend'"
            )

    def _init_db(self) -> None:
        """初始化数据库：创建所有必需的数据表并执行增量迁移。

        旧记忆表信息（仅用于 migrate 审计）
        已通过一次性迁移删除，所有记忆数据统一存储在 memory_items 中。
        """
        with self._conn() as conn:
            conn.executescript(
                """
                -- 会话表：每个 device_id 可有多个 session，status=active/closed
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL
                );
                -- 消息表：记录每轮对话的 user/assistant 消息
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                -- 人物画像表：JSON 格式存储完整 Profile Card
                CREATE TABLE IF NOT EXISTS person_profiles (
                    person_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            # 增量迁移：处理后续版本新增的字段
            self._migrate_memory_items(conn)
            self._migrate_sessions(conn)
            self._migrate_memory_relations(conn)
            self._migrate_relationship_state(conn)
            self._migrate_open_loops(conn)
            self._ensure_indexes(conn)

    # ============================
    # Session 管理
    # ============================

    def get_or_create_session(self, device_id: str, session_id: str | None) -> str:
        """获取或创建会话。

        参数:
            device_id:  设备标识
            session_id: 现有会话 ID，为空或无效时创建新会话

        返回:
            str: 有效的 session_id

        逻辑：
          1. 如果传了 session_id 且该会话状态为 active，复用它
          2. 如果复用成功但 active_person_id 为空，分配 tmp_* 访客 ID
          3. 否则创建新会话，分配 UUID 作为 session_id，
             同时分配 tmp_* 访客 ID 作为 active_person_id
        """
        from app.memory.identity import new_temp_person_id
        from app.memory.interlocutor import get_default_owner_person_id, MODE_GIRLFRIEND

        try:
            with self._conn() as conn:
                if session_id:
                    row = conn.execute(
                        "SELECT id, status, active_person_id FROM sessions WHERE id=?",
                        (session_id,),
                    ).fetchone()
                    if row and row["status"] == "active":
                        conn.execute(
                            "UPDATE sessions SET last_active=? WHERE id=?",
                            (_utc_now(), session_id),
                        )
                        if not row["active_person_id"]:
                            owner_id = get_default_owner_person_id(device_id)
                            pid = owner_id or new_temp_person_id()
                            conn.execute(
                                "UPDATE sessions SET active_person_id=? WHERE id=?",
                                (pid, session_id),
                            )
                        return session_id
                # 创建新会话
                new_id = str(uuid4())
                now = _utc_now()
                owner_id = get_default_owner_person_id(device_id)
                pid = owner_id or new_temp_person_id()
                conn.execute(
                    """
                    INSERT INTO sessions(
                        id, device_id, status, created_at, last_active,
                        active_person_id, guest_turn_count, identity_pending,
                        interlocutor_mode
                    ) VALUES (?,?,?,?,?,?,0,NULL,?)
                    """,
                    (new_id, device_id, "active", now, now, pid, MODE_GIRLFRIEND),
                )
                return new_id
        except Exception:
            import logging
            _log = logging.getLogger(__name__)
            _log.exception("get_or_create_session failed for device=%s session=%s, forcing new session",
                           device_id, session_id)
            # Last-resort: create a new session on a fresh connection
            try:
                with self._conn() as conn:
                    new_id = str(uuid4())
                    now = _utc_now()
                    owner_id = get_default_owner_person_id(device_id)
                    pid = owner_id or new_temp_person_id()
                    conn.execute(
                        """
                        INSERT INTO sessions(
                            id, device_id, status, created_at, last_active,
                            active_person_id, guest_turn_count, identity_pending,
                            interlocutor_mode
                        ) VALUES (?,?,?,?,?,?,0,NULL,?)
                        """,
                        (new_id, device_id, "active", now, now, pid, MODE_GIRLFRIEND),
                    )
                    return new_id
            except Exception:
                _log.exception("get_or_create_session double-fault for device=%s", device_id)
                return str(uuid4())

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """向工作上下文追加一条消息并更新会话 last_active 时间戳。

        参数:
            session_id: 会话 ID
            role:       "user" 或 "assistant"
            content:    消息文本
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, role, content, _utc_now()),
            )
            conn.execute("UPDATE sessions SET last_active=? WHERE id=?", (_utc_now(), session_id))

    def count_turns(self, session_id: str) -> int:
        """统计会话中 user 轮数（一条 user 消息 = 一轮）。

        用于判断是否达到工作上下文压缩阈值（working_context_turns）。
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id=? AND role='user'",
                (session_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def get_recent_messages(self, session_id: str, limit: int) -> list[dict]:
        """取工作上下文最近 limit 条消息（按 ID 倒序取，再反转为正序）。

        返回格式: [{"role": "user", "content": "你好"}, ...]
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at FROM messages
                WHERE session_id=? ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_oldest_messages(self, session_id: str, limit: int) -> list[dict]:
        """获取工作上下文中最早的 limit 条消息（按 ID 升序）。

        用于上下文压缩：取出最早的一批消息交给 LLM 压缩为统一记忆库条目。
        返回包含 id 字段（压缩后需要删除这些消息）。
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content FROM messages
                WHERE session_id=? ORDER BY id ASC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]

    def delete_messages_by_ids(self, message_ids: list[int]) -> None:
        """根据消息 ID 列表批量删除工作上下文中的旧消息。

        用于上下文压缩后清理已处理的旧消息。
        """
        if not message_ids:
            return
        placeholders = ",".join("?" * len(message_ids))
        with self._conn() as conn:
            conn.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", message_ids)

    def get_session_messages(self, session_id: str) -> list[dict]:
        """获取会话的所有消息（全部历史，按时间顺序）。

        用于会话结束时的上下文全量归档。
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def close_session(self, session_id: str) -> None:
        """标记会话为 closed 状态（不删除数据，保留审计记录）。"""
        with self._conn() as conn:
            conn.execute("UPDATE sessions SET status='closed', last_active=? WHERE id=?", (_utc_now(), session_id))

    def finalize_session(self, session_id: str) -> None:
        """结束会话：清空工作上下文消息 + 清除 active_person_id。

        与 close_session 的区别：
          finalize_session 会清空工作上下文和解除 person 绑定，
          用于访客结束会话（不保留上下文），画像数据保留不删。
        """
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute(
                """
                UPDATE sessions
                SET status='closed', active_person_id=NULL, last_active=?
                WHERE id=?
                """,
                (_utc_now(), session_id),
            )

    # ── Working Context 语义接口 ───────────────────────────────────────

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """向工作上下文追加一条消息（语义名，同 add_message）。"""
        self.add_message(session_id, role, content)

    def list_recent_messages(self, session_id: str, limit: int) -> list[dict]:
        """获取工作上下文最近的 N 条消息（语义名，同 get_recent_messages）。"""
        return self.get_recent_messages(session_id, limit)

    def compact_working_context(self, session_id: str, msg_limit: int) -> list[dict]:
        """压缩工作上下文：取出最旧的 msg_limit 条消息并删除，返回这些消息。

        用于工作上下文（Working Context）超过阈值时的压缩操作。
        调用方应将这些消息交给 LLM 压缩为统一记忆库条目后归档。

        Returns:
            list[dict]: 被移除的旧消息列表（含 id/role/content）。
        """
        oldest = self.get_oldest_messages(session_id, msg_limit)
        if oldest:
            self.delete_messages_by_ids([m["id"] for m in oldest])
        return oldest

    def list_idle_active_sessions(self, idle_before_iso: str) -> list[dict]:
        """列出在指定时间点之前无活动的活跃会话。

        参数:
            idle_before_iso: 截止时间（ISO 格式），last_active < 此时间的会话视为空闲

        用于 idle_session_sweeper 发现需要自动结束的 HTTP 会话。
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, device_id, last_active FROM sessions
                WHERE status='active' AND last_active < ?
                """,
                (idle_before_iso,),
            ).fetchall()
        return [dict(r) for r in rows]


    def list_active_recent_memory(
        self, device_id: str, person_id: str, limit: int = 30
    ) -> list[dict]:
        """获取用户的活跃近期记忆（基于 memory_items，查询 episode/emotion 类型）。

        返回按 created_at 倒序（最新的在前）。
        """
        del device_id
        pid = str(person_id or "").strip()
        if not pid:
            return []
        now = _utc_now()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content AS summary, context_json,
                       created_at, expires_at, emotional_weight, tags_json, deleted_at
                FROM memory_items
                WHERE person_id=? AND kind IN ('episode','emotion')
                  AND deleted_at='' AND (expires_at='' OR expires_at>?)
                ORDER BY created_at DESC LIMIT ?
                """,
                (pid, now, limit),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            try:
                ctx = json.loads(d.get("context_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            try:
                tags = json.loads(d.get("tags_json", "[]"))
            except (json.JSONDecodeError, TypeError):
                tags = []
            if not isinstance(tags, list):
                tags = []
            people = ", ".join(
                t.replace("person:", "") for t in tags
                if isinstance(t, str) and t.startswith("person:")
            )
            out.append({
                "id": d["id"],
                "summary": d.get("summary", ""),
                "topics": ctx.get("topics", "") if isinstance(ctx, dict) else "",
                "open_loops": ctx.get("open_loops", "") if isinstance(ctx, dict) else "",
                "emotion": (ctx.get("emotion", ctx.get("mood", ""))
                           if isinstance(ctx, dict) else ""),
                "created_at": d.get("created_at", ""),
                "expires_at": d.get("expires_at", ""),
                "importance": int(ctx.get("importance", d.get("emotional_weight", 3)))
                              if isinstance(ctx, dict) else int(d.get("emotional_weight", 3)),
                "people": people,
                "status": "active" if not d.get("deleted_at") else "archived",
            })
        return out

    def list_important_episodes(
        self, person_id: str, min_importance: int = 4, limit: int = 10,
    ) -> list[dict]:
        """获取用户的重要情景记忆（高 emotional_weight 事件，不受过期限制）。

        参数:
            person_id:       用户 ID
            min_importance:  最低重要性（emotional_weight >= 此值）
            limit:           最多条数

        返回:
            按重要性降序、时间倒序排列的事件列表。
        """
        pid = str(person_id or "").strip()
        if not pid:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content AS summary, context_json,
                       created_at, expires_at, emotional_weight, tags_json
                FROM memory_items
                WHERE person_id=? AND kind IN ('episode','emotion')
                  AND deleted_at='' AND emotional_weight>=?
                ORDER BY emotional_weight DESC, created_at DESC LIMIT ?
                """,
                (pid, min_importance, limit),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            try:
                ctx = json.loads(d.get("context_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            try:
                tags = json.loads(d.get("tags_json", "[]"))
            except (json.JSONDecodeError, TypeError):
                tags = []
            if not isinstance(tags, list):
                tags = []
            people = ", ".join(
                t.replace("person:", "") for t in tags
                if isinstance(t, str) and t.startswith("person:")
            )
            out.append({
                "id": d["id"],
                "summary": d.get("summary", ""),
                "topics": ctx.get("topics", "") if isinstance(ctx, dict) else "",
                "open_loops": ctx.get("open_loops", "") if isinstance(ctx, dict) else "",
                "emotion": (ctx.get("emotion", ctx.get("mood", ""))
                           if isinstance(ctx, dict) else ""),
                "created_at": d.get("created_at", ""),
                "expires_at": d.get("expires_at", ""),
                "importance": int(ctx.get("importance", d.get("emotional_weight", 3)))
                              if isinstance(ctx, dict) else int(d.get("emotional_weight", 3)),
                "people": people,
                "status": "active",
            })
        return out

    def list_expired_recent_memory(self, device_id: str | None = None) -> list[dict]:
        """获取已过期的近期记忆行（用于 recent_memory_rollup_sweeper 归档到长期记忆）。

        从 memory_items 中查询 kind=episode/emotion 且 expires_at 已过期且未归档的记录。
        返回按 person_id 分组 + created_at 升序排列（方便批量处理）。
        """
        del device_id
        now = _utc_now()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, device_id, person_id, source_session, content AS summary,
                       context_json, created_at, expires_at
                FROM memory_items
                WHERE kind IN ('episode','emotion') AND deleted_at=''
                  AND expires_at != '' AND expires_at <= ?
                ORDER BY person_id, created_at ASC
                """,
                (now,),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            # 解析 context_json 中的 topics/open_loops 字段
            try:
                ctx = json.loads(d.get("context_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            d["topics"] = ctx.get("topics", "")
            d["open_loops"] = ctx.get("open_loops", "")
            out.append(d)
        return out

    def archive_recent_memory(self, memory_item_ids: list[str]) -> None:
        """批量归档近期记忆记录（设置 deleted_at，不删除数据）。"""
        if not memory_item_ids:
            return
        now = _utc_now()
        placeholders = ",".join("?" * len(memory_item_ids))
        with self._conn() as conn:
            conn.execute(
                f"UPDATE memory_items SET deleted_at=? WHERE id IN ({placeholders})",
                [now] + memory_item_ids,
            )

    def delete_recent_memory_for_person(self, person_id: str) -> int:
        """软删除指定用户的所有近期记忆（memory_items 中 kind=episode/emotion）。"""
        pid = str(person_id or "").strip()
        if not pid:
            raise ValueError("person_id required")
        now = _utc_now()
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE memory_items SET deleted_at=? WHERE person_id=? AND kind IN ('episode','emotion') AND deleted_at=''",
                (now, pid),
            )
            return cur.rowcount

    # ============================
    # 记忆关联图（Memory Relations）
    # ============================

    def upsert_memory_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        strength: float,
    ) -> None:
        """插入或更新记忆关联边。

        参数:
            from_id:       源记忆的 key（格式：memory:{uuid}）
            to_id:         目标记忆的 key
            relation_type: 关系类型（related, cause_effect, same_event 等）
            strength:      关联强度 0.0-1.0

        冲突策略（ON CONFLICT）：
          同一条边已存在时，只更新 strength 为两者中的较大值，
          保留原有强度不会意外降低。
        """
        fr = str(from_id or "").strip()
        to = str(to_id or "").strip()
        rel = str(relation_type or "related").strip().lower()
        if not fr or not to or fr == to:
            return
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memory_relations(from_id, to_id, relation_type, strength, created_at)
                VALUES (?,?,?,?,?)
                ON CONFLICT(from_id, to_id, relation_type) DO UPDATE SET
                    strength = CASE
                        WHEN excluded.strength > memory_relations.strength
                        THEN excluded.strength
                        ELSE memory_relations.strength
                    END
                """,
                (fr, to, rel, float(strength), _utc_now()),
            )

    def get_memory_relations(
        self,
        memory_keys: list[str],
        *,
        min_strength: float = 0.6,
        limit: int = 24,
    ) -> list[dict]:
        """获取与给定记忆键相关的关联边（双向查找：from 和 to 都查）。

        参数:
            memory_keys:  记忆键列表（格式：memory:{uuid}）
            min_strength: 最低关联强度阈值
            limit:        返回上限

        返回按 strength 降序排列。
        """
        keys = [str(k).strip() for k in memory_keys if str(k).strip()]
        if not keys:
            return []
        placeholders = ",".join("?" * len(keys))
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT from_id, to_id, relation_type, strength FROM (
                    SELECT from_id, to_id, relation_type, strength
                    FROM memory_relations
                    WHERE from_id IN ({placeholders}) AND strength >= ?
                    UNION
                    SELECT from_id, to_id, relation_type, strength
                    FROM memory_relations
                    WHERE to_id IN ({placeholders}) AND strength >= ?
                )
                ORDER BY strength DESC
                LIMIT ?
                """,
                (*keys, min_strength, *keys, min_strength, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_relations_for_keys(self, memory_keys: list[str]) -> int:
        """删除与给定记忆键相关的所有关联边（双向删除），返回删除数量。"""
        keys = [str(k).strip() for k in memory_keys if str(k).strip()]
        if not keys:
            return 0
        total = 0
        with self._conn() as conn:
            for key in keys:
                cur = conn.execute(
                    """
                    DELETE FROM memory_relations
                    WHERE from_id=? OR to_id=?
                    """,
                    (key, key),
                )
                total += int(cur.rowcount)
        return total

    def delete_relations_with_id_prefix(self, prefix: str) -> int:
        """删除具有指定 ID 前缀的关联边，返回删除数量。"""
        p = str(prefix or "").strip()
        if not p:
            return 0
        like = f"{p}%"
        with self._conn() as conn:
            cur = conn.execute(
                """
                DELETE FROM memory_relations
                WHERE from_id LIKE ? OR to_id LIKE ?
                """,
                (like, like),
            )
            return int(cur.rowcount)

    def purge_persona_derived_memory(
        self,
        person_id: str,
    ) -> dict[str, int]:
        """清除由 persona 语料导入生成的长期记忆块/关联边。

        参数:
            person_id:         用户 ID（通常是 persona_global）

        返回:
            dict: {"memory_items": 删除的记忆条目数, "relations": 删除的关联边数}

        用途：
          重新导入语料时（ingest --reset），需要先清除所有由旧语料派生的记忆。
        """
        pid = str(person_id or "").strip()
        stats = {"memory_items": 0, "relations": 0}
        if not pid:
            return stats

        # 收集该用户的 memory_items 中属于语料导入的记录
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, source_table, source_id FROM memory_items
                WHERE person_id=? AND (
                    source_table='corpus'
                    OR source IN ('ingest', '旧版_upsert')
                )
                """,
                (pid,),
            ).fetchall()

        item_ids = [str(r["id"]) for r in rows]

        rel_keys = [f"memory:{item_id}" for item_id in item_ids]
        if rel_keys:
            stats["relations"] = self.delete_relations_for_keys(rel_keys)

        # 删除 memory_items + FTS
        with self._conn() as conn:
            for iid in item_ids:
                conn.execute("DELETE FROM memory_items_fts WHERE id=?", (iid,))
                conn.execute("DELETE FROM memory_items WHERE id=?", (iid,))
                stats["memory_items"] += 1

        return stats

















    # ============================
    # 会话身份管理
    # ============================

    def _ensure_session(self, conn: sqlite3.Connection, session_id: str, device_id: str = "") -> None:
        """确保会话行存在（防御性：生产流程中 get_or_create_session 已保证存在）。

        当 identity 模块被直接调用（如测试）绕过 get_or_create_session 时，
        UPDATE 会静默失败。此方法用 INSERT OR IGNORE 兜底创建。
        """
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions(
                id, device_id, status, created_at, last_active, guest_turn_count
            ) VALUES (?,?,'active',?,?,0)
            """,
            (session_id, device_id, _utc_now(), _utc_now()),
        )

    def get_session_active_person_id(self, session_id: str) -> str | None:
        """获取会话当前绑定的用户 person_id。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT active_person_id FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        if not row or not row["active_person_id"]:
            return None
        return str(row["active_person_id"])

    def set_session_active_person(self, session_id: str, person_id: str | None) -> None:
        """设置会话当前绑定的用户 person_id（实名识别成功后调用）。"""
        with self._conn() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                "UPDATE sessions SET active_person_id=? WHERE id=?",
                (person_id, session_id),
            )

    def get_session_identity_pending(self, session_id: str) -> str | None:
        """获取会话待处理的身份识别载荷（JSON 字符串）。

        存储用户声明了名字但 ID 尚未验证的情况，
        如"我是张三"但没说 ID，系统记录下来等待后续补充。
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT identity_pending FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        if not row or not row["identity_pending"]:
            return None
        return str(row["identity_pending"])

    def set_session_identity_pending(self, session_id: str, payload: str | None) -> None:
        """设置/清除会话待处理的身份识别载荷。"""
        with self._conn() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                "UPDATE sessions SET identity_pending=? WHERE id=?",
                (payload, session_id),
            )

    def clear_session_identity_pending(self, session_id: str) -> None:
        self.set_session_identity_pending(session_id, None)

    def get_session_interlocutor_mode(self, session_id: str) -> str | None:
        with self._conn() as conn:
            self._ensure_session(conn, session_id)
            row = conn.execute(
                "SELECT interlocutor_mode FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        mode = str(row["interlocutor_mode"] or "").strip()
        return mode or None

    def set_session_interlocutor_mode(self, session_id: str, mode: str) -> None:
        with self._conn() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                "UPDATE sessions SET interlocutor_mode=? WHERE id=?",
                (mode, session_id),
            )

    def increment_guest_turn_count(self, session_id: str) -> int:
        """增加访客对话轮数计数，返回新的计数值。

        用于 guest_identity_reminder_every 逻辑：
        每 N 轮提醒一次访客实名。
        """
        with self._conn() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                """
                UPDATE sessions SET guest_turn_count = guest_turn_count + 1, last_active=?
                WHERE id=?
                """,
                (_utc_now(), session_id),
            )
            row = conn.execute(
                "SELECT guest_turn_count FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        return int(row["guest_turn_count"]) if row else 0

    def reset_guest_turn_count(self, session_id: str) -> None:
        """重置访客对话轮数计数为零。

        实名成功后调用，访客变为已实名用户，不再需要计数。
        """
        with self._conn() as conn:
            self._ensure_session(conn, session_id)
            conn.execute(
                "UPDATE sessions SET guest_turn_count=0 WHERE id=?",
                (session_id,),
            )

    # ============================
    # 人物画像管理
    # ============================

    def save_person_profile(self, device_id: str, profile: dict) -> None:
        """保存或更新用户画像数据（JSON 格式存储完整 Profile Card）。

        参数:
            device_id: 设备标识
            profile:   画像字典，必须包含 person_id 字段。
                       字段结构见 persona/config/profile_card.md。

        使用 INSERT ... ON CONFLICT DO UPDATE 实现 upsert。
        自动标准化画像（normalize_profile）以保证数据一致性。
        """
        from app.memory.profile import normalize_profile

        profile = normalize_profile(profile)
        person_id = str(profile["person_id"])
        now = _utc_now()
        created = str(profile.get("create_time") or profile.get("created_at") or now)
        profile["update_time"] = profile.get("update_time") or now
        profile["create_time"] = profile.get("create_time") or created
        payload = json.dumps(profile, ensure_ascii=False)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO person_profiles(person_id, device_id, profile_json, created_at, updated_at)
                VALUES (?,?,?,?,?)
                ON CONFLICT(person_id) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    updated_at=excluded.updated_at
                """,
                (person_id, device_id, payload, created, profile["update_time"]),
            )

    def get_person_profile(self, person_id: str) -> dict | None:
        """获取指定用户的画像数据，返回标准化后的字典。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT profile_json FROM person_profiles WHERE person_id=?",
                (person_id,),
            ).fetchone()
        if not row:
            return None
        from app.memory.profile import normalize_profile

        return normalize_profile(json.loads(row["profile_json"]))

    # ============================
    # 统一记忆库 —— memory_items（所有长期/中期记忆的唯一存储层）
    # ============================

    @staticmethod
    def _memory_item_content_hash(content: str) -> str:
        """计算 memory_items content_hash（SHA256 前 16 位）。"""
        import hashlib
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def memory_item_upsert(
        self,
        *,
        person_id: str,
        device_id: str,
        kind: str,
        source: str,
        visibility: str,
        content: str,
        confidence: float = 1.0,
        emotional_weight: int = 3,
        recency_weight: int = 3,
        context_json: str = "{}",
        tags_json: str = "[]",
        embedding_json: str = "[]",
        source_table: str = "",
        source_id: str = "",
        source_session: str = "",
        created_at: str = "",
        expires_at: str = "",
    ) -> str:
        """插入或更新统一记忆库条目（按 source_table+source_id 幂等）。

        Args:
            person_id: 用户 ID
            device_id: 设备 ID
            kind: MemoryKind.value
            source: MemorySource.value
            visibility: MemoryVisibility.value
            content: 记忆内容
            confidence: 置信度 0.0-1.0
            emotional_weight: 情感重要性 1-5
            recency_weight: 时效重要性 1-5
            context_json: JSON 序列化的 context dict
            tags_json: JSON 序列化的 tags list
            embedding_json: JSON 序列化的 embedding 数组
            source_table: 来源表名（如 "migration" / "admin" / 历史来源）
            source_id: 来源表的主键
            source_session: 来源会话 ID
            expires_at: 过期时间（ISO 格式）

        Returns:
            str: 写入的 memory_items id（自动生成的 UUID）。

        幂等性保证：
          - 当 source_table 和 source_id 均非空时，按 UNIQUE 索引 upsert
          - 同时更新 memory_items_fts 全文索引
        """
        import uuid as _uuid

        item_id = str(_uuid.uuid4())
        c_hash = self._memory_item_content_hash(content)
        now = _utc_now()
        c_at = created_at or now
        body_safe = " ".join((content or "").strip().split())

        with self._conn() as conn:
            # 检查是否已有相同 source_table+source_id 的记录
            existing_id: str | None = None
            if source_table and source_id:
                row = conn.execute(
                    "SELECT id FROM memory_items WHERE source_table=? AND source_id=?",
                    (source_table, source_id),
                ).fetchone()
                if row:
                    existing_id = str(row["id"])

            if existing_id:
                # 更新已有记录
                conn.execute(
                    """
                    UPDATE memory_items SET
                        person_id=?, device_id=?, kind=?, source=?, visibility=?,
                        content=?, content_hash=?, confidence=?,
                        emotional_weight=?, recency_weight=?,
                        context_json=?, tags_json=?, embedding_json=?,
                        source_session=?, expires_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        person_id, device_id, kind, source, visibility,
                        body_safe, c_hash, float(confidence),
                        int(emotional_weight), int(recency_weight),
                        context_json, tags_json, embedding_json,
                        source_session, expires_at, now,
                        existing_id,
                    ),
                )
                item_id = existing_id
            else:
                # 插入新记录
                conn.execute(
                    """
                    INSERT INTO memory_items(
                        id, person_id, device_id, kind, source, visibility,
                        content, content_hash, confidence,
                        emotional_weight, recency_weight,
                        context_json, tags_json, embedding_json,
                        source_table, source_id, source_session,
                        expires_at, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?, ?,?,?, ?,?,?,?,?, ?,?,?, ?,?,?)
                    """,
                    (
                        item_id, person_id, device_id, kind, source, visibility,
                        body_safe, c_hash, float(confidence),
                        int(emotional_weight), int(recency_weight),
                        context_json, tags_json, embedding_json,
                        source_table, source_id, source_session,
                        expires_at, c_at, now,
                    ),
                )

            # 更新 FTS5 全文索引（使用 bigram 预处理，兼容中文）
            try:
                from app.store.chunks import prepare_fts_text
                fts_content = prepare_fts_text(body_safe)
                conn.execute("DELETE FROM memory_items_fts WHERE id=?", (item_id,))
                conn.execute(
                    "INSERT INTO memory_items_fts(id, content_fts) VALUES (?,?)",
                    (item_id, fts_content),
                )
            except Exception as exc:
                logger.warning("memory_items_fts upsert failed for %s: %s", item_id, exc)

        return item_id

    def memory_item_get_by_source(
        self, source_table: str, source_id: str,
    ) -> dict | None:
        """按 source_table+source_id 查找 memory_items 条目。"""
        if not source_table or not source_id:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, person_id, device_id, kind, source, visibility,
                       content, content_hash, confidence,
                       emotional_weight, recency_weight,
                       context_json, tags_json, embedding_json,
                       source_table, source_id, source_session,
                       expires_at, created_at, updated_at, deleted_at
                FROM memory_items
                WHERE source_table=? AND source_id=?
                """,
                (source_table, source_id),
            ).fetchone()
        return dict(row) if row else None

    def memory_item_list_for_person(
        self,
        person_id: str,
        *,
        limit: int = 50,
        kinds: list[str] | None = None,
        visibility: str | None = None,
    ) -> list[dict]:
        """列出指定用户的 memory_items 条目。

        Args:
            person_id: 用户 ID
            limit: 返回上限
            kinds: 按 kind 筛选（如 ["fact", "episode"]）
            visibility: 按 visibility 筛选（如 "always", "recall_only"）
        """
        pid = str(person_id or "").strip()
        if not pid:
            return []
        where = "person_id=?"
        params: list[str] = [pid]
        if kinds:
            placeholders = ",".join("?" * len(kinds))
            where += f" AND kind IN ({placeholders})"
            params.extend(kinds)
        if visibility:
            where += " AND visibility=?"
            params.append(visibility)
        params.append(str(limit))
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT id, person_id, device_id, kind, source, visibility,
                       content, content_hash, confidence,
                       emotional_weight, recency_weight,
                       context_json, tags_json, embedding_json,
                       source_table, source_id, source_session,
                       expires_at, created_at, updated_at, deleted_at
                FROM memory_items
                WHERE {where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def memory_item_search(
        self,
        person_id: str,
        *,
        kinds: list[str] | None = None,
        visibility: str | None = None,
        query: str = "",
        month_key: str = "",
        limit: int = 20,
        embedding_json: str = "",
        embedding_weight: float = 0.0,
        extra_person_ids: list[str] | None = None,
        include_expired: bool = False,
    ) -> list[dict]:
        """在 memory_items 统一表中执行语义检索（FTS + embedding 兜底 + 跨库补召）。

        检索逻辑：
          1. 当 query 非空时，优先走 memory_items_fts 全文索引命中
          2. FTS 无命中但提供了 embedding_json 时，从同 person_id 候选中 cosine rerank 兜底
          3. 当 query 为空时，按 updated_at DESC + confidence/emotional_weight/recency_weight 排序
          4. 默认排除已过期条目（expires_at 非空且 <= 当前时间）

        Args:
            person_id: 主用户 ID
            kinds: 按 kind 筛选（如 ["fact", "episode", "emotion"]）
            visibility: 按 visibility 筛选（如 "always", "recall_only"）
            query: FTS 查询文本
            month_key: 月份标识（如 "2025-06"），只返回包含此月份的条目
            limit: 返回上限
            embedding_json: 用于 rerank 的 query embedding 序列化 JSON
            embedding_weight: embedding rerank 的权重（0.0=仅 FTS，1.0=仅向量）
            extra_person_ids: 额外 person_id 列表，用于跨用户补召（persona/corpus 语料）
            include_expired: 是否包含已过期的条目（默认排除）

        Returns:
            list[dict]: memory_items 行列表，按相关性降序排列。
        """
        pids: list[str] = []
        pid = str(person_id or "").strip()
        if pid:
            pids.append(pid)
        if extra_person_ids:
            for ep in extra_person_ids:
                ep_s = str(ep or "").strip()
                if ep_s and ep_s not in pids:
                    pids.append(ep_s)
        if not pids:
            return []
        include_global_scope = bool(pid) and visibility != "always"

        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()

        def _month_needles(mk: str) -> list[str]:
            mk = str(mk or "").strip()
            if not mk:
                return []
            needles = [mk]
            if "-" in mk:
                year, month = mk.split("-", 1)
                if year.isdigit() and month.isdigit():
                    month_i = int(month)
                    needles.extend([
                        f"{year}年{month_i}月",
                        f"{year}年{month_i:02d}月",
                        f"{year}-{month_i}月",
                        f"{year}-{month_i:02d}月",
                    ])
            return list(dict.fromkeys(needles))

        def _row_has_month(row: dict, mk: str) -> bool:
            needles = _month_needles(mk)
            if not needles:
                return True
            haystack = " ".join(str(row.get(k, "") or "") for k in (
                "content", "source", "source_table", "source_id",
                "context_json", "tags_json",
            ))
            return any(n in haystack for n in needles)

        def _month_row_priority(row: dict) -> tuple[int, int, str]:
            """月份候选的轻量排序，保证关系查询不会被朋友/人物卡抢占。"""
            import re as _re

            q = str(query or "")
            haystack = " ".join(str(row.get(k, "") or "") for k in (
                "content", "source", "source_table", "source_id",
                "context_json", "tags_json",
            ))
            asks_relationship = bool(_re.search(
                r"我俩|我们俩|我们之间|咱俩|我和她|我和远慧|我跟你|远慧|刘远慧|秋雨",
                q,
            ))
            asks_friend = bool(_re.search(r"唐凯|伍钰涛|朋友群|兄弟们|朋友", q))
            if asks_relationship and asks_friend:
                asks_friend = False
            is_relationship = "刘远慧" in haystack or "远慧" in haystack
            is_friend = any(token in haystack for token in ("唐凯", "伍钰涛", "袁子翔", "朋友群"))
            if asks_relationship:
                group = 0 if is_relationship else 1 if is_friend else 2
            elif asks_friend:
                group = 0 if is_friend else 1 if is_relationship else 2
            else:
                group = 0 if is_relationship else 1 if is_friend else 2
            table_rank = 0 if str(row.get("source_table", "")) == "corpus" else 1
            return (group, table_rank, str(row.get("source_id", "")))

        def _build_person_clause() -> str:
            """构建 person_id IN (...) 子句。"""
            parts: list[str] = []
            if include_global_scope:
                parts.extend(["mi.person_id IS NULL", "mi.person_id=''"])
            if len(pids) == 1:
                parts.append("mi.person_id=?")
            else:
                placeholders = ",".join("?" * len(pids))
                parts.append(f"mi.person_id IN ({placeholders})")
            return "(" + " OR ".join(parts) + ")"

        def _build_where_extra() -> tuple[str, list]:
            """构建 kind/visibility/expires_at/deleted 筛选子句。"""
            clauses: list[str] = []
            params: list = list(pids)
            if kinds:
                placeholders = ",".join("?" * len(kinds))
                clauses.append(f"mi.kind IN ({placeholders})")
                params.extend(kinds)
            if visibility:
                clauses.append("mi.visibility=?")
                params.append(visibility)
            clauses.append("(mi.deleted_at IS NULL OR mi.deleted_at = '')")
            if not include_expired:
                clauses.append("(mi.expires_at IS NULL OR mi.expires_at = '' OR mi.expires_at > ?)")
                params.append(now_iso)
            return " AND ".join(clauses), params

        with self._conn() as conn:
            # ── 路径 A：FTS 全文检索 ──
            candidates: list[dict] = []
            if query.strip():
                from app.store.chunks import prepare_fts_text
                fts_q = prepare_fts_text(query)
                if fts_q.strip():
                    fts_terms = fts_q.split()
                    seen: set[str] = set()
                    unique_terms: list[str] = []
                    for t in fts_terms:
                        t_norm = t.strip().lower()
                        if t_norm and t_norm not in seen:
                            seen.add(t_norm)
                            unique_terms.append(t_norm)
                    if unique_terms:
                        fts_match = " OR ".join(
                            f'"{t}"' if " " not in t else t
                            for t in unique_terms
                        )
                        person_clause = _build_person_clause()
                        extra_clause, extra_params = _build_where_extra()
                        # 去掉已加到 extra_params 的 pids（保留唯一一份）
                        extra_params_for_fts = extra_params[len(pids):]
                        fts_rows = conn.execute(
                            f"""
                            SELECT mi.id, mi.person_id, mi.device_id, mi.kind, mi.source,
                                   mi.visibility, mi.content, mi.content_hash, mi.confidence,
                                   mi.emotional_weight, mi.recency_weight,
                                   mi.context_json, mi.tags_json, mi.embedding_json,
                                   mi.source_table, mi.source_id, mi.source_session,
                                   mi.expires_at, mi.created_at, mi.updated_at, mi.deleted_at
                            FROM memory_items mi
                            JOIN memory_items_fts fts ON mi.id = fts.id
                            WHERE {person_clause}
                              AND {extra_clause}
                              AND fts.content_fts MATCH ?
                            ORDER BY mi.updated_at DESC
                            LIMIT ?
                            """,
                            (*pids, *extra_params_for_fts, fts_match, limit * 3),
                        ).fetchall()
                        candidates = [dict(r) for r in fts_rows]

                # ── FTS 无命中且提供了 embedding_json → embedding 兜底 ──
                if not candidates and embedding_json:
                    try:
                        import json as _json
                        q_emb = _json.loads(embedding_json)
                        if q_emb and isinstance(q_emb, list) and len(q_emb) > 0:
                            from app.llm import cosine_similarity
                            # 从同 person_ids 的候选池中按 kind/visibility 筛选
                            extra_clause2, extra_params2 = _build_where_extra()
                            extra_params_for_emb = extra_params2[len(pids):]
                            pool_rows = conn.execute(
                                f"""
                                SELECT mi.id, mi.person_id, mi.device_id, mi.kind, mi.source,
                                       mi.visibility, mi.content, mi.content_hash, mi.confidence,
                                       mi.emotional_weight, mi.recency_weight,
                                       mi.context_json, mi.tags_json, mi.embedding_json,
                                       mi.source_table, mi.source_id, mi.source_session,
                                       mi.expires_at, mi.created_at, mi.updated_at, mi.deleted_at
                                FROM memory_items mi
                                WHERE {_build_person_clause()}
                                  AND {extra_clause2}
                                  AND mi.embedding_json IS NOT NULL
                                  AND mi.embedding_json != ''
                                  AND mi.embedding_json != '[]'
                                ORDER BY mi.updated_at DESC
                                LIMIT ?
                                """,
                                (*pids, *extra_params_for_emb, limit * 5),
                            ).fetchall()
                            pool = [dict(r) for r in pool_rows]
                            if pool:
                                scored: list[tuple[dict, float]] = []
                                for row in pool:
                                    try:
                                        row_emb = _json.loads(str(row.get("embedding_json", "[]") or "[]"))
                                        if row_emb and len(row_emb) > 0 and len(q_emb) > 0:
                                            sim = cosine_similarity(q_emb, row_emb)
                                            scored.append((row, sim))
                                        else:
                                            scored.append((row, 0.0))
                                    except (ValueError, TypeError, _json.JSONDecodeError):
                                        scored.append((row, 0.0))
                                lt_thresh = settings.long_term_memory_sim_threshold
                                scored.sort(key=lambda x: -x[1])
                                candidates = [r for r, s in scored
                                              if s >= lt_thresh or s > 0.0]
                    except Exception:
                        logger.warning("memory_item_search embedding fallback failed", exc_info=True)
            else:
                # ── 路径 B：无 FTS，直接按条件筛选 ──
                person_parts: list[str] = []
                if include_global_scope:
                    person_parts.extend(["person_id IS NULL", "person_id=''"])
                if len(pids) == 1:
                    person_parts.append("person_id=?")
                else:
                    person_parts.append(f"person_id IN ({','.join('?' * len(pids))})")
                person_where = "(" + " OR ".join(person_parts) + ")"
                where = f"{person_where} AND (deleted_at IS NULL OR deleted_at='')"
                params: list[str | int] = list(pids)
                if kinds:
                    placeholders = ",".join("?" * len(kinds))
                    where += f" AND kind IN ({placeholders})"
                    params.extend(kinds)
                if visibility:
                    where += " AND visibility=?"
                    params.append(visibility)
                if not include_expired:
                    where += " AND (expires_at IS NULL OR expires_at = '' OR expires_at > ?)"
                    params.append(now_iso)
                params.append(str(limit))
                rows = conn.execute(
                    f"""
                    SELECT id, person_id, device_id, kind, source, visibility,
                           content, content_hash, confidence,
                           emotional_weight, recency_weight,
                           context_json, tags_json, embedding_json,
                           source_table, source_id, source_session,
                           expires_at, created_at, updated_at, deleted_at
                    FROM memory_items
                    WHERE {where}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
                candidates = [dict(r) for r in rows]

            # ── 月份过滤（FTS 结果可能包含非月份内容） ──
            if month_key and candidates:
                filtered: list[dict] = []
                for c in candidates:
                    if _row_has_month(c, month_key):
                        filtered.append(c)
                if filtered:
                    candidates = filtered
                else:
                    # ── FTS 候选没有任何月份匹配 → 执行 month_key 专用补召 ──
                    # 用 month_key 重新执行 FTS 后仍需强过滤，避免 "20260702"
                    # 这类包含 2025/06 分词但并非目标月份的 近期记忆 rollup 抢占结果。
                    from app.store.chunks import build_fts_match_query

                    month_fts = build_fts_match_query(month_key)
                    if month_fts:
                        person_clause_month = _build_person_clause()
                        extra_clause_month, extra_params_month = _build_where_extra()
                        extra_params_for_fts_month = extra_params_month[len(pids):]
                        month_rows = conn.execute(
                            f"""
                            SELECT mi.id, mi.person_id, mi.device_id, mi.kind, mi.source,
                                   mi.visibility, mi.content, mi.content_hash, mi.confidence,
                                   mi.emotional_weight, mi.recency_weight,
                                   mi.context_json, mi.tags_json, mi.embedding_json,
                                   mi.source_table, mi.source_id, mi.source_session,
                                   mi.expires_at, mi.created_at, mi.updated_at, mi.deleted_at
                            FROM memory_items mi
                            JOIN memory_items_fts fts ON mi.id = fts.id
                            WHERE {person_clause_month}
                              AND {extra_clause_month}
                              AND fts.content_fts MATCH ?
                            ORDER BY mi.updated_at DESC
                            LIMIT ?
                            """,
                            (*pids, *extra_params_for_fts_month, month_fts, limit),
                        ).fetchall()
                        candidates = [
                            dict(r) for r in month_rows
                            if _row_has_month(dict(r), month_key)
                        ]

                    if len(candidates) < limit:
                        needles = _month_needles(month_key)
                        if needles:
                            person_clause_like = _build_person_clause()
                            extra_clause_like, extra_params_like = _build_where_extra()
                            extra_params_for_like = extra_params_like[len(pids):]
                            like_clauses: list[str] = []
                            like_params: list[str] = []
                            for n in needles:
                                pat = f"%{n}%"
                                like_clauses.append(
                                    "(mi.content LIKE ? OR mi.source LIKE ? OR mi.source_id LIKE ? "
                                    "OR mi.context_json LIKE ? OR mi.tags_json LIKE ?)"
                                )
                                like_params.extend([pat, pat, pat, pat, pat])
                            like_rows = conn.execute(
                                f"""
                                SELECT mi.id, mi.person_id, mi.device_id, mi.kind, mi.source,
                                       mi.visibility, mi.content, mi.content_hash, mi.confidence,
                                       mi.emotional_weight, mi.recency_weight,
                                       mi.context_json, mi.tags_json, mi.embedding_json,
                                       mi.source_table, mi.source_id, mi.source_session,
                                       mi.expires_at, mi.created_at, mi.updated_at, mi.deleted_at
                                FROM memory_items mi
                                WHERE {person_clause_like}
                                  AND {extra_clause_like}
                                  AND ({" OR ".join(like_clauses)})
                                ORDER BY mi.updated_at DESC
                                LIMIT ?
                                """,
                                (*pids, *extra_params_for_like, *like_params, max(limit * 5, 20)),
                            ).fetchall()
                            seen_ids = {str(c.get("id", "")) for c in candidates}
                            for r in like_rows:
                                row = dict(r)
                                rid = str(row.get("id", ""))
                                if rid and rid not in seen_ids:
                                    seen_ids.add(rid)
                                    candidates.append(row)

                if len(candidates) < limit:
                    needles = _month_needles(month_key)
                    if needles:
                        person_clause_like = _build_person_clause()
                        extra_clause_like, extra_params_like = _build_where_extra()
                        extra_params_for_like = extra_params_like[len(pids):]
                        like_clauses: list[str] = []
                        like_params: list[str] = []
                        for n in needles:
                            pat = f"%{n}%"
                            like_clauses.append(
                                "(mi.content LIKE ? OR mi.source LIKE ? OR mi.source_id LIKE ? "
                                "OR mi.context_json LIKE ? OR mi.tags_json LIKE ?)"
                            )
                            like_params.extend([pat, pat, pat, pat, pat])
                        like_rows = conn.execute(
                            f"""
                            SELECT mi.id, mi.person_id, mi.device_id, mi.kind, mi.source,
                                   mi.visibility, mi.content, mi.content_hash, mi.confidence,
                                   mi.emotional_weight, mi.recency_weight,
                                   mi.context_json, mi.tags_json, mi.embedding_json,
                                   mi.source_table, mi.source_id, mi.source_session,
                                   mi.expires_at, mi.created_at, mi.updated_at, mi.deleted_at
                            FROM memory_items mi
                            WHERE {person_clause_like}
                              AND {extra_clause_like}
                              AND ({" OR ".join(like_clauses)})
                            ORDER BY mi.updated_at DESC
                            LIMIT ?
                            """,
                            (*pids, *extra_params_for_like, *like_params, max(limit * 5, 20)),
                        ).fetchall()
                        seen_ids = {str(c.get("id", "")) for c in candidates}
                        for r in like_rows:
                            row = dict(r)
                            rid = str(row.get("id", ""))
                            if rid and rid not in seen_ids:
                                seen_ids.add(rid)
                                candidates.append(row)

                candidates.sort(key=_month_row_priority)

            # ── 结果数量控制 ──
            if len(candidates) > limit:
                candidates = candidates[:limit]

        return candidates

    def memory_item_counts(self, person_id: str = "") -> dict:
        """统计 memory_items 表中各 kind 的条目数和总新增数。

        Args:
            person_id: 可选，按用户过滤；为空时统计全局

        Returns:
            dict: {"total": N, "active": N, "by_kind": {"fact": N, ...}, "by_source_table": {...}}
        """
        where = "WHERE 1=1"
        params: list[str] = []
        if person_id:
            where += " AND person_id=?"
            params.append(person_id)
        with self._conn() as conn:
            # 总量
            total_row = conn.execute(
                f"SELECT COUNT(*) AS n FROM memory_items {where}", params,
            ).fetchone()
            total = int(total_row["n"]) if total_row else 0

            # 按 kind 分组
            kind_rows = conn.execute(
                f"""
                SELECT kind, COUNT(*) AS n FROM memory_items {where}
                GROUP BY kind ORDER BY n DESC
                """,
                params,
            ).fetchall()
            by_kind = {str(r["kind"]): int(r["n"]) for r in kind_rows}

            # 按 source_table 分组
            source_rows = conn.execute(
                f"""
                SELECT source_table, COUNT(*) AS n FROM memory_items {where}
                GROUP BY source_table ORDER BY n DESC
                """,
                params,
            ).fetchall()
            by_source = {str(r["source_table"]) or "none": int(r["n"]) for r in source_rows}

            # 未软删除条数（deleted_at 为空）
            active_where = f"{where} AND (deleted_at IS NULL OR deleted_at = '')"
            active_row = conn.execute(
                f"SELECT COUNT(*) AS n FROM memory_items {active_where}", params,
            ).fetchone()
            active = int(active_row["n"]) if active_row else 0

        return {
            "total": total,
            "active": active,
            "by_kind": by_kind,
            "by_source_table": by_source,
        }

    # ── Unified Memory 语义接口 ────────────────────────────────────────

    def write_memory_item(
        self,
        *,
        person_id: str,
        device_id: str,
        kind: str,
        source: str,
        visibility: str,
        content: str,
        confidence: float = 1.0,
        emotional_weight: int = 3,
        recency_weight: int = 3,
        context_json: str = "{}",
        tags_json: str = "[]",
        embedding_json: str = "[]",
        source_table: str = "",
        source_id: str = "",
        source_session: str = "",
        created_at: str = "",
        expires_at: str = "",
    ) -> str:
        """写入统一记忆库条目（语义名，同 memory_item_upsert）。"""
        return self.memory_item_upsert(
            person_id=person_id, device_id=device_id, kind=kind,
            source=source, visibility=visibility, content=content,
            confidence=confidence, emotional_weight=emotional_weight,
            recency_weight=recency_weight, context_json=context_json,
            tags_json=tags_json, embedding_json=embedding_json,
            source_table=source_table, source_id=source_id,
            source_session=source_session, created_at=created_at,
            expires_at=expires_at,
        )

    upsert_memory_item = write_memory_item  # 同名别名

    def get_memory_item(self, item_id: str) -> dict | None:
        """通过 ID 获取统一记忆库中的单条记录。"""
        if not item_id:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, person_id, device_id, kind, source, visibility,
                       content, content_hash, confidence,
                       emotional_weight, recency_weight,
                       context_json, tags_json, embedding_json,
                       source_table, source_id, source_session,
                       expires_at, created_at, updated_at, deleted_at
                FROM memory_items WHERE id=?
                """,
                (item_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_memory_items(
        self,
        person_id: str,
        *,
        limit: int = 50,
        kinds: list[str] | None = None,
        visibility: str | None = None,
    ) -> list[dict]:
        """列出指定用户的统一记忆库条目（语义名，同 memory_item_list_for_person）。"""
        return self.memory_item_list_for_person(
            person_id, limit=limit, kinds=kinds, visibility=visibility,
        )

    def search_memory_items(
        self,
        person_id: str,
        *,
        kinds: list[str] | None = None,
        visibility: str | None = None,
        query: str = "",
        month_key: str = "",
        limit: int = 20,
        embedding_json: str = "",
        embedding_weight: float = 0.0,
        extra_person_ids: list[str] | None = None,
        include_expired: bool = False,
    ) -> list[dict]:
        """在统一记忆库中执行语义检索（语义名，同 memory_item_search）。"""
        return self.memory_item_search(
            person_id, kinds=kinds, visibility=visibility,
            query=query, month_key=month_key, limit=limit,
            embedding_json=embedding_json, embedding_weight=embedding_weight,
            extra_person_ids=extra_person_ids, include_expired=include_expired,
        )

    def count_memory_items(self, person_id: str = "") -> dict:
        """统计统一记忆库中各 kind 的条目数（语义名，同 memory_item_counts）。"""
        return self.memory_item_counts(person_id=person_id)

    def count_corpus_memory_items(self) -> int:
        """统计由 persona/corpus/ 导入的长期语料块数量。

        识别规则：
          - source_table='corpus'（新规范，所有 corpus 导入行标记为 corpus）
          - 未被软删除
        """
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM memory_items
                WHERE (deleted_at IS NULL OR deleted_at = '')
                  AND source_table='corpus'
                """
            ).fetchone()
        return int(row["n"]) if row else 0

    def archive_memory_item(self, item_id: str) -> bool:
        """软归档指定记忆条目（标记 deleted_at，不删除）。

        管理员/调试用途，非运行时路径。
        """
        if not item_id:
            return False
        now = _utc_now()
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE memory_items SET deleted_at=? WHERE id=? AND (deleted_at IS NULL OR deleted_at='')",
                (now, item_id),
            )
            return cur.rowcount > 0

    def delete_memory_item(self, item_id: str) -> bool:
        """硬删除指定记忆条目（仅管理/调试用途）。"""
        if not item_id:
            return False
        with self._conn() as conn:
            conn.execute("DELETE FROM memory_items_fts WHERE id=?", (item_id,))
            cur = conn.execute("DELETE FROM memory_items WHERE id=?", (item_id,))
            return cur.rowcount > 0

    def reset_corpus_items(self) -> int:
        """清理所有 source_table='corpus' 的记忆条目及对应 FTS。

        Returns:
            删除的条目数。
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM memory_items WHERE source_table='corpus'"
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM memory_items_fts WHERE id IN ({placeholders})",
                    ids,
                )
                conn.execute(
                    f"DELETE FROM memory_items WHERE source_table='corpus'",
                )
            return len(ids)

    def list_corpus_source_ids(self) -> list[str]:
        """返回所有 source_table='corpus' 的非空 source_id 列表。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source_id FROM memory_items "
                "WHERE source_table='corpus' AND source_id != ''"
            ).fetchall()
            return [str(r["source_id"]) for r in rows]

    def delete_corpus_item_by_source_id(self, source_id: str) -> bool:
        """按 source_id 删除一条 corpus 条目（同时删 FTS）。

        对于 source_id 重复行仅删除第一条，重复行应在清理后用 audit 检测。"""
        row = self.memory_item_get_by_source("corpus", source_id)
        if row:
            return self.delete_memory_item(str(row["id"]))
        return False

    def list_corpus_source_id_counts(self) -> dict[str, int]:
        """统计 source_table='corpus' 下每个 source_id 的物理行数。

        Returns:
            dict[str, int]: {source_id: count, ...}
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source_id, COUNT(*) AS n FROM memory_items "
                "WHERE source_table='corpus' AND source_id != '' "
                "GROUP BY source_id"
            ).fetchall()
            return {str(r["source_id"]): int(r["n"]) for r in rows}

    def dedup_corpus_source_ids(self) -> int:
        """去重 source_table='corpus' 下重复的 source_id 行。

        保留每个 source_id 的第一个写入行（按 rowid 最早），删除多余的重复行。
        自动清除对应的 FTS 条目。

        Returns:
            删除的冗余行数（无重复返回 0）。
        """
        with self._conn() as conn:
            # 按 source_id 分组，每组只保留 rowid 最小的那条
            keep_rows = conn.execute(
                "SELECT source_id, MIN(rowid) AS r FROM memory_items "
                "WHERE source_table='corpus' AND source_id != '' "
                "GROUP BY source_id "
                "HAVING COUNT(*) > 1"
            ).fetchall()

            if not keep_rows:
                return 0

            total_deleted = 0
            for kr in keep_rows:
                sid = kr["source_id"]
                keep_r = int(kr["r"])
                # 删除该 source_id 下 rowid 不等于最小的行
                extra = conn.execute(
                    "SELECT id FROM memory_items "
                    "WHERE source_table='corpus' AND source_id=? "
                    "AND rowid != ?",
                    (sid, keep_r),
                ).fetchall()
                extra_ids = [r["id"] for r in extra]
                if extra_ids:
                    ph = ",".join("?" * len(extra_ids))
                    conn.execute(
                        f"DELETE FROM memory_items_fts WHERE id IN ({ph})",
                        extra_ids,
                    )
                    conn.execute(
                        "DELETE FROM memory_items WHERE source_table='corpus' AND source_id=? AND rowid != ?",
                        (sid, keep_r),
                    )
                    total_deleted += len(extra_ids)
            return total_deleted

    def delete_corpus_items_by_source_path(self, source_path: str) -> int:
        """删除与指定源文件相关的所有 corpus 条目（source_id LIKE '<path>#%'）。

        Args:
            source_path: 相对路径，如 'monthly/liu_yuanhui/2025-04.md'

        Returns:
            删除的条目数。
        """
        import sqlite3
        pattern = f"{source_path}#%"
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM memory_items "
                "WHERE source_table='corpus' AND source_id LIKE ?",
                (pattern,),
            ).fetchall()
            if not rows:
                return 0
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"DELETE FROM memory_items_fts WHERE id IN ({placeholders})",
                ids,
            )
            conn.execute(
                f"DELETE FROM memory_items WHERE id IN ({placeholders})",
                ids,
            )
            return len(ids)

    def list_corpus_source_ids_by_path(self, source_path: str) -> list[str]:
        """返回与指定源文件相关的所有 corpus source_id 列表。

        Args:
            source_path: 相对路径，如 'monthly/liu_yuanhui/2025-04.md'

        Returns:
            list[str]: source_id 列表（保留重复条目）。
        """
        pattern = f"{source_path}#%"
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source_id FROM memory_items "
                "WHERE source_table='corpus' AND source_id LIKE ?",
                (pattern,),
            ).fetchall()
            return [str(r["source_id"]) for r in rows]

    # ── 语义视图（均基于 memory_items 统一表） ─────────────────────────

    def list_core_facts(
        self, person_id: str, *, limit: int = 50,
    ) -> list[dict]:
        """列出核心事实：仅返回稳定、高置信、适合常驻注入的 memory_items。

        筛选规则：
          - visibility=always（常驻注入）
          - 按 kind 优先级排序（taboo > preference > relationship > entity > milestone）
        """
        pid = str(person_id or "").strip()
        if not pid:
            return []
        rows = self.memory_item_search(
            pid, kinds=None, visibility="always", limit=limit,
        )
        _CORE_PRIORITY = {
            "taboo": 0, "preference": 1, "relationship": 2,
            "entity": 3, "milestone": 4,
        }
        rows.sort(key=lambda r: _CORE_PRIORITY.get(r.get("kind", ""), 9))
        return rows[:limit]

    def search_recent_memory(
        self,
        person_id: str,
        query: str = "",
        *,
        limit: int = 20,
        month_key: str = "",
    ) -> list[dict]:
        """近期记忆语义视图：只查 episode / emotion / open_loop / milestone。

        用于获取近期发生的事件、情绪变化和活跃待办。
        """
        pid = str(person_id or "").strip()
        if not pid:
            return []
        return self.memory_item_search(
            pid,
            kinds=["episode", "emotion", "milestone"],
            visibility="recall_only",
            query=query,
            month_key=month_key,
            limit=limit,
        )

    def search_long_term_memory(
        self,
        person_id: str,
        query: str = "",
        *,
        limit: int = 20,
        month_key: str = "",
        embedding_json: str = "",
        extra_person_ids: list[str] | None = None,
    ) -> list[dict]:
        """长期记忆语义视图：查 fact / entity / wiki / relationship / episode。

        用于获取可长期召回的稳定知识、实体关系和里程碑事件。
        """
        pid = str(person_id or "").strip()
        if not pid:
            return []
        return self.memory_item_search(
            pid,
            kinds=["fact", "entity", "wiki", "relationship", "episode"],
            visibility="recall_only",
            query=query,
            month_key=month_key,
            limit=limit,
            embedding_json=embedding_json,
            extra_person_ids=extra_person_ids,
        )

    # ============================
    # 后台管理辅助方法
    # ============================

    def list_all_sessions(self, limit: int = 50, offset: int = 0, status: str | None = None) -> list[dict]:
        """后台管理：列出所有会话（分页）。"""
        where = "WHERE 1=1"
        params: list[str] = []
        if status:
            where += " AND status=?"
            params.append(status)
        params.append(str(limit))
        params.append(str(offset))
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT id, device_id, status, created_at, last_active,
                       active_person_id, guest_turn_count, interlocutor_mode
                FROM sessions {where}
                ORDER BY last_active DESC LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def count_sessions(self, status: str | None = None) -> int:
        """统计会话数量。"""
        where = "WHERE 1=1"
        params: list[str] = []
        if status:
            where += " AND status=?"
            params.append(status)
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM sessions {where}", params
            ).fetchone()
        return int(row["n"]) if row else 0

    def get_session_by_id(self, session_id: str) -> dict | None:
        """后台管理：获取单个会话详情。"""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, device_id, status, created_at, last_active,
                       active_person_id, guest_turn_count, interlocutor_mode
                FROM sessions WHERE id=?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def count_memory_stat(self) -> dict:
        """后台管理：统计统一记忆库、画像、会话等全局数据量。"""
        with self._conn() as conn:
            mi_total = conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]
            mi_active = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE deleted_at IS NULL OR deleted_at=''"
            ).fetchone()[0]
            profiles = conn.execute("SELECT COUNT(*) FROM person_profiles").fetchone()[0]
            sessions_active = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE status='active'"
            ).fetchone()[0]

            # 统一语义统计
            core = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE visibility='always' AND deleted_at=''"
            ).fetchone()[0]
            episodes = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE kind IN ('episode','emotion') AND deleted_at=''"
            ).fetchone()[0]
            long_term = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE kind NOT IN ('episode','emotion') AND visibility='recall_only' AND deleted_at=''"
            ).fetchone()[0]

            by_kind: dict[str, int] = {}
            for r in conn.execute(
                "SELECT kind, COUNT(*) AS n FROM memory_items WHERE deleted_at='' GROUP BY kind ORDER BY n DESC"
            ).fetchall():
                by_kind[str(r["kind"])] = r["n"]

        return {
            "memory_items": mi_total,
            "memory_items_active": mi_active,
            "core_memories": core,
            "episodes_total": episodes,
            "episodes_active": episodes,
            "long_term_memory": long_term,
            "profiles": profiles,
            "active_sessions": sessions_active,
            "by_kind": by_kind,
        }

    # ============================
    # 画像批量操作
    # ============================

    def list_all_person_profiles(self) -> list[dict]:
        """获取所有用户的画像数据（跨设备，按更新时间降序）。

        返回: [{"person_id": ..., "device_id": ..., "profile": ..., "updated_at": ...}, ...]
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    rowid,
                    CAST(person_id AS BLOB) AS person_id_blob,
                    CAST(device_id AS BLOB) AS device_id_blob,
                    profile_json,
                    CAST(updated_at AS BLOB) AS updated_at_blob
                FROM person_profiles ORDER BY updated_at DESC
                """
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            rowid = int(r["rowid"])
            person_id = _decode_text_cell(r["person_id_blob"], column="person_id", rowid=rowid)
            device_id = _decode_text_cell(r["device_id_blob"], column="device_id", rowid=rowid)
            updated_at = _decode_text_cell(r["updated_at_blob"], column="updated_at", rowid=rowid)
            if person_id is None or device_id is None or updated_at is None:
                continue
            try:
                profile = json.loads(r["profile_json"])
            except (TypeError, json.JSONDecodeError):
                logger.warning("跳过损坏的人物画像 JSON rowid=%s person_id=%s", rowid, person_id)
                continue
            out.append(
                {
                    "person_id": person_id,
                    "device_id": device_id,
                    "profile": profile,
                    "updated_at": updated_at,
                }
            )
        return out

    def list_recent_memory_since(
        self, person_id: str, since_iso: str, *, limit: int = 40
    ) -> list[dict]:
        """获取用户在指定时间点后创建的近期记忆（用于 Profile 增量更新）。

        从 memory_items 中查询。
        """
        pid = str(person_id or "").strip()
        if not pid:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content AS summary, context_json, created_at, expires_at
                FROM memory_items
                WHERE person_id=? AND kind IN ('episode','emotion')
                  AND deleted_at='' AND created_at >= ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (pid, since_iso, limit),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            try:
                ctx = json.loads(d.get("context_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            out.append({
                "id": d["id"],
                "summary": d.get("summary", ""),
                "topics": ctx.get("topics", "") if isinstance(ctx, dict) else "",
                "open_loops": ctx.get("open_loops", "") if isinstance(ctx, dict) else "",
                "created_at": d.get("created_at", ""),
                "expires_at": d.get("expires_at", ""),
            })
        return out

    def list_facts_since(
        self, person_id: str, since_iso: str, *, limit: int = 60
    ) -> list[dict]:
        """获取用户在指定时间点后创建的长期记忆块（基于 memory_items，用于 Profile 增量更新）。"""
        pid = str(person_id or "").strip()
        if not pid:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT content AS fact, kind AS category, confidence, created_at
                FROM memory_items
                WHERE person_id=? AND kind NOT IN ('episode','emotion')
                  AND deleted_at='' AND created_at >= ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (pid, since_iso, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_recall_person_ids(self, *, min_count: int = 3) -> list[str]:
        """列出有足够记忆条目的所有用户 ID。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT person_id, COUNT(*) AS cnt FROM memory_items
                WHERE kind NOT IN ('episode','emotion') AND deleted_at='' AND person_id != ''
                GROUP BY person_id HAVING cnt >= ?
                ORDER BY cnt DESC
                """,
                (min_count,),
            ).fetchall()
        return [str(r["person_id"]) for r in rows]

    # ═════════════════════════════════════════════════════════════════════════
    # 长期记忆查询接口
    # ═════════════════════════════════════════════════════════════════════════

    def list_person_long_term_memory(
        self, person_id: str, *, device_id: str = "", limit: int = 50
    ) -> list[dict]:
        """列出某用户的长期记忆项（从统一记忆库 memory_items 中查询）。"""
        pid = str(person_id or "").strip()
        if not pid:
            return []
        del device_id
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content AS text, kind AS category, confidence,
                       source, created_at, context_json, tags_json
                FROM memory_items
                WHERE person_id=? AND kind NOT IN ('episode','emotion')
                  AND deleted_at=''
                ORDER BY created_at DESC LIMIT ?
                """,
                (pid, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_memory_items_detailed(
        self, collection: str | None = None, person_id: str | None = None,
        *, limit: int = 100, offset: int = 0,
    ) -> list[dict]:
        """后台管理：列出统一记忆项。"""
        where = "WHERE deleted_at=''"
        params: list[str] = []
        if person_id:
            where += " AND person_id=?"
            params.append(person_id)
        if collection:
            where += " AND kind=?"
            params.append(collection)
        params.extend([str(limit), str(offset)])
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT id, content AS text, kind AS collection, device_id, person_id,
                       source, kind AS category, confidence, created_at
                FROM memory_items {where}
                ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_person_device_id(self, person_id: str) -> str:
        """获取用户关联的设备 ID（从画像表获取）。
        """
        pid = str(person_id or "").strip()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT device_id FROM person_profiles WHERE person_id=?",
                (pid,),
            ).fetchone()
            if row and row["device_id"]:
                return str(row["device_id"])
        return ""

    # ============================
    # 关系状态管理（Relationship State）
    # ============================

    def save_relationship_state(self, person_id: str, state_json: str) -> None:
        """保存或更新关系状态。

        参数:
            person_id:  用户 ID
            state_json: 关系状态的 JSON 字符串
        """
        pid = str(person_id or "").strip()
        if not pid or not state_json:
            return
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO relationship_states(person_id, state_json, created_at, updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(person_id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (pid, state_json, now, now),
            )

    def get_relationship_state(self, person_id: str) -> str | None:
        """获取用户的关系状态 JSON。

        参数:
            person_id: 用户 ID

        返回:
            JSON 字符串，不存在返回 None。
        """
        pid = str(person_id or "").strip()
        if not pid:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_json FROM relationship_states WHERE person_id=?",
                (pid,),
            ).fetchone()
        if not row or not row["state_json"]:
            return None
        return str(row["state_json"])

    def list_person_profiles(self, device_id: str) -> list[dict]:
        """列出指定设备的所有用户画像数据。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    rowid,
                    CAST(person_id AS BLOB) AS person_id_blob,
                    profile_json,
                    CAST(updated_at AS BLOB) AS updated_at_blob
                FROM person_profiles
                WHERE device_id=? ORDER BY updated_at DESC
                """,
                (device_id,),
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            rowid = int(r["rowid"])
            person_id = _decode_text_cell(r["person_id_blob"], column="person_id", rowid=rowid)
            updated_at = _decode_text_cell(r["updated_at_blob"], column="updated_at", rowid=rowid)
            if person_id is None or updated_at is None:
                continue
            try:
                profile = json.loads(r["profile_json"])
            except (TypeError, json.JSONDecodeError):
                logger.warning("跳过损坏的人物画像 JSON rowid=%s person_id=%s", rowid, person_id)
                continue
            out.append({"person_id": person_id, "profile": profile, "updated_at": updated_at})
        return out

    # ============================
    # person_id 重命名与删除
    # ============================

    def rename_person_id(self, old_id: str, new_id: str) -> dict[str, int]:
        """将 person_id 在所有记忆表/画像/会话中重命名（原子操作）。

        参数:
            old_id: 旧的 person_id
            new_id: 新的 person_id

        返回:
            dict: 各表被更新的记录数

        操作流程（全部在一个事务内）：
          1. 校验 old_id 存在、new_id 不冲突
          2. 删除旧画像，插入新画像（person_id 改为 new_id）
          3. 更新 memory_items 统一记忆库中的 person_id
          4. 更新 sessions 表中的 active_person_id

        为什么不用 UPDATE person_profiles：
          person_profiles 的主键是 person_id，UPDATE 会触发 UNIQUE 约束。
          所以走 DELETE + INSERT 路径。
        """
        old_id = str(old_id or "").strip()
        new_id = str(new_id or "").strip()
        if not old_id or not new_id or old_id == new_id:
            raise ValueError("invalid rename")
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT device_id, profile_json, created_at, updated_at
                FROM person_profiles WHERE person_id=?
                """,
                (old_id,),
            ).fetchone()
            if not row:
                raise ValueError("person not found")
            if conn.execute(
                "SELECT 1 FROM person_profiles WHERE person_id=?",
                (new_id,),
            ).fetchone():
                raise ValueError("new person_id already exists")

            profile = json.loads(row["profile_json"])
            profile["person_id"] = new_id
            profile["user_id"] = new_id
            now = _utc_now()
            profile["update_time"] = now
            payload = json.dumps(profile, ensure_ascii=False)
            created = str(profile.get("create_time") or row["created_at"] or now)

            # 画像表：删旧插新
            conn.execute("DELETE FROM person_profiles WHERE person_id=?", (old_id,))
            conn.execute(
                """
                INSERT INTO person_profiles(person_id, device_id, profile_json, created_at, updated_at)
                VALUES (?,?,?,?,?)
                """,
                (new_id, row["device_id"], payload, created, now),
            )

            counts: dict[str, int] = {"person_profiles": 1}

            # 统一记忆库（memory_items）
            cur = conn.execute(
                "UPDATE memory_items SET person_id=? WHERE person_id=?",
                (new_id, old_id),
            )
            counts["memory_items"] = int(cur.rowcount)

            # 会话表
            cur = conn.execute(
                "UPDATE sessions SET active_person_id=? WHERE active_person_id=?",
                (new_id, old_id),
            )
            counts["sessions"] = int(cur.rowcount)
        return counts

    def delete_person_id(self, person_id: str) -> dict[str, int]:
        """删除 person_id 及其全部关联记忆数据（不可逆操作）。

        参数:
            person_id: 要删除的用户 ID

        返回:
            dict: 各表被删除/更新的记录数

        删除范围：
          1. 删除 memory_items 统一记忆库（含 FTS5 索引）与关联图边
          2. 删除画像
          3. 清除活跃会话中的 person 绑定
        """
        pid = str(person_id or "").strip()
        if not pid:
            raise ValueError("person_id required")
        with self._conn() as conn:
            if not conn.execute(
                "SELECT 1 FROM person_profiles WHERE person_id=?",
                (pid,),
            ).fetchone():
                raise ValueError("person not found")

            # 收集统一记忆项 ID（用于关联图清理）
            mi_rows = conn.execute(
                "SELECT id FROM memory_items WHERE person_id=?",
                (pid,),
            ).fetchall()

        mi_ids = [str(r["id"]) for r in mi_rows]
        rel_keys = [f"memory:{item_id}" for item_id in mi_ids]
        rel_deleted = self.delete_relations_for_keys(rel_keys) if rel_keys else 0

        counts: dict[str, int] = {"memory_relations": rel_deleted}
        with self._conn() as conn:
            # 删除统一记忆库（memory_items + FTS）
            for mid in mi_ids:
                conn.execute("DELETE FROM memory_items_fts WHERE id=?", (mid,))
            cur = conn.execute("DELETE FROM memory_items WHERE person_id=?", (pid,))
            counts["memory_items"] = int(cur.rowcount)

            # 删除画像
            cur = conn.execute("DELETE FROM person_profiles WHERE person_id=?", (pid,))
            counts["person_profiles"] = int(cur.rowcount)

            # 清除会话中的 person 绑定
            cur = conn.execute(
                "UPDATE sessions SET active_person_id=NULL WHERE active_person_id=?",
                (pid,),
            )
            counts["sessions"] = int(cur.rowcount)

            # 清除 identity_pending 中的相关记录
            pending_rows = conn.execute(
                "SELECT id, identity_pending FROM sessions WHERE identity_pending IS NOT NULL"
            ).fetchall()
            pending_cleared = 0
            for row in pending_rows:
                raw = row["identity_pending"]
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and str(data.get("person_id") or "").strip() == pid:
                    conn.execute(
                        "UPDATE sessions SET identity_pending=NULL WHERE id=?",
                        (row["id"],),
                    )
                    pending_cleared += 1
            if pending_cleared:
                counts["sessions_pending_cleared"] = pending_cleared
        return counts

    # ============================
    # Open Loop 管理
    # ============================

    def create_open_loop(self, person_id: str, title: str, **kwargs) -> int | None:
        """创建一条待跟进事项。"""
        pid = str(person_id or "").strip()
        ttl = (title or "").strip()
        if not pid or not ttl:
            return None
        now = _utc_now()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO open_loops(
                    person_id, title, status, due_hint, emotional_weight,
                    created_at, last_mentioned_at, cooldown_until, source_session_id
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    pid, ttl, "open",
                    str(kwargs.get("due_hint", "")),
                    int(kwargs.get("emotional_weight", 3)),
                    now, now, "",
                    str(kwargs.get("source_session_id", "")),
                ),
            )
            return int(cur.lastrowid) if cur.lastrowid else None

    def resolve_open_loop(self, loop_id: int, evidence: str = "") -> bool:
        """将待跟进事项标记为已解决。"""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE open_loops SET status='done', resolved_evidence=? WHERE id=? AND status='open'",
                (evidence, loop_id),
            )
            return cur.rowcount > 0

    def list_open_loops(
        self, person_id: str, *, status: str = "open", limit: int = 10,
    ) -> list[dict]:
        """列出用户的待跟进事项。"""
        pid = str(person_id or "").strip()
        if not pid:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, title, status, due_hint, emotional_weight, created_at,
                       last_mentioned_at, cooldown_until, source_session_id
                FROM open_loops
                WHERE person_id=? AND status=?
                ORDER BY emotional_weight DESC, created_at DESC
                LIMIT ?
                """,
                (pid, status, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_open_loop_mentioned(self, loop_id: int) -> None:
        """更新待跟进事项的最后提及时间（冷却计时重置）。"""
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                "UPDATE open_loops SET last_mentioned_at=? WHERE id=?",
                (now, loop_id),
            )

    def count_open_loops(self, person_id: str) -> int:
        """统计用户当前待跟进事项数量。"""
        pid = str(person_id or "").strip()
        if not pid:
            return 0
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM open_loops WHERE person_id=? AND status='open'",
                (pid,),
            ).fetchone()
        return int(row["n"]) if row else 0


# 全局单例：所有模块通过 `from app.session import store` 引用
store = SessionStore()
