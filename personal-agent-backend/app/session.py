"""SQLite 持久层 —— 会话、记忆、画像、关联图的统一存储。

本模块是陪伴机器人所有数据的持久化入口，通过 SessionStore 类封装全部
数据库操作。使用 SQLite（单文件，零配置，适合单机部署）作为存储引擎。

============================
数据表总览
============================

  sessions              会话表（device_id + session_id + active_person_id）
  messages              消息表（L1 工作记忆：role + content）
  episodic_memories     情景记忆表（会话摘要 + 话题 + 未决事项 + 情绪）
  l3_chunks             向量块表（长期记忆 + 语料知识）
  l3_chunks_fts         L3 全文索引（FTS5 虚拟表）
  l0_core_memories      L0 核心记忆表（高置信核心事实）
  l3_recall_stats       L3 召回统计表（追踪高频召回用于 L0 升级）
  person_profiles       人物画像表（JSON 格式存储完整 Profile Card）
  memory_relations      记忆关联图表（from_id ↔ to_id 语义关系）

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
    """计算过期时间（默认使用配置的 L2 保留天数）。

    参数:
        days: 保留天数，None 时使用 settings.l2_retention_days
    返回:
        ISO 格式的过期时间字符串（UTC）
    """
    d = days if days is not None else settings.l2_retention_days
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
    """SQLite 持久层：管理会话、消息、L0/L2/L3、Facts、画像、关联图。

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

    def _migrate_episodic(self, conn: sqlite3.Connection) -> None:
        """迁移 episodic_memories 表，添加新字段。

        episodic_memories 表初始版本只有基本字段，
        后续迭代添加了 open_loops（未决事项）、expires_at（过期时间）、
        archived（归档标记）、emotion（情绪标签）。
        本方法检测这些列是否存在，不存在则 ALTER TABLE 添加。
        """
        cols = {row[1] for row in conn.execute("PRAGMA table_info(episodic_memories)")}
        if not cols:
            return  # 表不存在，无需迁移
        if "open_loops" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN open_loops TEXT")
        if "expires_at" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN expires_at TEXT")
        if "archived" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
        if "emotion" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN emotion TEXT")
        if "importance" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 3")
        if "people" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN people TEXT DEFAULT '[]'")
        if "status" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        # 补充现有的空 expires_at
        conn.execute(
            "UPDATE episodic_memories SET expires_at=? WHERE expires_at IS NULL OR expires_at=''",
            (_expires_at(),),
        )

    def _ensure_indexes(self, conn: sqlite3.Connection) -> None:
        """创建所有必要的索引以优化查询性能。

        索引策略：
          - messages: 按 session_id 查询（每轮都要拉 L1 历史）
          - episodic: 按 device_id + person_id 查询、按 person_id + expires_at 查询
          - facts: 按 person_id 查询
          - l3: 按 collection + device_id + person_id 查询
          - profiles: 按 device_id 查询
          - l0: 按 person_id + category 查询
          - recall: 按 person_id + recall_count 排序
          - relations: 按 from_id / to_id 双向查
        """
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_device ON episodic_memories(device_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_person ON episodic_memories(device_id, person_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_expires ON episodic_memories(device_id, expires_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_person_expires "
            "ON episodic_memories(person_id, expires_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_l3_person ON l3_chunks(collection, device_id, person_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_person_profiles_device ON person_profiles(device_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_l3_collection ON l3_chunks(collection, device_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_l0_person ON l0_core_memories(person_id, category)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_l3_recall_person ON l3_recall_stats(person_id, recall_count)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_from ON memory_relations(from_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_to ON memory_relations(to_id)")

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

    def _migrate_person_scope(self, conn: sqlite3.Connection) -> None:
        """为记忆表添加 person_id 字段以支持多用户隔离。

        早期版本只用 device_id 区分用户，一个设备可能有多人使用。
        引入 person_id 后，可以通过"名字+ID"精确绑定用户，
        一个人的记忆跨设备共享。
        """
        epi_cols = {row[1] for row in conn.execute("PRAGMA table_info(episodic_memories)")}
        if epi_cols and "person_id" not in epi_cols:
            conn.execute(
                "ALTER TABLE episodic_memories ADD COLUMN person_id TEXT NOT NULL DEFAULT ''"
            )
        l3_cols = {row[1] for row in conn.execute("PRAGMA table_info(l3_chunks)")}
        if l3_cols and "person_id" not in l3_cols:
            conn.execute("ALTER TABLE l3_chunks ADD COLUMN person_id TEXT NOT NULL DEFAULT ''")
        self._migrate_l0(conn)

    def _migrate_l0(self, conn: sqlite3.Connection) -> None:
        """创建 L0 核心记忆表和 L3 召回统计表。

        L0（l0_core_memories）：
          存储最高置信度的核心事实，每轮全量注入 prompt。
          按 person_id + category + content_hash 去重。
          分类：identity（身份）, taboo（忌讳）, key_people（关键人物）,
                milestone（里程碑）, preference（偏好）

        L3 召回统计（l3_recall_stats）：
          追踪 L3 中每条事实被召回的次数。
          当某事实的 recall_count >= l0_batch_min_recall
          且 confidence >= l0_batch_min_confidence 时，
          可被提升为 L0。
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS l0_core_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                device_id TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'user_declared',
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_l0_person_hash
            ON l0_core_memories(person_id, category, content_hash)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS l3_recall_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                recall_count INTEGER NOT NULL DEFAULT 0,
                last_recalled_at TEXT NOT NULL,
                UNIQUE(person_id, content_hash)
            )
            """
        )

    def _init_db(self) -> None:
        """初始化数据库：创建所有必需的数据表并执行增量迁移。

        建表顺序考虑了外键依赖：sessions → messages → episodic → facts → profiles → l3
        然后调用各 _migrate_* 方法处理增量字段和索引。
        FTS5 虚拟表在最后创建（不支持 IF NOT EXISTS 语法，通过 try/except 处理）。
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
                -- 消息表（L1 工作记忆）：记录每轮对话的 user/assistant 消息
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                -- L2 情景记忆表：会话压缩后的摘要
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    session_id TEXT,
                    summary TEXT NOT NULL,
                    topics TEXT,
                    open_loops TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    archived INTEGER NOT NULL DEFAULT 0
                );
                -- 人物画像表：JSON 格式存储完整 Profile Card
                CREATE TABLE IF NOT EXISTS person_profiles (
                    person_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                -- L3 向量块表：聊天记忆/语料知识的向量化存储
                -- embedding_json 存储浮点数向量的 JSON 序列化
                -- text_fts 用于 FTS5 全文索引
                CREATE TABLE IF NOT EXISTS l3_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    device_id TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL,
                    text_fts TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    source TEXT,
                    category TEXT,
                    confidence REAL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                -- FTS5 虚拟表：中文全文索引，unicode61 tokenizer 支持中文分词
                CREATE VIRTUAL TABLE IF NOT EXISTS l3_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text_fts,
                    tokenize='unicode61'
                );
                """
            )
            # 增量迁移：处理后续版本新增的字段
            self._migrate_episodic(conn)
            self._migrate_sessions(conn)
            self._migrate_person_scope(conn)
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
        """追加一条 L1 消息并更新会话 last_active 时间戳。

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

        用于判断是否达到 L1 压缩阈值（working_memory_turns）。
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id=? AND role='user'",
                (session_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def get_recent_messages(self, session_id: str, limit: int) -> list[dict]:
        """取最近 limit 条消息（按 ID 倒序取，再反转为正序）。

        返回格式: [{"role": "user", "content": "你好"}, ...]

        用于 L1 working memory 注入 prompt（最近 N 轮对话上下文）。
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
        """获取会话中最旧的 limit 条消息（按 ID 升序）。

        用于 L1 压缩：取出最早的一批消息交给 LLM 压缩为 L2 摘要。
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
        """根据消息 ID 列表批量删除 L1 消息。

        用于 L1 压缩后清理已处理的旧消息。
        """
        if not message_ids:
            return
        placeholders = ",".join("?" * len(message_ids))
        with self._conn() as conn:
            conn.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", message_ids)

    def get_session_messages(self, session_id: str) -> list[dict]:
        """获取会话的所有消息（全部历史，按时间顺序）。

        用于会话结束时的全量 L1→L2 压缩。
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
        """结束会话：清空 L1 消息 + 清除 active_person_id。

        与 close_session 的区别：
          finalize_session 会清除 L1 和解除 person 绑定，
          用于访客结束会话（丢弃 L1 不写入 L2）。
          L2/L3/画像数据保留不删。
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

    # ============================
    # L2 情景记忆（Episodic Memory）
    # ============================

    def add_episodic(
        self,
        device_id: str,
        session_id: str,
        summary: str,
        topics: str = "",
        open_loops: str = "",
        *,
        person_id: str = "",
        emotion: str = "",
        importance: int = 3,
        people: str = "[]",
        status: str = "active",
    ) -> None:
        """添加一条 L2 情景记忆记录。

        参数:
            device_id:  设备标识
            session_id: 来源会话 ID
            summary:    LLM 生成的会话摘要
            topics:     话题列表（逗号分隔）
            open_loops: 未决事项（需要后续跟进的话题）
            person_id:  用户 ID（用于多用户隔离）
            emotion:    情绪标签（如"开心"/"生气"/"焦虑"）
            importance: 重要性 1-5（3=普通，4=重要事件，5=里程碑）
            people:     涉及人物的 JSON 数组字符串
            status:     状态：active/archived/corrected

        自动设置 14 天过期时间（expires_at），比旧版 7 天长，
        以支持更久的情感近况感知。
        """
        now = _utc_now()
        exp = _expires_at()
        pid = str(person_id or "").strip()
        imp = max(1, min(5, int(importance))) if importance else 3
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO episodic_memories(
                    device_id, person_id, session_id, summary, topics, open_loops,
                    created_at, expires_at, archived, emotion,
                    importance, people, status
                ) VALUES (?,?,?,?,?,?,?,?,0,?, ?,?,?)
                """,
                (device_id, pid, session_id, summary, topics, open_loops,
                 now, exp, emotion, imp, people, status),
            )

    def list_episodic_active(
        self, device_id: str, person_id: str, limit: int = 30
    ) -> list[dict]:
        """获取用户的活跃 L2 情景记忆行（未归档且未过期）。

        返回按 ID 倒序（最新的在前），用于 prompt 中的 L2 注入。
        包含重要性评分（importance）和涉及人物（people）等新字段。
        """
        del device_id
        now = _utc_now()
        pid = str(person_id or "").strip()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, summary, topics, open_loops, emotion,
                       created_at, expires_at, importance, people, status
                FROM episodic_memories
                WHERE person_id=? AND archived=0 AND expires_at > ?
                ORDER BY id DESC LIMIT ?
                """,
                (pid, now, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_important_episodes(
        self, person_id: str, min_importance: int = 4, limit: int = 10,
    ) -> list[dict]:
        """获取用户的重要情景记忆（高重要性事件，不受过期限制）。

        参数:
            person_id:       用户 ID
            min_importance:  最低重要性（4=重要事件，5=里程碑）
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
                SELECT id, summary, topics, open_loops, emotion,
                       created_at, expires_at, importance, people, status
                FROM episodic_memories
                WHERE person_id=? AND importance>=? AND archived=0
                ORDER BY importance DESC, id DESC LIMIT ?
                """,
                (pid, min_importance, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_expired_episodic(self, device_id: str | None = None) -> list[dict]:
        """获取已过期的 L2 情景记忆行（用于 l2_rollup_sweeper 归档到 Corpus）。

        返回按 person_id 分组 + ID 升序排列（方便批量处理）。
        """
        del device_id
        now = _utc_now()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, device_id, person_id, session_id, summary, topics, open_loops, created_at, expires_at
                FROM episodic_memories
                WHERE archived=0 AND expires_at <= ?
                ORDER BY person_id, id ASC
                """,
                (now,),
            ).fetchall()
        return [dict(r) for r in rows]

    def archive_episodic(self, episodic_ids: list[int]) -> None:
        """批量归档 L2 情景记忆记录（标记 archived=1，不删除数据）。"""
        if not episodic_ids:
            return
        placeholders = ",".join("?" * len(episodic_ids))
        with self._conn() as conn:
            conn.execute(
                f"UPDATE episodic_memories SET archived=1 WHERE id IN ({placeholders})",
                episodic_ids,
            )

    def list_episodic(self, device_id: str, person_id: str, limit: int = 20) -> list[dict]:
        """获取用户活跃 L2 情景记忆（list_episodic_active 的别名方法）。"""
        return self.list_episodic_active(device_id, person_id, limit)

    def delete_episodic_for_person(self, person_id: str) -> int:
        """删除指定用户的所有 L2 情景记忆（不可逆），返回删除条数。"""
        pid = str(person_id or "").strip()
        if not pid:
            raise ValueError("person_id required")
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM episodic_memories WHERE person_id=?",
                (pid,),
            )
            return cur.rowcount

    # ============================
    # L3 Facts 管理
    # ============================

    def add_fact(
        self,
        device_id: str,
        person_id: str,
        fact: str,
        category: str,
        confidence: float,
        source_session: str,
    ) -> int | None:
        """添加一条事实记录（已废弃，当前版本保留为 no-op 以兼容旧调用）。"""
        return None

    def get_fact_by_id(self, fact_id: int) -> dict | None:
        """根据 ID 获取单条事实记录（已废弃，当前版本始终返回 None）。"""
        return None

    def get_fact_by_text(self, person_id: str, fact_text: str) -> dict | None:
        """（已废弃）旧 Facts 表已移除，始终返回 None。"""
        return None

    def list_facts(self, device_id: str, person_id: str, limit: int = 50) -> list[dict]:
        """（已废弃）旧 Facts 表已移除，始终返回空列表。"""
        return []

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
            from_id:       源记忆的 key（格式：fact:{id} 或 chunk:{chunk_id}）
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
            memory_keys:  记忆键列表（fact:{id} 或 chunk:{chunk_id}）
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
        """删除具有指定 ID 前缀的关联边（如"chunk:doc-"），返回删除数量。"""
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
        *,
        chunk_key_prefix: str = "chunk:doc-",
    ) -> dict[str, int]:
        """清除由 persona 语料导入生成的长期记忆块/关联边。

        参数:
            person_id:         用户 ID（通常是 persona_global）
            chunk_key_prefix:  文档块的 key 前缀

        返回:
            dict: {"l3_facts": 删除的块数, "relations": 删除的关联边数}

        用途：
          重新导入语料时（ingest --reset），需要先清除所有由旧语料派生的记忆。
        """
        pid = str(person_id or "").strip()
        stats = {"l3_facts": 0, "relations": 0}
        if not pid:
            return stats
        with self._conn() as conn:
            chunk_rows = conn.execute(
                """
                SELECT chunk_id FROM l3_chunks
                WHERE (collection='facts' OR chunk_id LIKE ?) AND person_id=?
                """,
                (f"{chunk_key_prefix}%", pid),
            ).fetchall()
            for row in chunk_rows:
                cid = str(row["chunk_id"])
                conn.execute("DELETE FROM l3_chunks WHERE chunk_id=?", (cid,))
                conn.execute("DELETE FROM l3_chunks_fts WHERE chunk_id=?", (cid,))
                stats["l3_facts"] += 1
        stats["relations"] += self.delete_relations_with_id_prefix(chunk_key_prefix)
        return stats

    # ============================
    # L3 向量块管理
    # ============================

    def l3_get_chunk(self, chunk_id: str) -> dict | None:
        """根据 chunk_id 获取单个 L3 向量块信息。"""
        cid = str(chunk_id or "").strip()
        if not cid:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT chunk_id, text, collection, person_id, device_id
                FROM l3_chunks WHERE chunk_id=?
                """,
                (cid,),
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def has_recent_emotional_event(self, person_id: str, title: str, hours: int = 24) -> bool:
        """检查当天是否已记录过相同标题的情感事件（用于去重）。

        Args:
            person_id: 用户 ID
            title:     情感事件标题
            hours:     回溯时间窗口（小时），默认 24

        Returns:
            True 如果已存在。
        """
        if not person_id or not title:
            return False
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM l3_chunks
                WHERE person_id=? AND source='emotional_event_turn'
                  AND created_at >= ? AND text LIKE ?
                LIMIT 1
                """,
                (str(person_id).strip(), cutoff, f"%{title}%"),
            ).fetchone()
        return row is not None

    def l3_delete_chunk(self, chunk_id: str) -> bool:
        """删除指定的 L3 向量块及其全文索引，返回是否成功删除。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM l3_chunks WHERE chunk_id=?", (chunk_id,)
            ).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM l3_chunks WHERE chunk_id=?", (chunk_id,))
            conn.execute("DELETE FROM l3_chunks_fts WHERE chunk_id=?", (chunk_id,))
            return True

    def l3_find_chunks_by_text(
        self,
        text: str,
        *,
        collection: str | None = None,
        device_id: str = "",
        limit: int = 8,
    ) -> list[dict]:
        """通过文本内容查找 L3 向量块。

        支持精确匹配和前缀匹配（length >= 12 时）。
        用于记忆修正时找到需要修改的具体块。
        """
        needle = text.strip()
        if not needle:
            return []
        collections = [collection] if collection else ["memory", "corpus"]
        found: list[dict] = []
        for coll in collections:
            if coll == "memory" and device_id:
                rows = self.l3_list_chunks(coll, device_id=device_id)
            else:
                rows = self.l3_list_chunks(coll)
            for row in rows:
                body = str(row.get("text", ""))
                if needle == body or (len(needle) >= 12 and needle in body):
                    found.append(
                        {
                            "chunk_id": row["chunk_id"],
                            "collection": coll,
                            "text": body[:400],
                        }
                    )
                if len(found) >= limit:
                    return found
        return found

    def _row_to_l3(self, row: sqlite3.Row) -> dict:
        """将 SQLite 行转换为 L3 向量块字典格式。

        自动解析 embedding_json，保留标准字段（chunk_id, text, embedding,
        collection, device_id, person_id, source, category, confidence, created_at）。
        """
        emb = json.loads(row["embedding_json"]) if row["embedding_json"] else []
        out = {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "embedding": emb,
            "collection": row["collection"],
            "device_id": row["device_id"],
        }
        if "person_id" in row.keys():
            out["person_id"] = row["person_id"]
        for key in ("source", "category", "confidence", "created_at"):
            if key in row.keys():
                out[key] = row[key]
        return out

    def l3_list_corpus_global(self) -> list[dict]:
        """列出全局语料库向量块（未绑定到特定用户，即 person_id 为空）。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, text, embedding_json, collection, device_id, person_id
                FROM l3_chunks
                WHERE collection='corpus' AND (person_id IS NULL OR person_id='')
                """
            ).fetchall()
        return [self._row_to_l3(r) for r in rows]

    def l3_list_corpus_for_person(self, person_id: str) -> list[dict]:
        """列出绑定到指定用户的语料库向量块。"""
        pid = str(person_id or "").strip()
        if not pid:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, text, embedding_json, collection, device_id, person_id
                FROM l3_chunks
                WHERE collection='corpus' AND person_id=?
                """,
                (pid,),
            ).fetchall()
        return [self._row_to_l3(r) for r in rows]

    def l3_list_recall_pool(
        self,
        person_id: str,
        *,
        extra_person_ids: list[str] | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """构建 L3 召回候选池：全局语料 + 当前用户 + 额外用户 + 系统 persona。

        这是向量检索的数据源，包含：
          1. 全局语料（person_id IS NULL，所有用户共享的知识）
          2. 当前用户绑定记忆
          3. persona_global 系统知识
          4. 额外指定的用户（如被提及的关联人物）

        限定最近 limit 条记录，避免用户数据积累后全量加载导致向量比对耗时过长。
        """
        pid = str(person_id or "").strip()
        extras = [str(x).strip() for x in (extra_person_ids or []) if str(x).strip()]
        clauses = ["(person_id IS NULL OR person_id='')"]
        params: list[str] = []
        if pid:
            clauses.append("person_id=?")
            params.append(pid)
        for e in extras:
            if e and e != pid:
                clauses.append("person_id=?")
                params.append(e)
        where = " OR ".join(clauses)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT chunk_id, text, embedding_json, collection, device_id, person_id,
                       source, category, confidence, created_at
                FROM l3_chunks
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params + [limit],
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            item = self._row_to_l3(r)
            item["source"] = r["source"]
            item["category"] = r["category"]
            item["confidence"] = r["confidence"]
            out.append(item)
        return out

    def l3_fts_search_pool(
        self,
        query: str,
        person_id: str,
        *,
        extra_person_ids: list[str] | None = None,
        limit: int = 24,
    ) -> list[dict]:
        """全文检索 L3，结果限定在 l3_list_recall_pool 同一作用域。

        使用 FTS5 联合查询 l3_chunks，先做全文检索拿到候选，
        再过滤 person_id 作用域，取前 limit 条。

        为什么需要作用域过滤：
          FTS5 返回的是全表匹配结果，需要限定到当前用户 +
          全局语料的范围内，防止用户 A 看到用户 B 的记忆。
        """
        from app.store.chunks import build_fts_match_query

        match_q = build_fts_match_query(query)
        if not match_q:
            return []
        pid = str(person_id or "").strip()
        extras = [str(x).strip() for x in (extra_person_ids or []) if str(x).strip()]
        allowed = {pid, *extras}
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.text, c.embedding_json, c.collection, c.device_id,
                       c.person_id, c.source, c.category, c.confidence, c.created_at
                FROM l3_chunks_fts f
                JOIN l3_chunks c ON c.chunk_id = f.chunk_id
                WHERE l3_chunks_fts MATCH ?
                LIMIT ?
                """,
                (match_q, limit * 4),  # 多取 4 倍候选，过滤后再截取 limit
            ).fetchall()
        out: list[dict] = []
        for r in rows:
            row_pid = str(r["person_id"] or "").strip()
            if row_pid and row_pid not in allowed:
                continue  # 不属于当前用户作用域，跳过
            item = self._row_to_l3(r)
            item["source"] = r["source"]
            item["category"] = r["category"]
            item["confidence"] = r["confidence"]
            out.append(item)
            if len(out) >= limit:
                break
        return out

    def l3_count(
        self, collection: str, *, device_id: str = "", person_id: str | None = None
    ) -> int:
        """统计指定集合中的向量块数量。

        参数:
            collection: 集合名（"memory", "corpus", "facts"）
            device_id:  可选，按设备过滤
            person_id:  可选，按用户过滤（仅 facts 集合生效）
        """
        with self._conn() as conn:
            if collection == "facts" and person_id is not None:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS n FROM l3_chunks
                    WHERE collection=? AND person_id=?
                    """,
                    (collection, str(person_id).strip()),
                ).fetchone()
            elif device_id:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM l3_chunks WHERE collection=? AND device_id=?",
                    (collection, device_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM l3_chunks WHERE collection=?",
                    (collection,),
                ).fetchone()
        return int(row["n"]) if row else 0

    def l3_reset_all(self) -> None:
        """清空所有 L3 向量块和全文索引（危险操作，不可逆）。"""
        with self._conn() as conn:
            conn.execute("DELETE FROM l3_chunks")
            conn.execute("DELETE FROM l3_chunks_fts")

    def l3_reset_corpus(self) -> None:
        """清空全局 L3 语料（corpus/memory 集合 + person_id 为空）。

        与 l3_reset_all 的区别：
          只清空全局语料，保留用户个人记忆（person_id 不为空的记录）。
        """
        with self._conn() as conn:
            ids = [
                r[0]
                for r in conn.execute(
                    """
                    SELECT chunk_id FROM l3_chunks
                    WHERE collection IN ('corpus', 'memory')
                      AND (person_id IS NULL OR person_id='')
                    """
                ).fetchall()
            ]
            conn.execute(
                """
                DELETE FROM l3_chunks
                WHERE collection IN ('corpus', 'memory')
                  AND (person_id IS NULL OR person_id='')
                """
            )
            for cid in ids:
                conn.execute("DELETE FROM l3_chunks_fts WHERE chunk_id=?", (cid,))

    def l3_clear(self, collection: str, *, device_id: str = "") -> None:
        """清空指定集合的 L3 向量块（可选按设备过滤）。"""
        with self._conn() as conn:
            if device_id:
                ids = [
                    r[0]
                    for r in conn.execute(
                        "SELECT chunk_id FROM l3_chunks WHERE collection=? AND device_id=?",
                        (collection, device_id),
                    ).fetchall()
                ]
                conn.execute(
                    "DELETE FROM l3_chunks WHERE collection=? AND device_id=?",
                    (collection, device_id),
                )
            else:
                ids = [
                    r[0]
                    for r in conn.execute(
                        "SELECT chunk_id FROM l3_chunks WHERE collection=?",
                        (collection,),
                    ).fetchall()
                ]
                conn.execute("DELETE FROM l3_chunks WHERE collection=?", (collection,))
            for cid in ids:
                conn.execute("DELETE FROM l3_chunks_fts WHERE chunk_id=?", (cid,))

    def l3_bulk_upsert(self, rows: list[dict]) -> None:
        """批量插入或更新 L3 向量块及其全文索引。

        参数:
            rows: list[dict]，每个元素包含:
                  chunk_id, collection, device_id, person_id, text,
                  embedding, source, category, confidence

        使用 INSERT ... ON CONFLICT DO UPDATE 实现 upsert。
        同时维护 FTS5 全文索引（先删后插，因为 FTS5 不支持 UPDATE）。
        """
        if not rows:
            return
        from app.store.chunks import prepare_fts_text

        now = _utc_now()
        with self._conn() as conn:
            for row in rows:
                cid = str(row["chunk_id"])
                text = str(row["text"])
                fts = prepare_fts_text(text)
                emb_json = json.dumps(row["embedding"])
                conn.execute(
                    """
                    INSERT INTO l3_chunks(
                        chunk_id, collection, device_id, person_id, text, text_fts, embedding_json,
                        source, category, confidence, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        collection=excluded.collection,
                        device_id=excluded.device_id,
                        person_id=excluded.person_id,
                        text=excluded.text,
                        text_fts=excluded.text_fts,
                        embedding_json=excluded.embedding_json,
                        source=excluded.source,
                        category=excluded.category,
                        confidence=excluded.confidence
                    """,
                    (
                        cid,
                        row["collection"],
                        str(row.get("device_id", "")),
                        str(row.get("person_id", "")),
                        text,
                        fts,
                        emb_json,
                        str(row.get("source", "")),
                        str(row.get("category", "")),
                        float(row.get("confidence", 0.0)),
                        now,
                    ),
                )
                # FTS5 不支持 INSERT OR REPLACE，所以先删后插
                conn.execute("DELETE FROM l3_chunks_fts WHERE chunk_id=?", (cid,))
                conn.execute(
                    "INSERT INTO l3_chunks_fts(chunk_id, text_fts) VALUES (?,?)",
                    (cid, fts),
                )

    def l3_list_chunks(
        self, collection: str, *, device_id: str = "", person_id: str | None = None
    ) -> list[dict]:
        """列出指定集合的 L3 向量块（可按设备或用户过滤）。"""
        with self._conn() as conn:
            if collection == "facts" and person_id is not None:
                rows = conn.execute(
                    """
                    SELECT chunk_id, text, embedding_json, collection, device_id, person_id
                    FROM l3_chunks WHERE collection=? AND person_id=?
                    """,
                    (collection, str(person_id).strip()),
                ).fetchall()
            elif device_id:
                rows = conn.execute(
                    """
                    SELECT chunk_id, text, embedding_json, collection, device_id, person_id
                    FROM l3_chunks WHERE collection=? AND device_id=?
                    """,
                    (collection, device_id),
                ).fetchall()
            elif collection == "corpus":
                rows = conn.execute(
                    """
                    SELECT chunk_id, text, embedding_json, collection, device_id
                    FROM l3_chunks WHERE collection=?
                    """,
                    (collection,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT chunk_id, text, embedding_json, collection, device_id
                    FROM l3_chunks WHERE collection=?
                    """,
                    (collection,),
                ).fetchall()
        return [self._row_to_l3(r) for r in rows]

    def l3_list_person_memory(
        self, person_id: str, *, device_id: str = "", limit: int = 50
    ) -> list[dict]:
        """列出某用户的 L3 长期记忆块（collection 为 memory 或 facts）。

        返回按 created_at 降序（最新的在前），最多 limit 条。
        """
        pid = str(person_id or "").strip()
        if not pid:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, text, embedding_json, collection, device_id, person_id,
                       source, category, confidence, created_at
                FROM l3_chunks
                WHERE person_id=? AND collection IN ('memory', 'facts')
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (pid, limit),
            ).fetchall()
        return [self._row_to_l3(r) for r in rows]

    def l3_fts_search(
        self,
        query: str,
        *,
        collection: str,
        device_id: str = "",
        person_id: str | None = None,
        limit: int = 24,
    ) -> list[dict]:
        """使用 FTS5 全文搜索引擎查找 L3 向量块（带作用域过滤）。"""
        from app.store.chunks import build_fts_match_query

        match_q = build_fts_match_query(query)
        if not match_q:
            return []
        with self._conn() as conn:
            if collection == "facts" and person_id is not None:
                rows = conn.execute(
                    """
                    SELECT c.chunk_id, c.text, c.embedding_json, c.collection, c.device_id, c.person_id
                    FROM l3_chunks_fts f
                    JOIN l3_chunks c ON c.chunk_id = f.chunk_id
                    WHERE l3_chunks_fts MATCH ? AND c.collection=? AND c.person_id=?
                    LIMIT ?
                    """,
                    (match_q, collection, str(person_id).strip(), limit),
                ).fetchall()
            elif device_id:
                rows = conn.execute(
                    """
                    SELECT c.chunk_id, c.text, c.embedding_json, c.collection, c.device_id, c.person_id
                    FROM l3_chunks_fts f
                    JOIN l3_chunks c ON c.chunk_id = f.chunk_id
                    WHERE l3_chunks_fts MATCH ? AND c.collection=? AND c.device_id=?
                    LIMIT ?
                    """,
                    (match_q, collection, device_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.chunk_id, c.text, c.embedding_json, c.collection, c.device_id, c.person_id
                    FROM l3_chunks_fts f
                    JOIN l3_chunks c ON c.chunk_id = f.chunk_id
                    WHERE l3_chunks_fts MATCH ? AND c.collection=?
                    LIMIT ?
                    """,
                    (match_q, collection, limit),
                ).fetchall()
        return [self._row_to_l3(r) for r in rows]

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
    # L0 核心记忆管理
    # ============================

    def l0_upsert(
        self,
        person_id: str,
        category: str,
        content: str,
        *,
        device_id: str = "",
        source: str = "user_declared",
        confidence: float = 1.0,
    ) -> None:
        """插入或更新 L0 核心记忆。

        参数:
            person_id:  用户 ID
            category:   分类（identity/taboo/key_people/milestone/preference）
            content:    记忆内容文本
            device_id:  来源设备
            source:     来源类型（user_declared/l3_promotion/system）
            confidence: 置信度 0.0-1.0

        基于 (person_id, category, content_hash) 去重：
          相同内容只存一条，更新时会刷新 confidence 和 updated_at。
        """
        import hashlib

        pid = str(person_id or "").strip()
        body = " ".join((content or "").strip().split())
        if not pid or not body:
            return
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO l0_core_memories(
                    person_id, device_id, category, content, content_hash,
                    source, confidence, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(person_id, category, content_hash) DO UPDATE SET
                    content=excluded.content,
                    source=excluded.source,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                (
                    pid,
                    device_id or "",
                    category,
                    body,
                    content_hash,
                    source,
                    float(confidence),
                    now,
                    now,
                ),
            )

    def l0_list(self, person_id: str, *, category: str | None = None) -> list[dict]:
        """列出用户的 L0 核心记忆（按优先级排序）。

        排序规则（CASE category）：
          identity(1) > taboo(2) > key_people(3) > milestone(4) > preference(5) > 其他(9)

        身份最关键，禁忌次之，偏好最轻—— prompt 中 L0 按此顺序注入。
        """
        pid = str(person_id or "").strip()
        if not pid:
            return []
        order = (
            "CASE category "
            "WHEN 'identity' THEN 1 WHEN 'taboo' THEN 2 WHEN 'key_people' THEN 3 "
            "WHEN 'milestone' THEN 4 WHEN 'preference' THEN 5 ELSE 9 END, id ASC"
        )
        with self._conn() as conn:
            if category:
                rows = conn.execute(
                    f"""
                    SELECT id, category, content, source, confidence, created_at
                    FROM l0_core_memories
                    WHERE person_id=? AND category=?
                    ORDER BY {order}
                    """,
                    (pid, category),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT id, category, content, source, confidence, created_at
                    FROM l0_core_memories
                    WHERE person_id=?
                    ORDER BY {order}
                    """,
                    (pid,),
                ).fetchall()
        return [dict(r) for r in rows]

    def l0_count(self, person_id: str, category: str) -> int:
        """统计用户指定类别的 L0 核心记忆数量。"""
        pid = str(person_id or "").strip()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n FROM l0_core_memories
                WHERE person_id=? AND category=?
                """,
                (pid, category),
            ).fetchone()
        return int(row["n"]) if row else 0

    def l0_find_by_content(
        self, person_id: str, content: str, category: str | None = None
    ) -> dict | None:
        """通过内容和类别查找 L0 核心记忆（基于 content_hash 去重匹配）。

        用途：检查某事实是否已存在于 L0 中，避免重复写入。
        """
        import hashlib

        pid = str(person_id or "").strip()
        body = " ".join((content or "").strip().split())
        if not pid or not body:
            return None
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        with self._conn() as conn:
            if category:
                row = conn.execute(
                    """
                    SELECT id, category, content FROM l0_core_memories
                    WHERE person_id=? AND category=? AND content_hash=?
                    """,
                    (pid, category, content_hash),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT id, category, content FROM l0_core_memories
                    WHERE person_id=? AND content_hash=?
                    """,
                    (pid, content_hash),
                ).fetchone()
        return dict(row) if row else None

    def l0_get(self, row_id: int) -> dict | None:
        """按主键获取单条 L0 记录。"""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, person_id, category, content, source, confidence, created_at, updated_at
                FROM l0_core_memories WHERE id=?
                """,
                (row_id,),
            ).fetchone()
        return dict(row) if row else None

    def l0_delete(self, row_id: int) -> bool:
        """删除指定 ID 的 L0 核心记忆记录，返回是否成功。"""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM l0_core_memories WHERE id=?", (row_id,))
            return cur.rowcount > 0

    def l0_update(
        self,
        row_id: int,
        *,
        category: str,
        content: str,
        source: str = "manual",
    ) -> bool:
        """更新指定 L0 记录的类别与内容（重算 content_hash）。"""
        import hashlib

        body = " ".join((content or "").strip().split())
        cat = str(category or "").strip()
        if not body or not cat:
            return False
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        now = _utc_now()
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE l0_core_memories
                SET category=?, content=?, content_hash=?, source=?, updated_at=?
                WHERE id=?
                """,
                (cat, body, content_hash, source, now, row_id),
            )
            return cur.rowcount > 0

    def l0_delete_matching(self, person_id: str, substring: str) -> int:
        """删除内容包含子字符串的 L0 记录（子串 >= 2 字符），返回删除数量。

        用于记忆修正：用户说"不对，我没有 XX"，删除匹配的 L0 条目。
        """
        pid = str(person_id or "").strip()
        sub = (substring or "").strip()
        if not pid or len(sub) < 2:
            return 0
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM l0_core_memories WHERE person_id=? AND content LIKE ?",
                (pid, f"%{sub}%"),
            )
            return cur.rowcount

    # ============================
    # L3 召回统计（用于 L0 升级）
    # ============================

    def l3_recall_bump(
        self,
        person_id: str,
        content: str,
        *,
        category: str = "",
        confidence: float = 0.0,
    ) -> None:
        """增加某条 L3 事实的召回计数。

        每次从 L3 检索命中某条内容时调用，追踪被高频访问的事实。
        当 recall_count >= l0_batch_min_recall 且
        confidence >= l0_batch_min_confidence 时，可提升为 L0。

        基于 (person_id, content_hash) 去重，use ON CONFLICT increment。
        """
        import hashlib

        pid = str(person_id or "").strip()
        body = " ".join((content or "").strip().split())
        if not pid or len(body) < 4:
            return
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO l3_recall_stats(
                    person_id, content, content_hash, category, confidence,
                    recall_count, last_recalled_at
                ) VALUES (?,?,?,?,?,1,?)
                ON CONFLICT(person_id, content_hash) DO UPDATE SET
                    recall_count=recall_count+1,
                    category=CASE WHEN excluded.category!='' THEN excluded.category ELSE category END,
                    confidence=CASE WHEN excluded.confidence>0 THEN excluded.confidence ELSE confidence END,
                    last_recalled_at=excluded.last_recalled_at
                """,
                (pid, body, content_hash, category or "", float(confidence), now),
            )

    def l3_recall_list(self, person_id: str, *, min_count: int = 1) -> list[dict]:
        """列出用户的 L3 召回统计（按召回次数降序）。

        用于 profile_batch_sweeper 中的 L0 批量升级判断。
        """
        pid = str(person_id or "").strip()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT content, category, confidence, recall_count, last_recalled_at
                FROM l3_recall_stats
                WHERE person_id=? AND recall_count>=?
                ORDER BY recall_count DESC, last_recalled_at DESC
                """,
                (pid, min_count),
            ).fetchall()
        return [dict(r) for r in rows]

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

    def list_episodic_for_person_admin(self, person_id: str, *, limit: int = 20) -> list[dict]:
        """后台管理：按用户列出情景摘要。"""
        pid = str(person_id or "").strip()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, device_id, person_id, session_id, summary, topics, open_loops,
                       created_at, expires_at, archived, emotion, importance, people, status
                FROM episodic_memories
                WHERE person_id=?
                ORDER BY created_at DESC LIMIT ?
                """,
                (pid, int(limit)),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_episodic_for_person(self, person_id: str) -> int:
        """后台管理：统计指定用户情景摘要数量。"""
        pid = str(person_id or "").strip()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM episodic_memories WHERE person_id=?",
                (pid,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def get_episodic_by_id(self, episodic_id: int) -> dict | None:
        """后台管理：获取单条 L2 情景记忆。"""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, device_id, person_id, session_id, summary, topics, open_loops,
                       created_at, expires_at, archived, emotion, importance, people, status
                FROM episodic_memories WHERE id=?
                """,
                (episodic_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_episodic_by_id(self, episodic_id: int) -> bool:
        """后台管理：删除单条 L2 情景记忆。"""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM episodic_memories WHERE id=?", (episodic_id,)
            )
            return cur.rowcount > 0

    def update_episodic_admin(
        self, episodic_id: int, **kwargs
    ) -> bool:
        """后台管理：更新 L2 情景记忆字段。"""
        allowed = {"summary", "topics", "emotion", "importance", "status"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return False
        sets = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [episodic_id]
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE episodic_memories SET {sets} WHERE id=?", vals
            )
            return cur.rowcount > 0

    def l3_list_chunks_detailed(
        self, collection: str | None = None, person_id: str | None = None,
        *, limit: int = 100, offset: int = 0,
    ) -> list[dict]:
        """后台管理：列出 L3 向量块（含 source/category/confidence/created_at）。"""
        where = "WHERE 1=1"
        params: list[str] = []
        if collection:
            where += " AND collection=?"
            params.append(collection)
        if person_id:
            where += " AND person_id=?"
            params.append(str(person_id))
        params.append(str(limit))
        params.append(str(offset))
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT chunk_id, text, collection, device_id, person_id,
                       source, category, confidence, created_at
                FROM l3_chunks {where}
                ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def count_l3_for_person(self, person_id: str) -> int:
        """后台管理：统计指定用户长期记忆块数量。"""
        pid = str(person_id or "").strip()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM l3_chunks WHERE person_id=?",
                (pid,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def list_memory_relations_admin(self, *, limit: int = 120) -> list[dict]:
        """后台管理：列出关联图边，用于关系图谱可视化。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT from_id, to_id, relation_type, strength, created_at
                FROM memory_relations
                ORDER BY strength DESC, created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def l3_update_chunk(self, chunk_id: str, *, text: str | None = None,
                        category: str | None = None) -> bool:
        """后台管理：更新 L3 向量块文本（同时更新 FTS5 索引）。"""
        updates: dict[str, object] = {}
        if text is not None:
            updates["text"] = text
            updates["text_fts"] = text
        if category is not None:
            updates["category"] = str(category).strip()
        if not updates:
            return False
        sets = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [chunk_id]
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE l3_chunks SET {sets} WHERE chunk_id=?", vals
            )
            ok = cur.rowcount > 0
            if ok and text is not None:
                conn.execute(
                    "UPDATE l3_chunks_fts SET text_fts=? WHERE chunk_id=?",
                    (text, chunk_id),
                )
            return ok

    def count_memory_stat(self) -> dict:
        """后台管理：统计各记忆数量和活跃会话数。"""
        with self._conn() as conn:
            l0 = conn.execute("SELECT COUNT(*) FROM l0_core_memories").fetchone()[0]
            l2 = conn.execute("SELECT COUNT(*) FROM episodic_memories WHERE archived=0").fetchone()[0]
            l2_total = conn.execute("SELECT COUNT(*) FROM episodic_memories").fetchone()[0]
            l3 = conn.execute("SELECT COUNT(*) FROM l3_chunks").fetchone()[0]
            profiles = conn.execute("SELECT COUNT(*) FROM person_profiles").fetchone()[0]
        return {
            "core_memories": l0, "episodes_active": l2, "episodes_total": l2_total,
            "long_term_memory": l3, "profiles": profiles,
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

    def list_episodic_since(
        self, person_id: str, since_iso: str, *, limit: int = 40
    ) -> list[dict]:
        """获取用户在指定时间点后创建的活跃情景记忆（用于 Profile 增量更新）。"""
        pid = str(person_id or "").strip()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, summary, topics, open_loops, created_at, expires_at
                FROM episodic_memories
                WHERE person_id=? AND archived=0 AND created_at >= ?
                ORDER BY id DESC LIMIT ?
                """,
                (pid, since_iso, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_facts_since(
        self, person_id: str, since_iso: str, *, limit: int = 60
    ) -> list[dict]:
        """获取用户在指定时间点后创建的长期记忆块（替换旧 Facts 表，用于 Profile 增量更新）。"""
        pid = str(person_id or "").strip()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT text AS fact, category, confidence, created_at FROM l3_chunks
                WHERE person_id=? AND created_at >= ?
                ORDER BY rowid DESC LIMIT ?
                """,
                (pid, since_iso, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_recall_person_ids(self, *, min_count: int = 3) -> list[str]:
        """列出有足够召回次数的所有用户 ID（用于 L0 批量升级扫描）。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT person_id FROM l3_recall_stats
                WHERE recall_count >= ? AND person_id != ''
                ORDER BY person_id
                """,
                (min_count,),
            ).fetchall()
        return [str(r["person_id"]) for r in rows]

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
        """将 person_id 在所有记忆表中重命名（在同一事务内完成，保证原子性）。

        参数:
            old_id: 旧的 person_id
            new_id: 新的 person_id

        返回:
            dict: 各表被更新的记录数

        操作流程（全部在一个事务内）：
          1. 校验 old_id 存在、new_id 不冲突
          2. 删除旧画像，插入新画像（person_id 改为 new_id）
          3. 更新 episodic_memories, l3_chunks,
             l0_core_memories, l3_recall_stats 中的 person_id
          4. 更新 sessions 表中的 active_person_id

        为什么不用 UPDATE：
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
            # 级联更新所有记忆表
            for table in (
                "episodic_memories",
                "l3_chunks",
                "l0_core_memories",
                "l3_recall_stats",
            ):
                cur = conn.execute(
                    f"UPDATE {table} SET person_id=? WHERE person_id=?",
                    (new_id, old_id),
                )
                counts[table] = int(cur.rowcount)
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
            dict: {"memory_relations": ..., "l3_chunks": ..., "episodic_memories": ...,
                   "l0_core_memories": ..., "l3_recall_stats": ...,
                   "person_profiles": ..., "sessions": ...}

        删除范围：
          1. 找到该用户的所有长期记忆块 → 收集 chunk key
          2. 删除所有关联图的边
          3. 删除长期记忆块（同时清理 FTS5 索引）
          4. 删除情景摘要/L0/recall_stats 记录
          5. 删除画像
          6. 清除活跃会话中的 person 绑定
          7. 清除 identity_pending 中的相关记录
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

            # 收集要清理的 chunk key
            chunk_rows = conn.execute(
                "SELECT chunk_id FROM l3_chunks WHERE person_id=?",
                (pid,),
            ).fetchall()

        chunk_ids = [str(r["chunk_id"]) for r in chunk_rows if r["chunk_id"]]

        # 删除关联图边
        rel_deleted = 0
        if chunk_ids:
            from app.memory.relations import chunk_key

            rel_keys = [chunk_key(cid) for cid in chunk_ids]
            rel_deleted = self.delete_relations_for_keys(rel_keys)

        counts: dict[str, int] = {"memory_relations": rel_deleted, "l3_chunks": 0}
        with self._conn() as conn:
            # 删除 L3 块和 FTS5 索引
            for cid in chunk_ids:
                conn.execute("DELETE FROM l3_chunks WHERE chunk_id=?", (cid,))
                conn.execute("DELETE FROM l3_chunks_fts WHERE chunk_id=?", (cid,))
                counts["l3_chunks"] += 1

            # 删除所有关联的记忆表数据
            for table in (
                "episodic_memories",
                "l0_core_memories",
                "l3_recall_stats",
            ):
                cur = conn.execute(f"DELETE FROM {table} WHERE person_id=?", (pid,))
                counts[table] = int(cur.rowcount)

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
