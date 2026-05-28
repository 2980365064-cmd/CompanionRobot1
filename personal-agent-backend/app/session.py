"""SQLite persistence for sessions and structured memory."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from app.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_at(days: int | None = None) -> str:
    d = days if days is not None else settings.l2_retention_days
    return (datetime.now(timezone.utc) + timedelta(days=d)).isoformat()


class SessionStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate_episodic(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(episodic_memories)")}
        if not cols:
            return
        if "open_loops" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN open_loops TEXT")
        if "expires_at" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN expires_at TEXT")
        if "archived" not in cols:
            conn.execute("ALTER TABLE episodic_memories ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            "UPDATE episodic_memories SET expires_at=? WHERE expires_at IS NULL OR expires_at=''",
            (_expires_at(),),
        )

    def _ensure_indexes(self, conn: sqlite3.Connection) -> None:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_device ON episodic_memories(device_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_expires ON episodic_memories(device_id, expires_at)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_device ON semantic_facts(device_id)")

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
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
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    confidence REAL DEFAULT 0.8,
                    source_session TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._migrate_episodic(conn)
            self._ensure_indexes(conn)

    def get_or_create_session(self, device_id: str, session_id: str | None) -> str:
        with self._conn() as conn:
            if session_id:
                row = conn.execute("SELECT id, status FROM sessions WHERE id=?", (session_id,)).fetchone()
                if row and row["status"] == "active":
                    conn.execute("UPDATE sessions SET last_active=? WHERE id=?", (_utc_now(), session_id))
                    return session_id
            new_id = str(uuid4())
            now = _utc_now()
            conn.execute(
                "INSERT INTO sessions(id, device_id, status, created_at, last_active) VALUES (?,?,?,?,?)",
                (new_id, device_id, "active", now, now),
            )
            return new_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, role, content, _utc_now()),
            )
            conn.execute("UPDATE sessions SET last_active=? WHERE id=?", (_utc_now(), session_id))

    def count_turns(self, session_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id=? AND role='user'",
                (session_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def get_recent_messages(self, session_id: str, limit: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id=? ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_oldest_messages(self, session_id: str, limit: int) -> list[dict]:
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
        if not message_ids:
            return
        placeholders = ",".join("?" * len(message_ids))
        with self._conn() as conn:
            conn.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", message_ids)

    def get_session_messages(self, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def close_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE sessions SET status='closed', last_active=? WHERE id=?", (_utc_now(), session_id))

    def add_episodic(
        self,
        device_id: str,
        session_id: str,
        summary: str,
        topics: str = "",
        open_loops: str = "",
    ) -> None:
        now = _utc_now()
        exp = _expires_at()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO episodic_memories(
                    device_id, session_id, summary, topics, open_loops, created_at, expires_at, archived
                ) VALUES (?,?,?,?,?,?,?,0)
                """,
                (device_id, session_id, summary, topics, open_loops, now, exp),
            )

    def list_episodic_active(self, device_id: str, limit: int = 30) -> list[dict]:
        now = _utc_now()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, summary, topics, open_loops, created_at, expires_at
                FROM episodic_memories
                WHERE device_id=? AND archived=0 AND expires_at > ?
                ORDER BY id DESC LIMIT ?
                """,
                (device_id, now, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_expired_episodic(self, device_id: str | None = None) -> list[dict]:
        now = _utc_now()
        with self._conn() as conn:
            if device_id:
                rows = conn.execute(
                    """
                    SELECT id, device_id, session_id, summary, topics, open_loops, created_at, expires_at
                    FROM episodic_memories
                    WHERE device_id=? AND archived=0 AND expires_at <= ?
                    ORDER BY id ASC
                    """,
                    (device_id, now),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, device_id, session_id, summary, topics, open_loops, created_at, expires_at
                    FROM episodic_memories
                    WHERE archived=0 AND expires_at <= ?
                    ORDER BY device_id, id ASC
                    """,
                    (now,),
                ).fetchall()
        return [dict(r) for r in rows]

    def archive_episodic(self, episodic_ids: list[int]) -> None:
        if not episodic_ids:
            return
        placeholders = ",".join("?" * len(episodic_ids))
        with self._conn() as conn:
            conn.execute(
                f"UPDATE episodic_memories SET archived=1 WHERE id IN ({placeholders})",
                episodic_ids,
            )

    def list_episodic(self, device_id: str, limit: int = 20) -> list[dict]:
        """Backward-compatible alias: active L2 only."""
        return self.list_episodic_active(device_id, limit)

    def add_fact(self, device_id: str, fact: str, category: str, confidence: float, source_session: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO semantic_facts(device_id, fact, category, confidence, source_session, created_at)
                VALUES (?,?,?,?,?,?)
                """,
                (device_id, fact, category, confidence, source_session, _utc_now()),
            )

    def list_facts(self, device_id: str, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT fact, category, confidence FROM semantic_facts
                WHERE device_id=? ORDER BY id DESC LIMIT ?
                """,
                (device_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]


store = SessionStore()
