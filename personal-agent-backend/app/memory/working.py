"""L1 working memory (session sliding window)."""

from __future__ import annotations

from app.config import settings
from app.session import store


class WorkingMemory:
    def get_recent(self, session_id: str) -> list[dict]:
        limit = settings.working_memory_turns * 2
        return store.get_recent_messages(session_id, limit)

    def append(self, session_id: str, role: str, content: str) -> None:
        store.add_message(session_id, role, content)

    def count_turns(self, session_id: str) -> int:
        return store.count_turns(session_id)


working_memory = WorkingMemory()
