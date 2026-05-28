"""L2 episodic memory (7-day active window)."""

from __future__ import annotations

from app.config import settings
from app.llm import cosine_similarity, embed_texts
from app.session import store


class EpisodicMemory:
    def recall(self, device_id: str, query: str, top_k: int | None = None) -> list[str]:
        top_k = top_k or settings.episodic_top_k
        rows = store.list_episodic_active(device_id, limit=50)
        if not rows:
            return []

        recent_n = settings.l2_recall_recent
        out: list[str] = []
        seen: set[str] = set()

        for row in rows[:recent_n]:
            text = row["summary"]
            if text not in seen:
                seen.add(text)
                out.append(text)

        if len(out) >= top_k or not query.strip():
            return out[:top_k]

        q_emb = embed_texts([query])[0]
        scored: list[tuple[float, str]] = []
        for row in rows[recent_n:]:
            summary = row["summary"]
            if summary in seen:
                continue
            s_emb = embed_texts([summary])[0]
            score = cosine_similarity(q_emb, s_emb)
            if score > 0.15:
                scored.append((score, summary))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _, summary in scored[: settings.l2_recall_query_k]:
            if summary not in seen:
                seen.add(summary)
                out.append(summary)
            if len(out) >= top_k:
                break

        return out[:top_k]

    def save_summary(
        self,
        device_id: str,
        session_id: str,
        summary: str,
        topics: str = "",
        open_loops: str = "",
    ) -> None:
        store.add_episodic(device_id, session_id, summary, topics, open_loops)


episodic_memory = EpisodicMemory()
