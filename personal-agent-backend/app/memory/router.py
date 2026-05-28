"""Unified memory recall: Profile + L1 + L2 + L3 (mode-dependent)."""

from __future__ import annotations

from app.config import settings
from app.memory.episodic import episodic_memory
from app.memory.extractor import rollup_expired_l2
from app.memory.intent import needs_l3_recall
from app.memory.semantic import semantic_memory
from app.memory.working import working_memory


class MemoryRouter:
    def recall(self, device_id: str, session_id: str, query: str) -> dict:
        rollup_expired_l2(device_id)

        mode = (settings.l3_recall_mode or "hybrid").lower()
        intent_hit = needs_l3_recall(query)
        semantic: list[str] = []
        l3_triggered = False

        if mode == "always":
            l3_triggered = True
            semantic = semantic_memory.recall(device_id, query, settings.rag_top_k)
        elif mode == "hybrid":
            light = semantic_memory.recall(device_id, query, settings.l3_light_top_k)
            if intent_hit:
                l3_triggered = True
                semantic = semantic_memory.recall(device_id, query, settings.rag_top_k)
            else:
                semantic = light
                l3_triggered = bool(light)
        else:
            if intent_hit:
                l3_triggered = True
                semantic = semantic_memory.recall(device_id, query, settings.rag_top_k)

        return {
            "working": working_memory.get_recent(session_id),
            "episodic": episodic_memory.recall(device_id, query, settings.episodic_top_k),
            "semantic": semantic,
            "l3_triggered": l3_triggered,
            "l3_intent": intent_hit,
        }


memory_router = MemoryRouter()
