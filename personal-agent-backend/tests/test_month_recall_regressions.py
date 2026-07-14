from __future__ import annotations

import unittest
from unittest.mock import patch

from app.memory.schema import MemoryItem, MemoryKind, MemorySource, MemoryVisibility
from app.memory.unified_store import MemorySearchQuery


class MonthKeyParsingRegressionTests(unittest.TestCase):
    def test_parse_month_key_accepts_year_dash_month_after_nian(self):
        from app.memory.router import _parse_month_key

        self.assertEqual(
            _parse_month_key("你知道2025年-6月我俩之间发生什么了吗"),
            "2025-06",
        )


class StartupCorpusIngestRegressionTests(unittest.TestCase):
    def test_startup_ingest_does_not_skip_when_corpus_incomplete(self):
        """审计发现 DB 中缺失 corpus 块时，应触发增量同步而非跳过。"""
        from app.persona.ingest import startup_ingest_corpus

        ingest_result = {
            "files": ["monthly/liu_yuanhui/2025-06.md"],
            "corpus_chunks": 1,
            "fact_stats": {"chunks": 0, "facts": 0, "skipped": 0},
            "wiki_synced": 0,
        }
        incomplete_audit = {
            "is_complete": False,
            "missing_source_ids": ["monthly/liu_yuanhui/2025-06.md#s0p0"],
            "stale_source_ids": [],
            "expected_chunk_count": 1,
            "actual_chunk_count": 0,
            "source_files": ["monthly/liu_yuanhui/2025-06.md"],
        }

        with patch("app.persona.ingest.settings.persona_ingest_on_startup", True), \
             patch("app.persona.ingest.settings.persona_ingest_reset_on_startup", False), \
             patch("app.persona.ingest.audit_corpus_sync_state", return_value=incomplete_audit), \
             patch("app.persona.ingest.ingest_directory", return_value=ingest_result) as mock_ingest:
            result = startup_ingest_corpus()

        mock_ingest.assert_called_once()
        self.assertFalse(result.get("skipped", False))
        self.assertTrue(result.get("audit_fixed", False))
        self.assertEqual(result.get("files"), ["monthly/liu_yuanhui/2025-06.md"])


class UnifiedSearchMonthRecallRegressionTests(unittest.TestCase):
    def test_long_term_search_includes_wiki_corpus_items(self):
        from app.memory.unified_store import UnifiedMemoryStore

        store = UnifiedMemoryStore()
        monthly_row = {
            "id": "mi-month-1",
            "person_id": "",
            "device_id": "",
            "kind": "wiki",
            "source": "monthly/liu_yuanhui/2025-06.md",
            "visibility": "recall_only",
            "content": "[人物: 叶鹏祥、刘远慧] [时间: 2025-06] ## 本月关键事件 - 科二考试通过 - 角色扮演争论",
            "content_hash": "abc",
            "confidence": 0.9,
            "emotional_weight": 3,
            "recency_weight": 3,
            "context_json": "{}",
            "tags_json": "[]",
            "embedding_json": "[]",
            "source_table": "",
            "source_id": "",
            "source_session": "",
            "expires_at": "",
            "created_at": "2026-07-09T00:00:00+00:00",
            "updated_at": "2026-07-09T00:00:00+00:00",
            "deleted_at": "",
        }

        def _fake_search_memory_items(person_id, **kwargs):
            kinds = kwargs.get("kinds") or []
            if kwargs.get("visibility") == "always":
                return []
            if "wiki" in kinds and kwargs.get("month_key") == "2025-06":
                return [monthly_row]
            return []

        with patch("app.memory.unified_store.store.search_memory_items", side_effect=_fake_search_memory_items):
            result = store.search(
                MemorySearchQuery(
                    device_id="web-chat",
                    person_id="123",
                    query="你知道2025年-6月我俩之间发生什么了吗",
                    include_long_term=True,
                    long_term_top_k=6,
                    month_key="2025-06",
                )
            )

        self.assertTrue(result.long_term_items)
        self.assertEqual(result.long_term_items[0].kind, MemoryKind.WIKI)
        self.assertIn("科二考试通过", result.long_term_items[0].content)
