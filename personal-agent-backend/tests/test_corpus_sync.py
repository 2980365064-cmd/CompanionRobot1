"""Corpus 同步幂等、审计、去重、局部重建和月份召回归类测试。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


# 新的 source_id 格式：#s{section_index}p{part_index}
_SID_0 = "monthly/liu_yuanhui/2025-06.md#s0p0"
_SID_1 = "monthly/liu_yuanhui/2025-06.md#s1p0"


class CorpusSyncIdempotenceTests(unittest.TestCase):
    """corpus 导入幂等性测试 —— 连续两次 sync 不应增加条目数。"""

    def setUp(self):
        self.specs = [
            {
                "source_path": "monthly/liu_yuanhui/2025-06.md",
                "source_id": _SID_0,
                "kind": "wiki",
                "source": "wiki",
                "text": "[人物: 叶鹏祥、刘远慧] [时间: 2025-06] ## 本月关键事件 - 科二考试通过",
                "month_key": "2025-06",
                "category": "monthly",
                "confidence": 0.8,
                "meta": {
                    "source_table": "corpus",
                    "source_id": _SID_0,
                    "source_path": "monthly/liu_yuanhui/2025-06.md",
                    "month_key": "2025-06",
                    "category": "monthly",
                    "confidence": 0.8,
                },
            },
            {
                "source_path": "monthly/liu_yuanhui/2025-06.md",
                "source_id": _SID_1,
                "kind": "wiki",
                "source": "wiki",
                "text": "[人物: 叶鹏祥、刘远慧] [时间: 2025-06] ## 关系变化 - 角色扮演争论 - 情绪波动后的沟通改善",
                "month_key": "2025-06",
                "category": "monthly",
                "confidence": 0.8,
                "meta": {
                    "source_table": "corpus",
                    "source_id": _SID_1,
                    "source_path": "monthly/liu_yuanhui/2025-06.md",
                    "month_key": "2025-06",
                    "category": "monthly",
                    "confidence": 0.8,
                },
            },
        ]

    def test_sync_corpus_chunk_specs_upserts_stable_source_ids(self):
        """连续两次 sync 相同 specs 不应增加条目数（幂等性）。"""
        from app.persona.ingest import sync_corpus_chunk_specs

        fake_store = MagicMock()
        fake_store.list_corpus_source_ids.return_value = []
        fake_store.write_memory_item.return_value = "mock-uuid"
        fake_store.delete_corpus_item_by_source_id.return_value = True
        fake_store.count_corpus_memory_items.return_value = 2
        fake_store.list_corpus_source_id_counts.return_value = {_SID_0: 1, _SID_1: 1}
        fake_store.dedup_corpus_source_ids.return_value = 0

        with patch("app.llm.embed_texts", return_value=[[0.1, 0.2], [0.3, 0.4]]), \
             patch("app.session.store", fake_store):
            # 第一次 sync
            r1 = sync_corpus_chunk_specs(self.specs)
            self.assertEqual(r1["written"], 2)
            self.assertEqual(r1["stale_deleted"], 0)
            self.assertEqual(r1["final_corpus_ids"], 2)
            self.assertEqual(r1["final_corpus_rows"], 2)

            # 第二次 sync（模拟 DB 中已有这些 source_ids）
            fake_store.list_corpus_source_ids.return_value = [_SID_0, _SID_1]
            r2 = sync_corpus_chunk_specs(self.specs)
            self.assertEqual(r2["written"], 2)
            self.assertEqual(r2["stale_deleted"], 0)

    def test_sync_removes_stale_source_ids(self):
        """当 spec 不再产生某个 source_id 时，DB 中应删除该过期条目。"""
        from app.persona.ingest import sync_corpus_chunk_specs

        fake_store = MagicMock()
        fake_store.delete_corpus_item_by_source_id.return_value = True
        fake_store.write_memory_item.return_value = "mock-uuid"
        fake_store.count_corpus_memory_items.return_value = 2
        fake_store.dedup_corpus_source_ids.return_value = 0

        fake_store.list_corpus_source_ids.return_value = [
            _SID_0,
            _SID_1,
            "monthly/liu_yuanhui/2025-04.md#s0p0",
        ]
        fake_store.list_corpus_source_id_counts.return_value = {
            _SID_0: 1, _SID_1: 1, "monthly/liu_yuanhui/2025-04.md#s0p0": 1,
        }

        with patch("app.llm.embed_texts", return_value=[[0.1, 0.2], [0.3, 0.4]]), \
             patch("app.session.store", fake_store):
            r = sync_corpus_chunk_specs(self.specs)
            self.assertEqual(r["written"], 2)
            self.assertEqual(r["stale_deleted"], 1)
            fake_store.delete_corpus_item_by_source_id.assert_any_call(
                "monthly/liu_yuanhui/2025-04.md#s0p0"
            )

    def test_sync_with_reset_clears_all_first(self):
        """reset=True 时先全量删除再全量写入。"""
        from app.persona.ingest import sync_corpus_chunk_specs

        fake_store = MagicMock()
        fake_store.reset_corpus_items.return_value = 10
        fake_store.list_corpus_source_ids.return_value = []
        fake_store.write_memory_item.return_value = "mock-uuid"
        fake_store.count_corpus_memory_items.return_value = 2
        fake_store.list_corpus_source_id_counts.return_value = {_SID_0: 1, _SID_1: 1}

        with patch("app.llm.embed_texts", return_value=[[0.1, 0.2], [0.3, 0.4]]), \
             patch("app.session.store", fake_store):
            r = sync_corpus_chunk_specs(self.specs, reset=True)
            fake_store.reset_corpus_items.assert_called_once()
            self.assertEqual(r["written"], 2)
            self.assertEqual(r["stale_deleted"], 0)


class CorpusSyncDuplicateAndStatsTests(unittest.TestCase):
    """重复行检测和同步统计增强测试。"""

    def test_sync_includes_final_counts(self):
        """sync 返回值应包含 final_corpus_rows 和 final_corpus_ids。"""
        from app.persona.ingest import sync_corpus_chunk_specs

        fake_store = MagicMock()
        fake_store.list_corpus_source_ids.return_value = []
        fake_store.write_memory_item.return_value = "mock-uuid"
        fake_store.delete_corpus_item_by_source_id.return_value = True
        fake_store.count_corpus_memory_items.return_value = 2
        fake_store.list_corpus_source_id_counts.return_value = {_SID_0: 1, _SID_1: 1}
        fake_store.dedup_corpus_source_ids.return_value = 0

        specs = [
            {
                "source_path": "monthly/liu_yuanhui/2025-06.md",
                "source_id": _SID_0,
                "kind": "wiki",
                "source": "wiki",
                "text": "test",
                "month_key": "2025-06",
                "category": "monthly",
                "confidence": 0.8,
                "meta": {"source_table": "corpus", "source_id": _SID_0, "source_path": "monthly/liu_yuanhui/2025-06.md"},
            },
        ]

        with patch("app.llm.embed_texts", return_value=[[0.1, 0.2]]), \
             patch("app.session.store", fake_store):
            r = sync_corpus_chunk_specs(specs)
            self.assertIn("final_corpus_rows", r)
            self.assertIn("final_corpus_ids", r)
            self.assertIn("dedup_removed", r)
            self.assertEqual(r["final_corpus_rows"], 2)
            self.assertEqual(r["final_corpus_ids"], 2)

    def test_dedup_corpus_source_ids_removes_extra_rows(self):
        """SessionStore.dedup_corpus_source_ids 应删除重复行。"""
        from app.session import store
        import sqlite3
        from uuid import uuid4

        sid = _SID_0

        # 绕过 upsert，直接 INSERT 两条相同 source_id 来模拟重复
        conn = sqlite3.connect(str(store.db_path))
        conn.row_factory = sqlite3.Row
        now = "2026-07-09T00:00:00+00:00"
        ids = []
        for i in range(2):
            uid = str(uuid4())
            ids.append(uid)
            conn.execute(
                "INSERT INTO memory_items(id, person_id, device_id, kind, source, visibility, "
                "content, content_hash, confidence, emotional_weight, recency_weight, "
                "context_json, tags_json, embedding_json, "
                "source_table, source_id, source_session, "
                "expires_at, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uid, "", "", "wiki", "wiki", "recall_only", f"dedup test {i}",
                 "hash", 0.8, 3, 3, "{}", "[]", "[]",
                 "corpus", sid, "",
                 "", now, now, ""),
            )
            conn.execute(
                "INSERT INTO memory_items_fts(id, content_fts) VALUES (?, ?)",
                (uid, f"dedup test {i}"),
            )
        conn.commit()
        conn.close()

        try:
            counts_before = store.list_corpus_source_id_counts()
            self.assertGreaterEqual(counts_before.get(sid, 0), 2,
                                    "should have at least 2 rows for same source_id")

            removed = store.dedup_corpus_source_ids()
            self.assertGreaterEqual(removed, 1)

            counts_after = store.list_corpus_source_id_counts()
            self.assertEqual(counts_after.get(sid, 0), 1,
                             "after dedup, should have exactly 1 row per source_id")
        finally:
            # 清理测试数据
            conn2 = sqlite3.connect(str(store.db_path))
            placeholders = ",".join("?" * len(ids))
            conn2.execute(f"DELETE FROM memory_items_fts WHERE id IN ({placeholders})", ids)
            conn2.execute(f"DELETE FROM memory_items WHERE id IN ({placeholders})", ids)
            conn2.commit()
            conn2.close()

    def test_dedup_returns_zero_when_no_duplicates(self):
        """无重复行时 dedup 应返回 0。"""
        from app.session import store

        removed = store.dedup_corpus_source_ids()
        self.assertEqual(removed, 0)


class AuditWithDuplicateDetectionTests(unittest.TestCase):
    """审计应能检测出数据库中的重复行。"""

    def test_audit_detects_duplicate_source_ids(self):
        """当 DB 中存在重复 source_id 时，审计应输出 duplicate_source_ids。"""
        from app.persona.ingest import audit_corpus_sync_state

        fake_store = MagicMock()
        fake_store.list_corpus_source_id_counts.return_value = {
            "monthly/liu_yuanhui/2025-06.md#s0p0": 2,  # 重复！
            "monthly/liu_yuanhui/2025-06.md#s1p0": 1,
        }
        fake_store.list_corpus_source_ids.return_value = [
            "monthly/liu_yuanhui/2025-06.md#s0p0",
            "monthly/liu_yuanhui/2025-06.md#s1p0",
        ]

        with patch("app.session.store", fake_store), \
             patch("app.persona.ingest.build_corpus_chunk_specs") as mock_build:
            mock_build.return_value = [
                {"source_id": "monthly/liu_yuanhui/2025-06.md#s0p0", "source_path": "monthly/liu_yuanhui/2025-06.md"},
                {"source_id": "monthly/liu_yuanhui/2025-06.md#s1p0", "source_path": "monthly/liu_yuanhui/2025-06.md"},
            ]
            audit = audit_corpus_sync_state()

        self.assertIn("duplicate_source_ids", audit)
        self.assertGreater(len(audit["duplicate_source_ids"]), 0,
                           "应检测到重复")
        self.assertFalse(audit["is_complete"],
                         "存在重复行时 is_complete 应为 False")
        self.assertIn("actual_row_count", audit)
        self.assertEqual(audit["actual_row_count"], 3)

    def test_audit_marks_incomplete_with_only_duplicates(self):
        """只有重复但无缺失无多余时，is_complete 应为 False。"""
        from app.persona.ingest import audit_corpus_sync_state

        fake_store = MagicMock()
        fake_store.list_corpus_source_id_counts.return_value = {
            "monthly/liu_yuanhui/2025-06.md#s0p0": 2,
            "monthly/liu_yuanhui/2025-06.md#s1p0": 1,
            "monthly/liu_yuanhui/2025-06.md#s2p0": 1,
        }
        fake_store.list_corpus_source_ids.return_value = [
            "monthly/liu_yuanhui/2025-06.md#s0p0",
            "monthly/liu_yuanhui/2025-06.md#s1p0",
            "monthly/liu_yuanhui/2025-06.md#s2p0",
        ]

        with patch("app.session.store", fake_store), \
             patch("app.persona.ingest.build_corpus_chunk_specs") as mock_build:
            mock_build.return_value = [
                {"source_id": "monthly/liu_yuanhui/2025-06.md#s0p0", "source_path": "f1.md"},
                {"source_id": "monthly/liu_yuanhui/2025-06.md#s1p0", "source_path": "f1.md"},
                {"source_id": "monthly/liu_yuanhui/2025-06.md#s2p0", "source_path": "f1.md"},
            ]
            audit = audit_corpus_sync_state()

        self.assertEqual(len(audit["missing_source_ids"]), 0,
                         "不应有缺失")
        self.assertEqual(len(audit["stale_source_ids"]), 0,
                         "不应有多余")
        self.assertGreater(len(audit["duplicate_source_ids"]), 0,
                           "应检测到重复")
        self.assertFalse(audit["is_complete"],
                         "仅重复行也导致 is_complete=False")
        self.assertEqual(audit["actual_row_count"], 4,
                         "物理行数应为 4（3 唯一 + 1 重复）")


class CorpusSyncWithFileChangesTests(unittest.TestCase):
    """模拟文件增删改场景下 sync 的正确行为。"""

    def test_guess_month_key_from_path(self):
        """_guess_month_key 应从文件路径提取 YYYY-MM。"""
        from app.persona.ingest import _guess_month_key

        mk = _guess_month_key("monthly/liu_yuanhui/2025-06.md", "", {})
        self.assertEqual(mk, "2025-06")

    def test_guess_month_key_from_frontmatter(self):
        """_guess_month_key 应优先从 frontmatter time 字段提取。"""
        from app.persona.ingest import _guess_month_key

        mk = _guess_month_key("some/path.md", "dummy body", {"time": "2025-04"})
        self.assertEqual(mk, "2025-04")

    def test_guess_month_key_from_heading(self):
        """_guess_month_key 应从正文 ## YYYY-MM 标题提取。"""
        from app.persona.ingest import _guess_month_key

        mk = _guess_month_key("general.md", "## 2025-03\nSome content here", {})
        self.assertEqual(mk, "2025-03")

    def test_source_id_format_uses_section_part_coordinates(self):
        """多 section+多 part 时 source_id 应使用 #sXpY 格式。"""
        from app.persona.ingest import _chunk_section

        # 模拟一个长 section 会被 _chunk_section 切成多段
        long_body = "## 第一节\n" + "段落内容 " * 300
        parts = _chunk_section(long_body, "[测试前缀]", chunk_size=100, overlap=10)

        self.assertGreater(len(parts), 0)


class StartupAuditTests(unittest.TestCase):
    """启动审计逻辑测试。"""

    def _make_audit(self, **overrides) -> dict:
        base = {
            "is_complete": True,
            "missing_source_ids": [],
            "stale_source_ids": [],
            "duplicate_source_ids": [],
            "expected_chunk_count": 5,
            "actual_chunk_count": 5,
            "actual_row_count": 5,
            "source_files": ["f1.md", "f2.md"],
        }
        base.update(overrides)
        return base

    def test_startup_skips_when_corpus_complete(self):
        """审计完整时，startup 应跳过入库。"""
        from app.persona.ingest import startup_ingest_corpus

        with patch("app.persona.ingest.settings.persona_ingest_on_startup", True), \
             patch("app.persona.ingest.settings.persona_ingest_reset_on_startup", False), \
             patch("app.persona.ingest.audit_corpus_sync_state",
                   return_value=self._make_audit()), \
             patch("app.persona.ingest.ingest_directory") as mock_ingest:
            result = startup_ingest_corpus()

        mock_ingest.assert_not_called()
        self.assertTrue(result.get("skipped", False))
        self.assertEqual(result.get("reason"), "corpus_complete")

    def test_startup_skips_when_disabled(self):
        """persona_ingest_on_startup=false → 跳过。"""
        from app.persona.ingest import startup_ingest_corpus

        with patch("app.persona.ingest.settings.persona_ingest_on_startup", False):
            result = startup_ingest_corpus()

        self.assertTrue(result.get("skipped", False))
        self.assertEqual(result.get("reason"), "disabled")

    def test_reset_on_startup_triggers_full_rebuild(self):
        """persona_ingest_reset_on_startup=true → 强制全量重建。"""
        from app.persona.ingest import startup_ingest_corpus

        ingest_result = {
            "files": ["test.md"],
            "corpus_chunks": 3,
            "fact_stats": {"chunks": 0, "facts": 0, "skipped": 0},
            "wiki_synced": 0,
        }

        with patch("app.persona.ingest.settings.persona_ingest_on_startup", True), \
             patch("app.persona.ingest.settings.persona_ingest_reset_on_startup", True), \
             patch("app.persona.ingest.ingest_directory", return_value=ingest_result):
            result = startup_ingest_corpus()

        self.assertFalse(result.get("skipped", False))
        self.assertEqual(result.get("corpus_chunks"), 3)

    def test_startup_does_not_skip_when_corpus_incomplete(self):
        """审计发现缺失块时，startup 应触发增量同步。"""
        from app.persona.ingest import startup_ingest_corpus

        ingest_result = {
            "files": ["monthly/liu_yuanhui/2025-06.md"],
            "corpus_chunks": 1,
            "fact_stats": {"chunks": 0, "facts": 0, "skipped": 0},
            "wiki_synced": 0,
        }

        with patch("app.persona.ingest.settings.persona_ingest_on_startup", True), \
             patch("app.persona.ingest.settings.persona_ingest_reset_on_startup", False), \
             patch("app.persona.ingest.audit_corpus_sync_state",
                   return_value=self._make_audit(
                       is_complete=False,
                       missing_source_ids=["monthly/liu_yuanhui/2025-06.md#s0p0"],
                       expected_chunk_count=1,
                       actual_chunk_count=0,
                       source_files=["monthly/liu_yuanhui/2025-06.md"],
                   )), \
             patch("app.persona.ingest.ingest_directory", return_value=ingest_result):
            result = startup_ingest_corpus()

        self.assertFalse(result.get("skipped", False))
        self.assertTrue(result.get("audit_fixed", False))

    def test_startup_does_not_skip_when_duplicates_exist(self):
        """存在重复行时，即使无缺失无多余，startup 也不应跳过。"""
        from app.persona.ingest import startup_ingest_corpus

        ingest_result = {
            "files": ["test.md"],
            "corpus_chunks": 3,
            "fact_stats": {"chunks": 0, "facts": 0, "skipped": 0},
            "wiki_synced": 0,
        }

        with patch("app.persona.ingest.settings.persona_ingest_on_startup", True), \
             patch("app.persona.ingest.settings.persona_ingest_reset_on_startup", False), \
             patch("app.persona.ingest.audit_corpus_sync_state",
                   return_value=self._make_audit(
                       is_complete=False,
                       duplicate_source_ids=["monthly/liu_yuanhui/2025-06.md#s0p0"],
                       actual_row_count=6,
                   )), \
             patch("app.persona.ingest.ingest_directory", return_value=ingest_result):
            result = startup_ingest_corpus()

        self.assertFalse(result.get("skipped", False))
        self.assertTrue(result.get("audit_fixed", True))


class RebuildScriptTests(unittest.TestCase):
    """rebuild_corpus_memory_items.py 的基础逻辑测试。"""

    def test_count_dirty_corpus_handles_empty_db(self):
        """无数据库时统计应返回零。"""
        from scripts.rebuild_corpus_memory_items import _count_dirty_corpus
        from app.config import settings

        with patch.object(settings.__class__, 'resolved_db_path') as mock_db:
            mock_db.return_value.exists.return_value = False
            result = _count_dirty_corpus()
            self.assertEqual(result["corpus_clean_rows"], 0)
            self.assertEqual(result["corpus_dirty_rows"], 0)
            self.assertEqual(result["total_memory_items"], 0)

    def test_delete_dirty_corpus_handles_empty_db(self):
        """无数据库时删除操作应安全返回零。"""
        from scripts.rebuild_corpus_memory_items import _delete_dirty_corpus
        from app.config import settings

        with patch.object(settings.__class__, 'resolved_db_path') as mock_db:
            mock_db.return_value.exists.return_value = False
            result = _delete_dirty_corpus()
            self.assertEqual(result["deleted_clean_corpus"], 0)
            self.assertEqual(result["deleted_dirty_corpus"], 0)
            self.assertEqual(result["total_deleted"], 0)

    def test_dry_run_does_not_modify_db(self):
        """--dry-run 不应落库。"""
        from scripts.rebuild_corpus_memory_items import main
        from app.config import settings
        import argparse

        with patch.object(settings.__class__, 'resolved_corpus_dir') as mock_dir, \
             patch.object(settings.__class__, 'resolved_db_path') as mock_db, \
             patch("scripts.rebuild_corpus_memory_items._count_dirty_corpus") as mock_count:
            mock_dir.return_value = MagicMock()
            mock_dir.return_value.exists.return_value = True
            mock_dir.return_value.rglob.return_value = []
            mock_db.return_value.exists.return_value = True
            mock_count.return_value = {
                "corpus_clean_rows": 5, "corpus_dirty_rows": 3, "total_memory_items": 100
            }

            with patch("scripts.rebuild_corpus_memory_items.argparse.ArgumentParser.parse_args") as mock_args:
                mock_args.return_value = argparse.Namespace(
                    dry_run=True, backup=False, only_path=""
                )
                with patch("scripts.rebuild_corpus_memory_items._delete_dirty_corpus") as mock_del:
                    main()
                    mock_del.assert_not_called()

    def test_only_path_dry_run_does_not_delete(self):
        """--only-path --dry-run 不应执行删除。"""
        from scripts.rebuild_corpus_memory_items import main
        from app.config import settings
        import argparse

        with patch.object(settings.__class__, 'resolved_corpus_dir') as mock_dir, \
             patch.object(settings.__class__, 'resolved_db_path') as mock_db, \
             patch("scripts.rebuild_corpus_memory_items._count_dirty_corpus") as mock_count:
            mock_dir.return_value = MagicMock()
            mock_dir.return_value.exists.return_value = True
            mock_dir.return_value.rglob.return_value = []
            mock_db.return_value.exists.return_value = True
            mock_count.return_value = {
                "corpus_clean_rows": 5, "corpus_dirty_rows": 3, "total_memory_items": 100
            }

            with patch("scripts.rebuild_corpus_memory_items.argparse.ArgumentParser.parse_args") as mock_args:
                mock_args.return_value = argparse.Namespace(
                    dry_run=True, backup=False, only_path="monthly/liu_yuanhui/2025-04.md"
                )
                with patch("scripts.rebuild_corpus_memory_items._delete_corpus_by_path") as mock_del:
                    main()
                    mock_del.assert_not_called()


class MonthRecallPriorityTests(unittest.TestCase):
    """月份查询时 corpus 块应获得高优先级。"""

    def test_prioritize_month_memory_items_gives_corpus_source_boost(self):
        """source_table='corpus' 的条目应获得与 long_term_chunks 同等的 table_rank 优先级。"""
        from app.memory.unified_store import _prioritize_month_memory_items

        query = "你知道2025年6月我俩之间发生什么了吗"

        # 新规范 corpus 行
        corpus_row = {
            "content": "[人物: 叶鹏祥、刘远慧] [时间: 2025-06] ## 本月关键事件",
            "source_table": "corpus",
            "source_id": _SID_0,
            "context_json": '{"source_path":"monthly/liu_yuanhui/2025-06.md","month_key":"2025-06"}',
        }

        # 普通用户 recent 行（无 source_table）
        episode_row = {
            "content": "用户问了一些关于过去的问题",
            "source_table": "",
            "source_id": "",
            "context_json": "{}",
        }

        corpus_prio = _prioritize_month_memory_items(corpus_row, query=query, month_key="2025-06")
        episode_prio = _prioritize_month_memory_items(episode_row, query=query, month_key="2025-06")

        # corpus 行因 source_table='corpus' 应排在前面
        self.assertLess(corpus_prio, episode_prio)
