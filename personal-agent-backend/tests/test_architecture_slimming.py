import ast
import unittest
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]


def _module_source(relative: str) -> str:
    return (BACKEND / relative).read_text("utf-8")


class ArchitectureSlimmingTests(unittest.TestCase):
    def test_main_is_a_thin_app_composition_root(self):
        source = _module_source("app/main.py")
        tree = ast.parse(source)
        top_level_defs = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        self.assertIn("create_app", top_level_defs)
        self.assertIn("lifespan", top_level_defs)
        self.assertLessEqual(len(top_level_defs), 8)
        self.assertNotIn("@app.get", source)
        self.assertNotIn("@app.post", source)
        self.assertNotIn("@app.websocket", source)

    def test_routes_are_split_by_surface_area(self):
        expected = [
            "app/routers/__init__.py",
            "app/routers/admin.py",
            "app/routers/audio_ws.py",
            "app/routers/chat_ws.py",
            "app/routers/health.py",
            "app/routers/pages.py",
        ]

        for relative in expected:
            self.assertTrue((BACKEND / relative).is_file(), relative)

    def test_session_facade_exposes_only_current_memory_write_api(self):
        source = _module_source("app/session.py")
        tree = ast.parse(source)
        session_store = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SessionStore"
        )
        method_names = {
            node.name
            for node in session_store.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertNotIn("add_fact", method_names)
        self.assertNotIn("get_fact_by_id", method_names)
        self.assertNotIn("get_fact_by_text", method_names)
        self.assertNotIn("list_facts", method_names)

    def test_runtime_artifacts_are_ignored(self):
        ignore_text = (BACKEND.parent / ".gitignore").read_text("utf-8")
        required_patterns = [
            "personal-agent-backend/*.db-shm",
            "personal-agent-backend/*.db-wal",
            "personal-agent-backend/app/*.db",
            "personal-agent-backend/.codegraph/",
            "personal-agent-backend/test_output.*",
        ]

        for pattern in required_patterns:
            self.assertIn(pattern, ignore_text)

    def test_public_routes_remain_registered_after_split(self):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 200)
        self.assertIn(client.get("/chat").status_code, {200, 404})
        self.assertIn(client.get("/admin").status_code, {200, 404})

        self.assertEqual(str(app.url_path_for("websocket_chat")), "/ws/v1/chat")
        self.assertEqual(str(app.url_path_for("websocket_audio")), "/ws/v2/audio")
        self.assertEqual(str(app.url_path_for("http_chat")), "/v1/chat")
        self.assertEqual(str(app.url_path_for("admin_list_persons")), "/v1/admin/persons")
        self.assertEqual(str(app.url_path_for("admin_list_tasks")), "/v1/admin/tasks")

    def test_prompt_context_exposes_semantic_memorypack_boundary_only(self):
        from app.memory.prompt_context import build_prompt_context
        from app.memory.schema import MemoryPackV2, MemoryItem, MemoryKind, MemorySource

        pack = MemoryPackV2(
            items=[
                MemoryItem(
                    kind=MemoryKind.PREFERENCE,
                    source=MemorySource.USER_DECLARED,
                    content="她不吃香菜",
                    confidence=0.95,
                )
            ],
            history=[{"role": "user", "content": "记住我不吃香菜"}],
            diagnostics={
                "person_id": "person-1",
                "interlocutor_mode": "girlfriend",
                "memory_miss": 0,
                "has_recent": True,
                "has_long_term": True,
            },
        )

        context = build_prompt_context(pack)

        self.assertEqual(context["history"], pack.history)
        self.assertEqual(context["memory_pack"], pack)
        self.assertEqual(
            set(context),
            {"history", "memory_pack", "diagnostics", "person_id", "guest_mode",
             "interlocutor_mode", "identity_hint", "memory_miss"},
        )

    def test_router_defines_retrieval_planner_instead_of_layer_first_policy(self):
        from app.memory.router import RetrievalPlanner

        planner = RetrievalPlanner()

        greeting = planner.plan("你好", working=[])
        self.assertFalse(greeting.needs_memory)
        self.assertFalse(greeting.search_long_term)
        self.assertEqual(greeting.long_term_top_k, 0)

        recall = planner.plan("你还记得我不吃香菜吗", working=[])
        self.assertTrue(recall.needs_memory)
        self.assertTrue(recall.search_recent_memory)
        self.assertTrue(recall.search_long_term)
        self.assertGreaterEqual(recall.long_term_top_k, 3)

    def test_agent_prompt_uses_humanized_memorypack_sections(self):
        from app.agent import build_messages
        from app.memory.prompt_context import build_prompt_context
        from app.memory.schema import MemoryPackV2, MemoryItem, MemoryKind, MemorySource

        pack = MemoryPackV2(
            items=[
                MemoryItem(
                    kind=MemoryKind.PREFERENCE,
                    source=MemorySource.USER_DECLARED,
                    content="她不吃香菜",
                    confidence=0.95,
                )
            ],
            history=[{"role": "user", "content": "我今天想吃火锅"}],
            diagnostics={
                "person_id": "person-1",
                "interlocutor_mode": "girlfriend",
                "memory_miss": 0,
            },
        )
        context = build_prompt_context(pack)

        messages = build_messages(
            "你是叶鹏祥。",
            context,
            "你记得我的忌口吗？",
            device_id="test-device",
            person_profile={"person_id": "person-1", "confirmed": True},
            memory_pack=pack,
        )

        system = messages[0]["content"]
        self.assertIn("## 你该记得的相关事", system)
        self.assertIn("她不吃香菜", system)
        for forbidden in ("工程层级", "向量", "检索", "命中"):
            self.assertNotIn(forbidden, system)

    # ── 第二阶段新增测试 ──────────────────────────────────────────────

    def test_unified_memory_store_is_the_only_router_storage_gateway(self):
        """MemoryRouter.recall() 通过 unified_memory_store.search 召回，
        不再直接调用旧 核心事实/近期记忆/长期记忆 读接口。"""
        source = _module_source("app/memory/router.py")

        # 断言包含 unified_memory_store.search
        self.assertIn("unified_memory_store.search", source)

        # 断言不直接调用旧接口
        self.assertNotIn("recall_scored", source)
        self.assertNotIn("list_core_cached", source)

    def test_router_returns_semantic_recall_shape(self):
        """MemoryRouter.recall() 返回语义 dict（history/items/diagnostics），
        不含旧 core/recent_hit/long_term_hit/matches/working。"""
        from unittest.mock import patch, MagicMock
        from app.memory.router import MemoryRouter, RetrievalPlanner
        from app.memory.unified_store import MemorySearchResult
        from app.memory.schema import MemoryItem, MemoryKind, MemorySource, MemoryVisibility

        router = MemoryRouter(planner=RetrievalPlanner())

        mock_result = MemorySearchResult(
            core_items=[
                MemoryItem(kind=MemoryKind.PREFERENCE, source=MemorySource.USER_DECLARED,
                           content="不吃香菜", confidence=0.95, visibility=MemoryVisibility.ALWAYS),
            ],
            recent_items=[],
            long_term_items=[],
            related_items=[],
            diagnostics={"has_recent": False, "has_long_term": False, "core_memory_count": 1,
                         "person_id": "person-1", "month_key": "", "recent": [], "long_term": [], "related": []},
        )

        with patch("app.memory.router.get_recent_context") as mock_wm, \
             patch("app.memory.router.memory_scoped_to_person", return_value=True), \
             patch("app.memory.router.unified_memory_store.search", return_value=mock_result), \
             patch("app.memory.router.emotion_trajectory", return_value=[]), \
             patch("app.memory.router.extract_self_name", return_value=None), \
             patch("app.memory.router._cached_embed", return_value=[]):
            mock_wm.return_value = [{"role": "user", "content": "测试"}]
            memory = router.recall("dev-1", "sess-1", "测试", person_id="person-1")

        # 断言含语义字段
        self.assertIn("history", memory)
        self.assertIn("items", memory)
        self.assertIn("diagnostics", memory)
        self.assertEqual(set(memory), {"history", "items", "diagnostics", "person_id", "guest_mode", "memory_miss"})

    def test_orchestrator_consumes_memory_items(self):
        """MemoryOrchestrator._build_v2_pack() 从 memory["items"] 构建
        MemoryPackV2。"""
        from app.memory.orchestrator import MemoryOrchestrator
        from app.memory.schema import MemoryItem, MemoryKind, MemorySource, MemoryVisibility

        orch = MemoryOrchestrator()
        # 构造语义 memory dict
        memory = {
            "history": [{"role": "user", "content": "你好"}],
            "items": [
                MemoryItem(kind=MemoryKind.PREFERENCE, source=MemorySource.USER_DECLARED,
                           content="不吃香菜", confidence=0.95, visibility=MemoryVisibility.ALWAYS),
            ],
            "diagnostics": {
                "has_recent": True, "has_long_term": False,
                "core_memory_count": 1, "recent": [], "long_term": [], "related": [],
            },
            "memory_miss": False,
            "person_id": "person-1",
            "guest_mode": False,
        }

        pack = orch._build_v2_pack(
            memory, [], "person-1", "girlfriend", "", "你好",
        )

        # 断言 items 正确
        self.assertGreaterEqual(len(pack.items), 1)
        pref_items = [it for it in pack.items if it.kind == MemoryKind.PREFERENCE]
        self.assertEqual(len(pref_items), 1)
        self.assertEqual(pref_items[0].content, "不吃香菜")


    def test_consolidator_writes_memory_items_through_unified_store(self):
        """Consolidator 的 _build_items_from_turn() 对"记住我不吃香菜"生成
        PREFERENCE/FACT 类型的 MemoryItem，confidence >= 0.9。"""
        from app.memory.consolidator import MemoryConsolidator, TurnClassification
        from app.memory.schema import MemoryKind

        consolidator = MemoryConsolidator()
        cls = TurnClassification(
            is_remember_intent=True,
            confidence=0.95,
            reason="explicit_remember",
        )

        items = consolidator._build_items_from_turn(
            "dev-1", "sess-1", "person-1",
            "记住我不吃香菜", "好的，我记住了",
            cls, turn_emotional_event=None,
        )

        # 断言至少写入一个 MemoryItem
        self.assertGreaterEqual(len(items), 1)

        # 断言包含 PREFERENCE 或 FACT
        kinds = {it.kind for it in items}
        self.assertTrue(kinds & {MemoryKind.PREFERENCE, MemoryKind.FACT})

        # 断言 confidence >= 0.9
        for it in items:
            self.assertGreaterEqual(it.confidence, 0.9)

    # ── 第二阶段风险收口新增测试 ──────────────────────────────────────

    def test_remember_intent_does_not_trigger_duplicate_旧版_writes(self):
        """is_remember_intent=True 时只走 unified write，
        不再触发 capture_user_stated_facts。"""
        from unittest.mock import patch, MagicMock
        from app.memory.consolidator import MemoryConsolidator, TurnClassification
        from app.memory.schema import MemoryItem, MemoryKind, MemorySource, MemoryVisibility

        consolidator = MemoryConsolidator()

        # 准备分类
        cls = TurnClassification(
            is_remember_intent=True, confidence=0.95, reason="explicit_remember",
        )

        # 构建 expected items
        items = consolidator._build_items_from_turn(
            "dev-1", "sess-1", "person-1",
            "记住我不吃香菜", "好的，我记住了",
            cls, turn_emotional_event=None,
        )
        self.assertGreaterEqual(len(items), 1, "记住指令应生成至少 1 个 MemoryItem")

        # mock 整个 process_turn 的依赖链
        with patch("app.memory.consolidator.is_verified_person_id", return_value=True), \
             patch("app.memory.consolidator.maybe_compact_working_context", return_value=False), \
             patch("app.memory.core_facts.capture_core_fact_from_message", return_value=[]), \
             patch("app.memory.guard.capture_user_stated_facts") as mock_capture, \
             patch("app.memory.consolidator.unified_memory_store.write_item", return_value="long_term:test") as mock_write, \
             patch("app.memory.consolidator.relationship_manager") as mock_rel, \
             patch("app.memory.consolidator.open_loop_manager") as mock_ol, \
             patch("app.memory.consolidator.store"):
            # 情感事件提取返回 None（非高重要性）
            mock_rel.load.return_value = MagicMock()
            mock_rel.expire_old_state.return_value = MagicMock()
            mock_rel.update_from_turn.return_value = MagicMock()
            mock_ol.detect_create.return_value = []
            mock_ol.detect_resolve.return_value = []
            # mock emotional_extractor
            with patch("app.memory.emotional_events.emotional_extractor") as mock_ee:
                mock_ee.extract_from_turn.return_value = None

                result = consolidator.process_turn(
                    "dev-1", "sess-1",
                    "记住我不吃香菜", "好的，我记住了",
                    {}, "person-1",
                )

        # 断言 unified write 被调用
        self.assertGreaterEqual(mock_write.call_count, 1,
                                "记住指令应触发 unified_memory_store.write_item")
        # 断言旧路径被跳过
        mock_capture.assert_not_called()
        # 断言统计字段
        self.assertGreaterEqual(result.unified_items_written, 1)

    def test_correction_applies_before_writing_correction_item(self):
        """纠错路径：先 try_apply_memory_corrections，再写 CORRECTION item。"""
        from unittest.mock import patch, MagicMock, call
        from app.memory.consolidator import MemoryConsolidator, TurnClassification

        consolidator = MemoryConsolidator()
        cls = TurnClassification(
            is_correction=True, confidence=1.0, reason="correction_signal: 不是",
        )

        # 记录调用顺序
        call_order = []

        def _record_corr(*args, **kwargs):
            call_order.append("correction")
            return {"stats": {"deleted_facts": 1, "deleted_chunks": 0,
                              "patched_chunks": 0, "added_facts": 1, "deleted_core_facts": 0}}

        def _record_write(*args, **kwargs):
            call_order.append("write_item")
            return "long_term:corr_test"

        with patch("app.memory.consolidator.is_verified_person_id", return_value=True), \
             patch("app.memory.consolidator.classify_turn", return_value=cls), \
             patch("app.memory.consolidator.maybe_compact_working_context", return_value=False), \
             patch("app.memory.core_facts.capture_core_fact_from_message", return_value=[]), \
             patch("app.memory.guard.capture_user_stated_facts"), \
             patch("app.memory.correction.try_apply_memory_corrections",
                   side_effect=_record_corr) as mock_corr, \
             patch("app.memory.consolidator.unified_memory_store.write_item",
                   side_effect=_record_write) as mock_write, \
             patch("app.memory.consolidator.relationship_manager") as mock_rel, \
             patch("app.memory.consolidator.open_loop_manager") as mock_ol, \
             patch("app.memory.consolidator.store"):
            mock_rel.load.return_value = MagicMock()
            mock_rel.expire_old_state.return_value = MagicMock()
            mock_rel.update_from_turn.return_value = MagicMock()
            mock_ol.detect_create.return_value = []
            mock_ol.detect_resolve.return_value = []
            with patch("app.memory.emotional_events.emotional_extractor") as mock_ee:
                mock_ee.extract_from_turn.return_value = None

                result = consolidator.process_turn(
                    "dev-1", "sess-1",
                    "不是上海，是杭州", "好的，我记住了",
                    {}, "person-1",
                )

        # 断言纠错先于 unified write
        self.assertIn("correction", call_order)
        self.assertIn("write_item", call_order)
        corr_idx = call_order.index("correction")
        write_idx = call_order.index("write_item")
        self.assertLess(corr_idx, write_idx,
                        "try_apply_memory_corrections 必须在 write_item 之前调用")
        # 断言纠错统计正确
        self.assertEqual(result.corrections_applied.get("deleted_facts"), 1)
        self.assertEqual(result.corrections_applied.get("added_facts"), 1)

    def test_single_source_evidence_does_not_force_unknown(self):
        """只有一种有效 evidence 时，不应强制机器人说"不确定"。
        （即 miss_lv=1 不再设置 should_admit_unknown=True）"""
        from app.memory.orchestrator import MemoryOrchestrator
        from app.memory.schema import MemoryItem, MemoryKind, MemorySource, MemoryVisibility

        orch = MemoryOrchestrator()

        # 场景：仅有 long_term_items，没有 recent
        memory = {
            "history": [{"role": "user", "content": "你记得我不吃香菜吗"}],
            "items": [
                MemoryItem(kind=MemoryKind.PREFERENCE, source=MemorySource.USER_DECLARED,
                           content="不吃香菜", confidence=0.95,
                           visibility=MemoryVisibility.ALWAYS),
            ],
            "diagnostics": {
                "has_recent": False,
                "has_long_term": True,
                "core_memory_count": 0,
                "recent": [],
                "long_term": [{"text": "她不吃香菜，点菜时要注意"}],
                "related": [],
                "evidence_count": 1,
                "evidence_weak": False,
                "evidence_sources": ["long_term"],
                "retrieval_plan": {"needs_memory": True},
            },
            "memory_miss": False,
            "person_id": "person-1",
            "guest_mode": False,
        }

        pack = orch._build_v2_pack(
            memory, [], "person-1", "girlfriend", "", "你记得我不吃香菜吗",
        )

        # 断言：有 evidence 时不强制说"不确定"
        self.assertFalse(
            pack.missing_memory.get("should_admit_unknown"),
            "单一来源但有有效 evidence 时，不应强制 should_admit_unknown"
        )

        # 场景 2：evidence_weak=True → 仍不应强制说"不确定"
        memory_weak = {
            **memory,
            "diagnostics": {**memory["diagnostics"], "evidence_weak": True},
        }
        pack_weak = orch._build_v2_pack(
            memory_weak, [], "person-1", "girlfriend", "", "你记得我不吃香菜吗",
        )

        # evidence_weak 时 miss_lv=1，但 should_admit_unknown 仍为 False
        self.assertFalse(
            pack_weak.missing_memory.get("should_admit_unknown"),
            "evidence_weak 时也仅提示不确定，不强制说完全不记得"
        )
        # miss_lv 应为 1（evidence_weak）
        self.assertEqual(pack_weak.diagnostics.get("memory_miss"), 1)

        # 场景 3：完全无 items 且 needs_memory=True → miss_lv=2
        memory_none = {
            "history": [{"role": "user", "content": "你还记得张三吗"}],
            "items": [],
            "diagnostics": {
                "has_recent": False, "has_long_term": False,
                "core_memory_count": 0, "recent": [], "long_term": [], "related": [],
                "evidence_count": 0, "evidence_weak": True, "evidence_sources": [],
                "retrieval_plan": {"needs_memory": True},
            },
            "memory_miss": True,
            "person_id": "person-1",
            "guest_mode": False,
        }
        pack_none = orch._build_v2_pack(
            memory_none, [], "person-1", "girlfriend", "", "你还记得张三吗",
        )
        self.assertTrue(
            pack_none.missing_memory.get("should_admit_unknown"),
            "完全未命中记忆时必须 should_admit_unknown"
        )

    def test_evidence_weak_adds_uncertainty_boundary_without_forcing_unknown(self):
        """证据较弱时 format_prompt_block 输出"隐约记得"级提示，不说"完全没印象"。
        同时验证 should_admit_unknown 仍为 False（不强制说不知道）。"""
        from app.memory.schema import MemoryPackV2, MemoryItem, MemoryKind, MemorySource

        pack = MemoryPackV2(
            items=[
                MemoryItem(
                    kind=MemoryKind.FACT,
                    source=MemorySource.INFERRED,
                    content="好像提过喜欢吃火锅",
                    confidence=0.6,
                ),
            ],
            missing_memory={
                "should_admit_unknown": False,
                "reason": "证据较弱 —— 有部分线索但不完全确定，坦诚表达不确定即可",
            },
        )

        block = pack.format_prompt_block()

        # 应包含"不太确定的记忆"区块
        self.assertIn("关于不太确定的记忆", block,
                      "证据较弱时也应输出不确定提示块")
        # 应包含"隐约记得"级别的温和提示
        self.assertIn("隐约记得", block,
                      "证据较弱时应使用'隐约记得'等温和措辞")
        # 不应包含"完全没印象"（那是 miss_lv=2 的语言）
        self.assertNotIn("完全没印象", block,
                         "证据较弱时不应使用'完全没印象'（那是 miss_lv=2 的措辞）")
        # 验证 should_admit_unknown 仍为 False
        self.assertFalse(pack.missing_memory.get("should_admit_unknown"),
                         "证据较弱时 should_admit_unknown 必须为 False")
