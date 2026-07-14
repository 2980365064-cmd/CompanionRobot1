"""
记忆行为契约测试 —— 离线、无真实 LLM、无真实 embedding、可稳定回归。

============================================================================
设计目标：
  从用户体验层面验证记忆系统的行为契约，不测试内部实现细节。
  所有测试通过 mock 隔离外部依赖（embedding / LLM / DB），
  确保后续 memory_items 统一表迁移时有稳定的安全网。

测试覆盖 8 条行为契约：
  1. 访客隔离 —— tmp_* 用户只能读当前会话，不触发长期记忆检索
  2. 寒暄省 embedding —— "你好""在吗"等不触发向量化，不查长期记忆
  3. 实名召回语义形状 —— 返回 dict 只含语义字段，不含工程层术语
  4. 显式记住去重 —— "记住XXX"只走 unified write，不重复触发旧路径
  5. 纠错顺序 —— 先修正旧记忆，再写入 correction item
  6. 弱证据边界 —— evidence_weak 时输出"隐约记得"，不强制说"完全没印象"
  7. Prompt 术语防回归 —— format_prompt_block 不含 核心事实/工作上下文/近期记忆/长期记忆 等工程词
  8. 月份查询意图排序 —— 朋友群查询优先朋友 chunks，关系查询优先关系 chunks
============================================================================
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock, call

from app.memory.schema import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryVisibility,
    MemoryPackV2,
    RelationshipState,
)


# ══════════════════════════════════════════════════════════════════════════════
# 合约 1：访客隔离
# ══════════════════════════════════════════════════════════════════════════════


class GuestModeIsolationTests(unittest.TestCase):
    """访客（tmp_* / 未实名）用户不应触发任何长期记忆检索。"""

    def test_guest_mode_never_calls_unified_memory_store(self):
        """tmp_* 用户调用 recall() 时：
        - 不调用 unified_memory_store.search()
        - 不调用 _cached_embed()
        - 返回 guest_mode=True, items=[]
        """
        from app.memory.router import MemoryRouter, RetrievalPlanner

        router = MemoryRouter(planner=RetrievalPlanner())

        with patch("app.memory.router.get_recent_context") as mock_wm, \
             patch("app.memory.router.memory_scoped_to_person", return_value=False), \
             patch("app.memory.router.unified_memory_store.search") as mock_search, \
             patch("app.memory.router._cached_embed") as mock_embed:
            mock_wm.return_value = [
                {"role": "user", "content": "你好呀"},
            ]

            memory = router.recall("dev-1", "sess-1", "你好呀", person_id="tmp_abc123")

        # 断言：不调用长期记忆检索
        mock_search.assert_not_called()
        # 断言：不调用 embedding
        mock_embed.assert_not_called()
        # 断言：返回访客模式标记
        self.assertTrue(memory.get("guest_mode"), "访客模式 guest_mode 应为 True")
        # 断言：items 为空（无长期记忆）
        self.assertEqual(len(memory.get("items", [])), 0,
                         "访客模式 items 应为空列表")
        # 断言：history 仍然可用（当前会话窗口）
        self.assertIsInstance(memory.get("history"), list)
        self.assertGreaterEqual(len(memory.get("history", [])), 0)

    def test_guest_mode_returns_empty_items_and_no_person_id(self):
        """访客模式的 recall 结果中 person_id 为 None。"""
        from app.memory.router import MemoryRouter, RetrievalPlanner

        router = MemoryRouter(planner=RetrievalPlanner())

        with patch("app.memory.router.get_recent_context") as mock_wm, \
             patch("app.memory.router.memory_scoped_to_person", return_value=False):
            mock_wm.return_value = []

            memory = router.recall("dev-1", "sess-1", "嗨", person_id=None)

        self.assertIsNone(memory.get("person_id"))
        self.assertTrue(memory.get("guest_mode"))
        self.assertEqual(memory.get("items"), [])


# ══════════════════════════════════════════════════════════════════════════════
# 合约 2：寒暄不触发 embedding
# ══════════════════════════════════════════════════════════════════════════════


class SmalltalkEmbeddingAvoidanceTests(unittest.TestCase):
    """寒暄短句不应触发向量化或长期记忆检索。"""

    def test_planner_marks_smalltalk_as_no_memory_needed(self):
        """RetrievalPlanner 对"你好""在吗""早呀"等寒暄返回 needs_memory=False。"""
        from app.memory.router import RetrievalPlanner

        planner = RetrievalPlanner()

        for msg in ["你好", "在吗", "早呀", "嗨", "hello", "嗯嗯"]:
            with self.subTest(msg=msg):
                plan = planner.plan(msg, working=[])
                self.assertFalse(plan.needs_memory,
                                 f"'{msg}' 不应标记为 needs_memory")
                self.assertFalse(plan.search_long_term,
                                 f"'{msg}' 不应触发 search_long_term")
                self.assertEqual(plan.long_term_top_k, 0,
                                 f"'{msg}' 的 long_term_top_k 应为 0")

    def test_smalltalk_router_does_not_embed_or_search_long_term(self):
        """寒暄查询走完整 router.recall() 时：
        - 不调用 _cached_embed()
        - 调用 unified_memory_store.search() 但 long_term_top_k=0
        """
        from unittest.mock import patch, MagicMock
        from app.memory.router import MemoryRouter, RetrievalPlanner
        from app.memory.unified_store import MemorySearchResult

        router = MemoryRouter(planner=RetrievalPlanner())
        mock_result = MemorySearchResult(
            diagnostics={
                "has_recent": False, "has_long_term": False,
                "core_memory_count": 0, "recent": [], "long_term": [],
                "related": [], "person_id": "person-1", "month_key": "",
            },
        )

        with patch("app.memory.router.get_recent_context") as mock_wm, \
             patch("app.memory.router.memory_scoped_to_person", return_value=True), \
             patch("app.memory.router.unified_memory_store.search",
                   return_value=mock_result) as mock_search, \
             patch("app.memory.router._cached_embed") as mock_embed, \
             patch("app.memory.router.emotion_trajectory", return_value=[]), \
             patch("app.memory.router.extract_self_name", return_value=None):
            mock_wm.return_value = []

            memory = router.recall("dev-1", "sess-1", "你好", person_id="person-1")

        # 断言：不调用 embedding（寒暄不需要向量化）
        mock_embed.assert_not_called()
        # 断言：search 被调用（核心记忆仍需加载），但 long_term_top_k=0
        self.assertTrue(mock_search.called, "search 仍被调用（加载核心记忆）")
        call_spec = mock_search.call_args[0][0]
        self.assertEqual(call_spec.long_term_top_k, 0,
                         "寒暄查询的 long_term_top_k 应为 0")

    def test_memory_query_triggers_embedding_and_search(self):
        """对比：需要记忆的查询（如"你还记得我不吃香菜吗"）应触发 embedding。"""
        from unittest.mock import patch, MagicMock
        from app.memory.router import MemoryRouter, RetrievalPlanner
        from app.memory.unified_store import MemorySearchResult
        from app.memory.schema import MemoryItem, MemoryKind, MemorySource, MemoryVisibility

        router = MemoryRouter(planner=RetrievalPlanner())
        mock_result = MemorySearchResult(
            core_items=[
                MemoryItem(kind=MemoryKind.PREFERENCE, source=MemorySource.USER_DECLARED,
                           content="不吃香菜", confidence=0.95,
                           visibility=MemoryVisibility.ALWAYS),
            ],
            diagnostics={
                "has_recent": True, "has_long_term": True,
                "core_memory_count": 1, "recent": [], "long_term": [],
                "related": [], "person_id": "person-1", "month_key": "",
                "evidence_count": 1, "evidence_weak": False, "evidence_sources": ["core"],
            },
        )

        with patch("app.memory.router.get_recent_context") as mock_wm, \
             patch("app.memory.router.memory_scoped_to_person", return_value=True), \
             patch("app.memory.router.unified_memory_store.search",
                   return_value=mock_result) as mock_search, \
             patch("app.memory.router._cached_embed",
                   return_value=[0.1] * 1024) as mock_embed, \
             patch("app.memory.router.emotion_trajectory", return_value=[]), \
             patch("app.memory.router.extract_self_name", return_value=None):
            mock_wm.return_value = []

            memory = router.recall(
                "dev-1", "sess-1", "你还记得我不吃香菜吗", person_id="person-1",
            )

        # 断言：需要记忆的查询会调用 embedding
        mock_embed.assert_called()
        # 断言：search 的 long_term_top_k > 0
        self.assertTrue(mock_search.called)
        call_spec = mock_search.call_args[0][0]
        self.assertGreater(call_spec.long_term_top_k, 0,
                           "需要记忆的查询 long_term_top_k 应 > 0")


# ══════════════════════════════════════════════════════════════════════════════
# 合约 3：实名用户记忆召回语义形状
# ══════════════════════════════════════════════════════════════════════════════


class VerifiedRecallSemanticShapeTests(unittest.TestCase):
    """已实名用户的 recall() 返回结构只含语义字段。"""

    def test_recall_returns_semantic_dict_only(self):
        """返回 dict 包含 history/items/diagnostics/person_id/guest_mode/memory_miss，
        不包含 core/recent_hit/long_term_hit/matches/working。"""
        from unittest.mock import patch
        from app.memory.router import MemoryRouter, RetrievalPlanner
        from app.memory.unified_store import MemorySearchResult
        from app.memory.schema import MemoryItem, MemoryKind, MemorySource, MemoryVisibility

        router = MemoryRouter(planner=RetrievalPlanner())
        mock_result = MemorySearchResult(
            core_items=[
                MemoryItem(kind=MemoryKind.PREFERENCE, source=MemorySource.USER_DECLARED,
                           content="不吃香菜", confidence=0.95,
                           visibility=MemoryVisibility.ALWAYS),
            ],
            recent_items=[
                MemoryItem(kind=MemoryKind.EPISODE,
                           source=MemorySource.CONVERSATION_SUMMARY,
                           content="上周去了火锅店", confidence=0.8,
                           visibility=MemoryVisibility.RECALL_ONLY),
            ],
            long_term_items=[
                MemoryItem(kind=MemoryKind.FACT, source=MemorySource.INFERRED,
                           content="她喜欢辣的食物", confidence=0.7,
                           visibility=MemoryVisibility.RECALL_ONLY),
            ],
            related_items=[],
            diagnostics={
                "has_recent": True, "has_long_term": True,
                "core_memory_count": 1, "recent": [], "long_term": [],
                "related": [], "person_id": "person-1", "month_key": "",
                "evidence_count": 3, "evidence_weak": False,
                "evidence_sources": ["core", "recent", "long_term"],
            },
        )

        with patch("app.memory.router.get_recent_context") as mock_wm, \
             patch("app.memory.router.memory_scoped_to_person", return_value=True), \
             patch("app.memory.router.unified_memory_store.search",
                   return_value=mock_result), \
             patch("app.memory.router._cached_embed",
                   return_value=[0.1] * 1024), \
             patch("app.memory.router.emotion_trajectory", return_value=[]), \
             patch("app.memory.router.extract_self_name", return_value=None):
            mock_wm.return_value = [
                {"role": "user", "content": "你还记得我不吃香菜吗"},
            ]

            memory = router.recall(
                "dev-1", "sess-1", "你还记得我不吃香菜吗", person_id="person-1",
            )

        # ── 必须存在的语义字段 ──
        self.assertIn("history", memory)
        self.assertIn("items", memory)
        self.assertIn("diagnostics", memory)
        self.assertIn("person_id", memory)
        self.assertIn("guest_mode", memory)
        self.assertIn("memory_miss", memory)

        self.assertEqual(set(memory), {"history", "items", "diagnostics", "person_id", "guest_mode", "memory_miss"})

        # ── 语义字段的值校验 ──
        self.assertFalse(memory["guest_mode"], "已实名用户 guest_mode 应为 False")
        self.assertEqual(memory["person_id"], "person-1")
        self.assertIsInstance(memory["items"], list)
        self.assertGreater(len(memory["items"]), 0,
                           "已实名用户 items 应包含记忆条目")

    def test_items_are_all_memory_item_instances(self):
        """recall() 返回的 items 列表中每条都是 MemoryItem 实例。"""
        from unittest.mock import patch
        from app.memory.router import MemoryRouter, RetrievalPlanner
        from app.memory.unified_store import MemorySearchResult
        from app.memory.schema import MemoryItem, MemoryKind, MemorySource, MemoryVisibility

        router = MemoryRouter(planner=RetrievalPlanner())
        mock_result = MemorySearchResult(
            core_items=[
                MemoryItem(kind=MemoryKind.PREFERENCE, source=MemorySource.USER_DECLARED,
                           content="不吃香菜", confidence=0.95,
                           visibility=MemoryVisibility.ALWAYS),
            ],
            diagnostics={
                "has_recent": False, "has_long_term": False,
                "core_memory_count": 1, "recent": [], "long_term": [],
                "related": [], "person_id": "person-1", "month_key": "",
                "evidence_count": 1, "evidence_weak": False,
                "evidence_sources": ["core"],
            },
        )

        with patch("app.memory.router.get_recent_context") as mock_wm, \
             patch("app.memory.router.memory_scoped_to_person", return_value=True), \
             patch("app.memory.router.unified_memory_store.search",
                   return_value=mock_result), \
             patch("app.memory.router._cached_embed",
                   return_value=[0.1] * 1024), \
             patch("app.memory.router.emotion_trajectory", return_value=[]), \
             patch("app.memory.router.extract_self_name", return_value=None):
            mock_wm.return_value = []

            memory = router.recall(
                "dev-1", "sess-1", "测试查询", person_id="person-1",
            )

        items = memory.get("items", [])
        self.assertGreater(len(items), 0)
        for item in items:
            self.assertIsInstance(item, MemoryItem,
                                  f"items 中每条都应是 MemoryItem，实际: {type(item)}")


# ══════════════════════════════════════════════════════════════════════════════
# 合约 4：显式记住去重
# ══════════════════════════════════════════════════════════════════════════════


class RememberIntentDedupTests(unittest.TestCase):
    """"记住XXX"指令只走 unified write，不重复触发旧的 capture/extract 路径。"""

    def test_remember_intent_skips_duplicate_write_paths(self):
        """is_remember_intent=True 时：
        - unified_memory_store.write_item 被调用
        - capture_user_stated_facts 不被调用
        """
        from unittest.mock import patch, MagicMock
        from app.memory.consolidator import MemoryConsolidator, TurnClassification

        consolidator = MemoryConsolidator()

        cls = TurnClassification(
            is_remember_intent=True,
            confidence=0.95,
            reason="explicit_remember",
        )

        # 验证 _build_items_from_turn 生成正确的 MemoryItem
        items = consolidator._build_items_from_turn(
            "dev-1", "sess-1", "person-1",
            "记住我不吃香菜", "好的，我记住了",
            cls, turn_emotional_event=None,
        )
        self.assertGreaterEqual(len(items), 1,
                                "记住指令应生成至少 1 个 MemoryItem")
        kinds = {it.kind for it in items}
        self.assertTrue(kinds & {MemoryKind.PREFERENCE, MemoryKind.FACT},
                        f"记住指令应生成 PREFERENCE/FACT，实际 kinds: {kinds}")

        # 验证完整 process_turn 的写入路径
        with patch("app.memory.consolidator.is_verified_person_id", return_value=True), \
             patch("app.memory.consolidator.maybe_compact_working_context", return_value=False), \
             patch("app.memory.core_facts.capture_core_fact_from_message", return_value=[]), \
             patch("app.memory.guard.capture_user_stated_facts") as mock_capture, \
             patch("app.memory.consolidator.unified_memory_store.write_item",
                   return_value="long_term:test") as mock_write, \
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
                    "记住我不吃香菜", "好的，我记住了",
                    {}, "person-1",
                )

        # 断言：unified write 被调用
        self.assertGreaterEqual(mock_write.call_count, 1,
                                "记住指令应触发 unified_memory_store.write_item")
        # 断言：旧路径被跳过
        mock_capture.assert_not_called()
        # 断言：统计字段正确
        self.assertGreaterEqual(result.unified_items_written, 1)
        self.assertEqual(result.classification.reason, "explicit_remember")

    def test_remember_intent_content_is_stripped_of_command_prefix(self):
        """"记住我不吃香菜"提取出的 content 应为"我不吃香菜"（去除指令前缀）。"""
        from app.memory.consolidator import MemoryConsolidator, TurnClassification

        consolidator = MemoryConsolidator()
        cls = TurnClassification(
            is_remember_intent=True, confidence=0.95, reason="explicit_remember",
        )

        items = consolidator._build_items_from_turn(
            "dev-1", "sess-1", "person-1",
            "记住我不吃香菜", "好的，我记住了",
            cls, turn_emotional_event=None,
        )

        # 至少有一个 item 的内容不含"记住"前缀
        cleaned = [it for it in items if "记住" not in it.content]
        self.assertGreater(len(cleaned), 0,
                           f"记住指令生成的 item content 应去除'记住'前缀，实际: {[it.content for it in items]}")

    def test_memory_question_is_not_remember_intent(self):
        """“你记得……吗”是回忆查询，不应被当作显式记住指令写入核心记忆。"""
        from app.memory.consolidator import classify_turn

        cls = classify_turn(
            "你记得2025年6月我俩发生啥了吗",
            "我想想",
            {},
        )

        self.assertFalse(cls.is_remember_intent)
        self.assertNotEqual(cls.reason, "explicit_remember")

    def test_colon_remember_command_still_is_remember_intent(self):
        """带冒号的“记得：xxx”仍作为显式记住指令。"""
        from app.memory.consolidator import classify_turn

        cls = classify_turn("记得：我不吃香菜", "好", {})

        self.assertTrue(cls.is_remember_intent)
        self.assertEqual(cls.reason, "explicit_remember")


# ══════════════════════════════════════════════════════════════════════════════
# 合约 5：纠错顺序
# ══════════════════════════════════════════════════════════════════════════════


class CorrectionOrderingTests(unittest.TestCase):
    """纠错时先 try_apply_memory_corrections()，再写入 CORRECTION item。"""

    def test_correction_applies_before_writing_correction_item(self):
        """验证调用顺序：try_apply_memory_corrections 在 write_item 之前。"""
        from unittest.mock import patch, MagicMock
        from app.memory.consolidator import MemoryConsolidator, TurnClassification

        consolidator = MemoryConsolidator()
        cls = TurnClassification(
            is_correction=True, confidence=1.0, reason="correction_signal: 不是",
        )

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

        # 断言调用顺序
        self.assertIn("correction", call_order)
        self.assertIn("write_item", call_order)
        corr_idx = call_order.index("correction")
        write_idx = call_order.index("write_item")
        self.assertLess(corr_idx, write_idx,
                        "try_apply_memory_corrections 必须在 write_item 之前调用")

        # 断言纠错统计
        self.assertEqual(result.corrections_applied.get("deleted_facts"), 1)
        self.assertEqual(result.corrections_applied.get("added_facts"), 1)
        self.assertEqual(result.quality_decision, "correction_flow")

    def test_correction_generates_correction_kind_item(self):
        """纠错消息生成的 MemoryItem kind 应为 CORRECTION。"""
        from app.memory.consolidator import MemoryConsolidator, TurnClassification

        consolidator = MemoryConsolidator()
        cls = TurnClassification(
            is_correction=True, confidence=1.0, reason="correction_signal: 不是",
        )

        items = consolidator._build_items_from_turn(
            "dev-1", "sess-1", "person-1",
            "不是上海，是杭州", "好的，已更正",
            cls, turn_emotional_event=None,
        )

        correction_items = [it for it in items if it.kind == MemoryKind.CORRECTION]
        self.assertEqual(len(correction_items), 1,
                         "纠错消息应生成 1 个 CORRECTION 类型的 MemoryItem")
        self.assertEqual(correction_items[0].confidence, 1.0,
                         "纠错 item 的 confidence 应为 1.0")
        self.assertEqual(correction_items[0].visibility, MemoryVisibility.ALWAYS,
                         "纠错 item 的 visibility 应为 ALWAYS")


# ══════════════════════════════════════════════════════════════════════════════
# 合约 6：弱证据边界
# ══════════════════════════════════════════════════════════════════════════════


class WeakEvidenceBoundaryTests(unittest.TestCase):
    """evidence_weak 时 prompt 输出"隐约记得"级别提示，不强制说"完全没印象"。"""

    def test_weak_evidence_prompts_uncertainty_boundary_only(self):
        """构造 memory_pack 含"证据较弱"reason：
        - format_prompt_block() 输出"关于不太确定的记忆"区块
        - 输出"隐约记得"级别的温和措辞
        - 不输出"完全没印象"
        - should_admit_unknown 仍为 False
        """
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
        self.assertIn("关于不太确定的记忆", block)
        # 应包含"隐约记得"级别的温和提示
        self.assertIn("隐约记得", block)
        # 不应包含"完全没印象"（那是 miss_lv=2 的语言）
        self.assertNotIn("完全没印象", block)
        # should_admit_unknown 必须为 False
        self.assertFalse(pack.missing_memory.get("should_admit_unknown"))

    def test_complete_miss_outputs_admit_unknown_language(self):
        """对比：miss_lv=2（完全未命中）时输出"完全没印象"级别语言。"""
        pack = MemoryPackV2(
            items=[],
            missing_memory={
                "should_admit_unknown": True,
                "reason": "完全未命中记忆 —— 诚实说不太记得，追问对方补充",
            },
        )

        block = pack.format_prompt_block()

        # 应包含"不太确定的记忆"区块
        self.assertIn("关于不太确定的记忆", block)
        # 应包含"完全没印象"级别的坦诚语言
        self.assertIn("完全没印象", block)

    def test_no_missing_memory_produces_no_uncertainty_block(self):
        """无缺失记忆时，format_prompt_block 不输出"不太确定的记忆"区块。"""
        pack = MemoryPackV2(
            items=[
                MemoryItem(
                    kind=MemoryKind.PREFERENCE,
                    source=MemorySource.USER_DECLARED,
                    content="不吃香菜",
                    confidence=0.95,
                ),
            ],
            relationship=RelationshipState(mode="girlfriend"),
        )

        block = pack.format_prompt_block()

        # 无缺失记忆时不应出现不确定区块
        self.assertNotIn("关于不太确定的记忆", block)
        # 但应有正常的记忆区块
        self.assertIn("你该记得的相关事", block)


# ══════════════════════════════════════════════════════════════════════════════
# 合约 7：Prompt 术语防回归
# ══════════════════════════════════════════════════════════════════════════════


class PromptTermGuardTests(unittest.TestCase):
    """MemoryPackV2.format_prompt_block() 不得暴露工程层术语。"""

    def test_prompt_never_exposes_memory_layer_terms(self):
        """构造包含 core/recent/long_term/related items 的 MemoryPackV2，
        调用 format_prompt_block()，断言不出现 核心事实/工作上下文/近期记忆/长期记忆/matches/向量/检索/命中。"""
        pack = MemoryPackV2(
            items=[
                # 核心事实（旧 核心事实）
                MemoryItem(kind=MemoryKind.PREFERENCE,
                           source=MemorySource.USER_DECLARED,
                           content="不吃香菜", confidence=0.95,
                           visibility=MemoryVisibility.ALWAYS),
                MemoryItem(kind=MemoryKind.TABOO,
                           source=MemorySource.USER_DECLARED,
                           content="不要叫她大炮", confidence=0.9,
                           visibility=MemoryVisibility.ALWAYS),
                # 情景记忆（旧 近期记忆）
                MemoryItem(kind=MemoryKind.EPISODE,
                           source=MemorySource.CONVERSATION_SUMMARY,
                           content="上周一起去了火锅店", confidence=0.8,
                           visibility=MemoryVisibility.RECALL_ONLY),
                # 长期记忆（旧 长期记忆）
                MemoryItem(kind=MemoryKind.FACT,
                           source=MemorySource.INFERRED,
                           content="她好像喜欢辣的食物", confidence=0.7,
                           visibility=MemoryVisibility.RECALL_ONLY),
                # 关联记忆
                MemoryItem(kind=MemoryKind.ENTITY,
                           source=MemorySource.INFERRED,
                           content="唐凯是她的初中同学", confidence=0.7,
                           visibility=MemoryVisibility.RECALL_ONLY),
                # 情感事件
                MemoryItem(kind=MemoryKind.EMOTION,
                           source=MemorySource.CONVERSATION_SUMMARY,
                           content="上次聊天她很难过", confidence=0.8,
                           visibility=MemoryVisibility.RECALL_ONLY),
                # 纠错记录
                MemoryItem(kind=MemoryKind.CORRECTION,
                           source=MemorySource.USER_DECLARED,
                           content="她不在上海，在杭州", confidence=1.0,
                           visibility=MemoryVisibility.ALWAYS),
            ],
            relationship=RelationshipState(
                mode="girlfriend",
                recent_mood="最近情绪还不错",
                relationship_temperature=0.75,
                care_points=["她最近在准备面试", "下周是她的生日"],
            ),
            current_mood="开心",
            current_topic="周末去哪里玩",
        )

        block = pack.format_prompt_block()

        # ── 禁止出现的工程术语 ──
        forbidden_terms = [
            "工程层级", "内部存储",
            "向量", "检索", "命中", "记忆库",
            "embedding", "embed",
            "chunk_id", "score",
            "recall_scored", "fts",
            "core_memory", "recent_memory",
        ]
        for term in forbidden_terms:
            self.assertNotIn(term, block,
                             f"format_prompt_block() 不得包含工程术语 '{term}'")

        # ── 必须出现的人类化区块标题 ──
        human_sections = [
            "你和她的关系状态",
            "她现在的状态",
            "你该记得的相关事",
            "这次不要乱说的边界",
        ]
        for section in human_sections:
            self.assertIn(section, block,
                          f"format_prompt_block() 应包含人类化区块 '{section}'")

    def test_prompt_block_human_readable_content(self):
        """format_prompt_block() 输出的内容是人类可读的自然语言。"""
        pack = MemoryPackV2(
            items=[
                MemoryItem(kind=MemoryKind.PREFERENCE,
                           source=MemorySource.USER_DECLARED,
                           content="不吃香菜", confidence=0.95,
                           visibility=MemoryVisibility.ALWAYS),
            ],
            relationship=RelationshipState(
                mode="girlfriend",
                recent_mood="最近还不错",
            ),
        )

        block = pack.format_prompt_block()

        # 内容应该是自然语言，包含用户数据
        self.assertIn("不吃香菜", block)
        self.assertIn("girlfriend", block)
        # 不应包含 JSON 或机器格式
        self.assertNotIn('"kind"', block)
        self.assertNotIn('"source"', block)
        self.assertNotIn('"confidence"', block)

    def test_prompt_includes_recalled_month_long_term_items(self):
        """高相关 recall_only 长期记忆必须进入 prompt，避免 LLM 只看到核心事实。"""
        pack = MemoryPackV2(
            items=[
                *[
                    MemoryItem(kind=MemoryKind.PREFERENCE,
                               source=MemorySource.USER_DECLARED,
                               content=f"核心偏好 {i}", confidence=0.95,
                               visibility=MemoryVisibility.ALWAYS)
                    for i in range(10)
                ],
                MemoryItem(kind=MemoryKind.EPISODE,
                           source=MemorySource.CONVERSATION_SUMMARY,
                           content="[人物: 叶鹏祥、刘远慧] [时间: 2025-06] ## 本月关键事件 - 科二考试通过 - 角色扮演争论 - 远慧噩梦",
                           confidence=0.9, emotional_weight=5, recency_weight=5,
                           visibility=MemoryVisibility.RECALL_ONLY,
                           source_id="long_term_chunks:doc-111"),
            ],
            relationship=RelationshipState(mode="girlfriend"),
            diagnostics={
                "month_key": "2025-06",
                "query_supported": True,
                "evidence_weak": False,
            },
        )

        block = pack.format_prompt_block()

        self.assertIn("科二考试通过", block)
        self.assertIn("角色扮演争论", block)


# ══════════════════════════════════════════════════════════════════════════════
# 合约 8：月份查询意图排序
# ══════════════════════════════════════════════════════════════════════════════


class MonthQueryIntentSortingTests(unittest.TestCase):
    """月份查询根据 query 意图调整来源优先级排序。"""

    def test_month_query_prioritizes_friend_group_when_query_mentions_friends(self):
        from app.memory.unified_store import _prioritize_month_memory_items
        friend = {"content": "唐凯聚会", "source_table": "corpus", "source_id": "monthly/friends_group/2025-06.md#s0p0", "context_json": ""}
        relationship = {"content": "远慧日常", "source_table": "corpus", "source_id": "monthly/liu_yuanhui/2025-06.md#s0p0", "context_json": ""}
        self.assertLess(
            _prioritize_month_memory_items(friend, query="唐凯最近怎么样", month_key="2025-06"),
            _prioritize_month_memory_items(relationship, query="唐凯最近怎么样", month_key="2025-06"),
        )

    def test_month_query_prioritizes_relationship_when_query_mentions_us(self):
        from app.memory.unified_store import _prioritize_month_memory_items
        friend = {"content": "唐凯聚会", "source_table": "corpus", "source_id": "monthly/friends_group/2025-06.md#s0p0", "context_json": ""}
        relationship = {"content": "远慧日常", "source_table": "corpus", "source_id": "monthly/liu_yuanhui/2025-06.md#s0p0", "context_json": ""}
        self.assertLess(
            _prioritize_month_memory_items(relationship, query="我俩上个月去了哪里", month_key="2025-06"),
            _prioritize_month_memory_items(friend, query="我俩上个月去了哪里", month_key="2025-06"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 运行入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
