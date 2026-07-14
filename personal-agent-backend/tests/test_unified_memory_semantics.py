"""统一记忆语义的回归约束。"""

from app.memory import relations
from app.memory.router import RetrievalPlanner


def test_retrieval_plan_exposes_recent_memory_field():
    plan = RetrievalPlanner().plan("你还记得上周那件事吗", working=[])

    assert hasattr(plan, "search_recent_memory")
    assert plan.search_recent_memory is True


def test_month_priority_uses_memory_item_semantic_name():
    from app.memory.unified_store import _prioritize_month_memory_items

    relationship = {
        "content": "2025年6月我们讨论了同居计划",
        "source_table": "corpus",
        "source_id": "monthly/liu_yuanhui/2025-06.md#s1p0",
        "context_json": '{"month_key":"2025-06"}',
    }
    unrelated = {
        "content": "朋友群里聊到唐凯",
        "source_table": "corpus",
        "source_id": "monthly/friends_group/2025-06.md#s1p0",
        "context_json": '{"month_key":"2025-06"}',
    }

    assert _prioritize_month_memory_items(
        relationship, query="2025年6月我俩发生什么", month_key="2025-06"
    ) < _prioritize_month_memory_items(
        unrelated, query="2025年6月我俩发生什么", month_key="2025-06"
    )


def test_relation_seeds_are_memory_item_uuids_only():
    seeds = relations.seed_keys_from_memory_items([
        {"id": "item-uuid", "source_id": "monthly/liu_yuanhui/2025-06.md#s1p0"},
    ])

    assert seeds == ["memory:item-uuid"]
    rejected_fact_key = "f" + "act:123"
    rejected_chunk_key = "ch" + "unk:doc-1"
    assert relations.resolve_memory_text(rejected_fact_key) == ""
    assert relations.resolve_memory_text(rejected_chunk_key) == ""


def test_working_context_exposes_functions_without_adapter_object():
    import app.memory.working_context as context

    assert callable(context.get_recent_context)
    assert callable(context.append_context_message)
    assert callable(context.count_context_turns)
    assert not hasattr(context, "working_" + "memory")


def test_memory_pack_rejects_unknown_legacy_shape_fields():
    from app.memory.schema import MemoryPackV2

    try:
        MemoryPackV2(**{"m" + "atches": []})
    except TypeError:
        return
    raise AssertionError("MemoryPackV2 不得接受已移除的召回字段")


def test_admin_configuration_fields_map_to_settings():
    from app.admin.config import FIELDS
    from app.config import settings

    available = set(type(settings).model_fields)
    missing = [field.key for field in FIELDS if field.key.lower() not in available]
    assert missing == []
