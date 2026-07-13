"""统一记忆关联图。

关联图只引用 ``memory_items.id``。实体和关系节点可用于描述关系，但不会替代
记忆项的稳定身份。
"""

from __future__ import annotations

from app.config import settings
from app.session import store

ENTITY_PREFIX = "entity:"
RELATIONSHIP_PREFIX = "relationship:"
MEMORY_PREFIX = "memory:"


def memory_key(item_id: str) -> str:
    """为一个统一记忆项生成关联图节点键。"""
    value = str(item_id or "").strip()
    return f"{MEMORY_PREFIX}{value}" if value else ""


def parse_relation_key(key: str) -> tuple[str, str]:
    """解析唯一支持的关联图节点键，未知格式直接拒绝。"""
    value = str(key or "").strip()
    if value.startswith(MEMORY_PREFIX):
        return "memory", value[len(MEMORY_PREFIX):]
    if value.startswith(ENTITY_PREFIX):
        return "entity", value[len(ENTITY_PREFIX):]
    if value.startswith(RELATIONSHIP_PREFIX):
        return "relationship", value[len(RELATIONSHIP_PREFIX):]
    return "unknown", ""


def resolve_memory_text(relation_key: str, person_id: str = "") -> str:
    """将关系图节点转换为可注入 prompt 的文本。"""
    del person_id
    kind, reference = parse_relation_key(relation_key)
    if kind in {"entity", "relationship"}:
        return reference
    if kind != "memory" or not reference:
        return ""
    row = store.get_memory_item(reference)
    return str(row.get("content", "")).strip() if row else ""


def expand_associative_recall(
    seed_keys: list[str], *, person_id: str = "",
    min_strength: float | None = None, limit: int = 8,
) -> list[dict]:
    """从统一记忆项 UUID 出发扩展关联记忆。"""
    threshold = (
        float(min_strength)
        if min_strength is not None
        else float(getattr(settings, "memory_relation_min_strength", 0.6))
    )
    seeds = list(dict.fromkeys(key for key in seed_keys if parse_relation_key(key)[0] != "unknown"))
    if not seeds:
        return []

    seed_set = set(seeds)
    rows = store.get_memory_relations(seeds, min_strength=threshold, limit=limit * 3)
    results: list[dict] = []
    seen_text: set[str] = set()
    for row in rows:
        source = str(row.get("from_id") or "")
        target = str(row.get("to_id") or "")
        neighbor = target if source in seed_set else source
        if neighbor in seed_set:
            continue
        text = resolve_memory_text(neighbor, person_id)
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        results.append({
            "memory_id": neighbor.removeprefix(MEMORY_PREFIX),
            "text": text,
            "relation_type": row.get("relation_type", "related"),
            "strength": round(float(row.get("strength", 0.5)), 3),
            "via": source if source in seed_set else target,
        })
        if len(results) >= limit:
            break
    return results


def seed_keys_from_memory_items(memory_items: list[dict]) -> list[str]:
    """只从统一记忆项 UUID 生成关联图种子。"""
    return list(dict.fromkeys(
        key for item in memory_items
        if (key := memory_key(str(item.get("id", "") or "")))
    ))
