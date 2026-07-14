"""后台核心记忆与人物画像管理 API 逻辑。

本模块管理统一记忆库中的核心事实（visibility=always）与 profile 归档字段。

与 person_admin.py 互补：person_admin 管身份（person_id / display_name），
本模块管用户绑定的核心事实与 profile 的增删改查。

所有核心事实的读写通过 memory_items 统一记忆库完成，不再使用旧核心事实表。
旧表数据仅由迁移脚本只读访问。
"""

from __future__ import annotations

from app.memory.guard import is_valid_person_name
from app.memory.identity import is_temp_person_id
from app.memory.core_facts import (
    CORE_FACT_CATEGORY_LABELS,
    CORE_FACT_KEY_PEOPLE,
    push_identity_on_profile_promotion,
)
from app.memory.profile import normalize_profile, profile_display_name
from app.session import store

_SOURCE_ADMIN = "manual"
_MAX_KEY_PEOPLE = 5
RECENT_MEMORY_KINDS = ("episode", "emotion", "milestone")
LONG_TERM_MEMORY_KINDS = ("fact", "entity", "wiki", "relationship", "correction")


def _validate_person(person_id: str) -> str:
    pid = str(person_id or "").strip()
    if not pid:
        raise ValueError("person_id required")
    if is_temp_person_id(pid):
        raise ValueError("cannot edit guest tmp_* id")
    if not store.get_person_profile(pid):
        raise ValueError("person not found")
    return pid


def _validate_category(category: str) -> str:
    cat = str(category or "").strip()
    if cat not in CORE_FACT_CATEGORY_LABELS:
        raise ValueError(f"invalid category; must be one of: {', '.join(CORE_FACT_CATEGORY_LABELS)}")
    return cat


def _normalize_content(content: str) -> str:
    body = " ".join((content or "").strip().split())
    if len(body) < 2:
        raise ValueError("content must be at least 2 characters")
    return body


def list_core_memory_admin(person_id: str) -> list[dict]:
    """列出用户的核心事实（统一记忆库中 visibility=always 的条目）。"""
    pid = _validate_person(person_id)
    rows = store.list_core_facts(pid)
    return [
        {
            "id": row.get("id", ""),
            "category": row.get("kind", ""),
            "category_label": CORE_FACT_CATEGORY_LABELS.get(row.get("kind", ""), row.get("kind", "")),
            "content": row.get("content", ""),
            "source": row.get("source", ""),
            "confidence": row.get("confidence", 1.0),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }
        for row in rows
    ]


def create_core_memory_admin(person_id: str, category: str, content: str) -> dict:
    """创建核心事实（写入统一记忆库，visibility=always）。"""
    pid = _validate_person(person_id)
    cat = _validate_category(category)
    body = _normalize_content(content)

    # 检查 key_people 数量上限
    if cat == CORE_FACT_KEY_PEOPLE:
        existing = store.search_memory_items(pid, kinds=[CORE_FACT_KEY_PEOPLE], visibility="always", limit=_MAX_KEY_PEOPLE + 1)
        if len(existing) >= _MAX_KEY_PEOPLE:
            # 检查是否已存在相同内容
            found = store.search_memory_items(pid, kinds=[CORE_FACT_KEY_PEOPLE], visibility="always", query=body, limit=1)
            if not found:
                raise ValueError(f"key_people limit reached ({_MAX_KEY_PEOPLE}); delete one first")

    device_id = store.get_person_device_id(pid) or "default"
    item_id = store.write_memory_item(
        person_id=pid,
        device_id=device_id,
        kind=cat,
        source=_SOURCE_ADMIN,
        visibility="always",
        content=body,
        confidence=1.0,
    )

    return {
        "id": item_id,
        "category": cat,
        "category_label": CORE_FACT_CATEGORY_LABELS[cat],
        "content": body,
        "source": _SOURCE_ADMIN,
    }


def update_core_memory_admin(
    person_id: str,
    memory_item_id: str | int,
    *,
    category: str | None = None,
    content: str | None = None,
) -> dict:
    """更新核心事实。"""
    pid = _validate_person(person_id)
    row = store.get_memory_item(str(memory_item_id))
    if not row or str(row.get("person_id")) != pid:
        raise ValueError("core fact entry not found")

    cat = _validate_category(category) if category is not None else str(row.get("kind", ""))
    body = _normalize_content(content) if content is not None else str(row.get("content", ""))

    # 重建条目（memory_items 不支持原地 content 变更，重新写入）
    device_id = store.get_person_device_id(pid) or "default"
    new_id = store.write_memory_item(
        person_id=pid,
        device_id=device_id,
        kind=cat,
        source=_SOURCE_ADMIN,
        visibility="always",
        content=body,
        confidence=1.0,
    )
    # 删除旧条目
    store.delete_memory_item(str(memory_item_id))

    return {
        "id": new_id,
        "category": cat,
        "category_label": CORE_FACT_CATEGORY_LABELS.get(cat, cat),
        "content": body,
        "source": _SOURCE_ADMIN,
    }


def delete_core_memory_admin(person_id: str, memory_item_id: str | int) -> dict:
    """删除核心事实。"""
    pid = _validate_person(person_id)
    row = store.get_memory_item(str(memory_item_id))
    if not row or str(row.get("person_id")) != pid:
        raise ValueError("core fact entry not found")
    if not store.delete_memory_item(str(memory_item_id)):
        raise ValueError("failed to delete core fact entry")
    return {"deleted": True, "id": memory_item_id, "content": row.get("content", "")}


def _memory_item_payload(row: dict) -> dict:
    return {
        "id": row.get("id", ""),
        "person_id": row.get("person_id", ""),
        "device_id": row.get("device_id", ""),
        "kind": row.get("kind", ""),
        "category": row.get("kind", ""),
        "visibility": row.get("visibility", ""),
        "source": row.get("source", ""),
        "content": row.get("content", ""),
        "summary": row.get("content", ""),
        "text": row.get("content", ""),
        "confidence": row.get("confidence", 1.0),
        "emotional_weight": row.get("emotional_weight", 3),
        "recency_weight": row.get("recency_weight", 3),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "expires_at": row.get("expires_at"),
    }


def _get_owned_memory_item(person_id: str, memory_item_id: str | int, *, kinds: tuple[str, ...]) -> dict:
    pid = _validate_person(person_id)
    row = store.get_memory_item(str(memory_item_id))
    if not row or str(row.get("person_id")) != pid or str(row.get("kind")) not in kinds:
        raise ValueError("memory item not found")
    return row


def list_recent_memory_admin(person_id: str, *, limit: int = 50) -> list[dict]:
    """列出情景记忆（episode/emotion/milestone）。"""
    pid = _validate_person(person_id)
    rows = store.list_memory_items(pid, kinds=list(RECENT_MEMORY_KINDS), limit=limit)
    return [_memory_item_payload(row) for row in rows]


def create_recent_memory_admin(
    person_id: str,
    *,
    content: str,
    kind: str = "episode",
    emotional_weight: int = 3,
    recency_weight: int = 3,
) -> dict:
    """新增情景记忆。"""
    pid = _validate_person(person_id)
    item_kind = str(kind or "episode").strip()
    if item_kind not in RECENT_MEMORY_KINDS:
        raise ValueError(f"invalid kind; must be one of: {', '.join(RECENT_MEMORY_KINDS)}")
    body = _normalize_content(content)
    device_id = store.get_person_device_id(pid) or "admin"
    item_id = store.write_memory_item(
        person_id=pid,
        device_id=device_id,
        kind=item_kind,
        source=_SOURCE_ADMIN,
        visibility="recall_only",
        content=body,
        emotional_weight=max(1, min(5, int(emotional_weight or 3))),
        recency_weight=max(1, min(5, int(recency_weight or 3))),
    )
    return _memory_item_payload(store.get_memory_item(item_id) or {})


def update_recent_memory_admin(
    person_id: str,
    memory_item_id: str | int,
    *,
    content: str | None = None,
    kind: str | None = None,
    emotional_weight: int | None = None,
    recency_weight: int | None = None,
) -> dict:
    """修改情景记忆；使用重写方式保持统一索引一致。"""
    row = _get_owned_memory_item(person_id, memory_item_id, kinds=RECENT_MEMORY_KINDS)
    item_kind = str(kind or row.get("kind") or "episode").strip()
    if item_kind not in RECENT_MEMORY_KINDS:
        raise ValueError(f"invalid kind; must be one of: {', '.join(RECENT_MEMORY_KINDS)}")
    body = _normalize_content(content) if content is not None else str(row.get("content") or "")
    new_id = store.write_memory_item(
        person_id=str(row.get("person_id") or person_id),
        device_id=str(row.get("device_id") or store.get_person_device_id(person_id) or "admin"),
        kind=item_kind,
        source=_SOURCE_ADMIN,
        visibility=str(row.get("visibility") or "recall_only"),
        content=body,
        confidence=float(row.get("confidence") or 1.0),
        emotional_weight=max(1, min(5, int(emotional_weight if emotional_weight is not None else row.get("emotional_weight") or 3))),
        recency_weight=max(1, min(5, int(recency_weight if recency_weight is not None else row.get("recency_weight") or 3))),
        context_json=str(row.get("context_json") or "{}"),
        tags_json=str(row.get("tags_json") or "[]"),
        source_session=str(row.get("source_session") or ""),
        expires_at=str(row.get("expires_at") or ""),
    )
    store.delete_memory_item(str(memory_item_id))
    return _memory_item_payload(store.get_memory_item(new_id) or {})


def delete_recent_memory_admin(person_id: str, memory_item_id: str | int) -> dict:
    row = _get_owned_memory_item(person_id, memory_item_id, kinds=RECENT_MEMORY_KINDS)
    if not store.delete_memory_item(str(memory_item_id)):
        raise ValueError("failed to delete memory item")
    return {"deleted": True, "id": memory_item_id, "content": row.get("content", "")}


def list_long_term_memory_admin(person_id: str, *, limit: int = 50) -> list[dict]:
    """列出长期记忆（recall_only 的 fact/entity/wiki/relationship/correction）。"""
    pid = _validate_person(person_id)
    rows = store.list_memory_items(
        pid,
        kinds=list(LONG_TERM_MEMORY_KINDS),
        visibility="recall_only",
        limit=limit,
    )
    return [_memory_item_payload(row) for row in rows]


def create_long_term_memory_admin(
    person_id: str,
    *,
    content: str,
    kind: str = "fact",
    confidence: float = 1.0,
) -> dict:
    """新增长期记忆。"""
    pid = _validate_person(person_id)
    item_kind = str(kind or "fact").strip()
    if item_kind not in LONG_TERM_MEMORY_KINDS:
        raise ValueError(f"invalid kind; must be one of: {', '.join(LONG_TERM_MEMORY_KINDS)}")
    body = _normalize_content(content)
    device_id = store.get_person_device_id(pid) or "admin"
    item_id = store.write_memory_item(
        person_id=pid,
        device_id=device_id,
        kind=item_kind,
        source=_SOURCE_ADMIN,
        visibility="recall_only",
        content=body,
        confidence=max(0.0, min(1.0, float(confidence or 1.0))),
    )
    return _memory_item_payload(store.get_memory_item(item_id) or {})


def update_long_term_memory_admin(
    person_id: str,
    memory_item_id: str | int,
    *,
    content: str | None = None,
    kind: str | None = None,
    confidence: float | None = None,
) -> dict:
    """修改长期记忆。"""
    row = _get_owned_memory_item(person_id, memory_item_id, kinds=LONG_TERM_MEMORY_KINDS)
    item_kind = str(kind or row.get("kind") or "fact").strip()
    if item_kind not in LONG_TERM_MEMORY_KINDS:
        raise ValueError(f"invalid kind; must be one of: {', '.join(LONG_TERM_MEMORY_KINDS)}")
    body = _normalize_content(content) if content is not None else str(row.get("content") or "")
    new_id = store.write_memory_item(
        person_id=str(row.get("person_id") or person_id),
        device_id=str(row.get("device_id") or store.get_person_device_id(person_id) or "admin"),
        kind=item_kind,
        source=_SOURCE_ADMIN,
        visibility=str(row.get("visibility") or "recall_only"),
        content=body,
        confidence=max(0.0, min(1.0, float(confidence if confidence is not None else row.get("confidence") or 1.0))),
        emotional_weight=int(row.get("emotional_weight") or 3),
        recency_weight=int(row.get("recency_weight") or 3),
        context_json=str(row.get("context_json") or "{}"),
        tags_json=str(row.get("tags_json") or "[]"),
        source_session=str(row.get("source_session") or ""),
        expires_at=str(row.get("expires_at") or ""),
    )
    store.delete_memory_item(str(memory_item_id))
    return _memory_item_payload(store.get_memory_item(new_id) or {})


def delete_long_term_memory_admin(person_id: str, memory_item_id: str | int) -> dict:
    row = _get_owned_memory_item(person_id, memory_item_id, kinds=LONG_TERM_MEMORY_KINDS)
    if not store.delete_memory_item(str(memory_item_id)):
        raise ValueError("failed to delete memory item")
    return {"deleted": True, "id": memory_item_id, "content": row.get("content", "")}


def get_profile_admin(person_id: str) -> dict:
    pid = _validate_person(person_id)
    profile = normalize_profile(store.get_person_profile(pid) or {})
    return {
        "person_id": pid,
        "display_name": profile_display_name(profile),
        "aliases": list(profile.get("aliases") or []),
        "relationship": str(profile.get("relationship") or ""),
        "personality": list(profile.get("personality") or []),
        "experiences": list(profile.get("experiences") or []),
        "emotional_habit": list(profile.get("emotional_habit") or []),
        "confirmed": bool(profile.get("confirmed")),
        "created_at": profile.get("created_at"),
        "updated_at": profile.get("updated_at"),
        "device_id": store.get_person_device_id(pid),
    }


def update_profile_admin(
    person_id: str,
    *,
    display_name: str | None = None,
    aliases: list[str] | None = None,
    relationship: str | None = None,
    personality: list[str] | None = None,
    experiences: list[str] | None = None,
    emotional_habit: list[str] | None = None,
    confirmed: bool | None = None,
    sync_core_facts: bool = True,
) -> dict:
    pid = _validate_person(person_id)
    raw = store.get_person_profile(pid)
    if not raw:
        raise ValueError("person not found")
    profile = normalize_profile(raw)
    device_id = store.get_person_device_id(pid) or "default"

    if display_name is not None:
        name = str(display_name).strip()
        if not is_valid_person_name(name):
            raise ValueError("invalid display_name")
        profile["display_name"] = name

    if aliases is not None:
        profile["aliases"] = [str(a).strip() for a in aliases if str(a).strip()]

    if relationship is not None:
        profile["relationship"] = str(relationship).strip()

    if personality is not None:
        profile["personality"] = [str(x).strip() for x in personality if str(x).strip()]

    if experiences is not None:
        profile["experiences"] = [str(x).strip() for x in experiences if str(x).strip()]

    if emotional_habit is not None:
        profile["emotional_habit"] = [str(x).strip() for x in emotional_habit if str(x).strip()]

    if confirmed is not None:
        profile["confirmed"] = bool(confirmed)

    from datetime import datetime, timezone

    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    store.save_person_profile(device_id, profile)

    if sync_core_facts and profile.get("confirmed"):
        push_identity_on_profile_promotion(device_id, pid, profile)

    return get_profile_admin(pid)


def delete_person_memory_admin(person_id: str) -> dict:
    """删除指定用户的所有记忆数据（通过 memory_items 统一库删除）。"""
    pid = _validate_person(person_id)
    items = store.list_memory_items(pid, limit=9999)
    deleted = 0
    for item in items:
        item_id = str(item.get("id", ""))
        if item_id:
            store.delete_memory_item(item_id)
            deleted += 1
    return {"deleted": True, "person_id": pid, "memory_items_deleted": deleted}
