"""后台 L0 与人物画像管理 API 逻辑。

与 person_admin.py 互补：person_admin 管身份（person_id / display_name），
本模块管用户绑定的 L0 核心记忆与 profile 归档字段的增删改查。
"""

from __future__ import annotations

from app.memory.guard import is_valid_person_name
from app.memory.identity import is_temp_person_id
from app.memory.l0 import (
    L0_CATEGORY_LABELS,
    L0_KEY_PEOPLE,
    push_identity_on_profile_promotion,
)
from app.memory.profile import normalize_profile, profile_display_name
from app.session import store

_SOURCE_ADMIN = "admin_manual"
_MAX_KEY_PEOPLE = 5


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
    if cat not in L0_CATEGORY_LABELS:
        raise ValueError(f"invalid category; must be one of: {', '.join(L0_CATEGORY_LABELS)}")
    return cat


def _normalize_content(content: str) -> str:
    body = " ".join((content or "").strip().split())
    if len(body) < 2:
        raise ValueError("content must be at least 2 characters")
    return body


def list_l0_admin(person_id: str) -> list[dict]:
    pid = _validate_person(person_id)
    rows = store.l0_list(pid)
    return [
        {
            "id": row["id"],
            "category": row["category"],
            "category_label": L0_CATEGORY_LABELS.get(row["category"], row["category"]),
            "content": row["content"],
            "source": row.get("source", ""),
            "confidence": row.get("confidence", 1.0),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        }
        for row in rows
    ]


def create_l0_admin(person_id: str, category: str, content: str) -> dict:
    pid = _validate_person(person_id)
    cat = _validate_category(category)
    body = _normalize_content(content)

    if cat == L0_KEY_PEOPLE and store.l0_count(pid, L0_KEY_PEOPLE) >= _MAX_KEY_PEOPLE:
        if not store.l0_find_by_content(pid, body, L0_KEY_PEOPLE):
            raise ValueError(f"key_people limit reached ({_MAX_KEY_PEOPLE}); delete one first")

    device_id = store.get_person_device_id(pid) or "default"
    store.l0_upsert(pid, cat, body, device_id=device_id, source=_SOURCE_ADMIN, confidence=1.0)

    row = store.l0_find_by_content(pid, body, cat)
    if not row:
        raise ValueError("failed to create L0 entry")
    return {
        "id": row["id"],
        "category": cat,
        "category_label": L0_CATEGORY_LABELS[cat],
        "content": body,
        "source": _SOURCE_ADMIN,
    }


def update_l0_admin(
    person_id: str,
    l0_id: int,
    *,
    category: str | None = None,
    content: str | None = None,
) -> dict:
    pid = _validate_person(person_id)
    row = store.l0_get(l0_id)
    if not row or str(row.get("person_id")) != pid:
        raise ValueError("L0 entry not found")

    cat = _validate_category(category) if category is not None else str(row["category"])
    body = _normalize_content(content) if content is not None else str(row["content"])

    if cat == L0_KEY_PEOPLE and cat != row["category"]:
        if store.l0_count(pid, L0_KEY_PEOPLE) >= _MAX_KEY_PEOPLE:
            raise ValueError(f"key_people limit reached ({_MAX_KEY_PEOPLE})")

    if not store.l0_update(l0_id, category=cat, content=body, source=_SOURCE_ADMIN):
        raise ValueError("failed to update L0 entry")

    updated = store.l0_get(l0_id)
    if not updated:
        raise ValueError("L0 entry not found after update")
    return {
        "id": updated["id"],
        "category": updated["category"],
        "category_label": L0_CATEGORY_LABELS.get(updated["category"], updated["category"]),
        "content": updated["content"],
        "source": updated.get("source", _SOURCE_ADMIN),
    }


def delete_l0_admin(person_id: str, l0_id: int) -> dict:
    pid = _validate_person(person_id)
    row = store.l0_get(l0_id)
    if not row or str(row.get("person_id")) != pid:
        raise ValueError("L0 entry not found")
    if not store.l0_delete(l0_id):
        raise ValueError("failed to delete L0 entry")
    return {"deleted": True, "id": l0_id, "content": row.get("content", "")}


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
    sync_l0: bool = True,
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

    if sync_l0 and profile.get("confirmed"):
        push_identity_on_profile_promotion(device_id, pid, profile)

    return get_profile_admin(pid)


def delete_l2_admin(person_id: str) -> dict:
    """删除指定用户的所有 L2 情景记忆（不可逆）。"""
    pid = _validate_person(person_id)
    deleted = store.delete_episodic_for_person(pid)
    return {"deleted": True, "person_id": pid, "l2_count": deleted}
