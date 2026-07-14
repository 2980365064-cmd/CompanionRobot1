"""后台第三方画像与关系图谱管理。

第三方画像复用 person_profiles 表，通过 profile_role=contact 与实名用户画像区分。
owner_person_id 表示这个第三方人物主要属于哪个实名用户的关系上下文。
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.memory.contacts import (
    ROBOT_OWNER_DISPLAY_NAME,
    ROBOT_OWNER_PERSON_ID,
    _contact_person_id,
)
from app.memory.guard import is_valid_person_name
from app.memory.identity import is_temp_person_id
from app.memory.profile import empty_profile, normalize_profile, profile_display_name
from app.session import store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: list[str] | None) -> list[str]:
    return [str(x).strip() for x in (value or []) if str(x).strip()]


def _is_contact(profile: dict) -> bool:
    return str(normalize_profile(profile).get("profile_role") or "owner") == "contact"


def is_robot_owner_person_id(owner_person_id: str) -> bool:
    return str(owner_person_id or "").strip() == ROBOT_OWNER_PERSON_ID


def _validate_owner_person_id(owner_person_id: str) -> str:
    owner_id = str(owner_person_id or "").strip()
    if is_robot_owner_person_id(owner_id):
        return owner_id
    if not owner_id or is_temp_person_id(owner_id) or not store.get_person_profile(owner_id):
        raise ValueError("valid owner_person_id required")
    return owner_id


def _owner_label(owner_person_id: str) -> str:
    if is_robot_owner_person_id(owner_person_id):
        return ROBOT_OWNER_DISPLAY_NAME
    owner = store.get_person_profile(owner_person_id)
    return profile_display_name(owner) if owner else owner_person_id


def _contact_payload(row: dict) -> dict:
    profile = normalize_profile(row["profile"])
    owner_id = str(profile.get("owner_person_id") or "")
    return {
        "person_id": row["person_id"],
        "device_id": row["device_id"],
        "display_name": profile_display_name(profile),
        "owner_person_id": owner_id,
        "owner_display_name": _owner_label(owner_id) if owner_id else "",
        "relationship": str(profile.get("relationship") or ""),
        "aliases": list(profile.get("aliases") or []),
        "personality": list(profile.get("personality") or []),
        "experiences": list(profile.get("experiences") or []),
        "emotional_habit": list(profile.get("emotional_habit") or []),
        "notes": list(profile.get("notes") or []),
        "mention_count": int(profile.get("mention_count") or 0),
        "last_mentioned_at": profile.get("last_mentioned_at") or "",
        "confirmed": bool(profile.get("confirmed")),
        "source": str(profile.get("source") or "manual"),
        "created_at": profile.get("created_at") or row.get("updated_at"),
        "updated_at": profile.get("updated_at") or row.get("updated_at"),
    }


def list_contacts_admin(
    *,
    owner_person_id: str | None = None,
    q: str | None = None,
    confirmed: bool | None = None,
) -> list[dict]:
    owner = str(owner_person_id or "").strip()
    needle = str(q or "").strip().lower()
    out: list[dict] = []
    for row in store.list_all_person_profiles():
        profile = normalize_profile(row["profile"])
        if not _is_contact(profile):
            continue
        if owner and str(profile.get("owner_person_id") or "") != owner:
            continue
        if confirmed is not None and bool(profile.get("confirmed")) is not confirmed:
            continue
        payload = _contact_payload(row)
        haystack = " ".join(
            [
                payload["display_name"],
                payload["relationship"],
                payload["owner_display_name"],
                " ".join(payload["aliases"]),
                " ".join(payload["notes"]),
            ]
        ).lower()
        if needle and needle not in haystack:
            continue
        out.append(payload)
    return out


def get_contact_admin(person_id: str) -> dict:
    pid = str(person_id or "").strip()
    profile = store.get_person_profile(pid)
    if not profile or not _is_contact(profile):
        raise ValueError("contact not found")
    row = {
        "person_id": pid,
        "device_id": store.get_person_device_id(pid),
        "profile": profile,
        "updated_at": profile.get("updated_at"),
    }
    return _contact_payload(row)


def create_contact_admin(
    *,
    owner_person_id: str,
    display_name: str,
    relationship: str = "",
    aliases: list[str] | None = None,
    personality: list[str] | None = None,
    experiences: list[str] | None = None,
    emotional_habit: list[str] | None = None,
    notes: list[str] | None = None,
    confirmed: bool = True,
) -> dict:
    owner_id = _validate_owner_person_id(owner_person_id)
    name = str(display_name or "").strip()
    if not is_valid_person_name(name):
        raise ValueError("invalid display_name")

    pid = _contact_person_id(owner_id, name)
    if store.get_person_profile(pid):
        raise ValueError("contact already exists")

    profile = empty_profile(name, person_id=pid, relationship=str(relationship or "").strip())
    profile["profile_role"] = "contact"
    profile["owner_person_id"] = owner_id
    profile["aliases"] = _as_list(aliases)
    profile["personality"] = _as_list(personality)
    profile["experiences"] = _as_list(experiences)
    profile["emotional_habit"] = _as_list(emotional_habit)
    profile["notes"] = _as_list(notes)
    profile["mention_count"] = 0
    profile["source"] = "manual"
    profile["confirmed"] = bool(confirmed)
    profile["created_at"] = _now()
    profile["updated_at"] = profile["created_at"]
    device_id = (
        "admin"
        if is_robot_owner_person_id(owner_id)
        else (store.get_person_device_id(owner_id) or "admin")
    )
    store.save_person_profile(device_id, profile)
    return get_contact_admin(pid)


def update_contact_admin(
    person_id: str,
    *,
    owner_person_id: str | None = None,
    display_name: str | None = None,
    relationship: str | None = None,
    aliases: list[str] | None = None,
    personality: list[str] | None = None,
    experiences: list[str] | None = None,
    emotional_habit: list[str] | None = None,
    notes: list[str] | None = None,
    confirmed: bool | None = None,
) -> dict:
    pid = str(person_id or "").strip()
    raw = store.get_person_profile(pid)
    if not raw or not _is_contact(raw):
        raise ValueError("contact not found")
    profile = normalize_profile(raw)

    if owner_person_id is not None:
        owner_id = _validate_owner_person_id(owner_person_id)
        profile["owner_person_id"] = owner_id
    if display_name is not None:
        name = str(display_name or "").strip()
        if not is_valid_person_name(name):
            raise ValueError("invalid display_name")
        profile["display_name"] = name
    if relationship is not None:
        profile["relationship"] = str(relationship or "").strip()
    for key, value in (
        ("aliases", aliases),
        ("personality", personality),
        ("experiences", experiences),
        ("emotional_habit", emotional_habit),
        ("notes", notes),
    ):
        if value is not None:
            profile[key] = _as_list(value)
    if confirmed is not None:
        profile["confirmed"] = bool(confirmed)

    profile["profile_role"] = "contact"
    profile["updated_at"] = _now()
    owner_id = str(profile.get("owner_person_id") or "")
    device_id = (
        "admin"
        if is_robot_owner_person_id(owner_id)
        else (store.get_person_device_id(owner_id) or store.get_person_device_id(pid) or "admin")
    )
    store.save_person_profile(device_id, profile)
    return get_contact_admin(pid)


def delete_contact_admin(person_id: str) -> dict:
    payload = get_contact_admin(person_id)
    stats = store.delete_person_id(payload["person_id"])
    return {"deleted": True, "contact": payload, "delete_stats": stats}


def build_memory_by_person_admin() -> dict:
    from app.admin.memory import list_core_memory_admin, list_long_term_memory_admin
    from app.admin.persons import list_persons_admin

    groups: list[dict] = []
    for person in list_persons_admin():
        pid = person["person_id"]
        core = list_core_memory_admin(pid)
        episodes = store.list_memory_items(pid, kinds=["episode", "emotion", "milestone"], limit=8)
        long_term = list_long_term_memory_admin(pid, limit=8)
        groups.append(
            {
                "person": person,
                "counts": {
                    "core_memories": len(core),
                    "episodes": len(store.list_memory_items(pid, kinds=["episode", "emotion"], limit=100)),
                    "long_term_memory": len(list_long_term_memory_admin(pid, limit=100)),
                    "contacts": len(list_contacts_admin(owner_person_id=pid)),
                },
                "core_memories": core[:8],
                "episodes": episodes,
                "long_term_memory": long_term,
            }
        )
    return {"groups": groups, "count": len(groups)}


def build_graph_admin() -> dict:
    from app.admin.memory import list_core_memory_admin
    from app.admin.persons import list_persons_admin

    robot_names = ("叶鹏祥", "SparkBot", "sparkbot", "机器人")

    def _memory_payload(person_id: str, *, limit: int = 5) -> dict:
        core = list_core_memory_admin(person_id)[:limit] if store.get_person_profile(person_id) else []
        episodes = store.list_memory_items(person_id, kinds=["episode", "emotion", "milestone"], limit=limit)
        long_term = store.search_long_term_memory(person_id, limit=limit)
        return {
            "core": core,
            "episodes": episodes,
            "long_term": long_term,
            "counts": {
                "core": len(core),
                "episodes": len(store.list_memory_items(person_id, kinds=["episode", "emotion"], limit=100)),
                "long_term": len(store.search_long_term_memory(person_id, limit=100)),
            },
        }

    def _memory_weight(memory: dict) -> int:
        counts = memory.get("counts") or {}
        return (
            int(counts.get("core") or 0) * 4
            + int(counts.get("episodes") or 0) * 2
            + int(counts.get("long_term") or 0)
        )

    def _relation_style(text: str, *, confirmed: bool = True) -> str:
        blob = str(text or "")
        if not confirmed or any(x in blob for x in ("陌生", "一次", "临时", "路人", "未确认")):
            return "stranger"
        if any(x in blob for x in ("冲突", "吵架", "矛盾", "讨厌", "拉黑", "伤害", "负面")):
            return "conflict"
        if any(x in blob for x in ("情侣", "女朋友", "男朋友", "老婆", "老公", "好友", "朋友", "同学", "亲人", "家人", "闺蜜")):
            return "close"
        return "daily"

    def _contact_relation_target(contact: dict, owner: dict | None) -> tuple[str, str]:
        """判断第三方画像关系边应该连向谁。

        owner_person_id 只表示画像管理上下文，不等于真实关系对象。
        图谱优先读取备注/来源中的关系主语；无法判断时不连线，避免编造关系。
        """
        relationship = str(contact.get("relationship") or "").strip()
        if not relationship:
            return "", ""
        notes = " ".join(str(x) for x in contact.get("notes") or [])
        experiences = " ".join(str(x) for x in contact.get("experiences") or [])
        aliases = " ".join(str(x) for x in contact.get("aliases") or [])
        haystack = f"{relationship} {notes} {experiences} {aliases} {contact.get('source') or ''}"
        owner_name = str((owner or {}).get("display_name") or "").strip()
        owner_id = str(contact.get("owner_person_id") or "").strip()

        if owner_name and owner_name in haystack:
            return f"person:{owner_id}", owner_name
        if any(name in haystack for name in robot_names):
            return "robot:sparkbot", "叶鹏祥"
        if str(contact.get("source") or "") == "wiki":
            return "robot:sparkbot", "叶鹏祥"
        if any(word in relationship for word in ("同学", "好友", "朋友", "室友", "同事")):
            return "robot:sparkbot", "叶鹏祥"
        return "", ""

    memory_stat = store.count_memory_stat()
    nodes: dict[str, dict] = {
        "robot:sparkbot": {
            "id": "robot:sparkbot",
            "type": "robot",
            "label": "叶鹏祥",
            "detail": "机器人主体",
            "basic": {
                "名称": "叶鹏祥 / SparkBot",
                "类型": "机器人",
                "说明": "陪伴机器人主体，负责对话、记忆召回与后台任务。",
            },
            "memory": {
                "core": [],
                "episodes": [],
                "long_term": [],
                "counts": {
                    "core": memory_stat.get("core_memories", 0),
                    "episodes": memory_stat.get("episodes_total", 0),
                    "long_term": memory_stat.get("long_term_memory", 0),
                },
            },
            "category": "robot",
            "memory_weight": int(memory_stat.get("core_memories", 0)) * 4
            + int(memory_stat.get("episodes_total", 0)) * 2
            + int(memory_stat.get("long_term_memory", 0)),
            "interaction_count": int(memory_stat.get("episodes_total", 0)),
            "last_interaction": "",
        }
    }
    links: list[dict] = []

    owners = list_persons_admin()
    owners = [
        p for p in owners
        if str(p.get("display_name") or "").strip() not in {"叶鹏祥", "SparkBot", "sparkbot", "机器人"}
    ]
    owners_by_id = {str(p["person_id"]): p for p in owners}
    for person in owners:
        profile = normalize_profile(store.get_person_profile(person["person_id"]) or {})
        relationship = str(profile.get("relationship") or "").strip() or "实名对话对象"
        memory = _memory_payload(person["person_id"])
        node_id = f"person:{person['person_id']}"
        nodes[node_id] = {
            "id": node_id,
            "type": "person",
            "label": person.get("display_name") or person["person_id"],
            "person_id": person["person_id"],
            "detail": relationship,
            "basic": {
                "person_id": person["person_id"],
                "显示名": person.get("display_name") or "",
                "设备": person.get("device_id") or "",
                "关系": relationship,
                "更新时间": person.get("updated_at") or "",
            },
            "profile": profile,
            "memory": memory,
            "category": "owner",
            "memory_weight": _memory_weight(memory),
            "interaction_count": int((memory.get("counts") or {}).get("episodes") or 0),
            "last_interaction": person.get("updated_at") or "",
        }
        if relationship:
            style = _relation_style(relationship)
            links.append({
                "source": "robot:sparkbot",
                "target": node_id,
                "type": relationship,
                "label": relationship,
                "strength": 0.9,
                "relation_style": style,
            })

    contacts = list_contacts_admin()
    for contact in contacts:
        relationship = str(contact.get("relationship") or "").strip()
        memory = _memory_payload(contact["person_id"])
        interaction_count = int(contact.get("mention_count") or 0) + int((memory.get("counts") or {}).get("episodes") or 0)
        node_id = f"contact:{contact['person_id']}"
        nodes[node_id] = {
            "id": node_id,
            "type": "contact",
            "label": contact["display_name"],
            "person_id": contact["person_id"],
            "detail": relationship or "第三方画像",
            "basic": {
                "person_id": contact["person_id"],
                "显示名": contact.get("display_name") or "",
                "管理上下文": contact.get("owner_display_name") or contact.get("owner_person_id") or "",
                "关系": relationship,
                "状态": "已确认" if contact.get("confirmed") else "未确认",
                "来源": contact.get("source") or "",
            },
            "profile": contact,
            "memory": memory,
            "category": "contact",
            "memory_weight": _memory_weight(memory) + interaction_count,
            "interaction_count": interaction_count,
            "last_interaction": contact.get("last_mentioned_at") or contact.get("updated_at") or "",
        }
        target_node, target_label = _contact_relation_target(
            contact, owners_by_id.get(str(contact.get("owner_person_id") or ""))
        )
        if target_node and target_node in nodes:
            nodes[node_id]["basic"]["关系对象"] = target_label
            style = _relation_style(relationship, confirmed=bool(contact.get("confirmed")))
            links.append({
                "source": target_node,
                "target": node_id,
                "type": relationship,
                "label": relationship,
                "strength": 0.72 if contact.get("confirmed") else 0.45,
                "relation_style": style,
            })

    return {"nodes": list(nodes.values()), "links": links}
