"""人物详情页 API —— 聚合单个实名用户的全部信息。

GET /v1/admin/persons/{person_id}/detail
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.admin.persons import list_persons_admin
from app.admin.memory import get_profile_admin, list_core_memory_admin
from app.session import store


def _get_long_term_memory_for_person(person_id: str) -> list[dict]:
    """获取指定用户的长期记忆块。"""
    try:
        return store.list_memory_items_detailed(
            person_id=person_id, limit=100, offset=0
        ) or []
    except Exception:
        return []


def _get_recent_memory_for_person(person_id: str, limit: int = 50) -> list[dict]:
    """获取指定用户的情景摘要（基于 memory_items 统一记忆库）。"""
    items = []
    conn = None
    try:
        conn = sqlite3.connect(str(settings.resolved_db_path()))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, content AS summary, context_json, tags_json,
                      emotional_weight, expires_at, created_at, updated_at, deleted_at
               FROM memory_items
               WHERE person_id=? AND kind IN ('episode','emotion')
               ORDER BY updated_at DESC LIMIT ?""",
            (person_id, limit),
        ).fetchall()
        for r in rows:
            import json
            d = dict(r)
            try:
                ctx = json.loads(d.get("context_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            try:
                tags = json.loads(d.get("tags_json", "[]"))
            except (json.JSONDecodeError, TypeError):
                tags = []
            people = ", ".join(
                t.replace("person:", "") for t in tags
                if isinstance(t, str) and t.startswith("person:")
            ) if isinstance(tags, list) else ""
            items.append({
                "id": d["id"],
                "summary": (d.get("summary") or "")[:300],
                "content": (d.get("summary") or "")[:300],
                "kind": "episode",
                "keywords": ctx.get("topics", "") if isinstance(ctx, dict) else "",
                "importance": int(ctx.get("importance", d.get("emotional_weight", 3)))
                              if isinstance(ctx, dict) else int(d.get("emotional_weight", 3)),
                "people": people,
                "expires_at": d.get("expires_at", ""),
                "archived": bool(d.get("deleted_at")),
                "created_at": d.get("created_at", ""),
                "updated_at": d.get("updated_at", ""),
            })
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()
    return items


def _get_contacts_for_person(person_id: str) -> list[dict]:
    """获取第三方画像中与用户关联的人物。"""
    from app.admin.contacts import list_contacts_admin
    try:
        contacts = list_contacts_admin(owner_person_id=person_id) or []
        return [
            {
                "person_id": c.get("person_id", ""),
                "display_name": c.get("display_name", ""),
                "relationship": c.get("relationship", ""),
                "confirmed": bool(c.get("confirmed", False)),
            }
            for c in contacts
        ]
    except Exception:
        return []


def _get_session_summary(person_id: str) -> dict:
    """返回该用户的会话统计摘要。"""
    try:
        conn = sqlite3.connect(str(settings.resolved_db_path()))
        total = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE person_id = ?", (person_id,)
        ).fetchone()[0]
        closed = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE person_id = ? AND status = 'closed'",
            (person_id,),
        ).fetchone()[0]
        msg_count = conn.execute(
            "SELECT COUNT(*) FROM messages m JOIN sessions s ON m.session_id = s.id "
            "WHERE s.person_id = ?",
            (person_id,),
        ).fetchone()[0]
        last_active = conn.execute(
            "SELECT MAX(updated_at) FROM sessions WHERE person_id = ?",
            (person_id,),
        ).fetchone()[0]
        conn.close()
        return {
            "total_sessions": total,
            "closed_sessions": closed,
            "active_sessions": total - closed,
            "total_messages": msg_count,
            "last_active": last_active,
        }
    except Exception:
        return {}


def _get_relation_sub(graph, person_id: str) -> list[dict]:
    """从关系图中提取与该用户相关的关系节点。"""
    related = []
    if not graph or not isinstance(graph, dict):
        return related
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    for edge in edges:
        if edge.get("source") == person_id:
            target = edge.get("target")
            target_node = next((n for n in nodes if n.get("id") == target), {})
            related.append({
                "person_id": target,
                "display_name": target_node.get("label", target),
                "relationship": edge.get("label", ""),
                "direction": "outgoing",
            })
        elif edge.get("target") == person_id:
            source = edge.get("source")
            source_node = next((n for n in nodes if n.get("id") == source), {})
            related.append({
                "person_id": source,
                "display_name": source_node.get("label", source),
                "relationship": edge.get("label", ""),
                "direction": "incoming",
            })
    return related


def _recent_profile_changes(person_id: str) -> list[dict]:
    """从审计日志查询最近与该用户画像相关的操作。"""
    from app.admin.audit import query_audit
    try:
        entries = query_audit(kind="memory", days_back=30)
        changes = [
            e for e in entries
            if person_id[:12] in str(e.get("detail", "")) + str(e.get("action", ""))
        ]
        return changes[:20]
    except Exception:
        return []


def build_person_detail(person_id: str) -> dict:
    """聚合一个人的全部信息。"""
    # 基础信息
    persons = list_persons_admin()
    person_info = next((p for p in persons if p.get("person_id") == person_id), None)
    if not person_info:
        raise ValueError(f"person not found: {person_id}")

    # 画像
    profile = {}
    try:
        profile = get_profile_admin(person_id) or {}
    except ValueError:
        pass

    # 核心记忆
    core_items = []
    try:
        core_items = list_core_memory_admin(person_id) or []
    except ValueError:
        pass

    # 近期情景摘要
    recent_items = _get_recent_memory_for_person(person_id)

    # 长期记忆
    long_term_items = _get_long_term_memory_for_person(person_id)

    # 第三方画像（contacts）
    contacts = _get_contacts_for_person(person_id)

    # 会话统计
    sessions = _get_session_summary(person_id)

    # 关系图信息
    from app.admin.contacts import build_graph_admin
    try:
        graph = build_graph_admin()
        relations = _get_relation_sub(graph, person_id)
    except Exception:
        relations = []

    # 最近变更
    recent_changes = _recent_profile_changes(person_id)

    return {
        "person": person_info,
        "profile": profile,
        "core_memory": core_items,
        "recent_episodes": recent_items,
        "long_term_memory": long_term_items,
        "contacts": contacts,
        "relations": relations,
        "sessions": sessions,
        "recent_changes": recent_changes,
    }
