"""
统一人物解析层 —— 作为第三方人物画像（contact）和 Wiki 人物页之间的桥梁。

职责：
  当对话中提到一个人名时，判断系统是否已经"认识"TA（通过 contact / Wiki / 统一记忆库 三个渠道），
  返回统一的 PersonResolution 结构，避免 Wiki 已知人物被误当作陌生人新建空画像。

Resolution 顺序（逐级降级）：
  1. contact（当前 owner 下的已存第三方画像，profile_role=contact）
  2. Wiki 人物页（persona/corpus/people/*.md 的 frontmatter 和 # 标题）
  3. 统一记忆库中 entity/relationship/fact 类型的条目
  4. unknown（系统不认识此人）
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings
from app.session import store

logger = logging.getLogger(__name__)

# ── Wiki 人物页索引缓存 ──────────────────────────────────────────────
_WIKI_CACHE: dict[str, Any] = {"last_scan": 0.0, "people": {}}
_WIKI_CACHE_TTL = 600  # 10 分钟过期


@dataclass
class PersonResolution:
    """人物解析结果。

    resolve_person() 的返回值，描述系统对某个名称的认知状态。
    caller 根据 known / source 决定后续行为：
      - contact: 已有画像，只更新提及次数
      - wiki: 已有人物页，自动同步为 contact，不新建
      - memory_items: 已通过统一记忆库认知，但不直接新建空画像
      - unknown: 系统不认识，允许按信号新建
    """
    known: bool
    source: str                     # contact | wiki | memory_items | unknown
    name: str                       # 查询时使用的原始名称
    display_name: str = ""          # 标准化后的显示名
    contact_profile: dict | None = None  # 如果命中 contact
    wiki_meta: dict | None = None   # 如果命中 wiki：frontmatter（people/aliases/relationship 等）
    wiki_body_facts: list[str] = field(default_factory=list)  # Wiki 正文中提取的短事实
    wiki_source_path: str = ""      # Wiki 文件相对路径（如 people/tang_kai.md）
    memory_summary: str = ""        # 如果命中统一记忆库：简洁摘要
    confidence: float = 0.0

    def to_log(self) -> str:
        """生成监控日志用的一行摘要。"""
        return f"resolve_person({self.name}) → {self.source} known={self.known}"


# ── 内部工具函数 ────────────────────────────────────────────────────

_NORM_RE = re.compile(r"\s+")


def _normalize(name: str) -> str:
    """去掉空白字符，用于名称比较。"""
    return _NORM_RE.sub("", (name or "").strip())


def _people_dir_abs() -> Path | None:
    """获取 persona/corpus/people/ 的绝对路径。"""
    corpus = settings.resolved_corpus_dir()
    if not corpus:
        return None
    people_dir = Path(str(corpus)) / "people"
    return people_dir if people_dir.is_dir() else None


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """简易 YAML frontmatter 解析（不依赖 pyyaml）。

    仅提取 person_resolver 需要的字段：people, aliases, relationship, confidence。
    """
    raw = raw.lstrip("﻿")
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    fm_text = raw[3:end].strip()
    body = raw[end + 4:].strip()
    fm: dict[str, Any] = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not val:
            continue
        # 列表值（如 people: [唐凯, 张三]）
        if val.startswith("["):
            items = re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\[\]\s]+)", val)
            fm[key] = [x for t in items for x in t if x]
        else:
            fm[key] = val
    return fm, body


def _extract_wiki_facts(body: str) -> list[str]:
    """从 Wiki 人物页正文中提取短事实，主要从 ## 身份关系 和 ## 人物特征 两节。"""
    facts: list[str] = []
    current_section = ""
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if current_section in ("身份关系", "人物特征") and len(line) > 4:
            facts.append(line.strip("- *"))
    return facts[:8]  # 最多 8 条短事实


# ── Wiki 人物页索引 ──────────────────────────────────────────────────

def _build_wiki_index() -> dict[str, dict]:
    """扫描 persona/corpus/people/*.md，构建 {归一化名称: info} 索引。

    索引结构：
      { normalized_name: {
            "people": [display_name, ...],
            "aliases": [...],
            "relationship": "...",
            "confidence": float,
            "source_path": "people/tang_kai.md",
            "body_facts": [...],
            "all_names": {归一化后的所有称呼},
        }
      }
    """
    now = time.time()
    if now - _WIKI_CACHE["last_scan"] < _WIKI_CACHE_TTL:
        return _WIKI_CACHE["people"]

    people_dir = _people_dir_abs()
    if not people_dir:
        _WIKI_CACHE["people"] = {}
        _WIKI_CACHE["last_scan"] = now
        return {}

    index: dict[str, dict] = {}
    for path in sorted(people_dir.glob("*.md")):
        rel = path.relative_to(people_dir.parent.parent)  # corpus/ 的父目录
        raw = path.read_text(encoding="utf-8", errors="ignore")
        fm, body = _parse_frontmatter(raw)
        people_list: list[str] = fm.get("people") or []
        if isinstance(people_list, str):
            people_list = [people_list]
        if not people_list:
            # 从 # 一级标题提取
            m = re.search(r"^#\s+(.+)", body, re.MULTILINE)
            if m:
                people_list = [m.group(1).strip()]

        if not people_list:
            continue

        aliases: list[str] = fm.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]

        relationship = str(fm.get("relationship") or "")
        confidence = float(fm.get("confidence") or 0.5)
        body_facts = _extract_wiki_facts(body)

        all_names = set()
        for p in people_list:
            n = _normalize(p)
            if n:
                all_names.add(n)
        for a in aliases:
            n = _normalize(a)
            if n:
                all_names.add(n)

        entry = {
            "people": people_list,
            "aliases": aliases,
            "relationship": relationship,
            "confidence": confidence,
            "source_path": str(rel),
            "body_facts": body_facts,
            "all_names": all_names,
        }

        for n in all_names:
            # 已存在的名称不覆盖（先扫描的优先）
            if n not in index:
                index[n] = entry

    _WIKI_CACHE["people"] = index
    _WIKI_CACHE["last_scan"] = now
    logger.debug("wiki person index rebuilt: %d entries for %d files", len(index), len(list(people_dir.glob("*.md"))))
    return index


def invalidate_wiki_cache() -> None:
    """强制刷新 Wiki 人物页索引缓存。

    在 ingest 完成后调用，确保索引与磁盘文件同步。
    """
    _WIKI_CACHE["last_scan"] = 0.0
    logger.info("wiki person index cache invalidated")


# ── 统一记忆库查询 ───────────────────────────────────────────────────

def _query_memory_person(name: str, person_id: str) -> dict | None:
    """在统一记忆库中检索该人物是否存在已知条目。

    仅作为二级 fallback（当 contact 和 wiki 都无命中时）。
    如果 memory_items 中有该人物的 entity/relationship/fact 类型内容，返回摘要。
    """
    try:
        from app.session import store

        name_norm = _normalize(name)
        if not name_norm:
            return None

        pid = str(person_id or "").strip()
        if not pid:
            return None

        rows = store.search_memory_items(
            pid,
            kinds=["entity", "relationship", "fact"],
            query=name_norm,
            limit=5,
        )

        if not rows:
            return None

        for row in rows:
            content = str(row.get("content", "") or "")
            if not content:
                continue
            # 检查文本开头是否包含该名称
            if name_norm in _normalize(content[:50]):
                return {"summary": content[:200], "source": "memory_items"}

        # 无精确前缀匹配时，返回第一个结果
        first_content = str(rows[0].get("content", "") or "")
        if first_content:
            return {"summary": first_content[:200], "source": "memory_items"}

        return None
    except Exception as exc:
        logger.warning("Memory person query failed: %s", exc)
        return None


# ── 公开 API ──────────────────────────────────────────────────────────

def resolve_person(
    name: str,
    device_id: str,
    owner_person_id: str,
) -> PersonResolution:
    """统一人物解析入口。

    按 contact → wiki → 统一记忆库 → unknown 的顺序判断系统是否认识此人。

    Args:
        name: 用户提到的原始名称（如"唐凯""伍钰涛"）
        device_id: 设备 ID
        owner_person_id: 当前对话对象的 person_id

    Returns:
        PersonResolution 结构，含 known/source/profile 等字段
    """
    from app.memory.contacts import find_contact_profile, list_contacts_for_owner

    name_norm = _normalize(name)
    if not name_norm:
        return PersonResolution(known=False, source="unknown", name=name)

    # ═══ 1. contact 检查 ═══
    contact = find_contact_profile(device_id, owner_person_id, name)
    if contact and _is_contact_profile(contact):
        from app.memory.profile import profile_display_name

        return PersonResolution(
            known=True,
            source="contact",
            name=name,
            display_name=profile_display_name(contact),
            contact_profile=contact,
            confidence=0.9,
        )

    # ═══ 2. Wiki 人物页检查 ═══
    index = _build_wiki_index()
    if name_norm in index:
        entry = index[name_norm]
        from app.memory.contacts import _empty_contact

        display = entry["people"][0] if entry["people"] else name
        return PersonResolution(
            known=True,
            source="wiki",
            name=name,
            display_name=display,
            wiki_meta={
                "people": entry["people"],
                "aliases": entry["aliases"],
                "relationship": entry["relationship"],
                "confidence": entry["confidence"],
            },
            wiki_body_facts=entry["body_facts"],
            wiki_source_path=entry["source_path"],
            confidence=entry["confidence"],
        )

    # ═══ 3. 统一记忆库检查（entity/relationship/fact） ═══
    memory_hit = _query_memory_person(name, owner_person_id)
    if memory_hit:
        return PersonResolution(
            known=True,
            source="memory_items",
            name=name,
            memory_summary=memory_hit["summary"],
            confidence=0.6,
        )

    # ═══ 4. unknown ═══
    return PersonResolution(known=False, source="unknown", name=name)


def _is_contact_profile(profile: dict | None) -> bool:
    if not profile:
        return False
    from app.memory.contacts import PROFILE_ROLE_CONTACT
    from app.memory.profile import normalize_profile

    p = normalize_profile(profile)
    return str(p.get("profile_role") or "") == PROFILE_ROLE_CONTACT


# ── 维护操作 ──────────────────────────────────────────────────────────

def sync_wiki_people_to_contacts(
    device_id: str | None = None,
    owner_person_id: str | None = None,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """扫描 Wiki 人物页，为每个人物页创建/更新一个 confirmed contact。

    规则：
      - 如果同名 contact 已存在，只补全缺失字段，不覆盖已有 notes/experiences
      - 新创建的 contact 自动 source="wiki"、confirmed=True
      - 如果同名 contact 存在且 source 不是 wiki，加 wiki_synced_at 但不覆盖 source
      - 返回操作记录列表

    Args:
        device_id: 设备 ID，不传则使用 settings 中的默认值或 "default"
        owner_person_id: 默认 owner person_id（谁名下的联系人），不传则使用 settings 中的默认值
        dry_run: 仅预览，不实际写入

    Returns:
        操作记录列表，每项含 {name, action, detail}
    """
    from app.memory.contacts import _contact_person_id, _empty_contact, find_contact_profile
    from app.memory.profile import normalize_profile, profile_display_name
    from datetime import datetime, timezone

    # ── 默认值 ──
    actual_device_id = device_id or getattr(settings, "default_device_id", "") or "default"
    actual_owner_person_id = (owner_person_id or "").strip() or str(
        getattr(settings, "default_owner_person_id", "") or ""
    ).strip()
    if not actual_owner_person_id:
        logger.warning("sync_wiki_people_to_contacts: no owner_person_id configured, using 'default'")
        actual_owner_person_id = "default"

    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    index = _build_wiki_index()
    seen_entries: dict[str, dict] = {}
    for _, entry in index.items():
        key = _normalize(entry["people"][0]) if entry["people"] else ""
        if key and key not in seen_entries:
            seen_entries[key] = entry

    # 排除 owner 本人和机器人自己（不应作为第三方联系人）
    owner_display = str(getattr(settings, "default_owner_display_name", "") or "").strip()
    exclude_self_names: set[str] = set()
    if owner_display:
        exclude_self_names.add(_normalize(owner_display))
    exclude_self_names.add(_normalize("叶鹏祥"))  # 机器人自己
    # 检查 actual_owner_person_id 对应的现有 profile 的名称
    try:
        owner_profile = store.get_person_profile(actual_owner_person_id)
        if owner_profile:
            from app.memory.profile import profile_nicknames
            for n in profile_nicknames(owner_profile):
                exclude_self_names.add(_normalize(n))
    except Exception:
        pass

    results: list[dict] = []
    for _, entry in seen_entries.items():
        display = entry["people"][0] if entry["people"] else "未知"
        # 跳过 owner 本人和机器人自己
        if _normalize(display) in exclude_self_names:
            logger.debug("wiki sync skip self: %s", display)
            continue
        rel = entry["relationship"]
        aliases = list(entry["aliases"])
        body_facts = entry["body_facts"][:4]
        path = entry["source_path"]

        existing = find_contact_profile(actual_device_id, actual_owner_person_id, display)
        now = _now()

        if existing:
            # 已存在 contact → 补全缺失字段，标记 wiki 同步
            p = normalize_profile(existing)
            changes: list[str] = []

            if not p.get("source") or p["source"] == "contact":
                if not dry_run:
                    p["source"] = "wiki"
                changes.append("source→wiki")

            if not dry_run:
                p["wiki_synced_at"] = now
                p["source_path"] = path
                p["confirmed"] = True
            changes.append("confirmed→True")

            # 补空字段但不覆盖已填的
            if not p.get("relationship") and rel:
                if not dry_run:
                    p["relationship"] = rel
                changes.append(f"relationship={rel}")

            existing_aliases = set(_normalize(a) for a in (p.get("aliases") or []))
            new_aliases = [a for a in aliases if _normalize(a) not in existing_aliases]
            if new_aliases:
                if not dry_run:
                    p.setdefault("aliases", []).extend(new_aliases)
                changes.append(f"aliases+{new_aliases}")

            existing_notes = set(str(n) for n in (p.get("notes") or []))
            new_facts = [f for f in body_facts if f not in existing_notes]
            if new_facts:
                if not dry_run:
                    p.setdefault("notes", []).extend(new_facts)
                changes.append(f"notes+{len(new_facts)}")

            if changes and not dry_run:
                p["updated_at"] = now
                store.save_person_profile(actual_device_id, p)

            results.append({
                "name": display,
                "action": "已存在，补全" if changes else "已存在，无需变更",
                "detail": "; ".join(changes) if changes else "完全一致",
            })
        else:
            # 新建 confirmed contact
            if not dry_run:
                profile = _empty_contact(actual_device_id, actual_owner_person_id, display, relationship=rel)
                profile["source"] = "wiki"
                profile["source_path"] = path
                profile["wiki_synced_at"] = now
                profile["confirmed"] = True
                profile["aliases"] = list(aliases)
                profile["notes"] = list(body_facts)
                profile["mention_count"] = 1
                profile["updated_at"] = now
                store.save_person_profile(actual_device_id, profile)

            results.append({
                "name": display,
                "action": "新建" if not dry_run else "将新建",
                "detail": f"display={display}, relationship={rel}, aliases={aliases}, notes_count={len(body_facts)}",
            })

    return results


def list_wiki_people() -> list[dict]:
    """列出所有 Wiki 人物页的基本信息（不依赖 DB）。"""
    index = _build_wiki_index()
    seen: dict[str, dict] = {}
    for _, entry in index.items():
        key = _normalize(entry["people"][0]) if entry["people"] else ""
        if key and key not in seen:
            seen[key] = {
                "name": entry["people"][0],
                "aliases": entry["aliases"],
                "relationship": entry["relationship"],
                "source_path": entry["source_path"],
                "confidence": entry["confidence"],
            }
    return list(seen.values())
