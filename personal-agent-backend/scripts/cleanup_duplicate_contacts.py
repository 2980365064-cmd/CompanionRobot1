#!/usr/bin/env python3
"""清理重复/低质第三方联系人画像，合并到 Wiki 同步档案。

用途：
  在引入 person_resolver 统一解析层后运行一次，清理之前因 contact 创建
  逻辑未查 Wiki 人物页而产生的重复记录（同一人物既有 Wiki 同步 profile，
  又有低置信的独立 contact）。

运行：
  python scripts/cleanup_duplicate_contacts.py          # dry-run（仅预览）
  python scripts/cleanup_duplicate_contacts.py --apply  # 实际执行

合并规则：
  1. 列出当前 owner 下所有 profile_role=contact 的记录
  2. 对每个 contact，通过 resolve_person 判断是否对应已知的 Wiki/长期记忆 人物
  3. 如果是 Wiki 已知但存在多份记录 → 保留最新的/wiki-synced 的，合并 mention_count/notes/experiences
  4. 删除未确认、仅一次提及、无实质内容的低质独立 contact
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.contacts import (
    PROFILE_ROLE_CONTACT,
    find_contact_profile,
    list_contacts_for_owner,
)
from app.memory.person_resolver import (
    PersonResolution,
    list_wiki_people,
    resolve_person,
    sync_wiki_people_to_contacts,
)
from app.memory.profile import (
    normalize_profile,
    profile_display_name,
    profile_nicknames,
)
from app.session import store


def _get_device_id() -> str:
    """从配置或现有数据中获取设备 ID。"""
    profiles = store.list_person_profiles("")
    if profiles:
        return profiles[0].get("person_id", "").split("_")[0]
    from app.config import settings
    return getattr(settings, "default_device_id", "") or "default"


def _get_owner_person_id() -> str:
    from app.config import settings
    owner = str(getattr(settings, "default_owner_person_id", "") or "").strip()
    if owner:
        return owner
    # 找第一个非 contact 的 profile 作为 owner
    device_id = _get_device_id()
    for row in store.list_person_profiles(device_id):
        p = normalize_profile(row["profile"])
        if str(p.get("profile_role") or "") != PROFILE_ROLE_CONTACT:
            return str(p["person_id"])
    return "default"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="清理重复/低质第三方联系人画像")
    parser.add_argument("--apply", action="store_true", help="实际执行合并和删除")
    parser.add_argument("--device-id", default="", help="设备 ID（可选）")
    parser.add_argument("--owner-id", default="", help="owner person_id（可选）")
    args = parser.parse_args()

    dry_run = not args.apply
    device_id = args.device_id or _get_device_id()
    owner_person_id = args.owner_id or _get_owner_person_id()

    print("=" * 60)
    print(f"联系人去重清理 — {'DRY-RUN' if dry_run else '实际执行'}")
    print(f"  device_id={device_id}, owner_id={owner_person_id}")
    print("=" * 60)

    # Step 1: 先同步 Wiki 人物页到 contact（确保基准正确）
    print("\n[Step 1] Wiki 人物页同步...")
    sync_results = sync_wiki_people_to_contacts(device_id, owner_person_id, dry_run=dry_run)
    for r in sync_results:
        print(f"  {r['name']}: {r['action']} — {r['detail']}")

    # Step 2: 列出所有 contact，找出重复的
    print("\n[Step 2] 扫描现有联系人...")
    all_contacts = list_contacts_for_owner(device_id, owner_person_id)
    wiki_names = {w["name"] for w in list_wiki_people()}
    wiki_contacts: list[dict] = []
    independent_contacts: list[dict] = []
    for p in all_contacts:
        name = profile_display_name(p)
        is_wiki = p.get("source") == "wiki" or name in wiki_names
        if is_wiki:
            wiki_contacts.append(p)
        else:
            independent_contacts.append(p)

    print(f"  总计 {len(all_contacts)} 个联系人")
    print(f"  Wiki 同步源: {len(wiki_contacts)}")
    print(f"  独立联系人: {len(independent_contacts)}")

    # Step 3: 标记可删除的低质独立 contact
    print("\n[Step 3] 标记低质/重复联系人...")
    to_delete: list[dict] = []
    to_preserve: list[dict] = []
    for p in independent_contacts:
        name = profile_display_name(p)
        mention = int(p.get("mention_count") or 0)
        has_content = bool(
            p.get("relationship") or p.get("notes") or p.get("experiences")
        )
        confirmed = bool(p.get("confirmed"))

        # 检查是否对应已知 Wiki/长期记忆 人物
        res = resolve_person(name, device_id, owner_person_id)
        is_duplicate_of_wiki = res.source in ("wiki", "memory_items") and res.known

        reasons: list[str] = []
        if is_duplicate_of_wiki:
            reasons.append(f"重复于 Wiki 人物({res.source})")

        if not confirmed and mention <= 1 and not has_content:
            reasons.append("未确认+仅1次提及+无内容")

        if reasons:
            to_delete.append({"profile": p, "reasons": "; ".join(reasons)})
        else:
            to_preserve.append(p)

    if to_delete:
        print(f"  → 可删除: {len(to_delete)}")
        for item in to_delete:
            name = profile_display_name(item["profile"])
            print(f"    - {name}: {item['reasons']}")
    else:
        print("  → 无可删除的联系人")
    print(f"  → 保留: {len(to_preserve)}")

    # Step 4: 合并 Wiki contact 中的重复（多个 wiki 源 contact 对应同一个人物）
    print("\n[Step 4] 合并 Wiki 联系人中的重复...")
    merged_count = 0
    wiki_by_name: dict[str, list[dict]] = {}
    for p in wiki_contacts:
        name = profile_display_name(p)
        wiki_by_name.setdefault(name, []).append(p)

    for name, profiles in wiki_by_name.items():
        if len(profiles) <= 1:
            continue
        # 找到 wiki_synced 或最新更新的作为主记录
        profiles_sorted = sorted(
            profiles,
            key=lambda x: (
                2 if x.get("source") == "wiki" else 1 if x.get("confirmed") else 0,
                str(x.get("updated_at", "")),
            ),
            reverse=True,
        )
        primary = profiles_sorted[0]
        primary_norm = normalize_profile(primary)
        merged_count += len(profiles_sorted) - 1
        if dry_run:
            keep = profile_display_name(primary)
            dupes = [profile_display_name(p) for p in profiles_sorted[1:]]
            print(f"  {keep}: 将合并 {len(dupes)} 个重复 -> {dupes}")
        else:
            # 合并注计数和笔记到主记录
            for dup in profiles_sorted[1:]:
                dp = normalize_profile(dup)
                primary_norm["mention_count"] = int(primary_norm.get("mention_count", 0)) + int(
                    dp.get("mention_count", 0)
                )
                for key in ("notes", "experiences", "personality"):
                    existing_notes = set(str(x) for x in primary_norm.get(key, []))
                    for item in dp.get(key, []):
                        s = str(item).strip()
                        if s and s not in existing_notes:
                            primary_norm.setdefault(key, []).append(s)
                            existing_notes.add(s)
                # 删除重复
                dup_id = dp.get("person_id", "")
                with store._conn() as conn:
                    conn.execute("DELETE FROM person_profiles WHERE person_id=?", (dup_id,))
            primary_norm["updated_at"] = __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat()
            store.save_person_profile(device_id, primary_norm)
            print(f"  {name}: 合并了 {len(profiles_sorted)-1} 个重复记录")

    if not merged_count:
        print("  无需要合并的重复 Wiki 联系人")

    # Step 5: 实际删除
    print("\n[Step 5] 删除低质联系人...")
    if dry_run:
        print(f"  将删除 {len(to_delete)} 个，--apply 以执行")
    else:
        deleted = 0
        for item in to_delete:
            pid = item["profile"].get("person_id", "")
            with store._conn() as conn:
                conn.execute("DELETE FROM person_profiles WHERE person_id=?", (pid,))
            name = profile_display_name(item["profile"])
            print(f"  已删除: {name} — {item['reasons']}")
            deleted += 1
        print(f"  共删除 {deleted} 个低质联系人")

    print("\n" + "=" * 60)
    if dry_run:
        print("DRY-RUN 完成。使用 --apply 执行实际清理。")
    else:
        print("清理完成。")


if __name__ == "__main__":
    main()
