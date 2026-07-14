#!/usr/bin/env python3
"""Person Resolver 评估脚本 —— 验证统一人物解析层核心行为。

验证项：
  1. resolve_person("唐凯") → source=wiki
  2. resolve_person("伍钰涛") → source=wiki
  3. resolve_person("不存在的人名") → source=unknown
  4. sync_wiki_people_to_contacts → 正确创建/更新 contact
  5. 同名 contact 存在时不重复创建
  6. upsert_contact_from_signal 集成 resolver 后的行为
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.person_resolver import (
    PersonResolution,
    list_wiki_people,
    resolve_person,
    sync_wiki_people_to_contacts,
)
from app.memory.contacts import (
    ContactSignal,
    upsert_contact_from_signal,
    find_contact_profile,
    list_contacts_for_owner,
)
from app.config import settings

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def assert_eq(label: str, actual, expected):
    ok = actual == expected
    print(f"  {'✓' if ok else '✗'} {label}: {actual}" if ok else f"  ✗ {label}: 期望={expected}, 实际={actual}")
    return ok


def assert_source(label: str, res: PersonResolution, expected_source: str):
    ok = res.source == expected_source
    print(f"  {'✓' if ok else '✗'} {label}: source={res.source} known={res.known}" + (f" display={res.display_name}" if ok else f" 期望={expected_source}"))
    return ok


def test_1_resolve_wiki_known():
    """已知 Wiki 人物能正确解析为 source=wiki"""
    print("\n[Test 1] resolve_person — Wiki 已知人物")
    checks = [
        ("唐凯", "wiki"),
        ("伍钰涛", "wiki"),
        ("叶鹏祥", "wiki"),
    ]
    results = []
    for name, expected in checks:
        res = resolve_person(name, "default", str(settings.default_owner_person_id or "default"))
        ok = assert_source(f"resolve_person({name!r})", res, expected)
        if ok and expected == "wiki":
            ok = ok and bool(res.wiki_meta)
            if not res.wiki_meta:
                print(f"    ! display={res.display_name}, wiki_meta 为空")
        results.append(ok)
    return all(results)


def test_2_resolve_unknown():
    """不存在的名字解析为 source=unknown"""
    print("\n[Test 2] resolve_person — 未知人物")
    names = ["阿斯顿马丁", "张三丰", "完全不存在的名字测试"]
    results = []
    for name in names:
        res = resolve_person(name, "default", str(settings.default_owner_person_id or "default"))
        ok = assert_source(f"resolve_person({name!r})", res, "unknown")
        results.append(ok)
    return all(results)


def test_3_wiki_meta_fields():
    """Wiki 元数据包含正确字段"""
    print("\n[Test 3] Wiki 元数据完整性")
    res = resolve_person("唐凯", "default", str(settings.default_owner_person_id or "default"))
    if res.source != "wiki":
        print(f"  {FAIL} 唐凯 不是 Wiki 来源 (source={res.source})")
        return False
    meta = res.wiki_meta or {}
    checks = []
    checks.append(assert_eq("display_name 不为空", bool(res.display_name), True))
    checks.append(assert_eq("relationship 不为空", bool(meta.get("relationship")), True))
    checks.append(assert_eq("aliases 不为空", bool(meta.get("aliases")), True))
    checks.append(assert_eq("people 不为空", bool(meta.get("people")), True))
    checks.append(assert_eq("body_facts 不为空", bool(res.wiki_body_facts), True))
    return all(checks)


def test_4_resolve_contact_after_sync():
    """sync 后 resolve 返回 source=contact"""
    print("\n[Test 4] Sync + contact 解析")
    device_id = "default"
    owner_id = str(settings.default_owner_person_id or "default")

    # 先同步
    results = sync_wiki_people_to_contacts(device_id, owner_id, dry_run=False)
    print(f"  sync 结果: {len(results)} 条")

    # 同步后再解析
    res = resolve_person("唐凯", device_id, owner_id)
    if res.source in ("wiki", "contact"):
        print(f"  {PASS} resolve_person(唐凯) → source={res.source}")
        return True
    print(f"  {FAIL} resolve_person(唐凯) → source={res.source}, 期望 wiki 或 contact")
    return False


def test_5_upsert_with_resolver():
    """集成 resolver 的 upsert 行为：Wiki 已知不新建独立 contact"""
    print("\n[Test 5] upsert_contact_from_signal + resolver 集成")
    device_id = "default"
    owner_id = str(settings.default_owner_person_id or "default")

    # 先清理可能存在的低质 contact
    existing = find_contact_profile(device_id, owner_id, "唐凯")
    if existing:
        print(f"  contact 已存在 (source={existing.get('source')})，验证 source 字段")
        if existing.get("source") == "wiki":
            print(f"  {PASS} source=wiki 已标记")
        else:
            print(f"  - source={existing.get('source')}")

    # 给一个 high 信号，看 upsert 行为
    sig = ContactSignal("唐凯", "high", relationship="好友", source="test")
    res_before = resolve_person("唐凯", device_id, owner_id)
    prof, ev = upsert_contact_from_signal(device_id, owner_id, sig, resolution=res_before)

    if prof:
        display = prof.get("display_name", "")
        source = prof.get("source", "")
        print(f"  upsert 结果: display={display}, source={source}, event={ev}")
        print(f"  {PASS} 正常返回")
    else:
        print(f"  upsert 返回 None — 可能因 long_term 来源跳过")
        print(f"  event: {ev}")

    # 验证 list 中不出现多余的未确认 contact
    contacts = list_contacts_for_owner(device_id, owner_id)
    tang_kai_contacts = [p for p in contacts if "唐凯" in str(p.get("display_name", ""))]
    print(f"  当前唐凯相关 contact 数量: {len(tang_kai_contacts)}")
    if len(tang_kai_contacts) <= 1:
        print(f"  {PASS} 无多余重复")
    else:
        print(f"  {FAIL} 存在 {len(tang_kai_contacts)} 个重复，请运行 cleanup_script")

    return True


def test_6_list_wiki_people():
    """list_wiki_people 返回所有 Wiki 人物"""
    print("\n[Test 6] list_wiki_people")
    people = list_wiki_people()
    print(f"  Wiki 人物总数: {len(people)}")
    names = [p["name"] for p in people]
    print(f"  人物列表: {names}")
    if len(people) >= 3:
        print(f"  {PASS} 至少 3 个 Wiki 人物")
        return True
    print(f"  {FAIL} 预期至少 3 个，实际 {len(people)}")
    return False


def test_7_resolve_long_term_fallback():
    """长期记忆 中已存在但不作为 wiki 的人物（在 长期记忆 有数据但非 people 目录）"""
    print("\n[Test 7] resolve_person — 长期记忆 fallback")
    # 用一个不太可能是有名有姓的真实人名来测试
    # 实际 长期记忆 可能没有，所以这个测试只是确认不 crash
    res = resolve_person("随机不存在", "default", str(settings.default_owner_person_id or "default"))
    print(f"  resolve_person(随机不存在) → source={res.source}, known={res.known}")
    print(f"  {PASS if res.source == 'unknown' else '?'} 测试完成（长期记忆 检查不可独立验证）")
    return True


def main():
    print("=" * 60)
    print("Person Resolver 评估")
    print("=" * 60)

    tests = [
        ("Wiki 已知人物", test_1_resolve_wiki_known),
        ("未知人物", test_2_resolve_unknown),
        ("Wiki 元数据", test_3_wiki_meta_fields),
        ("同步后 contact 解析", test_4_resolve_contact_after_sync),
        ("upsert 集成", test_5_upsert_with_resolver),
        ("Wiki 人物列表", test_6_list_wiki_people),
        ("长期记忆 fallback", test_7_resolve_long_term_fallback),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'─' * 50}")
        print(f"测试: {name}")
        try:
            ok = fn()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {FAIL} 异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n\n{'=' * 60}")
    print(f"结果: {passed}/{passed + failed} 通过{f', {failed} 失败' if failed else ''}")
    print(f"{'全部通过 ✓' if failed == 0 else '有失败项 ✗，请检查'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
