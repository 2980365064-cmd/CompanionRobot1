#!/usr/bin/env python3
"""Wiki 语料召回验收脚本。

用途：对固定问题集执行 L3 检索，验证新 Wiki 化语料的召回质量。
输出 top-5 的 source/category/score/text，并判定 top-1 是否命中期望类型。

依赖：需在 virtualenv 中执行，PYTHONPATH=.
    source .venv/bin/activate && PYTHONPATH=. python scripts/eval_wiki_memory.py
"""

import sqlite3
import sys
from pathlib import Path

# 确保可以从项目根目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.memory.l3 import semantic_memory
from app.store.chunks import prepare_fts_text, build_fts_match_query


# ── 验收问题集 ──

TEST_CASES = [
    ("刘远慧是谁？", ["person"],
     "人物识别：应命中 people/liu_yuanhui.md"),
    ("刘远慧喜欢什么？", ["person", "preference"],
     "偏好查询：应命中 person 的偏好块或 preference 文件"),
    ("刘远慧不喜欢吃什么？", ["preference"],
     "负向偏好：应命中 preference 文件的食物禁忌"),
    ("我能怎么称呼刘远慧？", ["person", "taboo"],
     "称呼查询：应命中 person 的可用称呼块 或 taboo 称呼规则"),
    ("哪些称呼不能用？", ["taboo"],
     "称呼禁忌：应命中 taboo 文件"),
    ("我们现在还有什么待跟进的事？", ["open_loop"],
     "待办查询：应命中 open_loops/active_open_loops.md"),
    ("2026年5月我们在杭州聊了什么？", ["monthly", "event"],
     "月份回忆：应命中 monthly 或 event 文件"),
    ("伍钰涛是谁？", ["person"],
     "朋友人物：应命中 people/wu_yutao.md"),
    ("唐凯维权那件事是什么？", ["event"],
     "朋友事件：应命中 events/friends/ 下的维权事件页"),
    ("我现在在哪实习？", ["person", "event", "monthly"],
     "现状查询：应命中 person/杭州实习事件/月度页"),
]


def get_type_from_source(source: str, category: str) -> str:
    """根据 source 路径和 category 判断块类型。"""
    if category in ('person', 'preference', 'taboo', 'open_loop', 'event', 'monthly'):
        return category
    if '/people/' in source:
        return "person"
    if 'taboos' in source:
        return "taboo"
    if '/preferences/' in source:
        return "preference"
    if '/open_loops/' in source:
        return "open_loop"
    if '/events/' in source:
        return "event"
    if '/monthly/' in source:
        return "monthly"
    if '/archive/' in source:
        return "archive"
    return "other"


def run_eval():
    print("=" * 78)
    print("  🧪 Wiki 记忆库召回验收脚本 v1")
    print("  " + "=" * 56)
    print(f"  时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    # ── 基础检查 ──
    db = sqlite3.connect(settings.resolved_db_path())
    db.row_factory = sqlite3.Row

    total = db.execute("SELECT COUNT(*) FROM l3_chunks").fetchone()[0]
    corpus = db.execute("SELECT COUNT(*) FROM l3_chunks WHERE collection IN ('corpus','memory')").fetchone()[0]
    archives = db.execute("SELECT COUNT(*) FROM l3_chunks WHERE source LIKE 'archive/%'").fetchone()[0]

    print(f"\n  总 L3 块: {total}  语料块: {corpus}  归档块: {archives}")
    if archives > 0:
        print(f"  ❌ 发现 archive 旧数据！应全量 --reset 重建")
    else:
        print(f"  ✅ archive 旧数据未进入 L3")

    # ── category 分布 ──
    cats = db.execute(
        "SELECT category, COUNT(*) as cnt FROM l3_chunks GROUP BY category ORDER BY cnt DESC"
    ).fetchall()
    empty_cats = sum(c['cnt'] for c in cats if not c['category'])
    valid_cats = sum(c['cnt'] for c in cats if c['category'])
    print(f"  Category 填充率: {valid_cats}/{total} ({valid_cats*100//total}%)  空: {empty_cats}")
    for c in cats:
        print(f"    {c['category'] or '(空)':>12}: {c['cnt']}")

    db.close()

    # ── 召回测试 ──
    print(f"\n  {'─'*74}")
    print(f"  📋 召回验证")
    print(f"  {'─'*74}")

    results_summary = {"PASS": 0, "FAIL": 0, "PARTIAL": 0}

    for query, expected_types, description in TEST_CASES:
        print(f"\n  [{description}]")
        print(f"  Q: {query}")
        print(f"  期望: {'/'.join(expected_types)}")

        hits = semantic_memory.recall_l3_scored(
            device_id="corpus",
            person_id="test_123",
            query=query,
            top_k=5,
        )

        if not hits:
            print(f"  ❌ NO HITS")
            results_summary["FAIL"] += 1
            continue

        # 分析 top-1 和整体类型分布
        top1 = hits[0]
        top1_src = top1['source']
        top1_cat = top1['category']
        top1_type = get_type_from_source(top1_src, top1_cat)

        type_counts = {}
        prefix_hits = 0
        for h in hits:
            t = get_type_from_source(h['source'], h['category'])
            type_counts[t] = type_counts.get(t, 0) + 1
            if '[人物:' in h['text'] or '[类型:' in h['text'] or '[时间:' in h['text']:
                prefix_hits += 1

        # 判定
        top1_ok = top1_type in expected_types
        any_ok = any(get_type_from_source(h['source'], h['category']) in expected_types for h in hits[:3])

        if top1_ok:
            verdict = "✅ TOP1"
            results_summary["PASS"] += 1
        elif any_ok:
            verdict = "🟡 PARTIAL (top-1 偏差但 top-3 包含期望类型)"
            results_summary["PARTIAL"] += 1
        else:
            verdict = "❌ FAIL (top-3 无期望类型)"
            results_summary["FAIL"] += 1

        # 输出 top-3
        for i, h in enumerate(hits[:3]):
            src = h['source']
            cat = h['category']
            score = h['score']
            txt = h['text'][:90].replace('\n', ' ')
            t = get_type_from_source(src, cat)
            mark = "←" if i == 0 else ""
            print(f"    {i+1}. [{t:>9}] cat={cat or '-':>7} score={score:.3f} {mark}")
            print(f"       src: {src}")
            print(f"       txt: {txt}")

        print(f"    → {verdict}  (类型分布: {type_counts}, 前缀注入率: {prefix_hits}/{len(hits)})")

    # ── 汇总 ──
    total_cases = len(TEST_CASES)
    print(f"\n  {'═'*74}")
    print(f"  📊 汇总: PASS={results_summary['PASS']}/{total_cases}  "
          f"PARTIAL={results_summary['PARTIAL']}  FAIL={results_summary['FAIL']}")
    pass_rate = results_summary['PASS'] / total_cases * 100
    effective_rate = (results_summary['PASS'] + results_summary['PARTIAL']) / total_cases * 100
    print(f"  准确率: {pass_rate:.0f}%  功能正确率: {effective_rate:.0f}%")
    print(f"  {'═'*74}")

    # 退出码
    if results_summary['FAIL'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_eval()
