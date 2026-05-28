#!/usr/bin/env python3
"""Analyze WeChat export → curated style/examples + persona stats."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.import_wechat import (
    DEFAULT_INPUT,
    dedupe_pairs,
    extract_qa_pairs,
    merge_blocks,
    parse_file,
    score_pair,
)

PERSONA = Path(__file__).resolve().parents[2] / "persona"
STYLE_OUT = PERSONA / "style" / "examples.md"
INTIMATE_OUT = PERSONA / "style" / "examples_intimate.md"
STYLE_ANALYSIS = PERSONA / "config" / "style_analysis.json"

PHRASES = [
    "你妈", "得了", "再说吧", "咋滴", "咋了", "咋个", "啥子", "嗯嗯", "嘻嘻",
    "笑死", "666", "我去", "逆天", "太屌", "okok", "行吧", "可以", "别整",
    "听我的", "宝贝", "老公", "远慧", "刘大炮", "烦死了", "累死了", "想你了",
    "爱你", "要得", "晓得", "嘛", "呗", "哦", "啊", "莫得", "整",
]

SKIP_NAME = re.compile(r"刘兴光|刘星光|江.?挽|引用\s")
BRACKET = re.compile(r"\[[^\]]+\]")
ANCHORS = [
    {"q": "我是刘远慧", "a": "咋滴，终于晓得回我了啊"},
    {"q": "在吗老公", "a": "在，咋了嘛"},
    {"q": "在干嘛", "a": "没干嘛，咋了"},
    {"q": "想你了", "a": "嗯，我也想你"},
    {"q": "你是不是不想理我", "a": "没得，刚忙到"},
    {"q": "咋不理我", "a": "理你啊，刚忙到"},
    {"q": "今天吃什么", "a": "辣的呗，别整鸡鸭鱼"},
    {"q": "你还健身吗", "a": "想啊，没时间"},
    {"q": "谢谢", "a": "行，客气啥"},
    {"q": "在吗", "a": "在，咋了"},
]


def analyze_my_messages(rows: list[dict]) -> dict:
    my = [r["text"] for r in rows if r["is_self"]]
    lens = [len(t) for t in my]
    phrase_hits = Counter()
    for t in my:
        for p in PHRASES:
            if p in t:
                phrase_hits[p] += 1
    return {
        "count": len(my),
        "avg_len": round(sum(lens) / len(lens), 1) if lens else 0,
        "pct_len_le_15": round(100 * sum(1 for l in lens if l <= 15) / len(lens), 1) if lens else 0,
        "pct_len_le_30": round(100 * sum(1 for l in lens if l <= 30) / len(lens), 1) if lens else 0,
        "top_phrases": phrase_hits.most_common(30),
    }


def curate_score(p: dict, *, intimate: bool) -> float:
    a, q = p["a"], p["q"]
    if SKIP_NAME.search(a) or SKIP_NAME.search(q):
        return -99
    if len(a) > 32 or len(a) < 3:
        return -99
    if len(q) > 55:
        return -99
    if BRACKET.search(a) or "[引用" in q:
        return -20
    q_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", q))
    a_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", a))
    if len(q) > 18 and q_tokens and a_tokens and not (q_tokens & a_tokens):
        return -15
    s = score_pair(p, private=intimate)
    if len(a) <= 12:
        s += 5
    elif len(a) <= 22:
        s += 3
    elif len(a) <= 35:
        s += 1
    else:
        s -= 2
    if len(q) > 80:
        s -= 3
    return s


def pick_pairs(blocks: list[dict], *, intimate: bool, limit: int) -> list[dict]:
    raw = extract_qa_pairs(blocks, max_answer_len=45)
    pool = [p for p in raw if bool(p.get("intimate")) == intimate]
    seen: set[str] = set()
    scored = sorted(pool, key=lambda p: curate_score(p, intimate=intimate), reverse=True)
    out: list[dict] = []
    for p in scored:
        if curate_score(p, intimate=intimate) < 0:
            continue
        key = re.sub(r"\s+", "", p["a"])[:35]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def write_examples_md(path: Path, pairs: list[dict], title: str, intro: str) -> None:
    lines = [f"# {title}", "", intro, ""]
    for p in pairs:
        a = BRACKET.sub("", p["a"]).strip()
        q = p["q"][:100] + ("…" if len(p["q"]) > 100 else "")
        lines += [f"问：{q}", f"答：{a}", ""]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    if not DEFAULT_INPUT.is_file():
        sys.exit(f"找不到 {DEFAULT_INPUT}")

    rows = parse_file(DEFAULT_INPUT, "打瓦女高手", "我")
    stats = analyze_my_messages(rows)
    blocks = merge_blocks(rows)

    daily = pick_pairs(blocks, intimate=False, limit=55)
    intimate = pick_pairs(blocks, intimate=True, limit=30)

    seen = {re.sub(r"\s+", "", p["a"])[:30] for p in daily}
    for a in ANCHORS:
        k = re.sub(r"\s+", "", a["a"])[:30]
        if k not in seen:
            daily.insert(0, a)
            seen.add(k)

    write_examples_md(
        STYLE_OUT,
        daily[:48],
        "口吻范例（微信 7.1 万条提炼）",
        f"真实数据：你平均 **{stats['avg_len']}** 字/条，{stats['pct_len_le_15']}% 在 15 字内。"
        " 回答必须同样短。禁止括号旁白、禁止客服腔、没问工作别扯代码。",
    )
    INTIMATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_examples_md(
        INTIMATE_OUT,
        intimate[:28],
        "女友私密口吻（微信提炼）",
        "可黏可骚，但仍短句口语，不要小作文。",
    )

    STYLE_ANALYSIS.write_text(
        json.dumps({"stats": stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Wrote", STYLE_OUT, INTIMATE_OUT)
    print("Top phrases:", stats["top_phrases"][:15])


if __name__ == "__main__":
    main()
