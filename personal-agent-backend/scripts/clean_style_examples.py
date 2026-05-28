#!/usr/bin/env python3
"""Remove semantically broken Q→A style examples (WeChat merge/compress artifacts)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PERSONA = Path(__file__).resolve().parents[2] / "persona"
STYLE_FILES = [
    PERSONA / "style" / "examples.md",
    PERSONA / "style" / "examples_intimate.md",
    PERSONA / "import" / "drafts" / "style_daily.md",
    PERSONA / "import" / "drafts" / "style_intimate.md",
    PERSONA / "import" / "drafts" / "style_group_friends.md",
]

PAIR_RE = re.compile(r"^问[:：](.*)$")
ANS_RE = re.compile(r"^答[:：](.*)$")
BRACKET = re.compile(r"\[[^\]]+\]")
GENERIC_A = frozenset({
    "还好吧 其实这个", "嗯嗯对对对", "笑死了", "666", "6666", "嗯嗯对对",
    "有点逆天", "逆天", "好吧晚安", "嗯嗯嗯呢", "还好吧说实话 没那么严重",
    "今天瘦了这么多逆天",
})
KEEP_ANCHORS = [
    ("在吗", "在，咋了"),
    ("谢谢", "行，客气啥"),
    ("想你了", "嗯，我也想你"),
    ("在干嘛", "没干嘛，咋了"),
    ("在吗老公", "在，咋了嘛"),
    ("我是刘远慧", "咋滴，终于晓得回我了啊"),
    ("今天吃什么", "辣的呗，别整鸡鸭鱼"),
    ("你还健身吗", "想啊，没时间"),
    ("咋不理我", "理你啊，刚忙到"),
    ("你是不是不想理我", "没得，刚忙到"),
]


def parse_pairs(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    header: list[str] = []
    pairs: list[tuple[str, str]] = []
    q: str | None = None
    started = False
    for line in text.splitlines():
        stripped = line.strip()
        mq = PAIR_RE.match(stripped)
        ma = ANS_RE.match(stripped)
        if mq:
            started = True
            q = mq.group(1).strip()
            continue
        if ma and q is not None:
            pairs.append((q, ma.group(1).strip()))
            q = None
            continue
        if not started and stripped:
            header.append(line)
    return header, pairs


def is_broken(q: str, a: str, *, strict: bool) -> bool:
    if not q or not a or len(a) < 2:
        return True
    max_a, max_q = (32, 50) if strict else (48, 85)
    if len(a) > max_a or len(q) > max_q:
        return True
    if len(a) > 18 and a.count(" ") >= 3:
        return True
    if len(q) > 28 and q.count(" ") >= 5:
        return True
    if "[引用" in q or "[引用" in a:
        return True
    if "&#x20;" in q or "&#x20;" in a or "转发的聊天记录" in q:
        return True
    if re.search(r"(.)\1{7,}", re.sub(r"\s+", "", a)):
        return True
    if q.count("[") >= 3:
        return True
    if q.count("?") + q.count("？") > 2:
        return True
    if a in GENERIC_A:
        return True
    if "嗯嗯对对" in a:
        return True
    if strict and len(q) > 42:
        return True
    if "还好吧" in a and "其实" in a:
        return True
    q_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", q))
    a_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}", a))
    if len(q) > 22 and len(a) > 10 and q_tokens and a_tokens:
        if len(q_tokens & a_tokens) == 0:
            return True
    chunks = [c for c in re.split(r"[\s，。！？]+", a) if len(c) >= 4]
    if len(chunks) >= 3 and len(a) > 22:
        return True
    return False


def clean_pairs(pairs: list[tuple[str, str]], *, strict: bool) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for q, a in pairs:
        if (q, a) in KEEP_ANCHORS:
            key = re.sub(r"\s+", "", a)[:30]
            if key not in seen:
                seen.add(key)
                out.append((q, a))
            continue
        a_clean = BRACKET.sub("", a).strip()
        a_clean = re.sub(r"\s+", " ", a_clean).strip()
        if is_broken(q, a_clean, strict=strict):
            continue
        key = re.sub(r"\s+", "", a_clean)[:30]
        if key in seen:
            continue
        seen.add(key)
        out.append((q, a_clean))
    return out


def write_file(path: Path, header: list[str], pairs: list[tuple[str, str]]) -> None:
    # 只保留标题段（到空行或首条问句之前）
    head: list[str] = []
    for line in header:
        if line.strip().startswith("问：") or line.strip().startswith("答："):
            break
        head.append(line)
    while head and not head[-1].strip():
        head.pop()
    lines = list(head)
    if lines:
        lines.append("")
    for q, a in pairs:
        lines += [f"问：{q}", f"答：{a}", ""]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    total_before = total_after = 0
    for path in STYLE_FILES:
        if not path.is_file():
            continue
        strict = path.parent.name == "style"
        header, pairs = parse_pairs(path.read_text(encoding="utf-8"))
        if path.name == "examples.md":
            continue  # 主文件已人工 curated，跳过自动清洗
        kept = clean_pairs(pairs, strict=strict)
        total_before += len(pairs)
        total_after += len(kept)
        write_file(path, header, kept)
        print(f"{path.name}: {len(pairs)} -> {len(kept)}")
    print(f"total {total_before} -> {total_after}")


if __name__ == "__main__":
    main()
