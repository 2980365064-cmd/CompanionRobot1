#!/usr/bin/env python3
"""WeChat TXT → persona/import/drafts/ (review before merging)."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PERSONA_DIR = Path(__file__).resolve().parents[2] / "persona"
WECHAT_RAW_DIR = PERSONA_DIR / "import" / "raw" / "wechat"
DEFAULT_INPUT = WECHAT_RAW_DIR / "clean_chat_memory.txt"
DRAFT_DIR = PERSONA_DIR / "import" / "drafts"
CORPUS_DIR = PERSONA_DIR / "corpus"
MEMORY_DAILY = CORPUS_DIR / "wechat_memory.md"
MEMORY_INTIMATE = CORPUS_DIR / "intimate.md"
MEMORY_GROUP = CORPUS_DIR / "wechat_group_friends.md"

# 群聊身份（微信昵称 → 真人）
GROUP_SELF_NICK = "一捧雪"
GROUP_FRIENDS: dict[str, str] = {
    "计算机原理": "伍钰涛",
    "唐凯": "唐凯",
    "袁子翔": "袁子翔",
}
GROUP_SKIP_NICKS = frozenset({"🐔🐔💦💦💧💧"})

LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+'([^']+)':\s*(.*)$")
GROUP_HEAD_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+'([^']+)'\s*$")
JUNK_TEXT = re.compile(
    r"^\[(表情包|图片|语音|视频|文件|其他消息|位置|名片|链接|红包|转账|撤回)"
)
COLLOQUIAL = ("你妈", "得了", "再说吧", "我去", "666", "笑死", "嗯嗯", "好吧", "逆天", "嘻嘻", "okok")
INTIMATE_HINTS = (
    "老公", "老婆", "宝贝", "宝宝", "亲亲", "抱抱", "想你", "害羞", "调皮", "爱爱", "乖", "爱你",
)
SKIP_ANSWER = re.compile(r"^(嗯|哦|好|行|ok|OK|好哒|好的|收到|1|111)$", re.I)
MAX_MEMORY_CHARS = 200


def is_memory_worthy(text: str) -> bool:
    t = text.strip()
    if len(t) > MAX_MEMORY_CHARS:
        return False
    if "[你的姓名]" in t or "共计1500字" in t or "学号：" in t:
        return False
    return True


def is_junk(text: str) -> bool:
    t = text.strip()
    return not t or len(t) < 2 or bool(JUNK_TEXT.match(t))


def is_intimate_text(text: str) -> bool:
    return any(h in text for h in INTIMATE_HINTS)


def find_group_txt() -> Path | None:
    files = sorted(WECHAT_RAW_DIR.glob("群聊*.txt"))
    return files[0] if files else None


def parse_file(path: Path, partner: str, self_name: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            who, text = m.group(2), m.group(3).strip()
            if who not in (partner, self_name) or is_junk(text):
                continue
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
            except ValueError:
                ts = None
            rows.append({
                "ts": ts, "who": who, "text": text,
                "is_self": who == self_name,
                "intimate": is_intimate_text(text),
            })
    return rows


def parse_group_file(path: Path) -> list[dict]:
    """Parse multiline WeChat group export: header line then message body."""
    allowed = {GROUP_SELF_NICK, *GROUP_FRIENDS.keys()}
    rows: list[dict] = []
    who: str | None = None
    ts: datetime | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal who, ts, body
        if not who or who not in allowed:
            who, ts, body = None, None, []
            return
        text = "\n".join(body).strip()
        if is_junk(text):
            who, ts, body = None, None, []
            return
        rows.append({
            "ts": ts,
            "who": who,
            "text": text,
            "is_self": who == GROUP_SELF_NICK,
            "intimate": is_intimate_text(text),
        })
        who, ts, body = None, None, []

    with path.open(encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            m = GROUP_HEAD_RE.match(line.strip())
            if m:
                flush()
                nick = m.group(2).strip()
                if nick in GROUP_SKIP_NICKS:
                    who, ts, body = None, None, []
                    continue
                try:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    ts = None
                who = nick
                body = []
                continue
            if who and line.strip():
                body.append(line.strip())
        flush()
    return rows


def merge_blocks(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    blocks = [{"who": rows[0]["who"], "is_self": rows[0]["is_self"], "texts": [rows[0]["text"]],
               "ts": rows[0]["ts"], "intimate": rows[0].get("intimate", False)}]
    for r in rows[1:]:
        if r["who"] == blocks[-1]["who"]:
            blocks[-1]["texts"].append(r["text"])
            blocks[-1]["intimate"] = blocks[-1]["intimate"] or r.get("intimate", False)
        else:
            blocks.append({"who": r["who"], "is_self": r["is_self"], "texts": [r["text"]],
                           "ts": r["ts"], "intimate": r.get("intimate", False)})
    for b in blocks:
        b["text"] = " ".join(b["texts"])
    return blocks


def extract_qa_pairs(blocks: list[dict], *, max_answer_len: int = 120) -> list[dict]:
    pairs = []
    for i, b in enumerate(blocks):
        if not b["is_self"] or i == 0 or blocks[i - 1]["is_self"]:
            continue
        q, a = blocks[i - 1]["text"], b["text"]
        if is_junk(q) or is_junk(a) or SKIP_ANSWER.match(a.strip()):
            continue
        if not (4 <= len(a) <= max_answer_len):
            continue
        intimate = blocks[i - 1].get("intimate") or b.get("intimate")
        pairs.append({"q": q[:150], "a": a, "ts": b.get("ts"), "intimate": intimate})
    return pairs


def score_pair(p: dict, *, private: bool) -> float:
    a = p["a"]
    s = 2.0 if 8 <= len(a) <= 55 else (1.0 if len(a) <= 80 else -0.5)
    s += sum(1.5 for w in COLLOQUIAL if w in a)
    if private and p.get("intimate"):
        s += 4.0
    return s


def dedupe_pairs(pairs: list[dict], limit: int, *, private: bool) -> list[dict]:
    seen: set[str] = set()
    out = []
    for p in sorted(pairs, key=lambda x: score_pair(x, private=private), reverse=True):
        key = re.sub(r"\s+", "", p["a"])[:40]
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def write_style_draft(pairs: list[dict], path: Path, *, intimate: bool, limit: int, partner: str) -> None:
    pool = [p for p in pairs if bool(p.get("intimate")) == intimate]
    top = dedupe_pairs(pool, limit, private=intimate)
    target = "style/examples_intimate.md" if intimate else "style/examples.md"
    lines = [f"# {'亲密' if intimate else '日常'}口吻草稿 → 合并到 {target}", ""]
    for p in top:
        lines += [f"问：{p['q']}", f"答：{p['a']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_memory_corpus(
    rows: list[dict],
    path: Path,
    *,
    intimate_only: bool,
    partner: str,
    preamble: list[str] | None = None,
) -> None:
    """Write directly to corpus/ for L3 ingest (long-term memory)."""
    by_month: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if not r["is_self"]:
            continue
        if intimate_only != bool(r.get("intimate")):
            continue
        if len(r["text"]) >= 6 and is_memory_worthy(r["text"]):
            month = r["ts"].strftime("%Y-%m") if r.get("ts") else "未标时间"
            by_month[month].append(r["text"])
    title = "微信亲密记忆" if intimate_only else "微信日常记忆"
    lines = [f"# {title}（L3 长期记忆）", ""]
    if preamble:
        lines.extend(preamble)
        lines.append("")
    lines += [
        f"来源：{partner}，由 import_wechat.py 自动生成。",
        "可直接 ingest；建议改写成 2～5 句叙述段落后入库。",
        "",
    ]
    for month in sorted(by_month):
        lines += [f"## {month}", ""]
        for s in by_month[month][:25]:
            lines.append(f"- {s}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def import_group_chat(path: Path) -> None:
    rows = parse_group_file(path)
    if not rows:
        sys.exit("未解析到群消息，请确认 TXT 格式与昵称（一捧雪/计算机原理/唐凯/袁子翔）")
    blocks = merge_blocks(rows)
    pairs = extract_qa_pairs(blocks, max_answer_len=120)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    friend_desc = "、".join(
        f"{real}（{nick}）" for nick, real in GROUP_FRIENDS.items()
    )
    preamble = [
        "叶鹏祥在群里微信昵称「一捧雪」。",
        f"群友：{friend_desc}，均为初中同学、高中同校。",
        "群里说话比跟女友更随意，互怼、打游戏、聊学校生活为主。",
    ]
    write_style_draft(
        pairs,
        DRAFT_DIR / "style_group_friends.md",
        intimate=False,
        limit=100,
        partner="好友群",
    )
    write_memory_corpus(
        rows,
        MEMORY_GROUP,
        intimate_only=False,
        partner=f"好友群（{path.name}）",
        preamble=preamble,
    )
    by_who = defaultdict(int)
    for r in rows:
        by_who[r["who"]] += 1
    print(f"group file: {path.name.encode('unicode_escape').decode()}")
    print(f"  messages {len(rows)}, qa pairs {len(pairs)}")
    for nick in [GROUP_SELF_NICK, *GROUP_FRIENDS]:
        print(f"  {nick}: {by_who.get(nick, 0)}")
    print("口吻草稿:", DRAFT_DIR / "style_group_friends.md")
    print("长期记忆:", MEMORY_GROUP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--partner", default="打瓦女高手")
    ap.add_argument("--self-name", default="我")
    ap.add_argument("--private", action="store_true")
    ap.add_argument(
        "--group",
        action="store_true",
        help="导入 persona/import/raw/wechat/群聊*.txt（一捧雪=叶鹏祥）",
    )
    args = ap.parse_args()

    if args.group:
        path = args.input or find_group_txt()
        if not path or not path.is_file():
            sys.exit(f"找不到群聊 TXT，请放到 {WECHAT_RAW_DIR}/群聊*.txt")
        import_group_chat(path)
        print("\n已写入 corpus/，重启服务会自动入库")
        return

    path = args.input or DEFAULT_INPUT
    if not path.is_file():
        sys.exit(f"找不到: {path}")

    rows = parse_file(path, args.partner, args.self_name)
    pairs = extract_qa_pairs(merge_blocks(rows), max_answer_len=150 if args.private else 120)

    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    write_style_draft(pairs, DRAFT_DIR / "style_daily.md", intimate=False, limit=100, partner=args.partner)
    write_memory_corpus(rows, MEMORY_DAILY, intimate_only=False, partner=f"与「{args.partner}」的单聊")
    print("口吻草稿:", DRAFT_DIR / "style_daily.md")
    print("长期记忆:", MEMORY_DAILY)

    if args.private:
        write_style_draft(pairs, DRAFT_DIR / "style_intimate.md", intimate=True, limit=80, partner=args.partner)
        write_memory_corpus(rows, MEMORY_INTIMATE, intimate_only=True, partner=f"与「{args.partner}」的单聊")
        print("亲密口吻草稿:", DRAFT_DIR / "style_intimate.md")
        print("亲密长期记忆:", MEMORY_INTIMATE)

    print("\n记忆已写入 corpus/，执行: python scripts/ingest.py")


if __name__ == "__main__":
    main()
