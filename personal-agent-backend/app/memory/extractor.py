"""L1→L2 compression, L2→L3 rollup, optional fact extraction."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

from app.config import settings
from app.llm import chat_completion
from app.memory.episodic import episodic_memory
from app.memory.semantic import semantic_memory
from app.session import store

logger = logging.getLogger(__name__)


def _format_transcript(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def compress_l1_to_l2(device_id: str, session_id: str) -> bool:
    """When L1 >= working_memory_turns, compress oldest batch into L2 only."""
    turns = store.count_turns(session_id)
    if turns < settings.working_memory_turns:
        return False

    batch = settings.l1_compress_batch_turns
    msg_limit = batch * 2
    oldest = store.get_oldest_messages(session_id, msg_limit)
    if len(oldest) < 4:
        return False

    transcript = _format_transcript(oldest)
    prompt = f"""将以下对话压缩为结构化摘要（中文 JSON），用于短期记忆，100～200字。

字段：
- summary: 一段话摘要（时间、决定、情绪、关键事实）
- topics: 逗号分隔主题词
- open_loops: 未说完/待跟进事项数组，无则 []

对话：
{transcript}

只输出 JSON：{{"summary":"...","topics":"...","open_loops":[]}}"""
    raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return False
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return False

    summary = str(data.get("summary", "")).strip()
    if not summary:
        return False

    topics = str(data.get("topics", ""))
    loops = data.get("open_loops", [])
    open_loops = json.dumps(loops, ensure_ascii=False) if loops else ""

    episodic_memory.save_summary(device_id, session_id, summary, topics, open_loops)
    store.delete_messages_by_ids([m["id"] for m in oldest])
    logger.debug("L1→L2 compressed %d messages for session=%s", len(oldest), session_id)
    return True


def consolidate_session(device_id: str, session_id: str) -> None:
    """Session end: compress any remaining L1 into L2."""
    messages = store.get_session_messages(session_id)
    if len(messages) < 2:
        store.close_session(session_id)
        return

    transcript = _format_transcript(messages[-40:])
    prompt = f"""将以下对话压缩为结构化摘要（中文 JSON）。

字段：summary（100～200字）、topics（逗号分隔）、open_loops（数组，无则[]）

{transcript}

只输出 JSON。"""
    raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            summary = str(data.get("summary", "")).strip()
            if summary:
                topics = str(data.get("topics", ""))
                loops = data.get("open_loops", [])
                open_loops = json.dumps(loops, ensure_ascii=False) if loops else ""
                episodic_memory.save_summary(device_id, session_id, summary, topics, open_loops)
        except json.JSONDecodeError:
            pass

    store.close_session(session_id)


def rollup_expired_l2(device_id: str | None = None) -> int:
    """Expired L2 → L3: narrative chunks (corpus) + facts. Returns count rolled up."""
    rows = store.list_expired_episodic(device_id)
    if not rows:
        return 0

    by_device: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_device[row["device_id"]].append(row)

    rolled = 0
    for dev_id, items in by_device.items():
        try:
            _rollup_device_batch(dev_id, items)
            store.archive_episodic([int(r["id"]) for r in items])
            rolled += len(items)
        except Exception as exc:
            logger.warning("L2→L3 rollup failed device=%s: %s", dev_id, exc)
    return rolled


def _rollup_device_batch(device_id: str, items: list[dict]) -> None:
    block = "\n\n".join(
        f"[{r.get('created_at', '')}] {r['summary']}"
        + (f" 主题:{r['topics']}" if r.get("topics") else "")
        for r in items
    )
    prompt = f"""以下是一批已过期（7天+）的短期会话摘要，请汇总为可长期检索的记忆。

输出 JSON：
{{
  "narrative": "200～400字叙述性段落，保留时间线、人物、决定、情绪",
  "facts": [
    {{"fact":"短句事实","category":"preference|event|person|general","confidence":0.0-1.0}}
  ]
}}

规则：
- facts 只写明确、可验证的信息，无则 []
- 不要编造摘要里没有的内容

摘要列表：
{block}"""

    raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return

    narrative = str(data.get("narrative", "")).strip()
    if narrative:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        chunk_id = f"l2-rollup-{device_id}-{ts}-{abs(hash(narrative)) % 10**8}"
        semantic_memory.ingest_chunks(
            [
                {
                    "id": chunk_id,
                    "text": narrative,
                    "meta": {"source": "l2_rollup", "device_id": device_id},
                }
            ],
        )

    for item in data.get("facts") or []:
        if not isinstance(item, dict) or not item.get("fact"):
            continue
        conf = float(item.get("confidence", 0.75))
        if conf < 0.7:
            continue
        semantic_memory.add_fact(
            device_id,
            str(item["fact"]),
            str(item.get("category", "general")),
            conf,
            "l2_rollup",
        )


def extract_facts(device_id: str, session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Disabled by default; only runs when auto_extract_facts=True."""
    if not settings.auto_extract_facts:
        return
    if not re.search(r"记住|别忘了|以后要|帮我记", user_msg):
        return

    prompt = f"""用户是否明确要求记住某事实？若是，返回 JSON 数组，否则 []。
每项：{{"fact":"...","category":"preference|event|person|general","confidence":0.0-1.0}}
confidence 仅当用户明确陈述时 >= 0.85

用户：{user_msg}
助手：{assistant_msg}"""
    raw = chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return
    for item in items:
        if isinstance(item, dict) and item.get("fact"):
            conf = float(item.get("confidence", 0.0))
            if conf >= 0.85:
                semantic_memory.add_fact(
                    device_id,
                    str(item["fact"]),
                    str(item.get("category", "general")),
                    conf,
                    session_id,
                )


def maybe_compress_l1(device_id: str, session_id: str) -> bool:
    did = False
    while store.count_turns(session_id) >= settings.working_memory_turns:
        if not compress_l1_to_l2(device_id, session_id):
            break
        did = True
    return did
