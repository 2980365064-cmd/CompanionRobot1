"""
Long-Term Memory — 长期记忆（混合检索 + 入库），自包含模块。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Long-Term Memory 是"长期记忆层"——存储所有已入库的文本语料块（chunks），
  通过混合检索（关键词 + 向量相似度）支持跨会话的长期回忆。

数据来源（多条路径汇聚）：
  1. 近期归档：过期（>14天）的近期摘要 → format_recent_to_long_term_block 转叙述块 → 入库
  2. 用户"记住"意图：用户说「记住这个」→ write_explicit_memory_request
  3. 纠错语料：用户纠正记忆 → capture_user_stated_facts → 入库
  4. 语料导入：外部脚本/定时任务批量导入

检索方式（统一 hybrid search）：
  - 通过 store.chunks 做 SQLite 全文搜索 + 向量相似度混合排序
  - 结果经噪声过滤后返回

存储结构：chunks 表，每条记录含 chunk_id、collection（统一用 "memory"）、
text、embedding 向量、device_id、person_id、source、category 等元数据。
============================================================================
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.config import settings
from app.embed_meta import load_embed_meta, save_embed_meta
from app.llm import embed_texts
from app.session import store
from app.store.chunks import count_unified_memory, search_unified_memory, ingest_chunks, reset_all, reset_corpus

logger = logging.getLogger(__name__)


# ── 结果过滤：去重 + 噪声过滤 ──────────────────────────────────────────────

def _scored_from_dicts(results: list[dict], top_k: int) -> list[dict]:
    """对检索结果做去重和噪声过滤，返回 top_k 条有效结果。"""
    from app.memory.guard import is_noise_memory

    seen: set[str] = set()
    out: list[dict] = []
    for row in results:
        text = str(row.get("text", "")).strip()
        if not text or text in seen or is_noise_memory(text):
            continue
        seen.add(text)
        out.append(row)
        if len(out) >= top_k:
            break
    return out


def _stored_dim() -> int | None:
    """从 embed_meta 读取已存储的向量维度。"""
    meta = load_embed_meta()
    if meta:
        return int(meta.get("dim", 0)) or None
    return None


class LongTermMemory:
    """长期记忆：封装语料块的写入（store）和混合检索（hybrid search）。

    所有写入统一使用 collection="memory"。
    读取时通过 person_id 和 device_id 隔离不同用户的数据。
    """

    def __init__(self) -> None:
        self._vector_dim: int | None = _stored_dim()

    def count_chunks(self) -> int:
        """返回长期记忆中的 chunk 总数。"""
        return int(count_unified_memory())

    def count_corpus_chunks(self) -> int:
        """返回 persona/corpus 导入的长期语料块总数。"""
        return self.count_chunks()

    def reset_all(self) -> None:
        """清空全部长期记忆数据（包括 corpus 和索引）。"""
        reset_all()

    def reset_corpus(self) -> None:
        """仅清空长期记忆 corpus 数据。"""
        reset_corpus()

    def store_long_term_chunks(
        self,
        chunks: list[dict],
        *,
        device_id: str = "",
        person_id: str = "",
        reset: bool = False,
    ) -> int:
        """按用户写入语料块到长期记忆。

        Args:
            chunks:    语料块列表
            device_id: 设备标识
            person_id: 用户 ID
            reset:     是否在写入前先清空

        Returns:
            成功写入的 chunk 数量。
        """
        if not chunks:
            return 0
        n = ingest_chunks(
            chunks,
            collection="memory",
            device_id=device_id,
            person_id=person_id,
            reset=reset,
        )
        if n:
            meta = save_embed_meta()
            self._vector_dim = int(meta.get("dim", 0)) or None
        return n

    def search_long_term(
        self,
        device_id: str,
        person_id: str,
        query: str,
        top_k: int,
        *,
        q_emb: list[float] | None = None,
        persona_person_id: str = "",
    ) -> list[dict]:
        """按 query 对长期记忆做混合检索（关键词 + 向量相似度），返回评分排序结果。

        Args:
            device_id:         设备标识
            person_id:         用户 ID
            query:             检索查询文本
            top_k:             返回数量上限
            q_emb:             可选的预计算查询向量
            persona_person_id: 人物画像事实的 person_id（跨用户共享的人物知识）

        Returns:
            评分后的记忆块列表，每条包含 text、score 等字段。
        """
        if not query.strip() or top_k <= 0:
            return []
        if q_emb is None:
            q_emb = embed_texts([query])[0]

        results = search_unified_memory(
            person_id=person_id,
            query=query,
            q_emb=q_emb,
            top_k=top_k,
            device_id=device_id,
            persona_person_id=persona_person_id,
        )
        return _scored_from_dicts(results, top_k)

    def search_corpus(self, device_id, person_id, query, top_k, *, q_emb=None):
        """检索语料块（内部调用 search_long_term）。

        会自动附加 persona_fact_person_id 以检索共享的人物知识。
        """
        ppid = str(getattr(settings, "persona_fact_person_id", "") or "").strip()
        return self.search_long_term(
            device_id, person_id, query, top_k, q_emb=q_emb, persona_person_id=ppid
        )

    def search_facts(self, device_id, person_id, query, top_k, *, q_emb=None):
        """检索事实（等同于 search_corpus）。"""
        return self.search_corpus(device_id, person_id, query, top_k, q_emb=q_emb)


# 模块级单例
long_term_memory = LongTermMemory()


# ══════════════════════════════════════════════════════════════════════════════
# 写入辅助函数 —— 将文本转换为 chunk 并写入长期记忆
# ══════════════════════════════════════════════════════════════════════════════

def store_long_term_chunk(
    device_id: str,
    person_id: str,
    text: str,
    *,
    source: str,
    chunk_id: str | None = None,
    category: str = "",
) -> str:
    """将一段文本包装为 chunk 并写入长期记忆。

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        text:      文本内容（至少 8 字符）
        source:    来源标签（如 "user_remember_intent", "recent_archived" 等）
        chunk_id:  可选的自定义 chunk ID
        category:  类别标签

    Returns:
        成功写入的 chunk_id；文本过短时返回空字符串。
    """
    body = str(text or "").strip()
    if len(body) < 8:
        return ""
    pid = str(person_id or "").strip()
    cid = chunk_id or f"mem-{pid or device_id}-{uuid4().hex[:12]}"
    long_term_memory.store_long_term_chunks(
        [
            {
                "id": cid,
                "text": body,
                "meta": {
                    "source": source,
                    "device_id": device_id,
                    "person_id": pid,
                    "category": category,
                },
            }
        ],
        device_id=device_id,
        person_id=pid,
    )
    return cid


def store_long_term_text(
    device_id: str,
    person_id: str,
    text: str,
    *,
    source: str,
    source_session: str = "",
    chunk_id: str | None = None,
    category: str = "",
) -> str:
    """将单段文本写入长期记忆（source_session 参数保留但不影响写入逻辑）。"""
    del source_session
    return store_long_term_chunk(
        device_id, person_id, text, source=source, chunk_id=chunk_id, category=category
    )


def store_corpus_chunk(
    device_id: str, person_id: str, text: str, *, source: str, chunk_id: str | None = None
) -> str:
    """写入语料块到长期记忆。"""
    return store_long_term_chunk(device_id, person_id, text, source=source, chunk_id=chunk_id)


# ══════════════════════════════════════════════════════════════════════════════
# 近期→长期归档：过期近期摘要转换为长期语料块
# ══════════════════════════════════════════════════════════════════════════════

def format_recent_to_long_term_block(row: dict) -> str:
    """将单条近期记忆记录格式化为长期语料叙述块。

    格式：[近期摘要 时间戳] 摘要文本\n主题: ...\n待办: ...

    Args:
        row: 近期记忆记录（含 summary, topics, open_loops, created_at）

    Returns:
        适合存入长期记忆的结构化叙述文本块。
    """
    ts = str(row.get("created_at") or "")
    summary = str(row.get("summary") or "").strip()
    topics = str(row.get("topics") or "").strip()
    loops = str(row.get("open_loops") or "").strip()
    parts = [f"[近期摘要 {ts}] {summary}" if ts else summary]
    if topics:
        parts.append(f"主题: {topics}")
    if loops and loops not in ("[]", "null", ""):
        parts.append(f"待办: {loops}")
    return "\n".join(parts)


def archive_recent_to_long_term(
    device_id: str, person_id: str, rows: list[dict]
) -> int:
    """将一批过期近期记忆记录转换为长期语料块并写入。

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        rows:      待归档的近期记忆记录列表

    Returns:
        成功归档的条数。
    """
    pid = str(person_id or "").strip()
    if not pid or not rows:
        return 0
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    count = 0
    for row in rows:
        block = format_recent_to_long_term_block(row)
        if len(block.strip()) < 12:
            continue
        rid = row.get("id", "")
        cid = f"archived-{pid}-{rid}-{ts}"
        store_long_term_text(
            device_id, pid, block, source="recent_archived",
            source_session=f"recent_archive:{rid}", chunk_id=cid, category="archive",
        )
        count += 1
    return count


def clear_derived_memory() -> dict[str, int]:
    """清理人物衍生记忆（画像衍生数据）。

    当人物画像更新后，清除旧的人物知识衍生数据，
    以便下次从头重新构建。

    Returns:
        各类型清除数量统计：{"facts": N, "relations": N, "long_term_cleared": N}
    """
    from app.config import settings

    pid = str(getattr(settings, "persona_fact_person_id", "persona_global") or "").strip()
    if not pid:
        return {"facts": 0, "relations": 0, "long_term_cleared": 0}
    stats = store.purge_persona_derived_memory(pid)
    stats.setdefault("facts", stats.pop("memory_items", 0))
    stats.setdefault("long_term_cleared", 0)
    if any(stats.values()):
        logger.info("cleared derived memory person=%s stats=%s", pid, stats)
    return stats
