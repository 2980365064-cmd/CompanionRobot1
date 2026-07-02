"""
L3 语义记忆（Semantic Memory）—— 长期向量检索，纯 SQLite 后端。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  L3 是"长期记忆层"——存储所有已入库的文本语料块（chunks），
  通过混合检索（关键词 + 向量相似度）支持跨会话的长期回忆。

数据来源（多条路径汇聚到 L3）：
  1. L2 rollup：过期（>7天）的 L2 摘要 → format_l2_corpus_block 转叙述块 → ingest
  2. 用户"记住"意图：用户说「记住这个」→ ingest_remember_to_l3
  3. 纠错语料：用户纠正记忆 → capture_user_stated_facts → ingest
  4. 语料导入：外部脚本/定时任务批量导入

检索方式（统一 hybrid search）：
  - 通过 store.hybrid_search_l3 做 SQLite 全文搜索 + 向量相似度混合排序
  - 结果经噪声过滤（is_noise_memory_for_l3）后返回

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
from app.store.chunks import count_l3, hybrid_search_l3, ingest_chunks, reset_all, reset_corpus

logger = logging.getLogger(__name__)


# ── 结果过滤：去重 + 噪声过滤 ──────────────────────────────────────────────

def _scored_from_dicts(results: list[dict], top_k: int) -> list[dict]:
    """对 L3 检索结果做去重和噪声过滤，返回 top_k 条有效结果。

    每条结果检查：
      1. text 非空且非重复（同一文本只保留一条）
      2. 通过 is_noise_memory_for_l3 噪声过滤（排除纯自称套话等）
    """
    from app.memory.guard import is_noise_memory_for_l3

    seen: set[str] = set()
    out: list[dict] = []
    for row in results:
        text = str(row.get("text", "")).strip()
        if not text or text in seen or is_noise_memory_for_l3(text):
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


class SemanticMemory:
    """L3 语义记忆：封装语料块的写入（ingest）和混合检索（hybrid search）。

    所有写入统一使用 collection="memory"。
    读取时通过 person_id 和 device_id 隔离不同用户的数据。
    """

    def __init__(self) -> None:
        self._vector_dim: int | None = _stored_dim()

    @property
    def corpus(self):
        """兼容旧的 corpus 接口：返回语料统计对象。"""
        class _CC:
            vector_dim = self._vector_dim

            def count(_self) -> int:
                return int(count_l3())

        return _CC()

    @property
    def facts(self):
        """兼容旧的 facts 接口（当前版本 facts 已合并入 memory collection）。"""
        class _FC:
            def count(_self) -> int:
                return 0

        return _FC()

    def reset_all(self) -> None:
        """清空全部 L3 数据（包括 corpus 和索引）。"""
        reset_all()

    def reset_corpus(self) -> None:
        """仅清空 L3 corpus 数据。"""
        reset_corpus()

    def ingest_chunks(self, chunks: list[dict], *, reset: bool = False) -> int:
        """批量写入语料块（兼容旧接口，映射到 ingest_person_corpus）。"""
        return self.ingest_person_corpus(chunks, device_id="", person_id="", reset=reset)

    def ingest_person_corpus(
        self,
        chunks: list[dict],
        *,
        device_id: str = "",
        person_id: str = "",
        reset: bool = False,
    ) -> int:
        """按用户写入语料块到 L3。

        每个 chunk 字典应包含：id（唯一标识）、text（文本内容）、
        meta（元数据：source/device_id/person_id/category 等）。

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
            # 写入成功后刷新向量维度缓存
            meta = save_embed_meta()
            self._vector_dim = int(meta.get("dim", 0)) or None
        return n

    def recall_l3_scored(
        self,
        device_id: str,
        person_id: str,
        query: str,
        top_k: int,
        *,
        q_emb: list[float] | None = None,
        persona_person_id: str = "",
    ) -> list[dict]:
        """按 query 对 L3 做混合检索（关键词 + 向量相似度），返回评分排序结果。

        这是 L3 的主要检索入口，供 router 调用。

        Args:
            device_id:         设备标识
            person_id:         用户 ID
            query:             检索查询文本
            top_k:             返回数量上限
            q_emb:             可选的预计算查询向量
            persona_person_id: 人物画像事实的 person_id（跨用户共享的人物知识）

        Returns:
            评分后的 L3 记忆块列表，每条包含 text、score 等字段。
        """
        if not query.strip() or top_k <= 0:
            return []
        if q_emb is None:
            q_emb = embed_texts([query])[0]

        results = hybrid_search_l3(
            person_id=person_id,
            query=query,
            q_emb=q_emb,
            top_k=top_k,
            device_id=device_id,
            persona_person_id=persona_person_id,
        )
        return _scored_from_dicts(results, top_k)

    def recall_corpus_scored(self, device_id, person_id, query, top_k, *, q_emb=None):
        """兼容旧接口：召回语料块（内部调用 recall_l3_scored）。

        会自动附加 persona_fact_person_id 以检索共享的人物知识。
        """
        ppid = str(getattr(settings, "persona_fact_person_id", "") or "").strip()
        return self.recall_l3_scored(
            device_id, person_id, query, top_k, q_emb=q_emb, persona_person_id=ppid
        )

    def recall_facts_scored(self, device_id, person_id, query, top_k, *, q_emb=None):
        """兼容旧接口：召回事实（等同于 recall_corpus_scored）。"""
        return self.recall_corpus_scored(device_id, person_id, query, top_k, q_emb=q_emb)

    def add_fact(self, *args, **kwargs) -> int | None:
        """兼容旧接口：当前版本事实合并到 memory collection，不再单独存储。"""
        return None


# 模块级单例
semantic_memory = SemanticMemory()


# ══════════════════════════════════════════════════════════════════════════════
# L3 写入辅助函数 —— 将文本转换为 chunk 并写入 L3 存储
# ══════════════════════════════════════════════════════════════════════════════

def store_l3_chunk(
    device_id: str,
    person_id: str,
    text: str,
    *,
    source: str,
    chunk_id: str | None = None,
    category: str = "",
) -> str:
    """将一段文本包装为 chunk 并写入 L3 存储。

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        text:      文本内容（至少 8 字符）
        source:    来源标签（如 "user_remember_intent", "l2_expired" 等）
        chunk_id:  可选的自定义 chunk ID（不提供则自动生成 mem-{person_id}-{uuid}）
        category:  类别标签

    Returns:
        成功写入的 chunk_id；文本过短时返回空字符串。
    """
    body = str(text or "").strip()
    if len(body) < 8:
        return ""
    pid = str(person_id or "").strip()
    cid = chunk_id or f"mem-{pid or device_id}-{uuid4().hex[:12]}"
    semantic_memory.ingest_person_corpus(
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


def ingest_l3_text(
    device_id: str,
    person_id: str,
    text: str,
    *,
    source: str,
    source_session: str = "",
    chunk_id: str | None = None,
    category: str = "",
) -> str:
    """将单段文本写入 L3（source_session 参数保留但不影响写入逻辑）。

    这是各模块写入 L3 的通用入口（extractor/guard/correction 等均调用此函数）。
    """
    del source_session  # 预留参数，当前未直接使用
    return store_l3_chunk(
        device_id, person_id, text, source=source, chunk_id=chunk_id, category=category
    )


def ingest_corpus_with_facts(
    device_id: str,
    person_id: str,
    text: str,
    *,
    source: str,
    source_session: str = "",
    chunk_id: str | None = None,
    extract: bool = True,
    min_confidence: float = 0.75,
    category: str = "",
) -> tuple[str, list[str]]:
    """写入语料至 L3（兼容旧接口，当前版本不从中提取 facts）。

    Returns:
        (chunk_id, []) —— 第二个元素始终为空列表（事实提取已移除）。
    """
    del extract, min_confidence  # 旧参数，当前版本不使用
    cid = ingest_l3_text(
        device_id, person_id, text, source=source, source_session=source_session,
        chunk_id=chunk_id, category=category,
    )
    return cid, []


def store_corpus_chunk(
    device_id: str, person_id: str, text: str, *, source: str, chunk_id: str | None = None
) -> str:
    """写入语料块到 L3（store_l3_chunk 的别名）。"""
    return store_l3_chunk(device_id, person_id, text, source=source, chunk_id=chunk_id)


def extract_facts_from_corpus(*args, **kwargs) -> list[str]:
    """兼容旧接口：当前版本事实提取已移除，始终返回空列表。"""
    return []


def batch_extract_facts_from_persona_chunks(chunks: list[dict], **kwargs) -> dict[str, int]:
    """兼容旧接口：批量从 persona chunks 提取事实（当前版本为 no-op）。"""
    return {"chunks": 0, "facts": 0, "skipped": 0}


# ══════════════════════════════════════════════════════════════════════════════
# L2 rollup → L3：过期 L2 摘要转换为 L3 语料块
# ══════════════════════════════════════════════════════════════════════════════

def format_l2_corpus_block(row: dict) -> str:
    """将单条 L2 记录格式化为 L3 语料叙述块。

    格式：[L2 时间戳] 摘要文本\n主题: ...\n待办: ...

    Args:
        row: L2 episodic_memories 记录（含 summary, topics, open_loops, created_at）

    Returns:
        适合存入 L3 的结构化叙述文本块。
    """
    ts = str(row.get("created_at") or "")
    summary = str(row.get("summary") or "").strip()
    topics = str(row.get("topics") or "").strip()
    loops = str(row.get("open_loops") or "").strip()
    parts = [f"[L2 {ts}] {summary}" if ts else summary]
    if topics:
        parts.append(f"主题: {topics}")
    if loops and loops not in ("[]", "null", ""):
        parts.append(f"待办: {loops}")
    return "\n".join(parts)


def rollup_l2_rows_to_corpus(
    device_id: str, person_id: str, rows: list[dict]
) -> int:
    """将一批过期 L2 记录转换为 L3 语料块并写入。

    由 extractor.rollup_expired_l2 调用，在 L2 摘要超过 7 天有效期后，
    将其转存为永久 L3 语料。

    Args:
        device_id: 设备标识
        person_id: 用户 ID
        rows:      待归档的 L2 episodic 记录列表

    Returns:
        成功归档的条数。
    """
    pid = str(person_id or "").strip()
    if not pid or not rows:
        return 0
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    count = 0
    for row in rows:
        block = format_l2_corpus_block(row)
        if len(block.strip()) < 12:
            continue
        rid = row.get("id", "")
        # chunk_id 格式：l2-{person_id}-{row_id}-{日期}
        cid = f"l2-{pid}-{rid}-{ts}"
        ingest_l3_text(
            device_id, pid, block, source="l2_expired",
            source_session=f"l2_rollup:{rid}", chunk_id=cid, category="rollup",
        )
        count += 1
    return count


def clear_persona_derived_memory() -> dict[str, int]:
    """清理 persona 衍生记忆（人物画像事实）。

    当人物画像更新后，清除旧的人物知识衍生数据，
    以便下次从头重新构建。

    Returns:
        各类型清除数量统计：{"facts": N, "relations": N, "l3_facts": N}
    """
    from app.config import settings

    pid = str(getattr(settings, "persona_fact_person_id", "persona_global") or "").strip()
    if not pid:
        return {"facts": 0, "relations": 0, "l3_facts": 0}
    stats = store.purge_persona_derived_memory(pid, chunk_key_prefix="chunk:doc-")
    if any(stats.values()):
        logger.info("cleared persona legacy facts person=%s stats=%s", pid, stats)
    return stats
