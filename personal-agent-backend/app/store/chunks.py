"""文本分块与检索工具模块（离线文本处理 + 统一记忆库检索辅助）。

本模块已降级为纯工具模块，不再承担"运行时长期记忆底层存储"的职责。
所有运行时记忆读写通过 SessionStore 新语义接口完成（memory_items 统一表）。

保留的能力：
  - 文本分块 + FTS5 预处理（prepare_fts_text / build_fts_match_query）
  - 意图检测（_detect_intent_boosts）
  - RRF 排序算法（_rrf_hybrid_rerank / search_unified_memory）
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import settings
from app.llm import cosine_similarity, embed_texts
from app.session import store

logger = logging.getLogger(__name__)

# 统一记忆存储集合名。
_COLLECTION_MEMORY = "memory"


# ═════════════════════════════════════════════════════════════════════════════
# 纯文本算法（运行时仍使用，不涉及 DB）
# ═════════════════════════════════════════════════════════════════════════════


def prepare_fts_text(text: str) -> str:
    """中文全文检索预处理：bigram + unigram 混合分词。

    策略：
    - CJK 汉字（U+4E00~U+9FFF）：先收集连续汉字序列，生成相邻两字 bigram
    - unigram 兜底：单字查询或短语中孤立字仍能命中
    - 字母数字字符保留原样并转小写

    Args:
        text: 原始文本

    Returns:
        FTS5 友好的空格分隔 token 字符串
    """
    import re

    parts: list[str] = []
    cjk_buffer: list[str] = []

    def _flush_cjk() -> None:
        if not cjk_buffer:
            return
        if len(cjk_buffer) >= 2:
            for i in range(len(cjk_buffer) - 1):
                parts.append(cjk_buffer[i] + cjk_buffer[i + 1])
        parts.extend(cjk_buffer)
        cjk_buffer.clear()

    for ch in text:
        if "一" <= ch <= "鿿":
            cjk_buffer.append(ch)
        else:
            _flush_cjk()
            if ch.isalnum():
                parts.append(ch.lower())

    _flush_cjk()

    for word in re.findall(r"[a-zA-Z0-9]{2,}", text):
        parts.append(word.lower())

    return " ".join(parts)


def build_fts_match_query(query: str) -> str | None:
    """将用户自然语言查询转换为 FTS5 MATCH 语句的查询字符串。

    Args:
        query: 用户原始查询文本

    Returns:
        FTS5 enhanced query 字符串；预处理后为空则返回 None
    """
    q = prepare_fts_text(query)
    if not q.strip():
        return None
    terms = q.split()
    if not terms:
        return None
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " OR ".join(unique)


def _detect_intent_boosts(query: str) -> dict[str, float]:
    """从查询文本中检测用户意图，返回类型到加分的映射。

    基于关键词模式匹配，识别多种意图并给出对应的 category 加分。
    加在 RRF 最终分数上，使匹配类别的 item 排名提升。

    Args:
        query: 用户原始查询文本

    Returns:
        {category: boost} 映射，如 {"person": 0.15}
    """
    boosts: dict[str, float] = {}
    q = query.strip()
    if not q:
        return boosts

    import re

    if any(w in q for w in ("是谁", "谁是", "哪位", "什么人", "哪一位")):
        boosts["person"] = max(boosts.get("person", 0), 0.20)
    if any(w in q for w in ("称呼", "叫什么", "名字")):
        boosts["person"] = max(boosts.get("person", 0), 0.10)
    if any(w in q for w in ("喜欢", "爱好", "口味", "吃什么", "喜欢什么")):
        boosts["preference"] = max(boosts.get("preference", 0), 0.15)
    if any(w in q for w in ("不喜欢", "讨厌", "讨厌吃", "不吃", "不爱")):
        boosts["preference"] = max(boosts.get("preference", 0), 0.20)
    if any(w in q for w in ("禁忌", "忌讳", "称呼不能用", "禁止称呼")):
        boosts["taboo"] = max(boosts.get("taboo", 0), 0.25)
    if any(w in q for w in ("不能叫", "不要叫", "不能称呼")):
        boosts["taboo"] = max(boosts.get("taboo", 0), 0.20)
    if "称呼" in q and any(w in q for w in ("不能", "禁止", "不可", "不要")):
        boosts["taboo"] = max(boosts.get("taboo", 0), 0.25)
    if any(w in q for w in ("待跟进", "待办", "还有什", "该做", "要做什么")):
        boosts["open_loop"] = max(boosts.get("open_loop", 0), 0.12)
    if re.search(r"\d{4}\s*[年/-]\s*\d{1,2}\s*月?|\d{4}年", q):
        boosts["monthly"] = max(boosts.get("monthly", 0), 0.10)

    return boosts


# ═════════════════════════════════════════════════════════════════════════════
# RRF 排序算法（基于 memory_items 搜索结果的 rerank 辅助）
# ═════════════════════════════════════════════════════════════════════════════


def _rrf_hybrid_rerank(
    *,
    query: str,
    q_emb: list[float],
    kw_ranked: list[tuple[str, dict[str, Any]]],
    vec_ranked: list[tuple[str, dict[str, Any]]],
    top_k: int,
    category_boosts: dict[str, float] | None = None,
) -> list[dict]:
    """RRF（倒数排名融合）算法合并关键词和向量两路召回结果。

    Args:
        query: 原始查询文本
        q_emb: 查询文本的 embedding 向量
        kw_ranked: 关键词 FTS 召回的排名结果 [(item_id, row_dict), ...]
        vec_ranked: 向量相似度召回的排名结果 [(item_id, row_dict), ...]
        top_k: 最终返回的最大命中数
        category_boosts: 类别加分映射 {category: boost_value}

    Returns:
        排序后的命中字典列表
    """
    rrf_scores: dict[str, float] = {}
    payload: dict[str, dict[str, Any]] = {}

    for hits in (kw_ranked, vec_ranked):
        for rank, (cid, row) in enumerate(hits, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))
            payload[cid] = row

    if not rrf_scores:
        return []

    max_rrf = max(rrf_scores.values())
    rrf_norm_map = {cid: s / max_rrf for cid, s in rrf_scores.items()}

    candidates = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    rerank_n = max(top_k, settings.es_rerank_top_n)
    min_score = settings.es_min_recall_score
    out: list[dict] = []

    for cid, _rrf in candidates[:rerank_n]:
        row = payload.get(cid, {})
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        emb = row.get("embedding") or []
        sim = 0.0
        if emb:
            try:
                sim = max(0.0, cosine_similarity(q_emb, emb))
            except Exception:
                sim = 0.0
        rrf_norm = rrf_norm_map.get(cid, 0.0)
        final_score = (0.72 * sim) + (0.28 * rrf_norm)
        category = str(row.get("category") or "")
        cat_boost = (category_boosts or {}).get(category, 0.0)
        final_score += cat_boost
        if final_score >= min_score:
            out.append({
                "text": text,
                "score": round(final_score, 3),
                "chunk_id": cid,
                "category": category,
                "source": str(row.get("source") or ""),
                "collection": str(row.get("collection") or _COLLECTION_MEMORY),
            })

    out.sort(key=lambda x: x.get("score") or 0, reverse=True)
    return out[:top_k]


# ═════════════════════════════════════════════════════════════════════════════
# 运行时 hybrid 检索（基于 memory_items 统一记忆库）
# ═════════════════════════════════════════════════════════════════════════════


def search_unified_memory(
    *,
    person_id: str,
    query: str,
    q_emb: list[float],
    top_k: int,
    device_id: str = "",
    persona_person_id: str = "",
) -> list[dict]:
    """统一长期记忆检索入口（基于 memory_items 统一记忆库）。

    基于 memory_items 表的 FTS 检索引擎。
    检索范围包括：
    - 核心事实（visibility=always）
    - 近期记忆（episode/emotion/milestone）
    - 长期记忆（fact/entity/wiki/relationship）

    Args:
        person_id: 当前对话对象的 person_id
        query: 用户自然语言查询
        q_emb: 查询文本的 embedding 向量
        top_k: 最终返回的最大 item 数
        device_id: 保留参数
        persona_person_id: 可选，跨用户补召的 person_id

    Returns:
        排序后的记忆项字典列表。
    """
    pid = str(person_id or "").strip()
    if not pid:
        return []
    ppid = str(persona_person_id or "").strip()
    extras = [ppid] if ppid and ppid != pid else None

    # 用统一记忆库检索（含 FTS + embedding rerank）
    kat = json.dumps(q_emb) if q_emb else ""
    items = store.search_memory_items(
        pid,
        kinds=None,  # 全部 kind
        query=query,
        limit=top_k * 2,
        embedding_json=kat,
        extra_person_ids=extras,
    )

    if not items:
        return []

    # 归一化为长期记忆搜索结果。
    result: list[dict] = []
    for item in items[:top_k]:
        try:
            ctx = json.loads(item.get("context_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            ctx = {}
        emb = item.get("embedding_json", "[]")
        try:
            emb_parsed = json.loads(emb) if isinstance(emb, str) else emb
        except (json.JSONDecodeError, TypeError):
            emb_parsed = []
        result.append({
            "text": str(item.get("content", "")),
            "score": 0.0,  # 调用方负责排序
            "memory_item_id": str(item.get("id", "")),
            "category": str(item.get("kind", "")),
            "source": str(item.get("source", "")),
            "collection": _COLLECTION_MEMORY,
            "embedding": emb_parsed,
            "created_at": item.get("created_at", ""),
        })

    return result


# ═════════════════════════════════════════════════════════════════════════════
def ingest_chunks(
    chunks: list[dict],
    *,
    collection: str = _COLLECTION_MEMORY,
    device_id: str = "",
    person_id: str = "",
    reset: bool = False,
) -> int:
    """批量写入记忆内容到统一记忆库。

    Args:
        chunks: 块字典列表，每项需含 id、text、meta
        collection: 保留参数
        device_id: 关联的设备 ID
        person_id: 关联的人物 ID
        reset: 是否先清空旧数据再写入

    meta 支持的字段（影响 memory_items 写入语义）：
        kind, source, source_table, source_id, visibility, confidence,
        category, device_id, person_id, source_session, month_key, tags
    """
    if not chunks:
        return 0

    pid = str(person_id or "").strip()
    did = str(device_id or "").strip()

    if reset and pid:
        existing = store.list_memory_items(pid, limit=9999)
        for item in existing:
            store.delete_memory_item(str(item.get("id", "")))

    docs = [c["text"] for c in chunks]
    embs = embed_texts(docs)
    if not embs:
        return 0

    written = 0
    for i, chunk in enumerate(chunks):
        meta = chunk.get("meta", {})
        kind = str(meta.get("kind", "wiki"))
        source = str(meta.get("source", "ingest"))
        source_table = str(meta.get("source_table", ""))
        source_id = str(meta.get("source_id", ""))
        visibility = str(meta.get("visibility", "recall_only"))
        confidence = float(meta.get("confidence", 0.8))

        context = {}
        for k in ("category", "device_id", "person_id", "source_session", "month_key", "source_path"):
            v = meta.get(k)
            if v is not None and str(v).strip():
                context[k] = str(v).strip()

        store.write_memory_item(
            person_id=pid,
            device_id=did,
            kind=kind,
            source=source,
            source_table=source_table,
            source_id=source_id,
            visibility=visibility,
            content=docs[i],
            confidence=confidence,
            embedding_json=json.dumps(embs[i]) if embs[i] else "[]",
            context_json=json.dumps(context, ensure_ascii=False),
            tags_json=json.dumps(meta.get("tags", []), ensure_ascii=False),
        )
        written += 1

    return written


def count_unified_memory() -> int:
    """统计 persona/corpus 导入的长期语料块数量。"""
    return int(store.count_corpus_memory_items())


def count_corpus() -> int:
    """统计 persona/corpus 导入的长期语料块数量。"""
    return int(store.count_corpus_memory_items())


def reset_corpus() -> None:
    """清空全部 corpus 记忆行及对应 FTS（仅清理 source_table='corpus' 的条目）。"""
    store.reset_corpus_items()


def reset_all() -> None:
    """清空全部 memory_items 记忆（谨慎使用！仅测试/管理用途）。"""
    store.reset_corpus_items()  # 当前仅清理 corpus 条目
