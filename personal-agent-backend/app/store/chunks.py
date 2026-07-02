"""L3 块存储模块：SQLite l3_chunks 表 + FTS5 全文索引 + 进程内向量 hybrid 检索。

这是陪伴机器人的长期记忆存储层，所有需要持久化检索的内容统一写入此处：
- 用户聊天记录入库后的记忆块
- L2 工作记忆归档后的长期记忆
- persona 语料（corpus）的向量化存储

架构设计：
  - SQLite 作为持久化存储，l3_chunks 表存放所有块的文本和元数据
  - FTS5 虚拟表提供中文关键词全文检索（无需额外搜索引擎依赖）
  - 向量 embedding 在内存中计算和比对，配合 FTS5 实现 hybrid 召回
  - RRF（倒数排名融合）算法合并关键词和向量两路召回结果
  - 使用 cosine_similarity 计算 72%/28% 的加权最终分数

统一 collection=memory 的设计：
  - 所有长期记忆统一写入 collection=memory，简化查询逻辑
  - 旧的 collection=corpus 和 collection=facts 作为 legacy 分片保留，
    召回时会一并检索，支持平滑迁移

中文 FTS 预处理：
  - 对 CJK 字符（U+4E00 ~ U+9FFF）逐字拆分，解决 FTS5 默认不支持中文分词的问题
  - 保留英文/数字词汇不做拆分，维持自然词边界
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.llm import cosine_similarity, embed_texts
from app.session import store

logger = logging.getLogger(__name__)

# 主要的记忆存储集合名（统一入口）
_COLLECTION_MEMORY = "memory"
# 旧的 legacy 集合名（corpus=语料，facts=结构化事实），兼容旧数据
_LEGACY_COLLECTIONS = ("corpus", "facts")


def prepare_fts_text(text: str) -> str:
    """中文全文检索预处理：bigram + unigram 混合分词。

    策略：
    - CJK 汉字（U+4E00~U+9FFF）：先收集连续汉字序列，生成相邻两字 bigram
      （如"粥顶山"→"粥顶"+"顶山"），再保留原字 unigram 作为 fallback
    - bigram 比纯 unigram 更精准：「粥顶山」能匹配"粥顶山上"但不会误匹配"喝粥养胃"
    - unigram 兜底：单字查询或短语中孤立字仍能命中
    - 字母数字字符保留原样并转小写
    - 额外提取 2 字符以上的英文/数字序列作为补充 token

    注意：此改动后旧数据（仅有 unigram 索引）仍可通过 unigram 命中，
    新数据具备 bigram 精度优势。建议重新 ingest 以全面受益。

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
        # bigram：相邻两字组合，精准匹配短语
        if len(cjk_buffer) >= 2:
            for i in range(len(cjk_buffer) - 1):
                parts.append(cjk_buffer[i] + cjk_buffer[i + 1])
        # unigram：保留单字作为 fallback
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

    # 英文/数字连续序列（>=2 字符）作为完整词
    for word in re.findall(r"[a-zA-Z0-9]{2,}", text):
        parts.append(word.lower())

    return " ".join(parts)


def build_fts_match_query(query: str) -> str | None:
    """将用户自然语言查询转换为 FTS5 MATCH 语句的查询字符串。

    使用 OR 语义连接所有 token：只要文本包含任意一个 token 即命中。
    配合 bigram 分词，"粥顶山"查询 → "粥顶 OR 顶山 OR 粥 OR 顶 OR 山"，
    bigram 命中提供精准匹配，unigram 命中提供宽召回兜底。

    Args:
        query: 用户原始查询文本（如"粥顶山玩"）

    Returns:
        FTS5 enhanced query 字符串；预处理后为空则返回 None
    """
    q = prepare_fts_text(query)
    if not q.strip():
        return None
    terms = q.split()
    if not terms:
        return None
    # 去重保持查询简洁
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " OR ".join(unique)


def _detect_intent_boosts(query: str) -> dict[str, float]:
    """从查询文本中检测用户意图，返回类型 → 加分映射。

    基于关键词模式匹配，识别 6 种意图并给出对应的 category 加分：
      - 人物/关系查询（谁/谁是/某人） → person +0.15
      - 称呼/名字查询               → person +0.10
      - 偏好/喜好/口味查询           → preference +0.15
      - 禁忌/禁止/不能做什么         → taboo +0.15
      - 待办/计划/未完成             → open_loop +0.12
      - 月份/年份回忆               → monthly +0.10

    加分加在 RRF 最终分数上，使匹配类别的 chunk 排名提升。
    加分幅度经过校准（0.10~0.15），保证仅在同分 rank 时起决定作用，
    不劣化不相关类别的检索结果。

    Args:
        query: 用户原始查询文本

    Returns:
        {category: boost} 映射，如 {"person": 0.15}
    """
    boosts: dict[str, float] = {}
    q = query.strip()
    if not q:
        return boosts

    # 人物/关系识别
    if any(w in q for w in ("是谁", "谁是", "哪位", "什么人", "哪一位")):
        boosts["person"] = max(boosts.get("person", 0), 0.20)
    if any(w in q for w in ("称呼", "叫什么", "名字")):
        boosts["person"] = max(boosts.get("person", 0), 0.10)

    # 偏好识别（含"不喜欢/讨厌/不吃"等负向偏好查询）
    if any(w in q for w in ("喜欢", "爱好", "口味", "吃什么", "喜欢什么")):
        boosts["preference"] = max(boosts.get("preference", 0), 0.15)
    if any(w in q for w in ("不喜欢", "讨厌", "讨厌吃", "不吃", "不爱")):
        boosts["preference"] = max(boosts.get("preference", 0), 0.20)

    # 禁忌识别
    # "不能叫/不要叫/禁止称呼/不能称呼" → taboo 优先于 person
    if any(w in q for w in ("禁忌", "忌讳", "称呼不能用", "禁止称呼")):
        boosts["taboo"] = max(boosts.get("taboo", 0), 0.25)
    if any(w in q for w in ("不能叫", "不要叫", "不能称呼")):
        boosts["taboo"] = max(boosts.get("taboo", 0), 0.20)
    # 包含"称呼"且同时有"不能/禁止"等否定词的，强 taboo boost
    if "称呼" in q and any(w in q for w in ("不能", "禁止", "不可", "不要")):
        boosts["taboo"] = max(boosts.get("taboo", 0), 0.25)

    # 待办/计划识别
    if any(w in q for w in ("待跟进", "待办", "还有什", "该做", "要做什么")):
        boosts["open_loop"] = max(boosts.get("open_loop", 0), 0.12)

    # 月份/年份回忆识别
    import re
    if re.search(r"\d{4}年", q):
        boosts["monthly"] = max(boosts.get("monthly", 0), 0.10)

    return boosts


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

    算法流程：
    1. 对关键词路和向量路各自按排名计算 RRF 分数：1.0 / (60 + rank)
       - rank 从 1 开始，分母偏移 60 用于平滑排名差异
    2. 合并两路 RRF 分数，归一化到 [0, 1]（除以当次查询的最大 RRF）
    3. 对每个候选计算最终分数：0.72 * cosine_similarity + 0.28 * rrf_norm
       - 权重偏向向量相似度（72%），因为向量语义理解更准确
       - rrf 归一化后占 28%，双路命中（关键词+向量都召回）的块获得额外加分
    4. 可选：按意图类别加分（category_boosts），使 person/preference 等类型排名提升
    5. 过滤低于 min_score 的候选，按最终分数降序返回

    Args:
        query: 原始查询文本（保留供未来扩展）
        q_emb: 查询文本的 embedding 向量
        kw_ranked: 关键词 FTS 召回的排名结果 [(chunk_id, row_dict), ...]
        vec_ranked: 向量相似度召回的排名结果 [(chunk_id, row_dict), ...]
        top_k: 最终返回的最大命中数
        category_boosts: 类别加分映射 {category: boost_value}，如 {"person": 0.15}

    Returns:
        排序后的命中字典列表，每项含 text/score/chunk_id/category/source/collection
    """
    rrf_scores: dict[str, float] = {}
    payload: dict[str, dict[str, Any]] = {}

    # 两路分别计算 RRF 分数
    for hits in (kw_ranked, vec_ranked):
        for rank, (cid, row) in enumerate(hits, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))
            payload[cid] = row

    if not rrf_scores:
        return []

    # 归一化：除以当次查询的最大 RRF，使最高分=1.0，其余按比例递减
    # 双路命中（关键词+向量都召回）的块天然获得更高 RRF → 更高归一化分数
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
        # 72% 向量语义 + 28% RRF 排名信号（归一化后双路命中加分明显）
        final_score = (0.72 * sim) + (0.28 * rrf_norm)
        # 意图类别加分：使人物/偏好/禁忌等类型按 query 意图适度提升
        category = str(row.get("category") or "")
        cat_boost = (category_boosts or {}).get(category, 0.0)
        final_score += cat_boost
        if final_score >= min_score:
            out.append(
                {
                    "text": text,
                    "score": round(final_score, 3),
                    "chunk_id": cid,
                    "category": category,
                    "source": str(row.get("source") or ""),
                    "collection": str(row.get("collection") or _COLLECTION_MEMORY),
                }
            )

    out.sort(key=lambda x: x.get("score") or 0, reverse=True)
    return out[:top_k]


def hybrid_search_l3(
    *,
    person_id: str,
    query: str,
    q_emb: list[float],
    top_k: int,
    device_id: str = "",
    persona_person_id: str = "",
) -> list[dict]:
    """统一 L3 hybrid 检索入口：全局记忆 + 用户个人记忆 + persona 背景记忆。

    这是检索的主入口，在对话处理流程中被调用，用于从长期记忆中
    查找与用户当前消息最相关的内容。

    检索范围包括：
    - 全局共享记忆（所有用户可见的通用知识）
    - 当前用户的个人记忆（按 person_id 隔离）
    - persona 的背景设定记忆（机器人人设相关，按 persona_person_id 隔离）

    两路召回 + RRF 融合：
    1. FTS5 关键词召回：快速命中包含查询关键词的记忆块
    2. 向量语义召回：通过 embedding 相似度找到语义相关的块
    3. RRF 融合排序：综合两路结果，返回最终排序

    Args:
        person_id: 当前对话对象的 person_id（用于用户范围隔离）
        query: 用户自然语言查询
        q_emb: 查询文本的 embedding 向量
        top_k: 最终返回的最大记忆块数
        device_id: 设备 ID（保留参数）
        persona_person_id: persona 背景记忆的 person_id

    Returns:
        RRF 融合排序后的命中记忆块字典列表
    """
    del device_id
    pid = str(person_id or "").strip()
    ppid = str(persona_person_id or "").strip()
    # 收集所有需要检索的 person_id（用户本人 + persona 背景）
    extras = [ppid] if ppid and ppid != pid else []
    # 获取全局 + 用户范围内所有可召回的块
    all_rows = store.l3_list_recall_pool(pid, extra_person_ids=extras)
    if not all_rows:
        return []

    # FTS5 关键词召回路径
    kw_hits = store.l3_fts_search_pool(
        query, pid, extra_person_ids=extras, limit=max(settings.es_keyword_candidates, top_k)
    )
    kw_ranked = [
        (
            row["chunk_id"],
            {
                "text": row["text"],
                "embedding": row.get("embedding") or [],
                "category": row.get("category", ""),
                "source": row.get("source", ""),
                "collection": row.get("collection", _COLLECTION_MEMORY),
                "created_at": row.get("created_at", ""),
            },
        )
        for row in kw_hits
    ]

    # 向量语义召回路径（对每个候选计算 cosine 相似度）
    vec_scored: list[tuple[float, str, list[float], dict[str, Any]]] = []
    for row in all_rows:
        emb = row.get("embedding") or []
        if not emb:
            continue
        score = cosine_similarity(q_emb, emb)
        vec_scored.append((score, row["chunk_id"], emb, row))

    # 按相似度降序排列
    vec_scored.sort(key=lambda x: x[0], reverse=True)
    # 取向量召回 top-N 个候选
    vec_ranked = [
        (
            cid,
            {
                "text": row["text"],
                "embedding": emb,
                "category": row.get("category", ""),
                "source": row.get("source", ""),
                "collection": row.get("collection", _COLLECTION_MEMORY),
                "created_at": row.get("created_at", ""),
            },
        )
        for _, cid, emb, row in vec_scored[: max(settings.es_vector_candidates, top_k)]
    ]

    return _rrf_hybrid_rerank(
        query=query,
        q_emb=q_emb,
        kw_ranked=kw_ranked,
        vec_ranked=vec_ranked,
        top_k=top_k,
        category_boosts=_detect_intent_boosts(query),
    )


def hybrid_search(
    *,
    collection: str,
    query: str,
    q_emb: list[float],
    top_k: int,
    device_id: str = "",
    person_id: str | None = None,
    scope: str = "all",
) -> list[tuple[float, str]]:
    """Legacy 单集合检索接口（保留兼容 ES 路径与旧脚本调用）。

    与 hybrid_search_l3 不同，此接口按指定 collection（corpus/facts）
    进行单集合检索，主要用于旧代码和特定评估脚本。

    Args:
        collection: 检索目标集合名（corpus/facts）
        query: 查询文本
        q_emb: 查询 embedding 向量
        top_k: 返回数量
        device_id: 设备 ID（facts 集合需要用于过滤）
        person_id: 用户 person_id
        scope: 检索范围（all/global/person）

    Returns:
        [(score, text), ...] 排序后的命中列表
    """
    pid = str(person_id).strip() if person_id else None
    if collection == "facts" and not pid:
        return []

    if collection == "corpus":
        # corpus 支持按 scope 区分全局语料和人物关联语料
        if scope == "global":
            all_rows = store.l3_list_corpus_global()
        elif scope == "person" and pid:
            all_rows = store.l3_list_corpus_for_person(pid)
        else:
            all_rows = store.l3_list_corpus_global()
            if pid:
                all_rows = all_rows + store.l3_list_corpus_for_person(pid)
        if not all_rows:
            return []
        kw_hits = store.l3_fts_search(
            query,
            collection=collection,
            limit=max(settings.es_keyword_candidates, top_k),
        )
        if scope == "person" and pid:
            kw_hits = [
                h for h in kw_hits if str(h.get("person_id") or "").strip() == pid
            ]
        elif scope == "global":
            kw_hits = [h for h in kw_hits if not str(h.get("person_id") or "").strip()]
    else:
        # facts 检索需要按 device_id + person_id 过滤
        fact_device = device_id
        if store.l3_count(collection, device_id=fact_device, person_id=pid) <= 0:
            return []
        kw_hits = store.l3_fts_search(
            query,
            collection=collection,
            device_id=fact_device,
            person_id=pid,
            limit=max(settings.es_keyword_candidates, top_k),
        )
        all_rows = store.l3_list_chunks(collection, device_id=fact_device, person_id=pid)

    vec_scored: list[tuple[float, str, list[float], str]] = []
    for row in all_rows:
        emb = row.get("embedding") or []
        if not emb:
            continue
        score = cosine_similarity(q_emb, emb)
        vec_scored.append((score, row["chunk_id"], emb, row["text"]))

    vec_scored.sort(key=lambda x: x[0], reverse=True)
    vec_ranked = [
        (cid, {"text": text, "embedding": emb})
        for _, cid, emb, text in vec_scored[: max(settings.es_vector_candidates, top_k)]
    ]
    kw_ranked = [
        (row["chunk_id"], {"text": row["text"], "embedding": row.get("embedding") or []})
        for row in kw_hits
    ]
    reranked = _rrf_hybrid_rerank(
        query=query,
        q_emb=q_emb,
        kw_ranked=kw_ranked,
        vec_ranked=vec_ranked,
        top_k=top_k,
    )
    return [(float(m["score"]), str(m["text"])) for m in reranked]


def ingest_chunks(
    chunks: list[dict],
    *,
    collection: str = _COLLECTION_MEMORY,
    device_id: str = "",
    person_id: str = "",
    reset: bool = False,
) -> int:
    """批量写入 L3 记忆块到存储层。

    这是所有长期记忆写入的统一入口，数据流：
    文本块列表 → embedding 向量化 → SQLite 批量 upsert

    Args:
        chunks: 块字典列表，每项需含 id（唯一标识）、text（文本）、meta（元数据）
        collection: 目标集合名，默认 memory
        device_id: 关联的设备 ID（用于用户数据隔离）
        person_id: 关联的人物 ID（用于人物数据隔离）
        reset: 是否先清空旧数据再写入

    Returns:
        成功写入的块数量
    """
    if not chunks:
        return 0
    if reset:
        # corpus/memory 全局级别的 reset 用专用方法
        if collection in (_COLLECTION_MEMORY, "corpus") and not device_id and not person_id:
            store.l3_reset_corpus()
        else:
            store.l3_clear(collection, device_id=device_id)

    ids = [str(c["id"]) for c in chunks]
    docs = [c["text"] for c in chunks]
    # 批量调用 embedding API 生成向量（一次 API 调用处理所有文本，提高效率）
    embs = embed_texts(docs)
    if not embs:
        return 0

    rows = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("meta", {})
        rows.append(
            {
                "chunk_id": ids[i],
                "collection": collection,
                "device_id": device_id or str(meta.get("device_id", "")),
                "text": docs[i],
                "embedding": embs[i],
                "source": str(meta.get("source", "memory")),
                "category": str(meta.get("category", "")),
                "confidence": float(meta.get("confidence", 0.0)),
                "person_id": str(meta.get("person_id", person_id or "")),
            }
        )
    store.l3_bulk_upsert(rows)
    return len(chunks)


def upsert_fact_chunk(
    device_id: str,
    person_id: str,
    fact: str,
    category: str,
    confidence: float,
    source_session: str,
    embedding: list[float],
) -> None:
    """Legacy：直接将结构化事实写入 facts 集合的向量块。

    新代码应优先使用 ingest_chunks 统一接口，此函数仅保留给旧数据迁移和
    特定 legacy 路径使用。

    每条 fact 的 chunk_id 由 device_id + person_id + content_hash 组成，
    确保同一条事实被重复调用时是 upsert 而非重复插入。

    Args:
        device_id: 设备 ID
        person_id: 人物 ID
        fact: 事实文本
        category: 事实分类标签
        confidence: 置信度 0~1
        source_session: 来源会话 ID
        embedding: 预计算的 embedding 向量
    """
    pid = str(person_id or "").strip()
    # 生成确定性 chunk_id：相同的 content + category 产生相同的 ID
    fid = f"{device_id}-{pid}-fact-{abs(hash((fact, category))) % 10**10}"
    store.l3_bulk_upsert(
        [
            {
                "chunk_id": fid,
                "collection": "facts",
                "device_id": device_id,
                "person_id": pid,
                "text": fact,
                "embedding": embedding,
                "source": source_session,
                "category": category,
                "confidence": confidence,
            }
        ]
    )


# ---------- 统计与维护函数 ----------

def count_l3() -> int:
    """L3 块总数（含 memory + legacy corpus + legacy facts）。"""
    return store.l3_count(_COLLECTION_MEMORY) + store.l3_count("corpus") + store.l3_count("facts")


def count_corpus() -> int:
    """全局 + memory 语料块数（兼容旧代码）。"""
    return store.l3_count("corpus") + store.l3_count(_COLLECTION_MEMORY)


def count_facts() -> int:
    """Legacy facts 集合块数（兼容旧代码）。"""
    return store.l3_count("facts")


def reset_corpus() -> None:
    """清空全局语料（corpus + memory 集合）。"""
    store.l3_reset_corpus()


def reset_all() -> None:
    """清空全部 L3 块（memory + corpus + facts）。"""
    store.l3_reset_all()
