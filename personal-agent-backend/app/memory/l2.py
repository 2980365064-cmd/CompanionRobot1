"""
L2 情景记忆（Episodic Memory）—— 近期会话摘要，7天有效期。

============================================================================
在陪伴型情感记忆体系中的角色：
  L2 是"近期记忆层"——存储最近 7 天内每次会话结束后 LLM 生成的摘要。
  比 L3 更细致、更即时，但有过期淘汰机制（expired → rollup 到 L3）。

检索机制：
  - 向量检索：对 query 做 embedding，与 L2 摘要向量做余弦相似度匹配
  - 兜底策略：向量检索无命中时，返回最近 N 条摘要作为 fallback
  - 门控阈值：相似度 < l2_sim_threshold 的摘要不参与召回

与其他记忆层的关系：
  L1 →（满 N 轮压缩）→ L2 →（7天后过期）→ L3 语料库
============================================================================
"""

from __future__ import annotations

import hashlib

from app.config import settings
from app.llm import cosine_similarity, embed_texts
from app.session import store


class EpisodicMemory:
    """L2 情景记忆：对 episodic_memories 表做向量检索，召回近期会话摘要。

    存储结构：每条记录包含 session_id、summary（LLM 生成的摘要文本）、
    topics（主题标签）、open_loops（未完结事项）、emotion（情感快照）等字段。

    设计要点：
    - 双模式检索：有 query 时走向量语义匹配，无 query 时走时间倒序兜底。
      这样在纯聊天（无需检索）场景下不会浪费 embedding 调用。
    - 门控阈值：相似度低于 l2_sim_threshold 的摘要不参与召回，避免
      不相关记忆污染上下文窗口。
    - 与 L3 的分工：L2 记录「最近几天发生了什么」，L3 记录「用户是什么样的人」。
      L2 偏叙事，L3 偏知识。
    - 摘要向量缓存：摘要文本很少变化，缓存其 embedding 避免每轮重新调用 API。
      缓存 key = md5(device_id + person_id + summary)，上限 500 条。
    """

    def __init__(self) -> None:
        self._emb_cache: dict[str, list[float]] = {}
        self._EMB_CACHE_MAX = 500

    def recall_scored(
        self,
        device_id: str,
        person_id: str,
        query: str,
        top_k: int | None = None,
        *,
        q_emb: list[float] | None = None,
    ) -> list[dict]:
        """按 query 做向量相似度检索，返回最相关的 L2 摘要。

        检索流程：
          1. 从 episodic_memories 表取该用户的活跃摘要（pool 条）
          2. 若 query 为空 → fallback 模式：返回最近 N 条摘要（无分数）
          3. 正常模式：对 query 和摘要分别做 embedding，计算余弦相似度
          4. 只保留相似度 >= l2_sim_threshold 的结果
          5. 按分数降序排列，返回 top_k 条

        Args:
            device_id:  设备标识
            person_id:  用户 ID
            query:      检索查询文本（用户当前消息或改写后的查询）
            top_k:      返回数量上限（默认取配置 episodic_top_k）
            q_emb:      可选的预计算查询向量（避免重复 embedding 调用）

        Returns:
            评分后的摘要列表，每条包含 text（摘要文本）、score（相似度分数）、
            note（可选标注，如 "recent" 表示兜底补充）。
        """
        top_k = top_k or settings.episodic_top_k
        # pool 比 top_k 大，保证有足够的候选来做相似度过滤：
        # 如果 pool == top_k，过滤后可能不足 top_k 条
        pool = max(settings.l2_embed_pool, settings.episodic_top_k)
        rows = store.list_episodic_active(device_id, person_id, limit=pool)
        if not rows:
            return []

        q = query.strip()
        # 空查询 → fallback 模式：不调用 embedding API，直接返回最近 N 条摘要。
        # 适用场景：用户消息太短/无实质内容时，router 传空查询触发此路径。
        # seen 集合用于去重——多条记录可能指向同一条摘要文本。
        if not q:
            out: list[dict] = []
            seen: set[str] = set()
            for row in rows[: settings.l2_recall_recent]:
                text = row["summary"]
                if text not in seen:
                    seen.add(text)
                    out.append({"text": text, "score": None, "note": "recent",
                                "created_at": row.get("created_at", "")})
                if len(out) >= top_k:
                    break
            return out

        # 正常向量检索模式
        min_score = settings.l2_sim_threshold
        if q_emb is None:
            # 对 query 做 embedding；若调用方已预计算（如 router 改写过查询），
            # 可通过 q_emb 参数传入，避免重复调用 embedding API
            q_emb = embed_texts([q])[0]
        # 摘要向量缓存：摘要文本很少变化，优先从缓存取 embedding
        # 只对缓存未命中的摘要做批量 API 调用，大幅减少重复开销
        summaries = [row["summary"] for row in rows]
        cache_keys = [
            hashlib.md5(f"{device_id}:{person_id}:{s}".encode()).hexdigest()[:16]
            for s in summaries
        ]
        embs: list[list[float]] = []
        miss_idx: list[int] = []
        for i, key in enumerate(cache_keys):
            emb = self._emb_cache.get(key)
            if emb is not None:
                embs.append(emb)
            else:
                embs.append([])  # placeholder
                miss_idx.append(i)
        if miss_idx:
            miss_texts = [summaries[i] for i in miss_idx]
            new_embs = embed_texts(miss_texts)
            for j, emb in enumerate(new_embs):
                idx = miss_idx[j]
                embs[idx] = emb
                self._emb_cache[cache_keys[idx]] = emb
            # 缓存放量保护：超过上限时删掉最旧一半
            if len(self._emb_cache) > self._EMB_CACHE_MAX:
                drop = list(self._emb_cache.keys())[: len(self._emb_cache) // 2]
                for k in drop:
                    del self._emb_cache[k]

        scored: list[dict] = []
        for row, summary, s_emb in zip(rows, summaries, embs):
            score = cosine_similarity(q_emb, s_emb)
            # 门控：低于阈值的摘要视为不相关，不参与召回，
            # 避免无关记忆挤占 prompt 空间
            if score >= min_score:
                scored.append({"text": summary, "score": round(score, 3),
                               "created_at": row.get("created_at", "")})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def save_summary(
        self,
        device_id: str,
        person_id: str,
        session_id: str,
        summary: str,
        topics: str = "",
        open_loops: str = "",
        *,
        emotion: str = "",
    ) -> None:
        """将一条会话摘要写入 L2 情景记忆库。

        由 extractor.compress_l1_to_l2 和 extractor.consolidate_session 调用。

        Args:
            device_id:  设备标识
            person_id:  用户 ID
            session_id: 会话标识
            summary:    LLM 生成的摘要文本（150~280 字）
            topics:     逗号分隔的主题标签
            open_loops: 未完结事项的 JSON 数组字符串
            emotion:    情感快照 JSON（含 mood/intensity/trigger/attitude）
        """
        store.add_episodic(
            device_id, session_id, summary, topics, open_loops,
            person_id=person_id, emotion=emotion,
        )


# 模块级单例，供 router 和 extractor 直接引用。
# EpisodicMemory 本身无内部状态，单例是安全的。
episodic_memory = EpisodicMemory()
