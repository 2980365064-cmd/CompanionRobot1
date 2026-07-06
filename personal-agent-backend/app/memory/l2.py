"""
L2 情景记忆（Episodic Memory）—— 近期会话摘要与情感事件记录。

============================================================================
在陪伴型情感记忆体系中的角色：
  L2 是"情景记忆层"——存储每次会话结束后 LLM 生成的摘要。
  分为两类：

  1. recent episodes （14-30天）：近期情感近况，用于感知"最近关系气氛"
  2. important episodes（永久）：高重要性情感事件，不受过期限制

  比 L3 更细致、更即时，但有过期淘汰机制（regular episodes → 到期归档 L3）。
  重要事件（importance >= 4）保留更久，支持长期情感时间线。

检索机制：
  - 向量检索：对 query 做 embedding，与摘要向量做余弦相似度匹配
  - 重要性加权：高重要性事件的相似度分数 +0.15 加成
  - 兜底策略：向量检索无命中时，返回最近 N 条摘要作为 fallback
  - 门控阈值：相似度 < l2_sim_threshold 的摘要不参与召回

数据流转：
  L1 →（满 N 轮压缩）→ L2 recent →（到期归档）→ L3 语料库
                                     ↕
                              L2 important（高重要性事件保留更久）
============================================================================
"""

from __future__ import annotations

import hashlib
import json

from app.config import settings
from app.llm import cosine_similarity, embed_texts
from app.session import store


class EpisodicMemory:
    """L2 情景记忆：对 episodic_memories 表做向量检索，召回会话摘要。

    扩展后支持结构化 episode 元数据：
      - importance: 重要性 1-5，4+ 视为重要事件
      - people: 涉及人物 JSON 数组
      - emotion: 情感快照 JSON
      - status: active / archived / corrected

    双模式检索：
      1. recent_chat: 近期情感，不在乎具体话题（无 query 时的时间倒序）
      2. semantic_recall: 有 query 的语义向量检索
    """

    def __init__(self) -> None:
        self._emb_cache: dict[str, list[float]] = {}
        self._EMB_CACHE_MAX = 500

    # ── 重要性加权 ────────────────────────────────────────────────────
    # 高重要性事件在向量匹配时获得分数加成，使其更易被召回
    # 即使 query 没有精确匹配关键词，重要情感事件也倾向于被想起
    _IMPORTANCE_BOOST = {
        5: 0.20,  # 里程碑 → 显著加分
        4: 0.10,  # 重要事件 → 适度加分
        3: 0.0,   # 普通摘要 → 无加成
        2: -0.05, # 次要 → 轻微扣分
        1: -0.10, # 琐事 → 扣分（让位给更重要的事）
    }

    def _apply_importance_boost(self, importance: int) -> float:
        """根据重要性返回分数偏移量。"""
        return self._IMPORTANCE_BOOST.get(importance, 0.0)

    def recall_scored(
        self,
        device_id: str,
        person_id: str,
        query: str,
        top_k: int | None = None,
        *,
        q_emb: list[float] | None = None,
        min_importance: int | None = None,
    ) -> list[dict]:
        """按 query 做向量相似度检索，返回最相关的 L2 摘要。

        与旧版兼容，新增参数：
          min_importance: 最低重要性过滤（None=不过滤）

        重要事件（importance >= 4）的匹配分数 +0.10~0.20 加成，
        使机器人更容易想起重要的事。
        """
        top_k = top_k or settings.episodic_top_k
        pool = max(settings.l2_embed_pool, settings.episodic_top_k)
        rows = store.list_episodic_active(device_id, person_id, limit=pool)

        # 按重要性过滤
        if min_importance is not None:
            rows = [
                r for r in rows
                if (r.get("importance") or 3) >= min_importance
            ]

        if not rows:
            return []

        q = query.strip()
        if not q:
            return self._fallback_recent(rows, top_k)

        return self._vector_search(rows, top_k, q, q_emb, device_id, person_id)

    def _fallback_recent(
        self, rows: list[dict], top_k: int,
    ) -> list[dict]:
        """空查询时的兜底模式：返回最近 N 条摘要。"""
        out: list[dict] = []
        seen: set[str] = set()
        for row in rows[: settings.l2_recall_recent]:
            text = row["summary"]
            if text not in seen:
                seen.add(text)
                out.append({
                    "text": text,
                    "score": None,
                    "note": "recent",
                    "created_at": row.get("created_at", ""),
                    "importance": row.get("importance", 3),
                    "emotion": row.get("emotion", ""),
                    "people": row.get("people", "[]"),
                })
            if len(out) >= top_k:
                break
        return out

    def _vector_search(
        self,
        rows: list[dict],
        top_k: int,
        query: str,
        q_emb: list[float] | None,
        device_id: str = "",
        person_id: str = "",
    ) -> list[dict]:
        """向量语义检索核心逻辑。

        Args:
            rows:      候选摘要行
            top_k:     返回条数
            query:     检索查询
            q_emb:     预计算查询向量
            device_id: 设备标识（用于 embedding 缓存键）
            person_id: 用户 ID（用于 embedding 缓存键）
        """
        min_score = settings.l2_sim_threshold

        if q_emb is None:
            q_emb = embed_texts([query])[0]

        summaries = [row["summary"] for row in rows]
        cache_keys = [
            hashlib.md5(f"{device_id}:{person_id}:{s}".encode()).hexdigest()[:16]
            for s in summaries
        ]

        # 保持对 device_id, person_id 的引用（用于缓存键，在方法内可用）
        embs: list[list[float]] = []
        miss_idx: list[int] = []
        for i, key in enumerate(cache_keys):
            emb = self._emb_cache.get(key)
            if emb is not None:
                embs.append(emb)
            else:
                embs.append([])
                miss_idx.append(i)

        if miss_idx:
            miss_texts = [summaries[i] for i in miss_idx]
            new_embs = embed_texts(miss_texts)
            for j, emb in enumerate(new_embs):
                idx = miss_idx[j]
                embs[idx] = emb
                self._emb_cache[cache_keys[idx]] = emb
            if len(self._emb_cache) > self._EMB_CACHE_MAX:
                drop = list(self._emb_cache.keys())[: len(self._emb_cache) // 2]
                for k in drop:
                    del self._emb_cache[k]

        scored: list[dict] = []
        for row, summary, s_emb in zip(rows, summaries, embs):
            score = cosine_similarity(q_emb, s_emb)
            imp = row.get("importance", 3) or 3
            boost = self._apply_importance_boost(imp)
            adjusted = score + boost
            if adjusted >= min_score:
                scored.append({
                    "text": summary,
                    "score": round(score, 3),
                    "adjusted_score": round(adjusted, 3),
                    "importance": imp,
                    "created_at": row.get("created_at", ""),
                    "emotion": row.get("emotion", ""),
                    "people": row.get("people", "[]"),
                })

        # 按调整后分数降序
        scored.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return scored[:top_k]

    def recall_important(
        self,
        person_id: str,
        min_importance: int = 4,
        limit: int = 5,
    ) -> list[dict]:
        """直接检索重要情景记忆（高重要性，不受过期限制）。

        用于情感时间线——当需要感知"你们之间发生过什么重要的事"
        时，调用此方法获取不受 14 天限制的永久事件记录。

        Args:
            person_id:       用户 ID
            min_importance:  最低重要性（4=重要事件，5=里程碑）
            limit:           最多条数

        Returns:
            带 metadata 的事件列表。
        """
        rows = store.list_important_episodes(person_id, min_importance, limit)
        out: list[dict] = []
        for row in rows:
            out.append({
                "text": row["summary"],
                "importance": row.get("importance", 4),
                "created_at": row.get("created_at", ""),
                "emotion": row.get("emotion", ""),
                "people": row.get("people", "[]"),
            })
        return out

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
        importance: int = 3,
        people: str = "",
    ) -> None:
        """将一条会话摘要写入 L2 情景记忆库。

        扩展参数:
            importance: 重要性 1-5（默认 3=普通）
            people:     涉及人物，JSON 数组字符串或逗号分隔名
        """
        # 兼容旧调用者：如果传入的是逗号分隔字符串，转成 JSON
        if people and not people.startswith("["):
            name_list = [n.strip() for n in people.split(",") if n.strip()]
            people = json.dumps(name_list, ensure_ascii=False)

        store.add_episodic(
            device_id, session_id, summary, topics, open_loops,
            person_id=person_id, emotion=emotion,
            importance=importance, people=people,
        )


# 模块级单例
episodic_memory = EpisodicMemory()
