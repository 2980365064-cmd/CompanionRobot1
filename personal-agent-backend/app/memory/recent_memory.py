"""
Recent Memory — 近期记忆（基于统一记忆库 memory_items 的语义视图）。

============================================================================
在陪伴型情感记忆体系中的角色：
  Recent Memory 是"近期记忆视图"——基于统一记忆库中 kind=episode/emotion/milestone
  的条目，提供向量检索 + 重要性加权。

  分为两类：

  1. recent episodes （14-30天）：近期情感近况，用于感知"最近关系气氛"
  2. important episodes（永久）：高重要性情感事件，不受过期限制

检索机制：
  - 向量检索：对 query 做 embedding，与摘要向量做余弦相似度匹配
  - 重要性加权：高重要性事件的相似度分数 +0.15 加成
  - 兜底策略：向量检索无命中时，返回最近 N 条摘要作为 fallback
  - 门控阈值：相似度 < recent_memory_sim_threshold 的摘要不参与召回
============================================================================
"""

from __future__ import annotations

import hashlib
import json

from app.config import settings
from app.llm import cosine_similarity, embed_texts
from app.session import store


class RecentMemory:
    """近期记忆：基于统一记忆库的语义视图，筛选 kind=episode/emotion 条目。

    支持结构化 episode 元数据（存于 context_json）：
      - emotional_weight (原 importance): 重要性 1-5，4+ 视为重要事件
      - context_json 中的 people, emotion, topics, open_loops

    双模式检索：
      1. recent_chat: 近期情感（无 query 时的时间倒序）
      2. semantic_recall: 有 query 的语义向量检索
    """

    def __init__(self) -> None:
        self._emb_cache: dict[str, list[float]] = {}
        self._EMB_CACHE_MAX = 500

    # ── 重要性加权 ────────────────────────────────────────────────────
    # 高重要性事件在向量匹配时获得分数加成，使其更易被召回
    _IMPORTANCE_BOOST = {
        5: 0.20,  # 里程碑 → 显著加分
        4: 0.10,  # 重要事件 → 适度加分
        3: 0.0,   # 普通摘要 → 无加成
        2: -0.05, # 次要 → 轻微扣分
        1: -0.10, # 琐事 → 扣分（让位给更重要的事）
    }

    def _apply_importance_boost(self, importance: int) -> float:
        return self._IMPORTANCE_BOOST.get(importance, 0.0)

    def search_recent_memory(
        self,
        device_id: str,
        person_id: str,
        query: str,
        top_k: int | None = None,
        *,
        q_emb: list[float] | None = None,
        min_importance: int | None = None,
    ) -> list[dict]:
        """按 query 做向量相似度检索，返回最相关的近期摘要（基于统一记忆库）。

        Args:
            min_importance: 最低重要性过滤（None=不过滤，基于 emotional_weight）

        重要事件（emotional_weight >= 4）的匹配分数 +0.10~0.20 加成。
        """
        top_k = top_k or settings.recent_memory_top_k
        pool = max(settings.recent_memory_embed_pool, settings.recent_memory_top_k)

        # 从统一记忆库读取近期条目（kind=episode/emotion）
        rows = store.search_recent_memory(person_id, limit=pool)

        def _get_importance(r: dict) -> int:
            return int(r.get("emotional_weight", 3) or 3)

        def _get_summary(r: dict) -> str:
            ctx = r.get("context_json", "{}")
            try:
                ctx_d = json.loads(ctx) if isinstance(ctx, str) else ctx
                return str(ctx_d.get("summary", r.get("content", "")))
            except (json.JSONDecodeError, TypeError):
                return str(r.get("content", ""))

        def _get_emotion(r: dict) -> str:
            ctx = r.get("context_json", "{}")
            try:
                ctx_d = json.loads(ctx) if isinstance(ctx, str) else ctx
                return str(ctx_d.get("emotion", ""))
            except (json.JSONDecodeError, TypeError):
                return ""

        def _get_people(r: dict) -> str:
            ctx = r.get("context_json", "{}")
            try:
                ctx_d = json.loads(ctx) if isinstance(ctx, str) else ctx
                return json.dumps(ctx_d.get("people", []), ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                return "[]"

        if min_importance is not None:
            rows = [r for r in rows if _get_importance(r) >= min_importance]

        if not rows:
            return []

        q = query.strip()
        if not q:
            return self._recent_fallback(rows, top_k, _get_summary)

        return self._vector_search_memory(
            rows, top_k, q, q_emb, device_id, person_id,
            _get_summary, _get_importance)

    def _recent_fallback(
        self, rows: list[dict], top_k: int,
        _get_summary=None,
    ) -> list[dict]:
        """空查询时的兜底模式：返回最近 N 条摘要。"""
        if _get_summary is None:
            _get_summary = lambda r: r.get("content", "")
        out: list[dict] = []
        seen: set[str] = set()
        for row in rows[: settings.recent_memory_recall_recent]:
            text = _get_summary(row)
            if text not in seen:
                seen.add(text)
                ctx = row.get("context_json", "{}")
                try:
                    ctx_d = json.loads(ctx) if isinstance(ctx, str) else {}
                except (json.JSONDecodeError, TypeError):
                    ctx_d = {}
                out.append({
                    "text": text,
                    "score": None,
                    "note": "recent",
                    "created_at": row.get("created_at", ""),
                    "importance": int(row.get("emotional_weight", 3) or 3),
                    "emotion": ctx_d.get("emotion", ""),
                    "people": json.dumps(ctx_d.get("people", []), ensure_ascii=False),
                })
            if len(out) >= top_k:
                break
        return out

    def _vector_search_memory(
        self,
        rows: list[dict],
        top_k: int,
        query: str,
        q_emb: list[float] | None,
        device_id: str = "",
        person_id: str = "",
        _get_summary=None,
        _get_importance=None,
    ) -> list[dict]:
        """向量语义检索核心逻辑。"""
        if _get_summary is None:
            _get_summary = lambda r: r.get("content", "")
        if _get_importance is None:
            _get_importance = lambda r: int(r.get("emotional_weight", 3) or 3)

        min_score = settings.recent_memory_sim_threshold

        if q_emb is None:
            q_emb = embed_texts([query])[0]

        summaries = [_get_summary(row) for row in rows]
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
            imp = _get_importance(row)
            boost = self._apply_importance_boost(imp)
            adjusted = score + boost
            if adjusted >= min_score:
                ctx = row.get("context_json", "{}")
                try:
                    ctx_d = json.loads(ctx) if isinstance(ctx, str) else {}
                except (json.JSONDecodeError, TypeError):
                    ctx_d = {}
                scored.append({
                    "text": summary,
                    "score": round(score, 3),
                    "adjusted_score": round(adjusted, 3),
                    "importance": imp,
                    "created_at": row.get("created_at", ""),
                    "emotion": ctx_d.get("emotion", ""),
                    "people": json.dumps(ctx_d.get("people", []), ensure_ascii=False),
                })

        scored.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return scored[:top_k]

    def search_important_memories(
        self,
        person_id: str,
        min_importance: int = 4,
        limit: int = 5,
    ) -> list[dict]:
        """直接检索重要近期记忆（高 emotional_weight，基于统一记忆库）。

        用于情感时间线——当需要感知"你们之间发生过什么重要的事"时调用。
        """
        # 从统一记忆库读取 episode/emotion/milestone，按 emotional_weight 排序
        rows = store.search_memory_items(
            person_id,
            kinds=["episode", "milestone"],
            visibility="recall_only",
            limit=limit * 3,
            include_expired=False,
        )
        # 筛选高重要性条目（emotional_weight >= min_importance）
        filtered = [
            r for r in rows
            if int(r.get("emotional_weight", 3) or 3) >= min_importance
        ]
        out: list[dict] = []
        for row in filtered[:limit]:
            ctx = row.get("context_json", "{}")
            try:
                ctx_d = json.loads(ctx) if isinstance(ctx, str) else {}
            except (json.JSONDecodeError, TypeError):
                ctx_d = {}
            out.append({
                "text": str(row.get("content", "")),
                "importance": int(row.get("emotional_weight", 4) or 4),
                "created_at": row.get("created_at", ""),
                "emotion": ctx_d.get("emotion", ""),
                "people": json.dumps(ctx_d.get("people", []), ensure_ascii=False),
            })
        return out

    def save_recent_summary(
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
        """将一条会话摘要写入统一记忆库（kind=episode）。

        Args:
            importance: 重要性 1-5（对应 memory_items emotional_weight）
            people:     涉及人物，JSON 数组字符串或逗号分隔名
        """
        if people and not people.startswith("["):
            name_list = [n.strip() for n in people.split(",") if n.strip()]
            people = json.dumps(name_list, ensure_ascii=False)

        context = {}
        if topics:
            context["topics"] = topics
        if open_loops:
            context["open_loops"] = open_loops
        if emotion:
            context["emotion"] = emotion
        if people and people != "[]":
            try:
                context["people"] = json.loads(people) if isinstance(people, str) else people
            except (json.JSONDecodeError, TypeError):
                context["people"] = people

        store.write_memory_item(
            person_id=person_id,
            device_id=device_id,
            kind="episode",
            source="conversation_summary",
            visibility="recall_only",
            content=summary,
            emotional_weight=importance,
            recency_weight=4 if importance >= 4 else 3,
            context_json=json.dumps(context, ensure_ascii=False),
            source_session=session_id,
        )


# 模块级单例
recent_memory = RecentMemory()
