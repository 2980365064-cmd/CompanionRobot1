"""
记忆召回路由 —— 按用户身份决定检索策略，协调记忆的联合召回。

============================================================================
在陪伴型情感机器人记忆体系中的角色：
  Router 是"记忆调度中心"——每轮对话开始时，根据用户身份（访客/已实名）
  决定检索哪些记忆类型、检索优先级、以及是否触发关联扩展。

召回策略：

  访客模式（tmp_* / 未实名）：
    → 仅返回当前会话窗口（history）
    → 不调用 embedding（节省 API 费用）
    → 核心记忆/情景记忆/长期记忆/关联记忆全部为空

  已实名模式（verified person_id）：
    1. 核心记忆全量加载（无向量门控，无条件注入）
    2. 情景记忆检索；无命中 → fallback 注入最近摘要
    3. 长期记忆检索（仅当 query 需要记忆回答时触发，含月份 FTS 补召）
    4. 关联记忆扩展：基于长期记忆命中做图扩展
    5. 情感轨迹：附加最近情感快照

输出结构（语义 memory dict）：
  {
    history:      当前会话消息列表
    items:        MemoryItem 列表（core + recent + long_term + related）
    diagnostics:  {person_id, guest_mode, has_recent, has_long_term,
                   evidence_count, evidence_weak, evidence_sources, ...}
    memory_miss:  是否完全未命中（用于反幻觉提示）
    person_id:    活跃用户 ID
    guest_mode:   是否访客模式
  }
============================================================================
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from app.config import settings
from app.llm import embed_texts
from app.memory.guard import extract_self_name, is_casual_smalltalk, query_needs_memory_answer
from app.memory.working_context import get_recent_context
from app.memory.emotion import emotion_trajectory
from app.memory.identity import memory_scoped_to_person
from app.memory.unified_store import unified_memory_store, MemorySearchQuery
from app.session import store

# ── Embedding 查询向量缓存 ─────────────────────────────────────────────────
# LRU + TTL 缓存：key = (person_id, query_hash)，最多缓存 128 条
# TTL = 300 秒（5分钟），与 Anthropic prompt cache TTL 对齐
_EMBED_CACHE_SIZE = 128
_EMBED_CACHE_TTL = 300.0
_embed_cache: dict[str, tuple[float, list[float]]] = {}


@dataclass(frozen=True)
class RetrievalPlan:
    """一次召回的语义检索计划。

    字段描述的是产品意图，不再让调用方直接关心底层存储层名称。
    """

    needs_memory: bool
    search_recent_memory: bool
    search_long_term: bool
    include_recent_episodes: bool
    recent_top_k: int
    long_term_top_k: int
    month_key: str = ""


class RetrievalPlanner:
    """按用户消息生成统一记忆检索计划。"""

    def plan(self, query: str, *, working: list[dict]) -> RetrievalPlan:
        q = (query or "").strip()
        month_key = _parse_month_key(q) or ""
        casual = is_casual_smalltalk(q)
        needs_memory = query_needs_memory_answer(q)
        has_context = bool(working)

        search_recent_memory = bool(q) and not casual and (needs_memory or has_context)
        search_long_term = bool(q) and needs_memory
        return RetrievalPlan(
            needs_memory=needs_memory,
            search_recent_memory=search_recent_memory,
            search_long_term=search_long_term,
            include_recent_episodes=bool(q) and not needs_memory and not casual,
            recent_top_k=settings.recent_memory_top_k if search_recent_memory else 0,
            long_term_top_k=(6 if month_key else 3) if search_long_term else 0,
            month_key=month_key,
        )


def _cached_embed(text: str, cache_key: str = "") -> list[float]:
    """带缓存的单文本向量化。"""
    if not text:
        return []
    key = cache_key or hashlib.md5(text.encode()).hexdigest()[:12]
    now = time.monotonic()
    entry = _embed_cache.get(key)
    if entry and now - entry[0] < _EMBED_CACHE_TTL:
        return entry[1]
    embs = embed_texts([text])
    vec = embs[0] if embs else []
    if vec:
        if len(_embed_cache) >= _EMBED_CACHE_SIZE:
            oldest = min(_embed_cache, key=lambda k: _embed_cache[k][0])
            del _embed_cache[oldest]
        _embed_cache[key] = (now, vec)
    return vec


def _batch_embed(texts: list[str]) -> list[list[float]]:
    """批量向量化（跳过缓存命中项，减少 API 调用）。

    对每个 text 检查缓存，只对未命中的做批量 API 调用，
    然后合并缓存结果。
    """
    if not texts:
        return []
    now = time.monotonic()
    results: list[list[float]] = []
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    for i, t in enumerate(texts):
        if not t:
            results.append([])
            continue
        key = hashlib.md5(t.encode()).hexdigest()[:12]
        entry = _embed_cache.get(key)
        if entry and now - entry[0] < _EMBED_CACHE_TTL:
            results.append(entry[1])
        else:
            results.append([])  # placeholder
            miss_indices.append(i)
            miss_texts.append(t)
    if miss_texts:
        embs = embed_texts(miss_texts)
        for j, emb in enumerate(embs):
            idx = miss_indices[j]
            results[idx] = emb
            key = hashlib.md5(miss_texts[j].encode()).hexdigest()[:12]
            if len(_embed_cache) >= _EMBED_CACHE_SIZE:
                oldest = min(_embed_cache, key=lambda k: _embed_cache[k][0])
                del _embed_cache[oldest]
            _embed_cache[key] = (now, emb)
    return results


def _enrich_long_term_query(query: str, working: list[dict]) -> str:
    """用 工作上下文 上下文扩充长期记忆检索查询——解决短追问丢失前文主题的问题。

    场景：用户先说「之前在粥顶山玩的时候...」，
    然后追问「你真的想起来了吗」。
    单独的「你真的想起来了吗」不包含任何可检索的实体词,
    需要从上下文提取「粥顶山」等关键信息拼入 query。

    Args:
        query:   当前用户消息
        working: 工作上下文 最近消息列表 [{"role": "user"/"assistant", "content": ...}, ...]

    Returns:
        扩充后的 query，无需扩充时返回原 query。
    """
    q = query.strip()
    # 消息足够长 → 自带上下文，不扩充
    if len(q) >= 18:
        return q
    # 消息虽短但包含具体实体词（地名/事件/人名）→ 不扩充
    if _has_content_entities(q):
        return q
    # 收集最近几轮用户消息（排除当前 query 本身）
    recent_user = [
        m["content"].strip()
        for m in working[-8:]
        if m.get("role") == "user" and m["content"].strip() != q
    ]
    if not recent_user:
        return q
    # 找前文中信息量最大的消息作为上下文：
    # 跳过短追问（< 12 字且无实体词），优先取包含具体内容的消息
    best = ""
    for msg in reversed(recent_user):
        if len(msg) > len(best) and not (len(msg) < 12 and not _has_content_entities(msg)):
            best = msg
        if len(best) >= 30:
            break  # 已经够长了
    if not best or len(best) <= 8:
        return q
    snippet = best[:80].strip()
    if snippet in q:
        return q
    return f"{q}（上下文：{snippet}）"


_CN_MONTHS: dict[str, int] = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


def _cn_month_to_int(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        n = int(token)
        return n if 1 <= n <= 12 else None
    if token in _CN_MONTHS:
        return _CN_MONTHS[token]
    if len(token) == 2 and token[0] == "十" and token[1] in _CN_MONTHS:
        return 10 + _CN_MONTHS[token[1]]
    return None


def _parse_month_key(query: str) -> str | None:
    """提取用户查询中的月份，归一化为 YYYY-MM。

    支持的输入格式：
      - 2025年6月, 2025年06月
      - 2025-6月, 2025-06月, 2025-6, 2025-06
      - 2025 年 6 月（空格分隔）
      - 二〇二五年六月（中文全数字）
    """
    import re

    q = query.strip()

    # 1. 2025年6月 / 2025年06月 / 2025年六月 / 2025年-6月
    m = re.search(r"(\d{4})\s*年\s*-?\s*(?:(\d{1,2})|([一二三四五六七八九十]{1,2}))\s*月", q)
    if m:
        year = int(m.group(1))
        month = _cn_month_to_int(m.group(2) or m.group(3) or "")
        if month and 1 <= month <= 12:
            return f"{year}-{month:02d}"

    # 2. 2025-6月 / 2025-06月 / 2025-6 / 2025-06
    m2 = re.search(r"(\d{4})\s*-\s*(\d{1,2})\s*月?", q)
    if m2:
        year = int(m2.group(1))
        month = int(m2.group(2))
        if 1 <= month <= 12:
            return f"{year}-{month:02d}"

    # 3. 2025 年 6 月（空格分隔）
    m3 = re.search(r"(\d{4})\s+年\s+(\d{1,2})\s+月", q)
    if m3:
        year = int(m3.group(1))
        month = int(m3.group(2))
        if 1 <= month <= 12:
            return f"{year}-{month:02d}"

    return None


def _expand_long_term_query_dates(query: str) -> str:
    """Append ISO month tags so FTS can hit ## 2024-04 style headings."""
    mk = _parse_month_key(query)
    if not mk:
        return query
    year, month = mk.split("-")
    extras = [mk, f"{year}年{int(month)}月"]
    mo = int(month)
    inv = {v: k for k, v in _CN_MONTHS.items() if v <= 12}
    if mo in inv:
        extras.append(f"{inv[mo]}月")
    return query + " " + " ".join(extras)


def _build_long_term_query(query: str, working: list[dict], self_name: str | None) -> str:
    base = _enrich_long_term_query(query, working)
    if self_name and self_name not in base:
        base = f"{base} {self_name}"
    return _expand_long_term_query_dates(base)


def _has_content_entities(text: str) -> bool:
    """检测短消息是否已包含具体内容实体（地名、事件、专有名词等）。
    有实体 → 不需要从上下文扩充；无实体 → 可能是追问，需要扩充。
    """
    import re
    # 地名/场所词
    if re.search(r"[一-鿿]{2,}(?:山|路|街|公园|广场|学校|医院|公司|餐厅|店|城|海|湖|河|岛)", text):
        return True
    # 具体动作+地点
    if re.search(r"(?:去|在|到|从)(?:过|了)?[一-鿿]{2,6}", text):
        return True
    # 具体事物/事件描述
    if re.search(r"(?:吃|喝|买|看|玩|唱|爬|逛|旅游|旅行|考试|面试|搬家|结婚)", text):
        return True
    # 具体时间
    if re.search(r"\d+[年月日天号]", text):
        return True
    return False


def _empty_recall(working: list[dict], *, reason: str) -> dict:
    """构建空的召回结果字典（访客模式或无数据时使用）。

    返回语义 dict：history/items/diagnostics，不含旧 核心事实/近期记忆/长期记忆 字段。
    """
    return {
        "history": working,
        "items": [],
        "diagnostics": {
            "person_id": None,
            "guest_mode": True,
            "reason": reason,
            "has_recent": False,
            "has_long_term": False,
            "core_memory_count": 0,
            "retrieval_plan": {},
        },
        "memory_miss": False,
        "person_id": None,
        "guest_mode": True,
    }


class MemoryRouter:
    """记忆召回路由器：通过 UnifiedMemoryStore 协调语义检索。

    核心设计：
      - 通过 RetrievalPlanner 生成语义检索计划，调用方不直接关心内部存储层
      - 访客模式下跳过所有向量检索，仅返回 history
      - 长期记忆检索按需触发（RetrievalPlan 门控），避免无效检索和 embedding 浪费
    """

    def __init__(self, planner: RetrievalPlanner | None = None) -> None:
        self.planner = planner or RetrievalPlanner()

    def _recall_guest(self, working: list[dict]) -> dict:
        """访客模式召回：仅返回 工作上下文 当前会话消息。

        访客模式的核心节约：不调用 embedding API，不查询任何持久化存储，
        仅从当前会话的 messages 表读取最近消息。这是系统的"浅度模式"，
        适用于未实名用户和匿名访客。
        """
        payload = _empty_recall(working, reason="guest_working_context_only")
        payload["person_id"] = None
        return payload

    def recall(
        self,
        device_id: str,
        session_id: str,
        query: str,
        *,
        person_id: str | None = None,
    ) -> dict:
        """执行完整的多层记忆召回，返回语义 dict。

        召回流程：
          1. 获取当前会话消息（history）
          2. 判断是否为已实名用户：否 → 访客模式，仅返回 history
          3. 生成 RetrievalPlan + embedding
          4. 调用 unified_memory_store.search() 统一检索（含月份增强 + 关联扩展）
          5. 情感轨迹附加
          6. 返回语义 dict（history/items/diagnostics）

        Returns:
            语义 dict: {history, items, diagnostics, memory_miss, person_id, guest_mode}
        """
        # Step 1: 工作上下文 工作记忆
        working = get_recent_context(session_id)
        pid = str(person_id or "").strip()

        # Step 2: 访客判断 → 跳过所有嵌入检索
        if not memory_scoped_to_person(pid):
            return self._recall_guest(working)

        # Step 3: 生成语义检索计划 + embedding
        q = query.strip()
        plan = self.planner.plan(q, working=working)
        self_name = extract_self_name(q)
        long_term_query = _build_long_term_query(q, working, self_name)

        q_emb = None
        long_term_emb = None
        if q and plan.search_recent_memory and plan.search_long_term and long_term_query and long_term_query != q:
            embs = _batch_embed([q, long_term_query])
            q_emb = embs[0] if embs[0] else None
            long_term_emb = embs[1] if embs[1] else q_emb
        elif q and (plan.search_recent_memory or plan.search_long_term):
            q_emb = _cached_embed(q) or None
            long_term_emb = q_emb

        # Step 4: 通过 UnifiedMemoryStore 统一检索（含月份增强 + 关联扩展）
        persona_pid = str(getattr(settings, "persona_fact_person_id", "") or "").strip()
        search_spec = MemorySearchQuery(
            device_id=device_id,
            person_id=pid,
            query=q,
            q_emb=q_emb,
            long_term_query=long_term_query,
            long_term_emb=long_term_emb,
            include_recent=plan.search_recent_memory,
            include_long_term=plan.search_long_term,
            include_recent_episodes=plan.include_recent_episodes,
            include_related=True,
            recent_top_k=plan.recent_top_k,
            long_term_top_k=plan.long_term_top_k,
            month_key=plan.month_key,
            persona_person_id=persona_pid,
        )
        result = unified_memory_store.search(search_spec)

        # Step 5: memory_miss 判定
        needs_memory = plan.needs_memory
        has_recent = result.diagnostics.get("has_recent", False)
        has_long_term = result.diagnostics.get("has_long_term", False)
        has_related = bool(result.related_items)
        memory_miss = needs_memory and not has_recent and not has_long_term and not has_related

        # Step 7: 情感轨迹
        emo_traj = emotion_trajectory(device_id, pid)

        # Step 8: 汇总语义结果
        all_items = (
            result.core_items
            + result.recent_items
            + result.long_term_items
            + result.related_items
        )
        return {
            "history": working,
            "items": all_items,
            "diagnostics": result.diagnostics | {
                "person_id": pid,
                "guest_mode": False,
                "retrieval_plan": plan.__dict__ if hasattr(plan, "__dict__") else {},
                "emotion_trajectory": emo_traj,
            },
            "memory_miss": memory_miss,
            "person_id": pid,
            "guest_mode": False,
        }


    def recall_fast(
        self,
        device_id: str,
        session_id: str,
        query: str,
        *,
        person_id: str | None = None,
    ) -> dict:
        """快速召回 —— 仅核心事实+历史+最近情景，跳过长期检索、关系图和情感事件。

        用于低延迟语音场景的首响。
        """
        working = get_recent_context(session_id)
        pid = str(person_id or "").strip()

        if not memory_scoped_to_person(pid):
            return self._recall_guest(working)

        # 通过 UnifiedMemoryStore 快速检索（core + recent recent only）
        search_spec = MemorySearchQuery(
            device_id=device_id,
            person_id=pid,
            query=query.strip(),
            q_emb=None,
            include_recent=False,
            include_long_term=False,
            include_recent_episodes=True,
            include_related=False,
            recent_top_k=0,
            long_term_top_k=0,
        )
        result = unified_memory_store.search(search_spec)

        all_items = result.core_items + result.recent_items
        return {
            "history": working,
            "items": all_items,
            "diagnostics": result.diagnostics | {
                "person_id": pid,
                "guest_mode": False,
                "retrieval_plan": {},
                "recall_mode": "fast",
            },
            "memory_miss": False,
            "person_id": pid,
            "guest_mode": False,
        }


# 模块级单例，供 agent 模块调用
memory_router = MemoryRouter()
