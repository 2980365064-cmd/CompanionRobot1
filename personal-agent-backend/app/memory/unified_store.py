"""
统一存储门面（UnifiedMemoryStore）—— 统一记忆库读写的唯一入口。

============================================================================
设计目标：
  对外只表达"核心事实 / 近期事件 / 长期记忆 / 关联记忆 / 状态寄存器"，
  所有读写通过 memory_items 统一表完成，不感知底层物理存储细节。

读写路径：
  - 主写：memory_items 统一表（唯一写入路径）
  - 主读：memory_items 统一表（FTS5 + embedding cosine rerank）
============================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.memory.schema import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryVisibility,
)
from app.session import store

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 查询与结果数据结构
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MemorySearchQuery:
    """一次统一记忆检索的语义查询规格。

    所有字段用产品语义命名。
    """
    device_id: str
    person_id: str
    query: str
    q_emb: list[float] | None = None
    long_term_query: str = ""
    long_term_emb: list[float] | None = None
    include_recent: bool = False
    include_long_term: bool = False
    include_recent_episodes: bool = False
    include_related: bool = True
    recent_top_k: int = 0
    long_term_top_k: int = 0
    month_key: str = ""
    persona_person_id: str = ""  # persona 语料 person_id，用于 FTS 跨库补召


@dataclass
class MemorySearchResult:
    """统一记忆检索结果。

    所有记忆以 MemoryItem 列表返回。
    diagnostics 保留 raw evidence 但 key 必须语义化。
    """
    core_items: list[MemoryItem] = field(default_factory=list)
    recent_items: list[MemoryItem] = field(default_factory=list)
    long_term_items: list[MemoryItem] = field(default_factory=list)
    related_items: list[MemoryItem] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# UnifiedMemoryStore
# ══════════════════════════════════════════════════════════════════════════════


def _prioritize_month_memory_items(row: dict, *, query: str, month_key: str) -> tuple[int, int, str]:
    """memory_items 月份查询排序。

    月份查询的 FTS/month_key 命中是硬证据，不应被 embedding rerank 过滤掉。
    这里仅做意图排序：关系查询优先刘远慧关系月度，朋友查询优先朋友/唐凯事件。
    """
    import re as _re

    q = str(query or "")
    content = str(row.get("content", "") or "")
    source_table = str(row.get("source_table", "") or "")
    source_id = str(row.get("source_id", "") or "")
    context = str(row.get("context_json", "") or "")
    haystack = f"{content} {source_table} {source_id} {context}"

    asks_relationship = bool(_re.search(
        r"我俩|我们俩|我们之间|咱俩|我和她|我和远慧|我跟你|远慧|刘远慧|秋雨",
        q,
    ))
    asks_friend = bool(_re.search(r"唐凯|伍钰涛|朋友群|兄弟们|朋友", q))
    if asks_relationship and asks_friend:
        asks_friend = False

    is_target_month = month_key in haystack
    is_relationship = "刘远慧" in haystack or "远慧" in haystack or "monthly/liu_yuanhui" in haystack
    is_friend = any(token in haystack for token in ("唐凯", "伍钰涛", "袁子翔", "朋友群", "monthly/friends_group"))
    is_corpus_source = source_table == "corpus"

    if asks_relationship:
        group = 0 if is_relationship else 1 if is_friend else 2
    elif asks_friend:
        group = 0 if is_friend else 1 if is_relationship else 2
    else:
        group = 0 if is_relationship else 1 if is_friend else 2

    # 目标月份优先；同组内 corpus 语料比会话摘要更能回答"发生了什么"。
    month_rank = 0 if is_target_month else 1
    table_rank = 0 if is_corpus_source else 1
    return (month_rank, group, f"{table_rank}:{source_id}")


class UnifiedMemoryStore:
    """统一长期记忆读写门面。

    对外只表达 MemoryItem 语义，内部适配 memory_items 物理表。
    """

    # ── 读取：核心事实 ──────────────────────────────────────────────────

    def load_core_items(self, person_id: str) -> list[MemoryItem]:
        """加载指定用户的全量核心事实（从 memory_items 统一表读取）。"""
        rows = store.search_memory_items(person_id, kinds=None, visibility="always", limit=50)
        items: list[MemoryItem] = []
        for row in rows:
            try:
                items.append(self._row_to_memory_item(row))
            except Exception:
                continue
        return items

    # ── 读取：统一语义检索 ──────────────────────────────────────────────

    def _row_to_memory_item(self, row: dict) -> MemoryItem:
        """将 memory_items 行转换为 MemoryItem。"""
        context: dict = {}
        try:
            context = json.loads(row.get("context_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass
        raw_source = str(row.get("source", "") or "").strip()
        source_value = raw_source or "inferred"
        if source_value not in {s.value for s in MemorySource}:
            if source_value.lower().endswith((".md", ".txt")) or "/" in source_value:
                source_value = MemorySource.WIKI.value
                context.setdefault("source_path", raw_source)
            else:
                source_value = MemorySource.INFERRED.value
        source_table = str(row.get("source_table", "") or "")
        source_id = str(row.get("source_id", "") or "")
        source_id_str = ""
        if source_table and source_id:
            source_id_str = f"{source_table}:{source_id}"
        elif row.get("id"):
            source_id_str = f"mi:{row['id']}"
        return MemoryItem(
            kind=MemoryKind(row.get("kind", "fact")),
            source=MemorySource(source_value),
            confidence=float(row.get("confidence", 1.0)),
            emotional_weight=int(row.get("emotional_weight", 3)),
            recency_weight=int(row.get("recency_weight", 3)),
            visibility=MemoryVisibility(row.get("visibility", "recall_only")),
            content=str(row.get("content", "")),
            context=context,
            source_id=source_id_str,
            created_at=str(row.get("created_at", "")),
            expires_at=str(row.get("expires_at", "") or None),
        )

    def _compute_evidence(self, result: MemorySearchResult, *, month_key: str = "") -> dict:
        """从 MemorySearchResult 计算 evidence 诊断。

        Args:
            result:     检索结果
            month_key:  月份查询标识（如 "2025-06"），用于判断是否命中目标月份

        Returns:
            dict: 包含 evidence_count / evidence_weak / evidence_sources /
                  query_supported / top_memory_sources / top_memory_samples
        """
        evidence_sources: list[str] = []
        if result.core_items:
            evidence_sources.append("core")
        if result.recent_items:
            evidence_sources.append("recent")
        if result.long_term_items:
            evidence_sources.append("long_term")
        if result.related_items:
            evidence_sources.append("related")
        evidence_count = (
            len(result.core_items)
            + len(result.recent_items)
            + len(result.long_term_items)
            + len(result.related_items)
        )

        # ── query_supported：只有命中目标月份的 long_term/episode 才算支持 ──
        query_supported = True  # 默认为 True，无月份查询时保持兼容
        if month_key:
            # 检查 long_term/episode 中是否有 target 月份内容
            has_target_month = False
            for item in result.long_term_items:
                content = str(item.content or "")
                source_id = str(item.source_id or "")
                source = str(item.source.value if hasattr(item.source, 'value') else item.source)
                context = item.context or {}
                if (month_key in content or month_key in source_id
                        or month_key in source or month_key in str(context)):
                    has_target_month = True
                    break
            if not has_target_month:
                for item in result.recent_items:
                    content = str(item.content or "")
                    source_id = str(item.source_id or "")
                    if month_key in content or month_key in source_id:
                        has_target_month = True
                        break
            query_supported = has_target_month

        # ── evidence_weak：月份查询时，只有 core + 旧摘要但无目标月份内容 → weak ──
        evidence_weak = False
        if month_key and not query_supported:
            # 月份查询但未命中目标月份 → 弱证据
            evidence_weak = True
        elif evidence_count == 0:
            evidence_weak = True
        elif len(evidence_sources) == 1 and evidence_count <= 1:
            evidence_weak = True
        elif evidence_sources == ["related"]:
            evidence_weak = True

        # ── top_memory_sources / top_memory_samples ──
        top_memory_sources: list[str] = []
        if result.long_term_items:
            top_memory_sources.append("long_term")
        if result.recent_items:
            top_memory_sources.append("recent")
        if result.core_items:
            top_memory_sources.append("core")
        if result.related_items:
            top_memory_sources.append("related")

        top_memory_samples: list[str] = []
        for item in result.long_term_items[:3]:
            content = str(item.content or "")[:120]
            if content:
                top_memory_samples.append(content)
        if len(top_memory_samples) < 3:
            for item in result.recent_items[:3]:
                content = str(item.content or "")[:120]
                if content and content not in top_memory_samples:
                    top_memory_samples.append(content)
        if len(top_memory_samples) < 3:
            for item in result.core_items[:3]:
                content = str(item.content or "")[:120]
                if content and content not in top_memory_samples:
                    top_memory_samples.append(content)

        return {
            "evidence_count": evidence_count,
            "evidence_weak": evidence_weak,
            "evidence_sources": evidence_sources,
            "query_supported": query_supported,
            "top_memory_sources": top_memory_sources,
            "top_memory_samples": top_memory_samples,
        }

    def _search_memory_items(self, spec: MemorySearchQuery) -> MemorySearchResult:
        """从统一记忆库检索 —— 完全基于 memory_items 表 + FTS5。

        不依赖任何旧表适配器，关联记忆扩展通过 memory_relations 表实现。
        """
        result = MemorySearchResult()
        diag: dict[str, Any] = {
            "person_id": spec.person_id,
            "has_recent": False,
            "has_long_term": False,
            "core_memory_count": 0,
            "month_key": spec.month_key,
            "read_path": "memory_items",
            "recent": [],
            "long_term": [],
            "related": [],
        }

        # ── 1. 核心事实：visibility=always ──
        core_rows = store.search_memory_items(
            spec.person_id,
            kinds=None,  # 全部 kind
            visibility="always",
            limit=50,
        )
        # 按优先级排序：preference/taboo/relationship/entity/milestone
        _CORE_PRIORITY = {
            "taboo": 0, "preference": 1, "relationship": 2,
            "entity": 3, "milestone": 4,
        }
        core_rows.sort(key=lambda r: _CORE_PRIORITY.get(r.get("kind", ""), 9))
        for row in core_rows:
            result.core_items.append(self._row_to_memory_item(row))
        diag["core_memory_count"] = len(core_rows)

        # ── 2. 情景记忆检索 ──
        if spec.include_recent and spec.recent_top_k > 0:
            epi_rows = store.search_memory_items(
                spec.person_id,
                kinds=["episode", "emotion"],
                visibility="recall_only",
                query=spec.query,
                limit=spec.recent_top_k * 2,
            )
            for row in epi_rows[:spec.recent_top_k]:
                result.recent_items.append(self._row_to_memory_item(row))
            diag["has_recent"] = bool(epi_rows)
            diag["recent"] = [
                {"text": r.get("content", ""), "source": r.get("source", ""),
                 "kind": r.get("kind", ""), "created_at": r.get("created_at", ""),
                 "content_hash": r.get("content_hash", "")}
                for r in epi_rows[:spec.recent_top_k]
            ]
        elif spec.include_recent_episodes:
            recent_rows = store.search_memory_items(
                spec.person_id,
                kinds=["episode", "emotion"],
                visibility="recall_only",
                query="",
                limit=settings.recent_memory_recall_recent,
            )
            for row in recent_rows:
                result.recent_items.append(self._row_to_memory_item(row))
            diag["has_recent"] = bool(recent_rows)
            diag["recent"] = [
                {"text": r.get("content", ""), "source": r.get("source", ""),
                 "kind": r.get("kind", ""), "created_at": r.get("created_at", "")}
                for r in recent_rows
            ]

        # ── 3. 长期记忆检索 ──
        if spec.include_long_term and spec.long_term_top_k > 0:
            long_term_query = spec.long_term_query or spec.query
            long_term_emb = spec.long_term_emb or spec.q_emb
            lt_kinds = ["fact", "episode", "emotion", "entity", "wiki", "correction", "milestone"]
            lt_emb_json = ""
            if long_term_emb:
                lt_emb_json = json.dumps(long_term_emb)
            lt_rows = store.search_memory_items(
                spec.person_id,
                kinds=lt_kinds,
                visibility="recall_only",
                query=long_term_query,
                month_key=spec.month_key,
                limit=spec.long_term_top_k * 3,
                embedding_json=lt_emb_json,
                extra_person_ids=([spec.persona_person_id] if spec.persona_person_id and spec.persona_person_id != spec.person_id else None),
            )

            if lt_rows and spec.month_key:
                lt_rows.sort(key=lambda r: _prioritize_month_memory_items(
                    r, query=long_term_query, month_key=spec.month_key,
                ))
            # 如果提供了 query embedding 且有 FTS 命中的候选，做 cosine rerank。
            # 月份查询已在上方按目标月份/查询意图排序，避免硬证据被低相似度误过滤。
            elif lt_rows and long_term_emb:
                from app.llm import cosine_similarity
                scored: list[tuple[dict, float]] = []
                for row in lt_rows:
                    emb_json = str(row.get("embedding_json", "[]") or "[]")
                    if emb_json and emb_json != "[]":
                        try:
                            row_emb = json.loads(emb_json)
                            if row_emb and len(row_emb) > 0 and len(long_term_emb) > 0:
                                sim = cosine_similarity(long_term_emb, row_emb)
                                scored.append((row, sim))
                            else:
                                scored.append((row, 0.0))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            scored.append((row, 0.0))
                    else:
                        # 无 embedding 时按默认顺序
                        scored.append((row, 0.0))
                scored.sort(key=lambda x: -x[1])
                lt_thresh = settings.long_term_memory_sim_threshold
                lt_rows = [r for r, s in scored[:spec.long_term_top_k * 2]
                           if s >= lt_thresh or s > 0]

            for row in lt_rows[:spec.long_term_top_k]:
                result.long_term_items.append(self._row_to_memory_item(row))
            diag["has_long_term"] = bool(lt_rows)
            diag["long_term"] = [
                {"text": r.get("content", ""), "kind": r.get("kind", ""),
                 "content_hash": r.get("content_hash", "")}
                for r in lt_rows[:10]
            ]

        # ── 4. 关联记忆扩展 ──
        # 从 long_term_items 中提取关联键，用于关联记忆扩展
        relation_keys: list[str] = []
        for item in result.long_term_items:
            sid = str(item.source_id or "")
            if sid.startswith("corpus:") or sid.startswith("long_term:"):
                cid = sid.split(":", 1)[1]
                relation_keys.append(f"memory:{cid}")
        if relation_keys:
            try:
                min_strength = getattr(settings, "memory_relation_min_strength", 0.6)
                related_matches = store.get_memory_relations(
                    relation_keys, min_strength=min_strength, limit=12,
                )
                # 去重：避免与 long_term_items 内容重复
                lt_texts = {it.content for it in result.long_term_items}
                seen_rel_keys: set[str] = set()
                for rel in related_matches:
                    to_key = str(rel.get("to_id", "") or "")
                    from_key = str(rel.get("from_id", "") or "")
                    target_key = to_key if to_key not in relation_keys else from_key
                    if target_key in seen_rel_keys:
                        continue
                    seen_rel_keys.add(target_key)
                    # 从 target_key 反查内容（memory_items 表）
                    # 统一按 memory:<uuid> 格式解析
                    rel_content = ""
                    if ":" in target_key:
                        cid = target_key.split(":", 1)[1]
                        row = store.get_memory_item(cid)
                    else:
                        row = None
                    if row:
                        rel_content = str(row.get("content", ""))
                    if rel_content and rel_content not in lt_texts and rel_content.strip():
                        result.related_items.append(MemoryItem(
                            kind=MemoryKind.FACT,
                            source=MemorySource.INFERRED,
                            confidence=0.7,
                            emotional_weight=3,
                            visibility=MemoryVisibility.RECALL_ONLY,
                            content=rel_content[:200],
                            context={
                                "relation_type": str(rel.get("relation_type", "related")),
                                "strength": float(rel.get("strength", 0.5)),
                            },
                        ))
                diag["related"] = [
                    {"text": it.content, "relation_type": it.context.get("relation_type", "related")}
                    for it in result.related_items
                ]
            except Exception as exc:
                logger.warning("memory_items related expansion failed: %s", exc)
                diag["related"] = []
        else:
            diag["related"] = []

        # ── 5. Evidence 诊断 ──
        ev = self._compute_evidence(result, month_key=spec.month_key)
        diag.update(ev)
        result.diagnostics = diag
        return result

    def search(self, spec: MemorySearchQuery) -> MemorySearchResult:
        """执行统一语义检索（仅 memory_items 路径）。"""
        return self._search_memory_items(spec)

    # ── 写入：统一 MemoryItem 路由 ─────────────────────────────────────

    def write_item(
        self,
        device_id: str,
        person_id: str,
        item: MemoryItem,
        *,
        source_session: str = "",
    ) -> str:
        """写入一条记忆到统一存储。

        写入 memory_items 统一表。

        Returns:
            写入确认标识（memory_items ID 或 "mi:{content_hash}" 或 ""）。

        路由规则：
          - TABOO/PREFERENCE/IDENTITY/MILESTONE + visibility=ALWAYS → core 类
          - EPISODE/EMOTION → episode 类
          - FACT/ENTITY/WIKI/CORRECTION → long_term 类
        """
        pid = str(person_id or "").strip()
        did = str(device_id or "").strip()
        if not pid or not item.content.strip():
            return ""

        kind = item.kind
        content = item.content.strip()

        # ── 步骤 1：先写 memory_items 统一表（主写路径）──
        import hashlib
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        is_core = kind in (MemoryKind.TABOO, MemoryKind.PREFERENCE,
                           MemoryKind.MILESTONE, MemoryKind.ENTITY, MemoryKind.RELATIONSHIP) \
                 and item.visibility == MemoryVisibility.ALWAYS

        try:
            if is_core:
                # core 类直接通过 memory_item_upsert 写入，source_table 标记 core
                mi_id = store.write_memory_item(
                    person_id=pid,
                    device_id=did,
                    kind=item.kind.value if isinstance(item.kind, MemoryKind) else str(item.kind),
                    source=item.source.value if isinstance(item.source, MemorySource) else str(item.source),
                    visibility=item.visibility.value if isinstance(item.visibility, MemoryVisibility) else str(item.visibility),
                    content=content,
                    confidence=item.confidence,
                    emotional_weight=item.emotional_weight,
                    recency_weight=item.recency_weight,
                    context_json=json.dumps(item.context, ensure_ascii=False),
                    tags_json=json.dumps(item.tags),
                    source_table="core",
                    source_id=f"mi:{content_hash}",
                    source_session=source_session,
                )
            else:
                # long_term/episode 类
                mi_id = store.write_memory_item(
                    person_id=pid,
                    device_id=did,
                    kind=item.kind.value if isinstance(item.kind, MemoryKind) else str(item.kind),
                    source=item.source.value if isinstance(item.source, MemorySource) else str(item.source),
                    visibility=item.visibility.value if isinstance(item.visibility, MemoryVisibility) else str(item.visibility),
                    content=content,
                    confidence=item.confidence,
                    emotional_weight=item.emotional_weight,
                    recency_weight=item.recency_weight,
                    context_json=json.dumps(item.context, ensure_ascii=False),
                    tags_json=json.dumps(item.tags),
                    source_table="long_term",
                    source_id=f"mi:{content_hash}",
                    source_session=source_session,
                )

            if not mi_id:
                logger.warning("memory_item_upsert returned empty for %s", content[:60])
                return ""
        except Exception as exc:
            logger.warning("UnifiedMemoryStore.write_item memory_items write failed: %s", exc)
            return ""

        return mi_id or f"mi:{content_hash}"


# 模块级单例
unified_memory_store = UnifiedMemoryStore()
