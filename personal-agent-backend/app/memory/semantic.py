"""L3 semantic memory: Elasticsearch hybrid retrieval + re-ranking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from app.config import settings
from app.embed_meta import load_embed_meta, save_embed_meta
from app.llm import cosine_similarity, embed_texts
from app.session import store

logger = logging.getLogger(__name__)


@dataclass
class _CollectionCounter:
    """Compatibility proxy: keeps `.count()` API used by scripts."""

    get_count: Any

    def count(self) -> int:
        return int(self.get_count())


def _stored_dim(collection) -> int | None:
    if hasattr(collection, "vector_dim"):
        dim = getattr(collection, "vector_dim")
        if isinstance(dim, int) and dim > 0:
            return dim
    meta = load_embed_meta()
    if meta:
        return int(meta.get("dim", 0)) or None
    return None


class SemanticMemory:
    def __init__(self) -> None:
        self._es_enabled = False
        self._vector_dim: int | None = None
        self._index_prefix = settings.es_index_prefix.strip() or "sparkbot"
        self._corpus_index = f"{self._index_prefix}-corpus"
        self._facts_index = f"{self._index_prefix}-facts"

        self.client = self._build_es_client()
        self._es_enabled = self.client is not None

        # compatibility for existing scripts
        self.corpus = _CollectionCounter(lambda: self._count_index(self._corpus_index))
        self.facts = _CollectionCounter(lambda: self._count_index(self._facts_index))
        setattr(self.corpus, "vector_dim", self._vector_dim)
        setattr(self.facts, "vector_dim", self._vector_dim)

    def _build_es_client(self) -> Elasticsearch | None:
        try:
            kwargs: dict[str, Any] = {
                "hosts": [settings.es_url],
                "request_timeout": settings.es_timeout_sec,
            }
            if settings.es_api_key:
                kwargs["api_key"] = settings.es_api_key
            elif settings.es_username and settings.es_password:
                kwargs["basic_auth"] = (settings.es_username, settings.es_password)
            client = Elasticsearch(**kwargs)
            if not client.ping():
                logger.warning("Elasticsearch unreachable at %s", settings.es_url)
                return None
            return client
        except Exception as exc:
            logger.warning("Elasticsearch init failed: %s", exc)
            return None

    def _count_index(self, index_name: str) -> int:
        if not self._es_enabled or self.client is None:
            return 0
        try:
            if not self.client.indices.exists(index=index_name):
                return 0
            return int(self.client.count(index=index_name).get("count", 0))
        except Exception as exc:
            logger.warning("ES count failed index=%s: %s", index_name, exc)
            return 0

    def _ensure_index(self, index_name: str, *, dim: int) -> None:
        if not self._es_enabled or self.client is None:
            return
        if self.client.indices.exists(index=index_name):
            return
        mapping = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "standard"}}},
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "device_id": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "confidence": {"type": "float"},
                    "embedding": {"type": "dense_vector", "dims": dim, "index": False},
                }
            },
        }
        self.client.indices.create(index=index_name, body=mapping)
        self._vector_dim = dim
        setattr(self.corpus, "vector_dim", dim)
        setattr(self.facts, "vector_dim", dim)

    def reset_all(self) -> None:
        if not self._es_enabled or self.client is None:
            return
        for name in (self._corpus_index, self._facts_index):
            try:
                self.client.indices.delete(index=name)
            except Exception:
                pass

    def reset_corpus(self) -> None:
        if not self._es_enabled or self.client is None:
            return
        try:
            self.client.indices.delete(index=self._corpus_index)
        except Exception:
            pass

    def ingest_chunks(self, chunks: list[dict], *, reset: bool = False) -> int:
        if not chunks:
            return 0
        if not self._es_enabled or self.client is None:
            logger.warning("ES unavailable, skip corpus ingest")
            return 0
        if reset:
            self.reset_all()

        ids = [str(c["id"]) for c in chunks]
        docs = [c["text"] for c in chunks]
        embs = embed_texts(docs)
        if not embs:
            return 0
        qdim = len(embs[0])
        self._ensure_index(self._corpus_index, dim=qdim)
        self._ensure_index(self._facts_index, dim=qdim)

        actions = []
        for i, chunk in enumerate(chunks):
            meta = chunk.get("meta", {})
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._corpus_index,
                    "_id": ids[i],
                    "_source": {
                        "text": docs[i],
                        "source": str(meta.get("source", "corpus")),
                        "embedding": embs[i],
                    },
                }
            )
        bulk(self.client, actions, refresh=True)
        save_embed_meta()
        return len(chunks)

    def _es_hybrid_search(
        self,
        *,
        index_name: str,
        query: str,
        q_emb: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[tuple[float, str]]:
        if not self._es_enabled or self.client is None:
            return []
        if self._count_index(index_name) <= 0:
            return []

        filters = [{"term": {k: v}} for k, v in (where or {}).items()]
        keyword_query: dict[str, Any]
        if query.strip():
            keyword_query = {
                "bool": {
                    "must": [{"match": {"text": {"query": query, "operator": "and"}}}],
                    "filter": filters,
                }
            }
        else:
            keyword_query = {"bool": {"must": [{"match_all": {}}], "filter": filters}}

        vector_query = {
            "script_score": {
                "query": {"bool": {"filter": filters or [{"match_all": {}}]}},
                "script": {
                    "source": "cosineSimilarity(params.q, 'embedding') + 1.0",
                    "params": {"q": q_emb},
                },
            }
        }

        try:
            kw_resp = self.client.search(
                index=index_name,
                query=keyword_query,
                size=max(settings.es_keyword_candidates, top_k),
                source=["text", "embedding"],
            )
            vec_resp = self.client.search(
                index=index_name,
                query=vector_query,
                size=max(settings.es_vector_candidates, top_k),
                source=["text", "embedding"],
            )
        except Exception as exc:
            logger.warning("ES hybrid search failed index=%s: %s", index_name, exc)
            return []

        rrf_scores: dict[str, float] = {}
        payload: dict[str, dict[str, Any]] = {}
        for hits, base in ((kw_resp.get("hits", {}).get("hits", []), 1.0), (vec_resp.get("hits", {}).get("hits", []), 1.25)):
            for rank, hit in enumerate(hits, start=1):
                hid = str(hit.get("_id", rank))
                src = hit.get("_source", {})
                text = str(src.get("text", "")).strip()
                if not text:
                    continue
                payload[hid] = src
                rrf_scores[hid] = rrf_scores.get(hid, 0.0) + (base / (50.0 + rank))

        candidates = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        rerank_n = max(top_k, settings.es_rerank_top_n)
        reranked: list[tuple[float, str]] = []
        for hid, rrf in candidates[:rerank_n]:
            src = payload.get(hid, {})
            text = str(src.get("text", "")).strip()
            emb = src.get("embedding") or []
            sim = 0.0
            if emb:
                try:
                    sim = max(0.0, cosine_similarity(q_emb, emb))
                except Exception:
                    sim = 0.0
            final_score = (0.72 * sim) + (0.28 * min(rrf * 100.0, 1.0))
            if final_score >= settings.es_min_recall_score:
                reranked.append((final_score, text))
        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k]

    def recall(self, device_id: str, query: str, top_k: int) -> list[str]:
        if not query.strip() or top_k <= 0:
            return []
        results: list[tuple[float, str]] = []
        q_emb = embed_texts([query])[0]

        for score, doc in self._es_hybrid_search(
            index_name=self._corpus_index,
            query=query,
            q_emb=q_emb,
            top_k=top_k,
        ):
            results.append((score, doc))

        for score, doc in self._es_hybrid_search(
            index_name=self._facts_index,
            query=query,
            q_emb=q_emb,
            top_k=top_k,
            where={"device_id": device_id},
        ):
            results.append((score + 0.03, doc))

        for row in store.list_facts(device_id, limit=30):
            fact = row["fact"]
            try:
                score = cosine_similarity(q_emb, embed_texts([fact])[0])
            except Exception:
                continue
            if score > max(0.32, settings.es_min_recall_score):
                results.append((score, fact))

        results.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        out: list[str] = []
        for _, text in results:
            if text not in seen:
                seen.add(text)
                out.append(text)
            if len(out) >= top_k:
                break
        return out

    def add_fact(self, device_id: str, fact: str, category: str, confidence: float, source_session: str) -> None:
        if confidence < 0.6 or len(fact.strip()) < 4:
            return
        store.add_fact(device_id, fact, category, confidence, source_session)
        if not self._es_enabled or self.client is None:
            return
        emb = embed_texts([fact])[0]
        self._ensure_index(self._facts_index, dim=len(emb))
        fid = f"{device_id}-{abs(hash((fact, category))) % 10**10}"
        try:
            self.client.index(
                index=self._facts_index,
                id=fid,
                document={
                    "text": fact,
                    "device_id": device_id,
                    "category": category,
                    "confidence": float(confidence),
                    "source": source_session,
                    "embedding": emb,
                },
                refresh=True,
            )
        except Exception as exc:
            logger.warning("ES fact upsert failed: %s", exc)


semantic_memory = SemanticMemory()
