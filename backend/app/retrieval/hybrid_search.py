"""Hybrid retrieval: pgvector semantic search + PostgreSQL full-text search
(+ exact identifier matching), combined with Reciprocal Rank Fusion."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import psycopg

from app.config import Settings
from app.ingestion.embedder import Embedder
from app.retrieval.keyword_search import exact_search, keyword_search
from app.retrieval.ranking import RankedList, dedupe_by_content, reciprocal_rank_fusion
from app.retrieval.synonyms import keyword_query_with_synonyms
from app.retrieval.types import RetrievedChunk
from app.retrieval.vector_search import vector_search

EMPHASIS_WEIGHTS: dict[str, tuple[float, float]] = {
    # (vector weight, keyword weight)
    "balanced": (1.0, 1.0),
    "semantic": (1.0, 0.6),
    "keyword": (0.6, 1.0),
}
EXACT_WEIGHT = 1.2


class QueryEmbeddingCache:
    """Small in-process LRU so repeated queries do not re-call the embedding API."""

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self._data: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compute(self, embedder: Embedder, text: str) -> tuple[list[float], bool]:
        key = f"{embedder.config.model}|{embedder.config.dimensions}|{text}"
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key], True
        vector = embedder.embed_query(text)
        with self._lock:
            self._data[key] = vector
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)
        return vector, False


_query_cache = QueryEmbeddingCache()


@dataclass
class SearchResult:
    chunks: list[RetrievedChunk]
    lists: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    query_embeddings: int = 0
    query_embedding_cache_hits: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    duplicates_collapsed: int = 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "lists": self.lists,
            "fused": [c.brief() for c in self.chunks],
            "query_embeddings": self.query_embeddings,
            "query_embedding_cache_hits": self.query_embedding_cache_hits,
            "duplicates_collapsed": self.duplicates_collapsed,
            "timings_ms": self.timings_ms,
        }


def hybrid_search(
    conn: psycopg.Connection,
    embedder: Embedder | None,
    queries: Sequence[str],
    *,
    settings: Settings,
    exact_terms: Sequence[str] = (),
    phrases: Sequence[str] = (),
    emphasis: str = "balanced",
    mode: str = "hybrid",
    top_k_final: int | None = None,
) -> SearchResult:
    """Run every query through vector and keyword search, add an exact-term list,
    and fuse all lists with weighted RRF. ``mode`` may be ``vector`` or
    ``keyword`` for the /search debugging endpoint."""
    if mode not in {"hybrid", "vector", "keyword"}:
        raise ValueError("mode must be hybrid, vector or keyword")
    vector_weight, keyword_weight = EMPHASIS_WEIGHTS.get(emphasis, EMPHASIS_WEIGHTS["balanced"])
    use_vector = mode in {"hybrid", "vector"}
    use_keyword = mode in {"hybrid", "keyword"}
    if use_vector and embedder is None:
        raise ValueError("vector search requires an embedder")

    pool: dict[str, RetrievedChunk] = {}
    lists: list[RankedList] = []
    result = SearchResult(chunks=[])

    def remember(chunk: RetrievedChunk, query: str) -> RetrievedChunk:
        existing = pool.get(chunk.chunk_id)
        if existing is None:
            pool[chunk.chunk_id] = chunk
            existing = chunk
        else:
            for attr in ("vector_rank", "keyword_rank", "exact_rank"):
                new, old = getattr(chunk, attr), getattr(existing, attr)
                if new is not None and (old is None or new < old):
                    setattr(existing, attr, new)
            for attr in ("vector_similarity", "keyword_score", "exact_hits"):
                new, old = getattr(chunk, attr), getattr(existing, attr)
                if new is not None and (old is None or new > old):
                    setattr(existing, attr, new)
        if query and query not in existing.matched_queries:
            existing.matched_queries.append(query)
        return existing

    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
    for idx, query in enumerate(unique_queries):
        if use_vector:
            started = time.perf_counter()
            embedding, hit = _query_cache.get_or_compute(embedder, query)  # type: ignore[arg-type]
            result.query_embeddings += 0 if hit else 1
            result.query_embedding_cache_hits += 1 if hit else 0
            vec = vector_search(conn, embedding, top_k=settings.top_k_vector)
            result.timings_ms[f"vector_{idx}"] = round((time.perf_counter() - started) * 1000, 1)
            lists.append(RankedList(f"vector:{idx}", [c.chunk_id for c in vec], vector_weight))
            result.lists[f"vector:{query}"] = [
                {"rank": c.vector_rank, "chunk_id": c.chunk_id, "title": c.title, "heading_path": c.heading_path,
                 "similarity": round(c.vector_similarity or 0.0, 4)}
                for c in vec
            ]
            for c in vec:
                remember(c, query)
        if use_keyword:
            started = time.perf_counter()
            kw_query = (keyword_query_with_synonyms(conn, query, file_path=settings.resolved_synonyms_path)
                        if settings.synonym_expansion else query)
            kw = keyword_search(conn, kw_query, top_k=settings.top_k_keyword, ts_config=settings.text_search_config,
                                phrases=phrases)
            result.timings_ms[f"keyword_{idx}"] = round((time.perf_counter() - started) * 1000, 1)
            lists.append(RankedList(f"keyword:{idx}", [c.chunk_id for c in kw], keyword_weight))
            result.lists[f"keyword:{kw_query}"] = [
                {"rank": c.keyword_rank, "chunk_id": c.chunk_id, "title": c.title, "heading_path": c.heading_path,
                 "score": round(c.keyword_score or 0.0, 4)}
                for c in kw
            ]
            for c in kw:
                remember(c, query)

    exact = list(dict.fromkeys([*exact_terms, *phrases]))
    if use_keyword and exact and settings.top_k_exact > 0:
        started = time.perf_counter()
        ex = exact_search(conn, exact, top_k=settings.top_k_exact)
        result.timings_ms["exact"] = round((time.perf_counter() - started) * 1000, 1)
        lists.append(RankedList("exact", [c.chunk_id for c in ex], EXACT_WEIGHT))
        result.lists["exact:" + ", ".join(exact)] = [
            {"rank": c.exact_rank, "chunk_id": c.chunk_id, "title": c.title, "heading_path": c.heading_path,
             "hits": c.exact_hits}
            for c in ex
        ]
        for c in ex:
            remember(c, "")

    fused = reciprocal_rank_fusion(lists, k=settings.rrf_k, max_rank=settings.rrf_max_rank or None)
    ordered_ids = [item.chunk_id for item in fused]
    kept, dropped = dedupe_by_content(ordered_ids, {cid: pool[cid].content_hash for cid in ordered_ids})
    result.duplicates_collapsed = sum(len(v) for v in dropped.values())
    scores = {item.chunk_id: item.score for item in fused}
    limit = top_k_final or settings.top_k_final
    for chunk_id in kept[:limit]:
        chunk = pool[chunk_id]
        chunk.rrf_score = scores[chunk_id]
        chunk.duplicate_of = dropped.get(chunk_id, [])
        result.chunks.append(chunk)
    return result
