"""Semantic retrieval with pgvector (cosine distance)."""

from __future__ import annotations

from collections.abc import Sequence

import psycopg
from pgvector import Vector

from app.retrieval.types import CHUNK_COLUMNS, RetrievedChunk


def vector_search(
    conn: psycopg.Connection,
    query_embedding: Sequence[float],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    """Nearest chunks by cosine distance (``<=>``).

    Cosine is used because gemini-embedding-001 vectors at reduced
    dimensionality are not unit-normalised, and cosine is scale-invariant.
    ``hnsw.ef_search`` is raised to at least ``top_k`` so the HNSW index can
    return the requested number of candidates.
    """
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(40, top_k * 2)),))
    rows = conn.execute(
        f"""
        SELECT {CHUNK_COLUMNS}, 1 - (c.embedding <=> %(q)s) AS similarity
        FROM chunks c JOIN documents d ON d.document_id = c.document_id
        WHERE d.status <> 'deleted'
        ORDER BY c.embedding <=> %(q)s, c.chunk_id
        LIMIT %(k)s
        """,
        {"q": Vector(list(query_embedding)), "k": top_k},
    ).fetchall()
    results: list[RetrievedChunk] = []
    for rank, row in enumerate(rows, start=1):
        chunk = RetrievedChunk.from_row(row)
        chunk.vector_rank = rank
        chunk.vector_similarity = float(row["similarity"])
        results.append(chunk)
    return results
