"""Semantic retrieval: pgvector cosine distance, or an in-app cosine fallback.

With pgvector (preferred) the database ranks chunks with ``<=>`` over an HNSW
index. Without pgvector (VECTOR_BACKEND=array) embeddings are stored as
``real[]`` in PostgreSQL and cosine similarity is computed here; the
embeddings are cached in memory and reloaded whenever the chunk table changes.
With numpy installed (default) a search over ~10,000 chunks takes a few
milliseconds; without numpy a pure-Python loop is used (slower, same results).
"""

from __future__ import annotations

import heapq
import math
import operator
import threading
from collections.abc import Sequence

import psycopg
from pgvector import Vector

try:  # numpy makes the array fallback fast; pure Python is used if it is missing
    import numpy as np
except ImportError:  # pragma: no cover - numpy is in requirements.txt
    np = None

from app.database.repository import uses_pgvector, vector_to_list
from app.retrieval.types import CHUNK_COLUMNS, RetrievedChunk


def vector_search(
    conn: psycopg.Connection,
    query_embedding: Sequence[float],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    """Nearest chunks by cosine similarity (best first).

    Cosine is used because gemini-embedding-001 vectors at reduced
    dimensionality are not unit-normalised, and cosine is scale-invariant.
    """
    if not uses_pgvector(conn):
        return _array_vector_search(conn, query_embedding, top_k=top_k)
    # hnsw.ef_search is raised to at least top_k so the HNSW index can return enough candidates.
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


# --------------------------------------------------------------------------- array fallback
class _EmbeddingMatrix:
    def __init__(self) -> None:
        self.signature: tuple | None = None
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []
        self.norms: list[float] = []
        self.matrix = None  # numpy (n, d) float32 array of unit-normalised rows
        self.lock = threading.Lock()


_matrix = _EmbeddingMatrix()


def _load_matrix(conn: psycopg.Connection) -> _EmbeddingMatrix:
    signature_row = conn.execute(
        """
        SELECT count(*) AS n, max(c.created_at) AS newest, max(d.updated_at) AS docs_changed
        FROM chunks c JOIN documents d ON d.document_id = c.document_id
        WHERE d.status <> 'deleted'
        """
    ).fetchone()
    signature = (signature_row["n"], signature_row["newest"], signature_row["docs_changed"])
    with _matrix.lock:
        if _matrix.signature == signature:
            return _matrix
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.embedding FROM chunks c JOIN documents d ON d.document_id = c.document_id
            WHERE d.status <> 'deleted'
            """
        ).fetchall()
        ids, vectors, norms = [], [], []
        for row in rows:
            vector = vector_to_list(row["embedding"])
            norm = math.sqrt(sum(x * x for x in vector))
            if norm == 0:
                continue
            ids.append(row["chunk_id"])
            vectors.append(vector)
            norms.append(norm)
        if np is not None and vectors:
            matrix = np.asarray(vectors, dtype=np.float32)
            matrix /= np.asarray(norms, dtype=np.float32)[:, None]
            _matrix.matrix, _matrix.vectors, _matrix.norms = matrix, [], []
        else:
            _matrix.matrix, _matrix.vectors, _matrix.norms = None, vectors, norms
        _matrix.ids, _matrix.signature = ids, signature
        return _matrix


def _array_vector_search(conn: psycopg.Connection, query_embedding: Sequence[float], *, top_k: int
                         ) -> list[RetrievedChunk]:
    query = [float(x) for x in query_embedding]
    query_norm = math.sqrt(sum(x * x for x in query))
    if query_norm == 0:
        return []
    matrix = _load_matrix(conn)
    if matrix.matrix is not None:
        scores = matrix.matrix @ (np.asarray(query, dtype=np.float32) / np.float32(query_norm))
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k] if k else []
        best = sorted(((float(scores[i]), matrix.ids[i]) for i in top), key=lambda item: (item[0], item[1]),
                      reverse=True)
    else:
        mul = operator.mul
        scored = (
            (sum(map(mul, vector, query)) / (norm * query_norm), chunk_id)
            for chunk_id, vector, norm in zip(matrix.ids, matrix.vectors, matrix.norms)
        )
        best = heapq.nlargest(top_k, scored, key=lambda item: (item[0], item[1]))
    if not best:
        return []
    similarity = {chunk_id: score for score, chunk_id in best}
    rows = conn.execute(
        f"""
        SELECT {CHUNK_COLUMNS}
        FROM chunks c JOIN documents d ON d.document_id = c.document_id
        WHERE c.chunk_id = ANY(%s)
        """,
        ([chunk_id for _, chunk_id in best],),
    ).fetchall()
    by_id = {row["chunk_id"]: row for row in rows}
    ordered = sorted((cid for cid in similarity if cid in by_id), key=lambda cid: (-similarity[cid], cid))
    results: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ordered, start=1):
        chunk = RetrievedChunk.from_row(by_id[chunk_id])
        chunk.vector_rank = rank
        chunk.vector_similarity = float(similarity[chunk_id])
        results.append(chunk)
    return results
