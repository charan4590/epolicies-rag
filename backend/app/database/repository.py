"""Data access layer. Every query is parameterised; no caller-supplied SQL is
ever executed."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from pgvector import Vector
from psycopg.types.json import Jsonb

# Session-level advisory lock key that serialises ingestion runs (CLI + API).
INGESTION_LOCK_KEY = 0x4550_4F4C  # "EPOL"


# --------------------------------------------------------------------------- helpers
def vector_to_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, Vector):
        return [float(x) for x in value.to_list()]
    return [float(x) for x in value]


def uses_pgvector(conn: psycopg.Connection) -> bool:
    """True when the pgvector type is registered on this connection (pgvector backend)."""
    return conn.adapters.types.get("vector") is not None


def embedding_param(conn: psycopg.Connection, values: list[float]):
    """Bind an embedding as pgvector `vector` or, in the array fallback, as a float array."""
    return Vector(values) if uses_pgvector(conn) else [float(v) for v in values]


@dataclass
class ChunkRecord:
    """A chunk ready to be written: everything except the DB-computed tsvector."""

    chunk_id: str
    chunk_index: int
    text: str
    heading_path: list[str]
    source_urls: list[str]
    metadata: dict[str, Any]
    content_hash: str
    token_count: int
    char_count: int
    embedding: list[float]
    embedding_model: str
    embedding_config: str


@dataclass
class DocumentRecord:
    document_id: str
    relative_path: str
    filename: str
    title: str
    content_hash: str
    source_url: str | None
    size_bytes: int
    encoding: str | None
    file_modified_at: datetime | None
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- documents
def get_all_documents(conn: psycopg.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT document_id, relative_path, title, content_hash, source_hash, status, chunk_count,
               embedding_config, failure_count, last_indexed_at, metadata->>'chunker_version' AS chunker_version
        FROM documents
        """
    ).fetchall()
    return {r["document_id"]: r for r in rows}


_TSV_SQL = (
    "setweight(to_tsvector(%(cfg)s::regconfig, %(title)s), 'A') || "
    "setweight(to_tsvector(%(cfg)s::regconfig, %(headings)s), 'B') || "
    "setweight(to_tsvector(%(cfg)s::regconfig, %(text)s), 'C')"
)


def replace_document(
    conn: psycopg.Connection,
    doc: DocumentRecord,
    chunks: Sequence[ChunkRecord],
    *,
    embedding_config: str,
    text_search_config: str,
) -> None:
    """Atomically (caller's transaction) write a document and replace all its chunks."""
    conn.execute(
        """
        INSERT INTO documents (document_id, relative_path, filename, title, content_hash, source_hash,
                               source_url, size_bytes, encoding, status, chunk_count, embedding_config,
                               metadata, last_error, failure_count, file_modified_at, last_indexed_at,
                               deleted_at, updated_at)
        VALUES (%(document_id)s, %(relative_path)s, %(filename)s, %(title)s, %(content_hash)s, %(content_hash)s,
                %(source_url)s, %(size_bytes)s, %(encoding)s, 'indexed', %(chunk_count)s, %(embedding_config)s,
                %(metadata)s, NULL, 0, %(file_modified_at)s, now(), NULL, now())
        ON CONFLICT (document_id) DO UPDATE SET
            relative_path = EXCLUDED.relative_path, filename = EXCLUDED.filename, title = EXCLUDED.title,
            content_hash = EXCLUDED.content_hash, source_hash = EXCLUDED.source_hash,
            source_url = EXCLUDED.source_url, size_bytes = EXCLUDED.size_bytes, encoding = EXCLUDED.encoding,
            status = 'indexed', chunk_count = EXCLUDED.chunk_count, embedding_config = EXCLUDED.embedding_config,
            metadata = EXCLUDED.metadata, last_error = NULL, failure_count = 0,
            file_modified_at = EXCLUDED.file_modified_at, last_indexed_at = now(), deleted_at = NULL,
            updated_at = now()
        """,
        {
            "document_id": doc.document_id,
            "relative_path": doc.relative_path,
            "filename": doc.filename,
            "title": doc.title,
            "content_hash": doc.content_hash,
            "source_url": doc.source_url,
            "size_bytes": doc.size_bytes,
            "encoding": doc.encoding,
            "chunk_count": len(chunks),
            "embedding_config": embedding_config,
            "metadata": Jsonb(doc.metadata),
            "file_modified_at": doc.file_modified_at,
        },
    )
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc.document_id,))
    if not chunks:
        return
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO chunks (chunk_id, document_id, chunk_index, text, heading_path, source_urls, metadata,
                                content_hash, token_count, char_count, embedding, embedding_model,
                                embedding_config, search_tsv)
            VALUES (%(chunk_id)s, %(document_id)s, %(chunk_index)s, %(text)s, %(heading_path)s, %(source_urls)s,
                    %(metadata)s, %(content_hash)s, %(token_count)s, %(char_count)s, %(embedding)s,
                    %(embedding_model)s, %(embedding_config)s, {_TSV_SQL})
            """,
            [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": doc.document_id,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "heading_path": list(c.heading_path),
                    "source_urls": list(c.source_urls),
                    "metadata": Jsonb(c.metadata),
                    "content_hash": c.content_hash,
                    "token_count": c.token_count,
                    "char_count": c.char_count,
                    "embedding": embedding_param(conn, c.embedding),
                    "embedding_model": c.embedding_model,
                    "embedding_config": c.embedding_config,
                    "cfg": text_search_config,
                    "title": doc.title,
                    "headings": " ".join(c.heading_path),
                }
                for c in chunks
            ],
        )


def mark_document_failed(
    conn: psycopg.Connection,
    *,
    document_id: str,
    relative_path: str,
    filename: str,
    source_hash: str,
    size_bytes: int,
    error: str,
    file_modified_at: datetime | None,
) -> None:
    """Record a failure. A previously indexed version (if any) stays searchable;
    its content_hash is left unchanged so the next run retries the new content."""
    conn.execute(
        """
        INSERT INTO documents (document_id, relative_path, filename, title, content_hash, source_hash,
                               size_bytes, status, last_error, failure_count, file_modified_at)
        VALUES (%s, %s, %s, %s, NULL, %s, %s, 'failed', %s, 1, %s)
        ON CONFLICT (document_id) DO UPDATE SET
            status = 'failed', source_hash = EXCLUDED.source_hash, last_error = EXCLUDED.last_error,
            failure_count = documents.failure_count + 1, size_bytes = EXCLUDED.size_bytes,
            file_modified_at = EXCLUDED.file_modified_at, deleted_at = NULL, updated_at = now()
        """,
        (document_id, relative_path, filename, filename, source_hash, size_bytes, error[:2000], file_modified_at),
    )


def mark_document_deleted(conn: psycopg.Connection, document_id: str) -> int:
    """Remove indexed chunks; keep the document row as an audit record."""
    cur = conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    removed = cur.rowcount or 0
    conn.execute(
        """
        UPDATE documents SET status = 'deleted', chunk_count = 0, deleted_at = now(), updated_at = now()
        WHERE document_id = %s
        """,
        (document_id,),
    )
    return removed


def list_documents(
    conn: psycopg.Connection,
    *,
    page: int,
    page_size: int,
    query: str | None = None,
    status: str | None = None,
    include_deleted: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    where = ["TRUE"]
    params: list[Any] = []
    if not include_deleted and status != "deleted":
        where.append("status <> 'deleted'")
    if status:
        where.append("status = %s")
        params.append(status)
    if query:
        where.append("(title ILIKE %s OR relative_path ILIKE %s)")
        like = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        params.extend([like, like])
    clause = " AND ".join(where)
    total = conn.execute(f"SELECT count(*) AS n FROM documents WHERE {clause}", params).fetchone()["n"]
    rows = conn.execute(
        f"""
        SELECT document_id, relative_path, filename, title, status, chunk_count, source_url, size_bytes,
               embedding_config, last_error, last_indexed_at, file_modified_at, created_at, updated_at,
               (metadata->>'empty')::boolean AS is_empty
        FROM documents WHERE {clause}
        ORDER BY relative_path
        LIMIT %s OFFSET %s
        """,
        params + [page_size, (page - 1) * page_size],
    ).fetchall()
    return rows, int(total)


def indexed_document_titles(conn: psycopg.Connection) -> list[tuple[str, str]]:
    """(title, relative_path) of every indexed document."""
    rows = conn.execute(
        "SELECT title, relative_path FROM documents WHERE status = 'indexed' ORDER BY title, relative_path"
    ).fetchall()
    return [(row["title"], row["relative_path"]) for row in rows]


def get_document(conn: psycopg.Connection, document_id: str) -> dict[str, Any] | None:
    doc = conn.execute(
        """
        SELECT document_id, relative_path, filename, title, status, chunk_count, source_url, size_bytes,
               encoding, content_hash, source_hash, embedding_config, metadata, last_error, failure_count,
               last_indexed_at, file_modified_at, deleted_at, created_at, updated_at
        FROM documents WHERE document_id = %s
        """,
        (document_id,),
    ).fetchone()
    if doc is None:
        return None
    doc["chunks"] = conn.execute(
        """
        SELECT chunk_id, chunk_index, heading_path, token_count, char_count, source_urls,
               left(text, 280) AS preview
        FROM chunks WHERE document_id = %s ORDER BY chunk_index
        """,
        (document_id,),
    ).fetchall()
    return doc


def get_chunks_by_ids(conn: psycopg.Connection, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    rows = conn.execute(
        """
        SELECT c.chunk_id, c.document_id, c.chunk_index, c.text, c.heading_path, c.source_urls, c.content_hash,
               d.title, d.relative_path, d.status, d.source_url
        FROM chunks c JOIN documents d ON d.document_id = c.document_id
        WHERE c.chunk_id = ANY(%s)
        """,
        (list(chunk_ids),),
    ).fetchall()
    return {r["chunk_id"]: r for r in rows}


def index_stats(conn: psycopg.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
          (SELECT count(*) FROM documents WHERE status = 'indexed') AS indexed_documents,
          (SELECT count(*) FROM documents WHERE status = 'failed') AS failed_documents,
          (SELECT count(*) FROM documents WHERE status = 'pending') AS pending_documents,
          (SELECT count(*) FROM documents WHERE status = 'deleted') AS deleted_documents,
          (SELECT count(*) FROM chunks) AS chunks,
          (SELECT count(*) FROM embedding_cache) AS cached_embeddings,
          (SELECT array_agg(DISTINCT embedding_config) FROM chunks) AS embedding_configs,
          (SELECT max(last_indexed_at) FROM documents) AS last_indexed_at
        """
    ).fetchone()
    row["embedding_configs"] = [c for c in (row["embedding_configs"] or []) if c]
    return row


# --------------------------------------------------------------------------- embedding cache
def get_cached_embeddings(
    conn: psycopg.Connection, content_hashes: Iterable[str], embedding_config: str
) -> dict[str, list[float]]:
    hashes = list(dict.fromkeys(content_hashes))
    if not hashes:
        return {}
    rows = conn.execute(
        """
        SELECT content_hash, embedding FROM embedding_cache
        WHERE embedding_config = %s AND content_hash = ANY(%s)
        """,
        (embedding_config, hashes),
    ).fetchall()
    return {r["content_hash"]: vector_to_list(r["embedding"]) for r in rows}


def put_cached_embeddings(
    conn: psycopg.Connection,
    items: Iterable[tuple[str, list[float]]],
    *,
    embedding_config: str,
    embedding_model: str,
    dimensions: int,
) -> int:
    rows = [
        {
            "h": h,
            "cfg": embedding_config,
            "model": embedding_model,
            "dims": dimensions,
            "emb": embedding_param(conn, v),
        }
        for h, v in items
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO embedding_cache (content_hash, embedding_config, embedding_model, dimensions, embedding)
            VALUES (%(h)s, %(cfg)s, %(model)s, %(dims)s, %(emb)s)
            ON CONFLICT (content_hash, embedding_config) DO UPDATE SET last_used_at = now()
            """,
            rows,
        )
    return len(rows)


def touch_cached_embeddings(conn: psycopg.Connection, content_hashes: Sequence[str], embedding_config: str) -> None:
    if content_hashes:
        conn.execute(
            "UPDATE embedding_cache SET last_used_at = now() WHERE embedding_config = %s AND content_hash = ANY(%s)",
            (embedding_config, list(content_hashes)),
        )


# --------------------------------------------------------------------------- ingestion runs
def create_run(conn: psycopg.Connection, *, mode: str, triggered_by: str, embedding_config: str) -> uuid.UUID:
    run_id = uuid.uuid4()
    conn.execute(
        """
        INSERT INTO ingestion_runs (run_id, mode, status, triggered_by, embedding_config)
        VALUES (%s, %s, 'running', %s, %s)
        """,
        (run_id, mode, triggered_by, embedding_config),
    )
    return run_id


def update_run(conn: psycopg.Connection, run_id: uuid.UUID, *, report: dict[str, Any],
               status: str | None = None, error: str | None = None) -> None:
    conn.execute(
        """
        UPDATE ingestion_runs SET report = %s, heartbeat_at = now(),
               status = COALESCE(%s::text, status), error = COALESCE(%s::text, error),
               finished_at = CASE WHEN %s::text IS NOT NULL AND %s::text <> 'running' THEN now() ELSE finished_at END
        WHERE run_id = %s
        """,
        (Jsonb(report), status, error, status, status, run_id),
    )


def add_event(
    conn: psycopg.Connection,
    run_id: uuid.UUID,
    *,
    stage: str,
    status: str,
    document_id: str | None = None,
    relative_path: str | None = None,
    error: str | None = None,
    retry_count: int = 0,
    chunk_count: int | None = None,
    embeddings_generated: int | None = None,
    embeddings_reused: int | None = None,
    embedding_config: str | None = None,
    started_at: datetime | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_events (run_id, document_id, relative_path, stage, status, error, retry_count,
                                      chunk_count, embeddings_generated, embeddings_reused, embedding_config,
                                      started_at, finished_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()), now())
        """,
        (run_id, document_id, relative_path, stage, status, error[:2000] if error else None, retry_count,
         chunk_count, embeddings_generated, embeddings_reused, embedding_config, started_at),
    )


def mark_stale_runs_interrupted(conn: psycopg.Connection, exclude: uuid.UUID | None = None) -> int:
    """Called while holding the ingestion lock: any other 'running' run is dead."""
    cur = conn.execute(
        """
        UPDATE ingestion_runs SET status = 'interrupted', finished_at = now(),
               error = COALESCE(error, 'Run did not finish (process stopped). Re-run ingestion to resume.')
        WHERE status = 'running' AND (%s::uuid IS NULL OR run_id <> %s::uuid)
        """,
        (exclude, exclude),
    )
    return cur.rowcount or 0


def list_runs(conn: psycopg.Connection, limit: int = 20) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT run_id, mode, status, triggered_by, embedding_config, report, error, started_at, finished_at,
               heartbeat_at
        FROM ingestion_runs ORDER BY started_at DESC LIMIT %s
        """,
        (limit,),
    ).fetchall()


def get_run(conn: psycopg.Connection, run_id: uuid.UUID) -> dict[str, Any] | None:
    run = conn.execute(
        """
        SELECT run_id, mode, status, triggered_by, embedding_config, report, error, started_at, finished_at,
               heartbeat_at
        FROM ingestion_runs WHERE run_id = %s
        """,
        (run_id,),
    ).fetchone()
    if run is None:
        return None
    run["failures"] = conn.execute(
        """
        SELECT relative_path, stage, error, retry_count, finished_at FROM ingestion_events
        WHERE run_id = %s AND status = 'failed' ORDER BY event_id LIMIT 200
        """,
        (run_id,),
    ).fetchall()
    return run


def try_ingestion_lock(conn: psycopg.Connection) -> bool:
    return bool(conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (INGESTION_LOCK_KEY,)).fetchone()["ok"])


def release_ingestion_lock(conn: psycopg.Connection) -> None:
    conn.execute("SELECT pg_advisory_unlock(%s)", (INGESTION_LOCK_KEY,))


def ingestion_lock_held(conn: psycopg.Connection) -> bool:
    row = conn.execute(
        "SELECT count(*) AS n FROM pg_locks WHERE locktype = 'advisory' AND objid = %s AND granted",
        (INGESTION_LOCK_KEY,),
    ).fetchone()
    return bool(row["n"])


# --------------------------------------------------------------------------- conversations
def ensure_conversation(conn: psycopg.Connection, conversation_id: uuid.UUID | None, title: str) -> uuid.UUID:
    if conversation_id is not None:
        row = conn.execute(
            "SELECT conversation_id FROM conversations WHERE conversation_id = %s", (conversation_id,)
        ).fetchone()
        if row:
            return conversation_id
    new_id = conversation_id or uuid.uuid4()
    conn.execute(
        "INSERT INTO conversations (conversation_id, title) VALUES (%s, %s)",
        (new_id, title[:120] or "New conversation"),
    )
    return new_id


def add_message(
    conn: psycopg.Connection,
    conversation_id: uuid.UUID,
    *,
    role: str,
    content: str,
    standalone_question: str | None = None,
    payload: dict[str, Any] | None = None,
) -> uuid.UUID:
    message_id = uuid.uuid4()
    conn.execute(
        """
        INSERT INTO messages (message_id, conversation_id, role, content, standalone_question, payload)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (message_id, conversation_id, role, content, standalone_question, Jsonb(payload or {})),
    )
    conn.execute("UPDATE conversations SET updated_at = now() WHERE conversation_id = %s", (conversation_id,))
    return message_id


def recent_messages(conn: psycopg.Connection, conversation_id: uuid.UUID, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT role, content, standalone_question, created_at FROM messages
        WHERE conversation_id = %s ORDER BY created_at DESC LIMIT %s
        """,
        (conversation_id, limit),
    ).fetchall()
    return list(reversed(rows))


def list_conversations(conn: psycopg.Connection, limit: int = 30) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT c.conversation_id, c.title, c.created_at, c.updated_at,
               (SELECT count(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS message_count
        FROM conversations c ORDER BY c.updated_at DESC LIMIT %s
        """,
        (limit,),
    ).fetchall()


def get_conversation(conn: psycopg.Connection, conversation_id: uuid.UUID) -> dict[str, Any] | None:
    conv = conn.execute(
        "SELECT conversation_id, title, created_at, updated_at FROM conversations WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchone()
    if conv is None:
        return None
    conv["messages"] = conn.execute(
        """
        SELECT message_id, role, content, payload, created_at FROM messages
        WHERE conversation_id = %s ORDER BY created_at
        """,
        (conversation_id,),
    ).fetchall()
    return conv


def delete_conversation(conn: psycopg.Connection, conversation_id: uuid.UUID) -> bool:
    cur = conn.execute("DELETE FROM conversations WHERE conversation_id = %s", (conversation_id,))
    return (cur.rowcount or 0) > 0
