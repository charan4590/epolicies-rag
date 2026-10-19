"""Schema initialisation, verification and (explicit) destructive rebuild."""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg

from app.config import Settings, get_settings
from app.database.connection import DatabaseUnavailableError, raw_connect

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
SCHEMA_FILE = Path(__file__).with_name("schema.sql")
HNSW_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
    "ON chunks USING hnsw (embedding vector_cosine_ops)"
)


class SchemaMismatchError(RuntimeError):
    pass


def render_schema(dimensions: int, backend: str = "pgvector") -> str:
    dims = int(dimensions)
    vector_type = f"vector({dims})" if backend == "pgvector" else f"real[] CHECK (cardinality(embedding) = {dims})"
    return (SCHEMA_FILE.read_text(encoding="utf-8")
            .replace("__VECTOR_TYPE__", vector_type)
            .replace("__EMBEDDING_DIMENSIONS__", str(dims)))


def pgvector_available(conn: psycopg.Connection) -> bool:
    row = conn.execute("SELECT 1 AS ok FROM pg_available_extensions WHERE name = 'vector'").fetchone()
    return row is not None


def existing_vector_backend(conn: psycopg.Connection, table: str = "chunks") -> str | None:
    """'pgvector' or 'array' for an existing <table>.embedding column, None if absent."""
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS type
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = %s AND a.attname = 'embedding' AND n.nspname = current_schema()
          AND NOT a.attisdropped
        """,
        (table,),
    ).fetchone()
    if row is None:
        return None
    return "pgvector" if str(row["type"]).startswith("vector") else "array"


def existing_vector_dimensions(conn: psycopg.Connection, table: str = "chunks") -> int | None:
    """Declared embedding dimension of <table>.embedding, or None if the table is absent."""
    row = conn.execute(
        """
        SELECT a.atttypmod AS dims
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = %s AND a.attname = 'embedding' AND n.nspname = current_schema()
          AND NOT a.attisdropped
        """,
        (table,),
    ).fetchone()
    if row is None:
        return None
    if row["dims"] and row["dims"] > 0:
        return int(row["dims"])
    # Array fallback: real[] has no typmod; the dimension is recorded in schema_meta.
    meta = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'embedding_dimensions'"
    ).fetchone() if conn.execute("SELECT to_regclass('schema_meta') AS t").fetchone()["t"] else None
    return int(meta["value"]) if meta else None


def choose_backend(conn: psycopg.Connection, settings: Settings) -> str:
    """Decide the vector backend for a new or existing schema."""
    existing = existing_vector_backend(conn)
    wanted = settings.vector_backend
    if existing is not None:
        if wanted != "auto" and wanted != existing:
            raise SchemaMismatchError(
                f"The database was created with VECTOR_BACKEND={existing} but VECTOR_BACKEND={wanted}. "
                "Set VECTOR_BACKEND=auto, or run `python -m app.cli rebuild --confirm` to switch."
            )
        return existing
    if wanted == "array":
        return "array"
    if pgvector_available(conn):
        return "pgvector"
    if wanted == "pgvector":
        raise DatabaseUnavailableError(
            "pgvector is not installed on this PostgreSQL server. Install it first "
            "(macOS: `brew install pgvector`; Debian/Ubuntu: `sudo apt install postgresql-17-pgvector`; "
            "Windows: ask IT), or set VECTOR_BACKEND=auto to use the built-in array fallback."
        )
    logger.warning("pgvector is not installed; using the built-in array fallback (VECTOR_BACKEND=array).")
    return "array"


def init_schema(settings: Settings | None = None) -> dict[str, object]:
    """Create tables and indexes if they do not exist (and enable pgvector when used).

    Refuses to proceed when an existing schema was created with different
    embedding dimensions (no silent truncation/padding of vectors).
    """
    settings = settings or get_settings()
    dims = settings.embedding_dimensions
    with raw_connect(settings) as conn:
        backend = choose_backend(conn, settings)
        current = existing_vector_dimensions(conn)
        if current is not None and current != dims:
            raise SchemaMismatchError(
                f"The database stores {current}-dimensional embeddings but EMBEDDING_DIMENSIONS={dims}. "
                "Either restore EMBEDDING_DIMENSIONS or run `python -m app.cli rebuild --confirm` "
                "(destructive: drops indexed chunks and cached embeddings)."
            )
        if backend == "pgvector":
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except psycopg.errors.InsufficientPrivilege as exc:
                raise DatabaseUnavailableError(
                    "Permission denied while enabling pgvector. Run `CREATE EXTENSION vector;` once as a "
                    "superuser in this database, then re-run init-db."
                ) from exc
        conn.execute(render_schema(dims, backend))
        if backend == "pgvector" and settings.vector_index_supported:
            conn.execute(HNSW_INDEX_SQL)
        elif backend == "pgvector":
            logger.warning(
                "EMBEDDING_DIMENSIONS=%s exceeds pgvector's 2000-dimension index limit; "
                "vector search will use exact (sequential) scans.",
                dims,
            )
        for key, value in (
            ("schema_version", SCHEMA_VERSION),
            ("embedding_model", settings.embedding_model),
            ("embedding_dimensions", str(dims)),
            ("vector_backend", backend),
        ):
            conn.execute(
                """
                INSERT INTO schema_meta (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                (key, value),
            )
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        conn.commit()
    return {
        "schema_version": SCHEMA_VERSION,
        "vector_backend": backend,
        "pgvector_version": version["extversion"] if version and backend == "pgvector" else None,
        "embedding_dimensions": dims,
        "hnsw_index": backend == "pgvector" and settings.vector_index_supported,
    }


def drop_index_data(settings: Settings | None = None) -> None:
    """Destructive: drop chunk/embedding tables so they can be recreated with new dimensions.

    Documents, ingestion history and conversations are retained; documents are
    reset to 'pending' so the next ingestion rebuilds everything.
    """
    settings = settings or get_settings()
    with raw_connect(settings) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS embedding_cache")
        conn.execute(
            """
            DO $$ BEGIN
              IF to_regclass('documents') IS NOT NULL THEN
                UPDATE documents SET status = CASE WHEN status = 'deleted' THEN 'deleted' ELSE 'pending' END,
                                     content_hash = NULL, chunk_count = 0, embedding_config = NULL,
                                     updated_at = now();
              END IF;
            END $$;
            """
        )
        conn.commit()
    logger.warning("Dropped chunks and embedding_cache tables (explicit rebuild requested).")


def check_schema(conn: psycopg.Connection, settings: Settings | None = None) -> dict[str, object]:
    """Lightweight verification used by /health."""
    settings = settings or get_settings()
    ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    dims = existing_vector_dimensions(conn)
    return {
        "vector_backend": existing_vector_backend(conn),
        "pgvector_enabled": ext is not None,
        "pgvector_version": ext["extversion"] if ext else None,
        "schema_initialized": dims is not None,
        "vector_dimensions": dims,
        "dimensions_match": dims == settings.embedding_dimensions if dims is not None else False,
    }
