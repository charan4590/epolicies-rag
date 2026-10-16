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


def render_schema(dimensions: int) -> str:
    return SCHEMA_FILE.read_text(encoding="utf-8").replace("__EMBEDDING_DIMENSIONS__", str(int(dimensions)))


def pgvector_available(conn: psycopg.Connection) -> bool:
    row = conn.execute("SELECT 1 AS ok FROM pg_available_extensions WHERE name = 'vector'").fetchone()
    return row is not None


def existing_vector_dimensions(conn: psycopg.Connection, table: str = "chunks") -> int | None:
    """Return the declared vector dimension of <table>.embedding, or None if absent."""
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
    return int(row["dims"]) if row["dims"] and row["dims"] > 0 else None


def init_schema(settings: Settings | None = None) -> dict[str, object]:
    """Create the extension, tables and indexes if they do not exist.

    Refuses to proceed when an existing schema was created with different
    embedding dimensions (no silent truncation/padding of vectors).
    """
    settings = settings or get_settings()
    dims = settings.embedding_dimensions
    with raw_connect(settings) as conn:
        if not pgvector_available(conn):
            raise DatabaseUnavailableError(
                "pgvector is not installed on this PostgreSQL server. Install it first "
                "(macOS: `brew install pgvector`; Debian/Ubuntu: `sudo apt install postgresql-17-pgvector`), "
                "then re-run `python -m app.cli init-db`."
            )
        current = existing_vector_dimensions(conn)
        if current is not None and current != dims:
            raise SchemaMismatchError(
                f"The database stores {current}-dimensional embeddings but EMBEDDING_DIMENSIONS={dims}. "
                "Either restore EMBEDDING_DIMENSIONS or run `python -m app.cli rebuild --confirm` "
                "(destructive: drops indexed chunks and cached embeddings)."
            )
        try:
            conn.execute(render_schema(dims))
        except psycopg.errors.InsufficientPrivilege as exc:
            raise DatabaseUnavailableError(
                "Permission denied while enabling pgvector. Run `CREATE EXTENSION vector;` once as a "
                "superuser in this database, then re-run init-db."
            ) from exc
        if settings.vector_index_supported:
            conn.execute(HNSW_INDEX_SQL)
        else:
            logger.warning(
                "EMBEDDING_DIMENSIONS=%s exceeds pgvector's 2000-dimension index limit; "
                "vector search will use exact (sequential) scans.",
                dims,
            )
        for key, value in (
            ("schema_version", SCHEMA_VERSION),
            ("embedding_model", settings.embedding_model),
            ("embedding_dimensions", str(dims)),
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
        "pgvector_version": version["extversion"] if version else None,
        "embedding_dimensions": dims,
        "hnsw_index": settings.vector_index_supported,
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
        "pgvector_enabled": ext is not None,
        "pgvector_version": ext["extversion"] if ext else None,
        "schema_initialized": dims is not None,
        "vector_dimensions": dims,
        "dimensions_match": dims == settings.embedding_dimensions if dims is not None else False,
    }
