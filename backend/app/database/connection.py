"""PostgreSQL connection management (psycopg 3 + psycopg_pool)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


class DatabaseUnavailableError(RuntimeError):
    """Raised when PostgreSQL cannot be reached or is misconfigured."""


def _configure(conn: psycopg.Connection, statement_timeout_ms: int) -> None:
    from pgvector.psycopg import register_vector

    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        # No pgvector type in this database: the array fallback (VECTOR_BACKEND=array) is in use,
        # or init-db has not been run yet. Embeddings are then plain real[] values.
        conn.rollback()
    conn.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
    conn.commit()


def raw_connect(settings: Settings | None = None, *, autocommit: bool = False) -> psycopg.Connection:
    """A plain connection without pgvector registration (used before the extension exists)."""
    settings = settings or get_settings()
    try:
        return psycopg.connect(
            settings.database_url.get_secret_value(),
            autocommit=autocommit,
            row_factory=dict_row,
            connect_timeout=10,
        )
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailableError(_describe_connect_error(exc)) from exc


def get_pool(settings: Settings | None = None) -> ConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            settings = settings or get_settings()
            timeout = settings.db_statement_timeout_ms
            pool = ConnectionPool(
                conninfo=settings.database_url.get_secret_value(),
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
                kwargs={"row_factory": dict_row, "connect_timeout": 10},
                configure=lambda conn: _configure(conn, timeout),
                open=False,
                name="epolicies",
            )
            try:
                pool.open(wait=True, timeout=15)
            except Exception as exc:  # psycopg_pool.PoolTimeout or configure errors
                pool.close()
                raise DatabaseUnavailableError(
                    "Could not open a PostgreSQL connection pool. Check DATABASE_URL, that the "
                    "server is running, and that `python -m app.cli init-db` has been run."
                ) from exc
            _pool = pool
    return _pool


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Borrow a pooled connection. The block runs inside a transaction that is
    committed on success and rolled back on error."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def _describe_connect_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "password authentication failed" in message:
        hint = "authentication failed — check the user/password in DATABASE_URL"
    elif "does not exist" in message and "database" in message:
        hint = "the database does not exist — create it with `createdb epolicies`"
    elif "connection refused" in message or "could not connect" in message:
        hint = "connection refused — is PostgreSQL running on the host/port in DATABASE_URL?"
    else:
        hint = "check DATABASE_URL and that PostgreSQL is running"
    return f"Cannot connect to PostgreSQL: {hint}."
