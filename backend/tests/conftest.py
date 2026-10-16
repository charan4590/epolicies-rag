"""Test configuration.

Unit tests need nothing external. PostgreSQL integration tests run when
TEST_DATABASE_URL points at a *dedicated* database whose name contains "test"
(it is wiped). pgvector must be installed on that server.

    TEST_DATABASE_URL=postgresql://postgres@localhost:5432/epolicies_test pytest
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "markdown"
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

# Environment must be set before any `app` module reads settings.
os.environ.update({
    "APP_ENV": "test",
    "GOOGLE_API_KEY": "test-key-not-real",
    "GEMINI_MODEL": "test-model",
    "EMBEDDING_MODEL": "gemini-embedding-001",
    "EMBEDDING_DIMENSIONS": "768",
    "ADMIN_API_TOKEN": "test-admin-token",
    "LOG_FORMAT": "text",
    "LOG_LEVEL": "WARNING",
    "PLANNER_MODE": "auto",
    "MAX_RETRIEVAL_ROUNDS": "3",
    "LLM_CITATION_VERIFICATION": "true",
    "DATABASE_URL": TEST_DATABASE_URL or "postgresql://invalid:invalid@127.0.0.1:1/none",
})


def _reset_settings() -> None:
    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture()
def markdown_dir(tmp_path, monkeypatch) -> Path:
    """A writable copy of the synthetic fixtures configured as MARKDOWN_DIR."""
    target = tmp_path / "markdown"
    shutil.copytree(FIXTURES, target)
    monkeypatch.setenv("MARKDOWN_DIR", str(target))
    monkeypatch.setenv("EMBEDDING_JSONL_PATH", str(tmp_path / "artifacts" / "embedding.jsonl"))
    _reset_settings()
    yield target
    _reset_settings()


requires_db = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set (PostgreSQL+pgvector integration tests)"
)


@pytest.fixture(scope="session")
def pg_schema():
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    import psycopg

    dbname = psycopg.conninfo.conninfo_to_dict(TEST_DATABASE_URL).get("dbname", "")
    if "test" not in str(dbname):
        pytest.exit(f"Refusing to run destructive tests against database '{dbname}' (name must contain 'test').")
    _reset_settings()
    from app.database.connection import close_pool, raw_connect
    from app.database.schema import init_schema

    with raw_connect() as conn:
        conn.execute("""
            DROP TABLE IF EXISTS messages, conversations, ingestion_events, ingestion_runs,
                                 embedding_cache, chunks, documents, schema_meta CASCADE
        """)
        conn.execute("DROP EXTENSION IF EXISTS vector CASCADE")
        conn.commit()
    info = init_schema()
    yield info
    close_pool()


@pytest.fixture()
def db(pg_schema, markdown_dir):
    """Empty tables before each integration test."""
    from app.database.connection import close_pool, connection

    close_pool()  # pick up per-test settings
    with connection() as conn:
        conn.execute("""
            TRUNCATE messages, conversations, ingestion_events, ingestion_runs, embedding_cache, chunks,
                     documents RESTART IDENTITY CASCADE
        """)
    yield
    close_pool()
