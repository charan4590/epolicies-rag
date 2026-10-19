"""Array fallback (no pgvector): embeddings stored as real[] in PostgreSQL, cosine computed in the app.

Runs against a separate database (`<TEST_DATABASE_URL db>_array`) and simulates a server where the
pgvector extension is not installed.
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from app.config import get_settings
from app.database import repository as repo
from app.database import schema as schema_module
from app.database.connection import close_pool, connection
from app.ingestion import jsonl_store
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.vector_search import vector_search
from app.schemas.api_models import ChatRequest
from app.services.rag_service import RAGService
from tests.conftest import TEST_DATABASE_URL, requires_db
from tests.fakes import FakeEmbedder, FakeLLM, first_source_label, hashed_vector

pytestmark = requires_db


@pytest.fixture()
def array_db(markdown_dir, monkeypatch):
    params = conninfo_to_dict(TEST_DATABASE_URL)
    dbname = f"{params.get('dbname', 'epolicies_test')}_array"
    with psycopg.connect(make_conninfo(TEST_DATABASE_URL, dbname="postgres"), autocommit=True) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone():
            admin.execute(f'CREATE DATABASE "{dbname}"')
    url = make_conninfo(TEST_DATABASE_URL, dbname=dbname)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("VECTOR_BACKEND", "auto")
    # Simulate a PostgreSQL server without the pgvector extension (e.g. a locked-down Windows laptop).
    monkeypatch.setattr(schema_module, "pgvector_available", lambda conn: False)
    get_settings.cache_clear()
    close_pool()
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("""DROP TABLE IF EXISTS messages, conversations, ingestion_events, ingestion_runs,
                        embedding_cache, chunks, documents, schema_meta CASCADE""")
        conn.execute("DROP EXTENSION IF EXISTS vector CASCADE")
    info = schema_module.init_schema()
    yield info
    close_pool()
    get_settings.cache_clear()


def ingest(embedder: FakeEmbedder | None = None, **kwargs):
    embedder = embedder or FakeEmbedder()
    pipeline = IngestionPipeline(get_settings(), embedder_factory=lambda: embedder, progress=lambda m: None)
    return pipeline.run(**kwargs), embedder


def test_schema_uses_real_arrays_without_extension(array_db):
    assert array_db["vector_backend"] == "array" and array_db["pgvector_version"] is None
    with connection() as conn:
        assert conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone() is None
        assert schema_module.existing_vector_backend(conn) == "array"
        assert schema_module.existing_vector_dimensions(conn) == 768
        status = schema_module.check_schema(conn)
        assert status["schema_initialized"] and status["dimensions_match"] and not status["pgvector_enabled"]
        assert not repo.uses_pgvector(conn)
    # init-db is idempotent and keeps the backend
    assert schema_module.init_schema()["vector_backend"] == "array"


def test_dimension_constraint_still_enforced(array_db):
    with pytest.raises(psycopg.errors.CheckViolation):
        with connection() as conn:
            conn.execute("""INSERT INTO embedding_cache (content_hash, embedding_config, embedding_model, dimensions,
                            embedding) VALUES ('h', 'c', 'm', 768, %s)""", ([0.1, 0.2],))


def test_ingest_search_jsonl_and_answer_in_array_mode(array_db, markdown_dir):
    report, _ = ingest()
    assert report.status == "completed" and report.documents_indexed == 9 and report.chunks_created > 0
    again, embedder = ingest()
    assert again.unchanged_documents == 9 and embedder.stats.requests == 0  # cache reuse works

    settings = get_settings()
    with connection() as conn:
        vec = vector_search(conn, hashed_vector("hotel accommodation reimbursed per night", 768), top_k=5)
        hy = hybrid_search(conn, FakeEmbedder(), ["How much is the internet reimbursement for remote work?"],
                           settings=settings, exact_terms=["HR-POL-019"])
        row = conn.execute("SELECT pg_typeof(embedding)::text AS t FROM chunks LIMIT 1").fetchone()
    assert row["t"] == "real[]"
    assert vec[0].relative_path.startswith("finance/") and vec[0].vector_rank == 1
    assert all(a.vector_similarity >= b.vector_similarity for a, b in zip(vec, vec[1:]))
    assert hy.chunks[0].relative_path == "hr/remote-work-policy.md"
    assert any(k.startswith("vector:") for k in hy.lists) and any(k.startswith("keyword:") for k in hy.lists)

    # A modified document is picked up by the in-app vector index (cache invalidation).
    path = markdown_dir / "hr" / "leave-policy.md"
    path.write_text(path.read_text() + "\n\n## Sabbatical\n\nEmployees may take a sabbatical of 6 months after 10 years.\n")
    ingest()
    with connection() as conn:
        top = vector_search(conn, hashed_vector("sabbatical after 10 years months", 768), top_k=3)
    assert any("Sabbatical" in c.text for c in top)

    # JSONL round trip without pgvector
    jsonl = settings.resolved_jsonl_path
    assert jsonl_store.validate_jsonl(jsonl, expected_dimensions=768).valid
    with connection() as conn:
        before = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        conn.execute("TRUNCATE chunks, documents, embedding_cache CASCADE")
        conn.commit()
        loaded = jsonl_store.load_jsonl(conn, jsonl, embedding_model="gemini-embedding-001", dimensions=768,
                                        text_search_config="english")
    assert loaded.chunks_loaded == before

    # Full agentic answer with citations
    def answer(prompt: str):
        label = first_source_label(prompt, "$250")
        return {"answer_markdown": f"Hotels are reimbursed up to $250 per night [{label}].",
                "claims": [{"statement": "Hotel costs are reimbursed up to $250 per night", "source_ids": [label],
                            "support": "explicit"}],
                "insufficient_evidence": False, "missing_information": [], "conflicts": []}

    llm = FakeLLM({"evidence_check": {"sufficient": True, "confidence": "high", "relevant_source_ids": [],
                                      "missing_information": [], "follow_up_queries": []}, "answer": answer})
    response = RAGService(settings, embedder=FakeEmbedder(), llm=llm).answer(
        ChatRequest(question="What is the hotel cap per night?", persist=False))
    assert response.verification.status == "verified" and "travel-expenses" in response.citations[0].relative_path


def test_switching_backend_requires_rebuild(array_db, monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    get_settings.cache_clear()
    with pytest.raises(schema_module.SchemaMismatchError, match="VECTOR_BACKEND=array"):
        schema_module.init_schema()
