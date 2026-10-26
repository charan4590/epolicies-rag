"""PostgreSQL + pgvector integration tests (require TEST_DATABASE_URL).

Uses a deterministic fake embedder, so no Gemini calls are made.
"""

from __future__ import annotations

import shutil

import psycopg
import pytest

from app.config import get_settings
from app.database import repository as repo
from app.database.connection import connection
from app.database.schema import SchemaMismatchError, existing_vector_dimensions, init_schema
from app.ingestion import jsonl_store
from app.ingestion.pipeline import IngestionPipeline, UnsafeDeletionError
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.keyword_search import exact_search, keyword_search, unknown_terms
from app.retrieval.synonyms import keyword_query_with_synonyms
from app.retrieval.vector_search import vector_search
from app.schemas.api_models import ChatRequest
from app.services.rag_service import RAGService
from tests.conftest import requires_db
from tests.fakes import FakeEmbedder, FakeLLM, first_source_label, hashed_vector

pytestmark = requires_db

FIXTURE_FILES = 9
EXPECTED_READABLE = 9


def ingest(embedder: FakeEmbedder | None = None, **kwargs):
    embedder = embedder or FakeEmbedder()
    pipeline = IngestionPipeline(get_settings(), embedder_factory=lambda: embedder, progress=lambda m: None)
    return pipeline.run(**kwargs), embedder


def counts() -> dict:
    with connection() as conn:
        return conn.execute("""
            SELECT (SELECT count(*) FROM documents WHERE status = 'indexed') AS docs,
                   (SELECT count(*) FROM chunks) AS chunks,
                   (SELECT count(DISTINCT chunk_id) FROM chunks) AS distinct_chunks,
                   (SELECT count(*) FROM embedding_cache) AS cache
        """).fetchone()


# --------------------------------------------------------------------------- schema
def test_schema_objects_exist(db):
    with connection() as conn:
        tables = {r["tablename"] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        indexes = {r["indexname"] for r in conn.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")}
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        dims = existing_vector_dimensions(conn)
    assert {"documents", "chunks", "embedding_cache", "ingestion_runs", "ingestion_events", "conversations",
            "messages", "schema_meta"} <= tables
    assert {"chunks_embedding_hnsw_idx", "chunks_search_tsv_gin_idx", "chunks_document_order_uniq"} <= indexes
    assert ext is not None and dims == 768


def test_init_schema_is_idempotent_and_refuses_dimension_change(db, monkeypatch):
    init_schema()
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    get_settings.cache_clear()
    with pytest.raises(SchemaMismatchError, match="768-dimensional"):
        init_schema()
    get_settings.cache_clear()


def test_constraints_reject_bad_rows(db):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with connection() as conn:
            conn.execute("""INSERT INTO chunks (chunk_id, document_id, chunk_index, text, content_hash, token_count,
                            char_count, embedding, embedding_model, embedding_config, search_tsv)
                            VALUES ('x', 'missing', 0, 't', 'h', 1, 1, %s, 'm', 'c', ''::tsvector)""",
                         (str([0.1] * 768),))
    with pytest.raises(psycopg.errors.DataException):
        with connection() as conn:
            conn.execute("SELECT %s::vector(768)", (str([0.1] * 3),))


# --------------------------------------------------------------------------- ingestion
def test_initial_ingestion_report_and_storage(db, markdown_dir):
    report, embedder = ingest()
    assert report.status == "completed"
    assert report.total_files_discovered == FIXTURE_FILES and report.readable_files == EXPECTED_READABLE
    assert report.empty_files == ["misc/empty.md"]
    assert report.duplicate_content_groups == [["finance/archive/travel-expenses-copy.md", "finance/travel-expenses.md"]]
    assert len(report.new_documents) == FIXTURE_FILES and report.failed_documents == []
    assert any("cp1252" in w for w in report.warnings)
    c = counts()
    assert c["docs"] == FIXTURE_FILES and c["chunks"] == report.chunks_created == c["distinct_chunks"]
    # duplicate file content is embedded once
    assert report.embeddings_generated == c["cache"] < report.chunks_created
    assert report.embeddings_generated + report.embeddings_reused == report.chunks_created
    with connection() as conn:
        row = conn.execute("""SELECT c.heading_path, c.source_urls, c.metadata, c.embedding_model, c.embedding_config,
                                     d.title, d.source_url
                              FROM chunks c JOIN documents d USING (document_id)
                              WHERE d.relative_path = 'hr/leave-policy.md' AND c.metadata->>'has_table' = 'true'""").fetchone()
        runs = repo.list_runs(conn)
    assert row["heading_path"] == ["Employee Leave Policy", "Leave Entitlements"]
    assert row["embedding_model"] == "gemini-embedding-001" and "dim=768" in row["embedding_config"]
    assert row["title"] == "Employee Leave Policy (HR-POL-012)"
    assert row["source_url"] == "https://policies.example.com/hr/leave"
    assert runs[0]["status"] == "completed" and runs[0]["report"]["chunks_created"] == report.chunks_created


def test_rerun_is_idempotent_and_reuses_embeddings(db, markdown_dir):
    first, _ = ingest()
    before = counts()
    second, embedder = ingest()
    assert second.unchanged_documents == FIXTURE_FILES and second.new_documents == []
    assert second.embeddings_generated == 0 and embedder.stats.requests == 0
    assert counts() == before
    forced, embedder = ingest(mode="force")
    assert len(forced.modified_documents) == FIXTURE_FILES and forced.embeddings_generated == 0
    assert forced.embeddings_reused == forced.chunks_created and counts() == before


def test_modified_document_replaces_obsolete_chunks(db, markdown_dir):
    ingest()
    path = markdown_dir / "hr" / "remote-work-policy.md"
    with connection() as conn:
        old_ids = {r["chunk_id"] for r in conn.execute(
            "SELECT chunk_id FROM chunks c JOIN documents d USING (document_id) WHERE d.relative_path = %s",
            ("hr/remote-work-policy.md",))}
    path.write_text(path.read_text().replace("up to 3 days per week", "up to 4 days per week"))
    report, embedder = ingest()
    assert report.modified_documents == ["hr/remote-work-policy.md"] and report.unchanged_documents == FIXTURE_FILES - 1
    assert 1 <= report.embeddings_generated < report.chunks_created  # only changed chunks re-embedded
    with connection() as conn:
        rows = conn.execute(
            "SELECT chunk_id, text FROM chunks c JOIN documents d USING (document_id) WHERE d.relative_path = %s",
            ("hr/remote-work-policy.md",)).fetchall()
    new_ids = {r["chunk_id"] for r in rows}
    assert new_ids != old_ids and not any("3 days per week" in r["text"] for r in rows)
    assert any("4 days per week" in r["text"] for r in rows)


def test_deleted_and_renamed_documents(db, markdown_dir):
    ingest()
    (markdown_dir / "misc" / "injection-test.md").unlink()
    shutil.move(markdown_dir / "it" / "acceptable-use.md", markdown_dir / "it" / "aup-renamed.md")
    report, embedder = ingest()
    assert sorted(report.deleted_documents) == ["it/acceptable-use.md", "misc/injection-test.md"]
    assert report.renamed_documents == [{"from": "it/acceptable-use.md", "to": "it/aup-renamed.md"}]
    assert report.embeddings_generated == 0  # rename reuses cached embeddings
    with connection() as conn:
        deleted = conn.execute("SELECT status, chunk_count FROM documents WHERE relative_path = 'misc/injection-test.md'").fetchone()
        orphan = conn.execute("""SELECT count(*) AS n FROM chunks c JOIN documents d USING (document_id)
                                 WHERE d.status = 'deleted'""").fetchone()
    assert deleted == {"status": "deleted", "chunk_count": 0} and orphan["n"] == 0


def test_mass_deletion_guard(db, markdown_dir):
    ingest()
    for child in markdown_dir.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    with pytest.raises(UnsafeDeletionError):
        ingest()
    assert counts()["docs"] == FIXTURE_FILES
    report, _ = ingest(allow_delete_all=True)
    assert len(report.deleted_documents) == FIXTURE_FILES and counts()["chunks"] == 0


def test_dry_run_writes_nothing(db, markdown_dir):
    report, embedder = ingest(dry_run=True)
    assert len(report.new_documents) == FIXTURE_FILES and embedder.stats.requests == 0
    assert counts()["docs"] == 0


def test_interrupted_embedding_resumes_without_rework(db, markdown_dir):
    report, embedder = ingest(FakeEmbedder(fail_after_requests=2, batch_size=4))
    assert report.status == "completed_with_errors" and report.failed_documents
    assert report.embeddings_generated == 8  # two batches were persisted before the failure
    with connection() as conn:
        failed = conn.execute("SELECT count(*) AS n FROM documents WHERE status = 'failed'").fetchone()["n"]
    assert failed == len(report.failed_documents)
    total_needed = 0
    resumed, embedder2 = ingest(FakeEmbedder(batch_size=4))
    assert resumed.status == "completed" and resumed.failed_documents == []
    total_needed = counts()["cache"]
    assert resumed.embeddings_generated == total_needed - 8  # nothing embedded twice
    assert counts()["docs"] == FIXTURE_FILES


def test_failed_transaction_keeps_previous_version(db, markdown_dir, monkeypatch):
    ingest()
    path = markdown_dir / "finance" / "travel-expenses.md"
    path.write_text(path.read_text().replace("$250", "$275"))
    original = repo.replace_document

    def exploding(conn, doc, chunks, **kw):
        original(conn, doc, chunks, **kw)  # writes happen, then the transaction must roll back
        if doc.relative_path == "finance/travel-expenses.md":
            raise psycopg.errors.InternalError("simulated failure after chunk insert")

    monkeypatch.setattr(repo, "replace_document", exploding)
    report, _ = ingest()
    assert [f["path"] for f in report.failed_documents] == ["finance/travel-expenses.md"]
    with connection() as conn:
        doc = conn.execute("SELECT status, chunk_count, last_error FROM documents WHERE relative_path = %s",
                           ("finance/travel-expenses.md",)).fetchone()
        texts = [r["text"] for r in conn.execute(
            "SELECT text FROM chunks c JOIN documents d USING (document_id) WHERE d.relative_path = %s",
            ("finance/travel-expenses.md",))]
    assert doc["status"] == "failed" and doc["chunk_count"] > 0 and "simulated" in doc["last_error"]
    assert any("$250" in t for t in texts) and not any("$275" in t for t in texts)  # old version intact
    monkeypatch.setattr(repo, "replace_document", original)
    retry, _ = ingest(retry_failed_only=True)
    assert retry.modified_documents == ["finance/travel-expenses.md"] and retry.failed_documents == []


# --------------------------------------------------------------------------- retrieval
def test_vector_keyword_exact_and_hybrid_search(db, markdown_dir):
    ingest()
    settings = get_settings()
    embedder = FakeEmbedder()
    with connection() as conn:
        vec = vector_search(conn, hashed_vector("hotel accommodation reimbursed per night", 768), top_k=5)
        kw = keyword_search(conn, "bereavement leave entitlement", top_k=5)
        table_kw = keyword_search(conn, "SW-042 restricted software", top_k=5)
        ex = exact_search(conn, ["HR-POL-019"], top_k=5)
        hy = hybrid_search(conn, embedder, ["How much is the internet reimbursement for remote work?"],
                           settings=settings, exact_terms=["HR-POL-019"])
    assert vec[0].relative_path.startswith("finance/") and vec[0].vector_similarity > 0
    assert kw[0].relative_path == "hr/leave-policy.md" and "Bereavement" in kw[0].text
    assert any("SW-042" in c.text for c in table_kw)
    assert ex and all(c.relative_path == "hr/remote-work-policy.md" for c in ex)
    assert hy.chunks[0].relative_path == "hr/remote-work-policy.md"
    assert {"vector:", "keyword:", "exact:"} <= {k.split(":")[0] + ":" for k in hy.lists}
    top = hy.chunks[0]
    assert top.rrf_score > 0 and (top.vector_rank or top.keyword_rank)
    # duplicate files collapse to a single result
    with connection() as conn:
        dup = hybrid_search(conn, embedder, ["hotel cap per night"], settings=settings)
    travel = [c for c in dup.chunks if "travel-expenses" in c.relative_path and "$250" in c.text]
    assert len(travel) == 1 and travel[0].duplicate_of and dup.duplicates_collapsed >= 1


def test_identifier_not_matched_inside_other_words(db, markdown_dir):
    ingest()
    with connection() as conn:
        hits = exact_search(conn, ["PTO"], top_k=20)
    assert hits and all("PTO" in c.text or "PTO" in c.title for c in hits)


# --------------------------------------------------------------------------- JSONL round trip
def test_jsonl_export_validate_load_and_reconcile(db, markdown_dir):
    report, _ = ingest()
    settings = get_settings()
    path = settings.resolved_jsonl_path
    assert report.jsonl["records"] == counts()["chunks"]
    validation = jsonl_store.validate_jsonl(path, expected_dimensions=768)
    assert validation.valid and validation.manifest_ok and validation.dimensions == {768}
    with connection() as conn:
        assert jsonl_store.reconcile(conn, path, 768)["consistent"]
        before = counts()
        conn.execute("TRUNCATE chunks, documents, embedding_cache CASCADE")
        conn.commit()
        drift = jsonl_store.reconcile(conn, path, 768)
        assert not drift["consistent"] and "load-jsonl" in drift["advice"]
        loaded = jsonl_store.load_jsonl(conn, path, embedding_model="gemini-embedding-001", dimensions=768,
                                        text_search_config="english")
        loaded_again = jsonl_store.load_jsonl(conn, path, embedding_model="gemini-embedding-001", dimensions=768,
                                              text_search_config="english")
    assert loaded.chunks_loaded == loaded_again.chunks_loaded == before["chunks"]
    after = counts()
    assert after["chunks"] == before["chunks"] == after["distinct_chunks"]  # no duplicates after retry
    # incremental ingestion after a JSONL restore needs no API calls
    report2, embedder = ingest()
    assert embedder.stats.requests == 0 and report2.unchanged_documents == FIXTURE_FILES - 1  # empty doc not in JSONL
    with connection() as conn:
        assert jsonl_store.reconcile(conn, path, 768)["consistent"]


def test_load_rejects_wrong_dimensions(db, markdown_dir, tmp_path):
    ingest()
    path = get_settings().resolved_jsonl_path
    with connection() as conn:
        with pytest.raises(ValueError, match="database expects 1536"):
            jsonl_store.load_jsonl(conn, path, embedding_model="gemini-embedding-001", dimensions=1536,
                                   text_search_config="english")


# --------------------------------------------------------------------------- full agentic flow
def d_notes(response) -> list[str]:
    return response.diagnostics["plan"].get("notes", [])


def test_unknown_terms_finds_words_missing_from_every_policy(db, markdown_dir):
    ingest()
    with connection() as conn:
        assert unknown_terms(conn, "jpoint partership for sick leave") == ["jpoint", "partership"]
        assert unknown_terms(conn, "sick leave days for employees") == []


def test_agentic_flow_with_follow_up_retrieval_and_persistence(db, markdown_dir):
    ingest()

    def evidence(prompt: str):
        if "approved critical project" in prompt:
            return {"sufficient": True, "confidence": "high", "relevant_source_ids": [first_source_label(prompt, "critical project")],
                    "missing_information": [], "follow_up_queries": []}
        return {"sufficient": False, "confidence": "low", "relevant_source_ids": [],
                "missing_information": ["the carryover exception"],
                "follow_up_queries": ["exception approved critical project carry over 10 days department head"]}

    def answer(prompt: str):
        label = first_source_label(prompt, "critical project")
        return {"answer_markdown": f"Up to 10 days may be carried over when leave was cancelled for an approved "
                                   f"critical project, with written department head approval [{label}].",
                "claims": [{"statement": "Employees may carry over up to 10 days when leave was cancelled because of "
                                         "an approved critical project", "source_ids": [label], "support": "explicit"}],
                "insufficient_evidence": False, "missing_information": [], "conflicts": []}

    # "rules" occurs in no fixture policy, so the planner (spelling correction) runs for this question.
    plan_leave = {"standalone_question": "Tell me about leave rules", "is_policy_question": True,
                  "intent": "leave overview", "key_terms": ["leave"], "identifiers": [], "exact_phrases": [],
                  "search_queries": ["leave rules"], "retrieval_emphasis": "balanced",
                  "needs_multiple_documents": False, "ambiguity": ""}
    llm = FakeLLM({"evidence_check": evidence, "answer": answer, "planning": plan_leave})
    service = RAGService(get_settings(), embedder=FakeEmbedder(), llm=llm)
    response = service.answer(ChatRequest(question="Tell me about leave rules", include_diagnostics=True))
    assert any("rules" in note for note in d_notes(response))
    d = response.diagnostics
    assert d["retrieval_rounds"] >= 1 and d["stop_reason"] in {"sufficient_evidence", "no_new_evidence",
                                                               "max_rounds_reached", "no_follow_up_queries"}
    assert response.citations and response.citations[0].relative_path == "hr/leave-policy.md"
    assert response.citations[0].heading_path[-1] == "Exception for Approved Projects"
    assert response.verification.status == "verified" and response.conversation_id is not None
    assert llm.stages().count("evidence_check") <= get_settings().max_retrieval_rounds

    # follow-up: stored history is used by the planner; the answer is again grounded in retrieved chunks
    llm.handlers["planning"] = {
        "standalone_question": "Who must approve the carryover exception for approved critical projects?",
        "is_policy_question": True, "intent": "approval", "key_terms": ["department head"], "identifiers": [],
        "exact_phrases": [], "search_queries": ["carryover exception approval department head"],
        "retrieval_emphasis": "balanced", "needs_multiple_documents": False, "ambiguity": ""}
    follow = service.answer(ChatRequest(question="Who approves that exception?", conversation_id=response.conversation_id))
    follow_planning = [c for c in llm.calls if c["stage"] == "planning"][-1]
    assert "Tell me about leave rules" in follow_planning["prompt"]
    assert follow.standalone_question.startswith("Who must approve")
    with connection() as conn:
        conv = repo.get_conversation(conn, response.conversation_id)
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["messages"][1]["payload"]["citations"][0]["chunk_id"] == response.citations[0].chunk_id


def test_prompt_injection_content_is_wrapped_as_untrusted(db, markdown_dir):
    ingest()
    captured = {}

    def answer(prompt: str):
        captured["prompt"] = prompt
        label = first_source_label(prompt, "kitchen closes")
        return {"answer_markdown": f"The kitchen closes at 8 pm [{label}].",
                "claims": [{"statement": "The kitchen closes at 8 pm", "source_ids": [label], "support": "explicit"}],
                "insufficient_evidence": False, "missing_information": [], "conflicts": []}

    llm = FakeLLM({"evidence_check": lambda p: {"sufficient": True, "confidence": "high", "relevant_source_ids": [],
                                                "missing_information": [], "follow_up_queries": []},
                   "answer": answer})
    service = RAGService(get_settings(), embedder=FakeEmbedder(), llm=llm)
    response = service.answer(ChatRequest(question="When does the office kitchen close?", persist=False))
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in captured["prompt"]  # present only as quoted data…
    assert "untrusted" in llm.calls[-1]["system"]  # …under a system rule that it is untrusted
    assert response.citations[0].relative_path == "misc/injection-test.md"


def test_catalog_questions_are_answered_from_the_index_without_gemini(db, markdown_dir):
    ingest()
    with connection() as conn:
        titles = repo.indexed_document_titles(conn)
    service = RAGService(get_settings(), embedder=FakeEmbedder(), llm=FakeLLM({}))
    count = service.answer(ChatRequest(question="How many policies are there?", persist=False))
    assert f"**{len(titles)}** policy documents" in count.answer
    assert count.verification.status == "not_applicable" and not count.insufficient_evidence
    listed = service.answer(ChatRequest(question="list all policies about remote work", persist=False))
    assert "Remote Work Policy" in listed.answer and "Acceptable Use" not in listed.answer


def test_keyword_search_adds_acronyms_learned_from_the_policies(db, markdown_dir):
    ingest()
    with connection() as conn:
        expanded = keyword_query_with_synonyms(conn, "How many PTO days can I carry over?")
    assert expanded.endswith("paid time off")
    settings = get_settings()
    with connection() as conn:
        result = hybrid_search(conn, FakeEmbedder(), ["paid time off carry over"], settings=settings)
    assert any(name.startswith("keyword:") and "pto" in name for name in result.lists)
