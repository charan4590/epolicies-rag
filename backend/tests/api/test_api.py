"""FastAPI endpoint tests. DB-backed tests require TEST_DATABASE_URL."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.config import ConfigurationError, get_settings
from app.database.connection import DatabaseUnavailableError
from app.ingestion.pipeline import IngestionPipeline
from app.main import app
from app.schemas.api_models import ChatResponse, VerificationInfo
from app.services.rag_service import RAGService
from tests.conftest import requires_db
from tests.fakes import FakeEmbedder, FakeLLM, first_source_label

ADMIN = {"X-Admin-Token": "test-admin-token"}


class StubService:
    """Minimal service used for error-path tests that need no database."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.embedder = FakeEmbedder()

    def answer(self, request, on_event=None, is_cancelled=None):
        if on_event:
            on_event({"type": "stage", "stage": "planning", "message": "Understanding the question"})
        if self.exc:
            raise self.exc
        return ChatResponse(answer="ok", citations=[], insufficient_evidence=False,
                            verification=VerificationInfo(status="not_applicable"),
                            standalone_question=request.question)


@pytest.fixture()
def client():
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def use_service(service) -> None:
    app.dependency_overrides[deps.get_rag_service] = lambda: service


# --------------------------------------------------------------------------- no database needed
def test_chat_validation_errors(client):
    use_service(StubService())
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = client.post("/chat", json={"question": "q", "unexpected": "field"})
    assert r.status_code == 422
    r = client.post("/chat", content=b"x" * (70 * 1024), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_internal_errors_do_not_leak_details(client):
    use_service(StubService(RuntimeError("secret connection string postgresql://user:pw@host")))
    r = client.post("/chat", json={"question": "What is PTO?"})
    body = r.json()
    assert r.status_code == 500 and body["error"]["code"] == "internal_error"
    assert "postgresql" not in r.text and "Traceback" not in r.text and body["error"]["request_id"]
    assert r.headers["X-Request-ID"] == body["error"]["request_id"]


@pytest.mark.parametrize("exc,status,code", [
    (ConfigurationError("GOOGLE_API_KEY is not set."), 503, "configuration_error"),
    (DatabaseUnavailableError("Cannot connect to PostgreSQL."), 503, "database_unavailable"),
])
def test_configuration_and_database_failures(client, exc, status, code):
    use_service(StubService(exc))
    r = client.post("/chat", json={"question": "What is PTO?"})
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_stream_emits_stage_and_final_events(client):
    use_service(StubService())
    with client.stream("POST", "/chat/stream", json={"question": "What is PTO?"}) as r:
        body = "".join(r.iter_text())
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["stage", "final"]
    assert events[-1]["response"]["answer"] == "ok"


def test_stream_reports_errors_as_events(client):
    use_service(StubService(ConfigurationError("GEMINI_MODEL is not set.")))
    with client.stream("POST", "/chat/stream", json={"question": "What is PTO?"}) as r:
        body = "".join(r.iter_text())
    error = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")][-1]
    assert error["type"] == "error" and error["error"]["code"] == "configuration_error"


def test_admin_endpoints_require_token(client, monkeypatch):
    assert client.post("/ingest", json={}).status_code == 401
    assert client.post("/ingest", json={}, headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.get("/ingest/runs").status_code == 401
    r = client.post("/ingest", json={"mode": "incremental", "path": "/etc"}, headers=ADMIN)
    assert r.status_code == 422
    monkeypatch.setenv("ADMIN_API_TOKEN", "")
    get_settings.cache_clear()
    try:
        r = client.post("/ingest", json={}, headers=ADMIN)
        assert r.status_code == 403 and "disabled" in r.json()["error"]["message"]
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_health_reports_database_down(client, monkeypatch):
    def boom():
        raise DatabaseUnavailableError("Cannot connect to PostgreSQL: connection refused.")

    monkeypatch.setattr("app.api.routes_documents.connection", boom)
    r = client.get("/health")
    body = r.json()
    assert r.status_code == 503 and body["status"] == "error" and body["database"]["ok"] is False
    assert "test-key-not-real" not in r.text and "password" not in r.text.lower()


def test_cors_allows_local_frontend_only(client):
    r = client.options("/chat", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
    r = client.options("/chat", headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_openapi_docs_available(client):
    schema = client.get("/openapi.json").json()
    assert {"/health", "/chat", "/chat/stream", "/documents", "/documents/{document_id}", "/search", "/ingest"} <= set(schema["paths"])


# --------------------------------------------------------------------------- database-backed
@requires_db
def test_health_documents_search_and_chat(db, client, markdown_dir):
    IngestionPipeline(get_settings(), embedder_factory=FakeEmbedder, progress=lambda m: None).run()
    embedder = FakeEmbedder()

    def answer(prompt: str):
        label = first_source_label(prompt, "$250")
        return {"answer_markdown": f"| Item | Cap |\n|---|---|\n| Hotel | $250 per night [{label}] |",
                "claims": [{"statement": "Hotel costs are reimbursed up to $250 per night", "source_ids": [label],
                            "support": "explicit"}],
                "insufficient_evidence": False, "missing_information": [], "conflicts": []}

    llm = FakeLLM({"evidence_check": {"sufficient": True, "confidence": "high", "relevant_source_ids": ["S1"],
                                      "missing_information": [], "follow_up_queries": []},
                   "answer": answer})
    use_service(RAGService(get_settings(), embedder=embedder, llm=llm))

    health = client.get("/health").json()
    assert health["status"] == "ok" and health["database"]["pgvector_enabled"]
    assert health["index"]["indexed_documents"] == 9 and health["database"]["vector_dimensions"] == 768

    docs = client.get("/documents", params={"page_size": 5}).json()
    assert docs["total"] == 9 and len(docs["items"]) == 5
    filtered = client.get("/documents", params={"q": "remote"}).json()
    assert {d["relative_path"] for d in filtered["items"]} == {"hr/remote-work-policy.md", "it/remote-work-policy.md"}
    doc_id = filtered["items"][0]["document_id"]
    detail = client.get(f"/documents/{doc_id}").json()
    assert detail["chunks"] and detail["chunk_count"] == len(detail["chunks"])
    assert client.get("/documents/doc_000000000000000000000000").status_code == 404
    assert client.get("/documents/../../etc/passwd").status_code == 404

    search = client.get("/search", params={"q": "HR-POL-019 internet reimbursement", "mode": "hybrid"}).json()
    assert search["hits"][0]["relative_path"] == "hr/remote-work-policy.md" and "HR-POL-019" in search["identifiers"]
    assert any(k.startswith("vector:") for k in search["diagnostics"]["lists"])
    kw_only = client.get("/search", params={"q": "bereavement", "mode": "keyword"}).json()
    assert kw_only["hits"] and kw_only["hits"][0]["vector_rank"] is None

    r = client.post("/chat", json={"question": "What is the hotel cap per night?", "include_diagnostics": True})
    body = r.json()
    assert r.status_code == 200, body
    assert "travel-expenses" in body["citations"][0]["relative_path"]  # either identical duplicate
    assert "| Hotel |" in body["answer"] and body["verification"]["status"] == "verified"
    assert body["diagnostics"]["models"]["embedding"] == "gemini-embedding-001"
    conv = client.get(f"/conversations/{body['conversation_id']}").json()
    assert len(conv["messages"]) == 2
    assert any(c["conversation_id"] == body["conversation_id"] for c in client.get("/conversations").json())
    assert client.delete(f"/conversations/{body['conversation_id']}").status_code == 204
    assert client.get(f"/conversations/{body['conversation_id']}").status_code == 404

    status = client.get("/ingest/source-status").json()
    assert status["discovered"] == 9 and status["unchanged"] == 9 and status["new"] == []


@requires_db
def test_ingest_endpoint_runs_in_background(db, client, markdown_dir, monkeypatch):
    monkeypatch.setattr("app.api.routes_ingestion.IngestionPipeline",
                        lambda settings: IngestionPipeline(settings, embedder_factory=FakeEmbedder,
                                                           progress=lambda m: None))
    dry = client.post("/ingest", json={"dry_run": True}, headers=ADMIN).json()
    assert dry["status"] == "dry_run" and dry["message"].startswith("9 new")
    r = client.post("/ingest", json={"mode": "incremental"}, headers=ADMIN)
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    for _ in range(100):
        run = client.get(f"/ingest/runs/{run_id}", headers=ADMIN).json()
        if run["status"] != "running":
            break
        time.sleep(0.1)
    assert run["status"] == "completed", run
    assert run["report"]["documents_indexed"] == 9 and "source_dir" not in run["report"]
    assert client.get("/ingest/runs", headers=ADMIN).json()[0]["run_id"] == run_id
