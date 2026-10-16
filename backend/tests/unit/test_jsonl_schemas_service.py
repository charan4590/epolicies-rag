"""JSONL serialisation/validation, API schemas, and RAG service logic without a database."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agents.retrieval_agent import RetrievalAgent, RetrievalOutcome
from app.config import get_settings
from app.ingestion import jsonl_store
from app.retrieval.types import RetrievedChunk
from app.schemas.api_models import ChatRequest, IngestRequest
from app.services.rag_service import NO_EVIDENCE_REPLY, SMALL_TALK_REPLY, RAGService
from tests.fakes import FakeEmbedder, FakeLLM, first_source_label


def record(i: int, dims: int = 4, **overrides) -> dict:
    base = {
        "document_id": "doc_a", "chunk_id": f"chk_{i}", "source_file": "hr/a.md", "text": f"text {i}",
        "embedding_model": "gemini-embedding-001", "embedding_dimensions": dims, "embedding": [0.1] * dims,
        "metadata": {"title": "A", "heading_path": ["A"], "source_urls": [], "chunk_index": i,
                     "content_hash": f"h{i}"},
    }
    base.update(overrides)
    return base


def write_jsonl(path: Path, records: list[dict], *, manifest: bool = True) -> None:
    def writer(emit):
        for r in records:
            emit((json.dumps(r) + "\n").encode())

    sha = jsonl_store._atomic_write_bytes(path, writer)
    if manifest:
        jsonl_store.manifest_path(path).write_text(json.dumps({"complete": True, "sha256": sha,
                                                               "records": len(records)}))


# --------------------------------------------------------------------------- JSONL
def test_valid_jsonl_with_manifest(tmp_path: Path):
    path = tmp_path / "embedding.jsonl"
    write_jsonl(path, [record(0), record(1)])
    result = jsonl_store.validate_jsonl(path, expected_dimensions=4)
    assert result.valid and result.records == 2 and result.documents == 1 and result.manifest_ok
    for line in path.read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["embedding_dimensions"] == 4


def test_jsonl_detects_problems(tmp_path: Path):
    path = tmp_path / "e.jsonl"
    bad = [record(0), record(0), record(2, embedding=[0.1, 0.2]), record(3, dims=4, embedding=[0.1, float("nan"), 0, 0])]
    write_jsonl(path, bad[:3])
    path.write_text(path.read_text() + "{not json\n" + json.dumps(bad[3], allow_nan=True) + "\n")
    errors = " | ".join(jsonl_store.validate_jsonl(path, expected_dimensions=4).errors)
    assert "duplicate chunk_id" in errors
    assert "embedding has 2 values" in errors
    assert "invalid JSON" in errors
    assert "non-finite" in errors
    assert "manifest missing or does not match" in errors  # file changed after manifest was written


def test_partial_file_without_manifest_is_not_complete(tmp_path: Path):
    path = tmp_path / "e.jsonl"
    write_jsonl(path, [record(0)], manifest=False)
    result = jsonl_store.validate_jsonl(path)
    assert not result.valid and not result.manifest_ok


def test_validate_record_missing_fields():
    assert "missing field 'embedding'" in jsonl_store.validate_record({k: v for k, v in record(0).items()
                                                                       if k != "embedding"})
    rec = record(0)
    del rec["metadata"]["heading_path"]
    assert jsonl_store.validate_record(rec) == ["metadata missing 'heading_path'"]


def test_atomic_write_leaves_no_partial_file(tmp_path: Path):
    target = tmp_path / "out.jsonl"
    target.write_text("previous complete content")

    def failing(emit):
        emit(b"partial")
        raise RuntimeError("crash mid-write")

    with pytest.raises(RuntimeError):
        jsonl_store._atomic_write_bytes(target, failing)
    assert target.read_text() == "previous complete content"
    assert list(tmp_path.iterdir()) == [target]


def test_chunk_row_to_record_shape():
    from pgvector import Vector

    row = {"document_id": "doc_x", "chunk_id": "chk_x", "relative_path": "a/b.md", "text": "body",
           "embedding_model": "gemini-embedding-001", "embedding": Vector([0.5, -0.25]), "title": "T",
           "heading_path": ["H"], "source_urls": ["https://x.example.com"], "chunk_index": 0, "content_hash": "h",
           "token_count": 1, "char_count": 4, "embedding_config": "cfg", "document_content_hash": "dh",
           "document_source_url": None, "filename": "b.md", "chunk_metadata": {"has_table": False}}
    rec = jsonl_store.chunk_row_to_record(row)
    assert rec["embedding"] == [0.5, -0.25] and rec["embedding_dimensions"] == 2
    assert jsonl_store.validate_record(rec) == []
    assert "GOOGLE_API_KEY" not in json.dumps(rec)


# --------------------------------------------------------------------------- schemas
def test_chat_request_validation():
    assert ChatRequest(question="  What is PTO?  ").question == "What is PTO?"
    for bad in ({"question": "   "}, {"question": "x" * 2001}, {"question": "q", "sql": "DROP"},
                {"question": "q", "history": [{"role": "system", "content": "x"}]},
                {"question": "q", "history": [{"role": "user", "content": "x"}] * 11},
                {"question": "q", "conversation_id": "not-a-uuid"}):
        with pytest.raises(ValidationError):
            ChatRequest(**bad)


def test_ingest_request_cannot_choose_paths():
    with pytest.raises(ValidationError):
        IngestRequest(mode="incremental", path="/etc")
    with pytest.raises(ValidationError):
        IngestRequest(mode="delete_everything")


# --------------------------------------------------------------------------- service (no DB)
def rc(cid: str, text: str, title: str = "Employee Leave Policy") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id="doc_" + cid, chunk_index=0, text=text, title=title,
                          relative_path=f"hr/{cid}.md", heading_path=[title, "Carryover"], source_urls=[],
                          document_source_url=None, content_hash="h" + cid)


@pytest.fixture()
def service(monkeypatch):
    svc = RAGService(get_settings(), embedder=FakeEmbedder(), llm=FakeLLM())
    lookup = {}
    monkeypatch.setattr(svc, "_lookup", lambda ids: {i: lookup[i] for i in ids if i in lookup})
    svc.lookup_rows = lookup
    return svc


def test_small_talk_skips_retrieval(service, monkeypatch):
    monkeypatch.setattr(RetrievalAgent, "run", lambda *a, **k: pytest.fail("retrieval must not run"))
    response = service.answer(ChatRequest(question="hello", persist=False))
    assert response.answer == SMALL_TALK_REPLY and response.citations == []


def test_no_evidence_is_reported_without_generation(service, monkeypatch):
    monkeypatch.setattr(RetrievalAgent, "run", lambda *a, **k: RetrievalOutcome(chunks=[], all_retrieved_ids=set(),
                                                                                  stop_reason="no_results"))
    response = service.answer(ChatRequest(question="What is the policy on pet llamas?", persist=False,
                                          include_diagnostics=True))
    assert response.insufficient_evidence and response.answer == NO_EVIDENCE_REPLY
    assert response.citations == [] and service.llm.calls == []
    assert response.diagnostics["stop_reason"] == "no_results"


def test_answer_with_verified_citations_and_revision(service, monkeypatch):
    chunks = [rc("a", "Employees may carry over up to 5 unused PTO days into the next calendar year."),
              rc("b", "Hotel costs are reimbursed up to $250 per night.", "Travel Policy")]
    for c in chunks:
        service.lookup_rows[c.chunk_id] = {"document_id": c.document_id, "status": "indexed"}
    monkeypatch.setattr(RetrievalAgent, "run", lambda *a, **k: RetrievalOutcome(
        chunks=chunks, all_retrieved_ids={"a", "b"}, stop_reason="sufficient_evidence"))

    def answer(prompt: str):
        label = first_source_label(prompt, "carry over")
        if "Citation verification found these problems" in prompt:
            return {"answer_markdown": f"You may carry over up to 5 unused PTO days [{label}].",
                    "claims": [{"statement": "Employees may carry over up to 5 unused PTO days",
                                "source_ids": [label], "support": "explicit"}],
                    "insufficient_evidence": False, "missing_information": [], "conflicts": []}
        return {"answer_markdown": f"You may carry over up to 5 days [{label}]. You also get a bonus [S7].",
                "claims": [{"statement": "Employees may carry over up to 5 unused PTO days", "source_ids": [label],
                            "support": "explicit"},
                           {"statement": "Employees get a carryover bonus of 3 days", "source_ids": [label],
                            "support": "explicit"}],
                "insufficient_evidence": False, "missing_information": [], "conflicts": []}

    service.llm.handlers.update({
        "answer": answer,
        "revision": answer,
        "citation_verification": lambda prompt: {"verdicts": [
            {"claim_index": 1, "verdict": "unsupported", "missing_qualifications": [], "explanation": "not stated"}]},
    })
    response = service.answer(ChatRequest(question="How many PTO days can I carry over?", persist=False,
                                          include_diagnostics=True))
    assert service.llm.stages() == ["answer", "citation_verification", "revision"]
    assert response.verification.status == "revised" and response.verification.revised
    assert response.answer == "You may carry over up to 5 unused PTO days [S1]."
    assert [c.chunk_id for c in response.citations] == ["a"]
    assert response.citations[0].label == "S1" and response.citations[0].verification == "supported"
    assert response.diagnostics["llm_usage"]["calls"] == 3


def test_answer_keeps_conflicts(service, monkeypatch):
    chunks = [rc("h", "Employees may work remotely up to 3 days per week.", "Remote Work Policy"),
              rc("i", "IT department employees may work remotely up to 2 days per week.", "IT Remote Access")]
    for c in chunks:
        service.lookup_rows[c.chunk_id] = {"document_id": c.document_id, "status": "indexed"}
    monkeypatch.setattr(RetrievalAgent, "run", lambda *a, **k: RetrievalOutcome(
        chunks=chunks, all_retrieved_ids={"h", "i"}, stop_reason="sufficient_evidence"))
    service.llm.handlers["answer"] = {
        "answer_markdown": "HR allows 3 days [S1] but IT staff are limited to 2 days [S2].",
        "claims": [{"statement": "Employees may work remotely up to 3 days per week", "source_ids": ["S1"],
                    "support": "explicit"},
                   {"statement": "IT department employees may work remotely up to 2 days per week",
                    "source_ids": ["S2"], "support": "explicit"}],
        "insufficient_evidence": False, "missing_information": [],
        "conflicts": [{"description": "Remote days differ between HR and IT", "source_ids": ["S1", "S2"]}]}
    response = service.answer(ChatRequest(question="How many remote days are allowed?", persist=False))
    assert response.verification.status == "verified"
    assert response.conflicts[0].labels == ["S1", "S2"] and len(response.citations) == 2
    assert service.llm.stages() == ["answer"]  # no LLM verification needed for well-supported claims


def test_conversation_history_is_bounded(service):
    req = ChatRequest(question="q", persist=False,
                      history=[{"role": "user", "content": f"m{i}"} for i in range(10)])
    assert len(service._load_history(req)) == get_settings().max_history_turns * 2
    assert uuid.UUID(int=0)  # keep uuid import used
