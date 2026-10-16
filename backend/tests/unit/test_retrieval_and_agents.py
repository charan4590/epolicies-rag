"""RRF, keyword sanitisation, query planning, citation verification."""

from __future__ import annotations

import pytest

from app.agents.answer_generator import Claim, GeneratedAnswer
from app.agents.citation_verifier import (
    lexical_support,
    parse_marker_labels,
    revision_feedback,
    strip_invalid_markers,
    verify_deterministic,
)
from app.agents.query_planner import (
    deterministic_plan,
    extract_acronyms,
    extract_identifiers,
    extract_quoted_phrases,
    is_complex,
    is_small_talk,
    plan_query,
)
from app.llm.gemini import GeminiUnavailableError, LLMUsage
from app.retrieval.keyword_search import keyword_terms, word_boundary_pattern
from app.retrieval.ranking import RankedList, dedupe_by_content, reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk
from app.services.rag_service import renumber_markers
from tests.fakes import FakeLLM


# --------------------------------------------------------------------------- RRF
def test_rrf_formula_and_ordering():
    fused = reciprocal_rank_fusion(
        [RankedList("vector", ["a", "b", "c"]), RankedList("keyword", ["c", "a", "d"])], k=60
    )
    scores = {f.chunk_id: f.score for f in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [f.chunk_id for f in fused] == ["a", "c", "b", "d"]
    assert fused[0].contributions == {"vector": 1, "keyword": 2}


def test_rrf_weights_duplicates_and_determinism():
    fused = reciprocal_rank_fusion([RankedList("x", ["a", "a", "b"]), RankedList("y", ["b"], weight=2.0),
                                    RankedList("z", ["a"], weight=0)], k=10)
    scores = {f.chunk_id: f.score for f in fused}
    assert scores["a"] == pytest.approx(1 / 11)  # duplicate within a list counted once; weight 0 ignored
    assert scores["b"] == pytest.approx(1 / 12 + 2 / 11)
    tie = reciprocal_rank_fusion([RankedList("x", ["q"]), RankedList("y", ["p"])], k=60)
    assert [f.chunk_id for f in tie] == ["p", "q"]  # equal score & rank → chunk id order
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)


def test_dedupe_by_content_keeps_best_ranked():
    kept, dropped = dedupe_by_content(["a", "b", "c", "d"], {"a": "h1", "b": "h2", "c": "h1", "d": "h3"})
    assert kept == ["a", "b", "d"] and dropped == {"a": ["c"]}


# --------------------------------------------------------------------------- keyword sanitisation
def test_keyword_terms_strip_operators_and_question_words():
    assert keyword_terms("What is the PTO carryover policy for HR-POL-012?") == ["pto", "carryover", "hr", "pol", "012"]
    assert keyword_terms("a & b | !c <-> (d)") == []
    assert keyword_terms("'; DROP TABLE chunks; --") == ["drop", "table", "chunks"]


def test_word_boundary_pattern_escapes_everything():
    assert word_boundary_pattern("401(k)") == r"(^|[^[:alnum:]_])401\(k\)($|[^[:alnum:]_])"
    assert word_boundary_pattern("HR-POL-012") == r"(^|[^[:alnum:]_])HR\-POL\-012($|[^[:alnum:]_])"
    assert r"\s+" in word_boundary_pattern("Section 4.2")


# --------------------------------------------------------------------------- planner
def test_identifier_and_phrase_extraction():
    ids = extract_identifiers("Does HR-POL-012 or Section 4.2 mention the 401(k) and ISO27001?")
    assert {"HR-POL-012", "Section 4.2", "401(k)", "ISO27001"} <= set(ids)
    assert extract_acronyms("Is MFA required on the VPN for PTO requests?") == ["MFA", "VPN", "PTO"]
    assert extract_quoted_phrases('What does "critical project" mean?') == ["critical project"]
    assert is_small_talk("Hello!") and not is_small_talk("Hello, what is the leave policy?")
    assert is_complex("Compare the remote work rules in HR and IT")
    assert not is_complex("How many sick days do I get?")


def test_shouting_does_not_create_acronym_identifiers():
    plan = deterministic_plan("WHAT IS THE LEAVE POLICY")
    assert plan.identifiers == []


def test_simple_question_uses_no_llm():
    llm = FakeLLM()
    plan = plan_query("How many sick days do I get?", [], llm=llm, mode="auto", usage=LLMUsage(), max_queries=3)
    assert llm.calls == [] and plan.planner == "deterministic"
    assert plan.search_queries == ["How many sick days do I get?"]


def test_follow_up_is_rewritten_by_llm():
    llm = FakeLLM({"planning": {
        "standalone_question": "What is the exception to the PTO carryover limit?",
        "is_policy_question": True, "intent": "exception lookup", "key_terms": ["carryover", "exception"],
        "identifiers": ["PTO"], "exact_phrases": [], "search_queries": ["PTO carryover exception", "carryover 10 days"],
        "retrieval_emphasis": "balanced", "needs_multiple_documents": False, "ambiguity": "",
    }})
    history = [{"role": "user", "content": "How many PTO days can I carry over?"},
               {"role": "assistant", "content": "Up to 5 days [S1]."}]
    plan = plan_query("What about the exception?", history, llm=llm, mode="auto", usage=LLMUsage(), max_queries=3)
    assert plan.planner == "llm"
    assert plan.standalone_question == "What is the exception to the PTO carryover limit?"
    assert plan.search_queries[0] == plan.standalone_question and len(plan.search_queries) <= 3
    assert "Up to 5 days" in llm.calls[0]["prompt"]  # history provided as context


def test_planner_falls_back_when_llm_fails():
    llm = FakeLLM({"planning": GeminiUnavailableError("down")})
    history = [{"role": "user", "content": "Tell me about remote work"}]
    plan = plan_query("and the stipend?", history, llm=llm, mode="auto", usage=LLMUsage(), max_queries=3)
    assert plan.planner == "deterministic_fallback"
    assert any("remote work" in q for q in plan.search_queries)
    assert plan.notes


# --------------------------------------------------------------------------- citation verification
def chunk(cid: str, doc: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id=doc, chunk_index=0, text=text, title="Leave Policy",
                          relative_path="hr/leave.md", heading_path=["Leave"], source_urls=[],
                          document_source_url=None, content_hash=cid)


def test_marker_parsing_and_stripping():
    assert parse_marker_labels("A [S1]. B [S2, S3]. C [S1][S4]") == ["S1", "S2", "S3", "S4"]
    assert strip_invalid_markers("A [S1]. B [S2, S9].", {"S1", "S2"}) == "A [S1]. B [S2]."
    assert renumber_markers("x [S3] y [S1][S3] z [S2]", {"S3": "S1", "S1": "S2"}) == "x [S1] y [S2][S1] z "


def test_lexical_support_requires_numbers():
    coverage, missing = lexical_support("Employees may carry over up to 5 PTO days",
                                        "Employees may carry over up to 5 unused PTO days into the next year")
    assert coverage == 1.0 and missing == []
    _, missing = lexical_support("Employees may carry over 7 days", "carry over up to 5 days")
    assert missing == ["7"]


def test_verify_deterministic_checks_ids_and_support():
    s1 = chunk("c1", "d1", "Employees may carry over up to 5 unused PTO days.")
    s2 = chunk("c2", "d2", "Hotel costs are reimbursed up to $250 per night.")
    s3 = chunk("c3", "d3", "not retrieved")
    label_map = {"S1": s1, "S2": s2, "S3": s3}
    answer = GeneratedAnswer(
        answer_markdown="You can carry over 5 days [S1]. Hotels: $250 [S2]. Ghost [S9]. Other [S3].",
        claims=[
            Claim(statement="Employees may carry over up to 5 PTO days", source_ids=["S1"], support="explicit"),
            Claim(statement="Hotel costs are reimbursed up to $300 per night", source_ids=["S2"], support="explicit"),
            Claim(statement="Unicorns approve leave", source_ids=[], support="explicit"),
        ],
        insufficient_evidence=False, missing_information=[], conflicts=[],
    )
    db_rows = {"c1": {"document_id": "d1", "status": "indexed"}, "c2": {"document_id": "WRONG", "status": "indexed"}}
    cleaned, report = verify_deterministic(answer, label_map, {"c1", "c2"}, lambda ids: db_rows)
    assert report.valid_labels == ["S1"]
    assert set(report.removed_labels) == {"S2", "S3", "S9"}
    assert "[S9]" not in cleaned and "[S2]" not in cleaned and "[S1]" in cleaned
    statuses = [c.status for c in report.claims]
    assert statuses == ["supported", "uncited", "uncited"]  # S2 invalid → its claim becomes uncited
    feedback = revision_feedback(report)
    assert "Uncited statement" in feedback and "S9" in feedback


def test_weak_claim_detected_when_number_missing():
    s1 = chunk("c1", "d1", "Hotel costs are reimbursed up to $250 per night.")
    answer = GeneratedAnswer(answer_markdown="Up to $300 [S1].",
                             claims=[Claim(statement="Hotel costs are reimbursed up to $300 per night",
                                           source_ids=["S1"], support="explicit")],
                             insufficient_evidence=False, missing_information=[], conflicts=[])
    _, report = verify_deterministic(answer, {"S1": s1}, {"c1"},
                                     lambda ids: {"c1": {"document_id": "d1", "status": "indexed"}})
    assert report.claims[0].status == "weak" and report.claims[0].missing_numbers == ["300"]
