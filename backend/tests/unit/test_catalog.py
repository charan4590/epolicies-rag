"""Questions about the document collection are answered from the index, not from retrieval."""

from __future__ import annotations

import pytest

from app.agents.catalog import build_catalog_answer, parse_catalog_question

TITLES = [(t, f"{t}.md") for t in ["Abortion - Policy", "Ambulance - Medicare HRP Policy",
                                    "Coordination Of Benefits (COB) - Policy",
                                    "Ambulance - Medicare HRP Claim Processing"]]


@pytest.mark.parametrize("question, kind, terms", [
    ("how many policies are there", "count", []),
    ("How many policies are there?", "count", []),
    ("What is the total number of policies?", "count", []),
    ("how many documents do you have", "count", []),
    ("name all the policies", "list", []),
    ("List all policies", "list", []),
    ("show me all the documents", "list", []),
    ("Which policies do you have?", "list", []),
    ("list all policies about Medicare HRP", "list", ["Medicare", "HRP"]),
    ("how many policies related to ambulance", "count", ["ambulance"]),
])
def test_recognises_catalog_questions(question, kind, terms):
    parsed = parse_catalog_question(question)
    assert parsed is not None and parsed.kind == kind and parsed.filter_terms == terms


@pytest.mark.parametrize("question", [
    "how many policies cover abortion?",          # needs the policy text
    "What is the abortion policy?",
    "Which policies cover ambulance transport?",
    "What are the policies on coordination of benefits?",
    "Under COB, how many policies apply?",
    "list all policies about",                    # filter word missing
])
def test_leaves_content_questions_to_rag(question):
    assert parse_catalog_question(question) is None


def test_count_answer_uses_real_number_of_titles():
    answer = build_catalog_answer(parse_catalog_question("how many policies are there"), TITLES)
    assert "**4** policy documents" in answer and "Abortion" not in answer


def test_list_answer_names_every_document_sorted():
    answer = build_catalog_answer(parse_catalog_question("name all the policies"), TITLES)
    lines = [line for line in answer.splitlines() if line[:1].isdigit()]
    assert lines[0] == "1. Abortion - Policy" and len(lines) == 4


def test_documents_sharing_a_title_are_both_counted_and_told_apart():
    docs = TITLES + [("Abortion - Policy", "archive/Abortion - Policy.md")]
    count = build_catalog_answer(parse_catalog_question("how many policies are there"), docs)
    assert "**5** policy documents" in count
    listed = build_catalog_answer(parse_catalog_question("list all policies"), docs)
    assert "Abortion - Policy (archive/Abortion - Policy.md)" in listed
    assert "Abortion - Policy (Abortion - Policy.md)" in listed


def test_filtered_answer_and_no_match_falls_back():
    answer = build_catalog_answer(parse_catalog_question("list all policies about ambulance"), TITLES)
    assert "**2**" in answer and "Coordination" not in answer
    assert build_catalog_answer(parse_catalog_question("list all policies about dental"), TITLES) is None
