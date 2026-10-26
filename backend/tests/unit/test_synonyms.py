"""Synonym/acronym expansion for keyword search."""

from __future__ import annotations

import json

import pytest

from app.retrieval.synonyms import BUILTIN_GROUPS, acronym_from, expansion_terms, load_file_groups


@pytest.mark.parametrize("before, acronym, phrase", [
    ("The rules for Coordination Of Benefits ", "COB", "Coordination Of Benefits"),
    ("we follow the Order of Benefit Determination ", "OBD", "Order of Benefit Determination"),
    ("rented Durable Medical Equipment ", "DME", "Durable Medical Equipment"),
    ("and multi-factor authentication ", "MFA", None),   # hyphenated word: initials do not spell it
    ("This policy explains paid time off ", "PTO", "paid time off"),
    ("Annual leave ", "PTO", None),                      # not a definition of the acronym
])
def test_acronym_definitions_are_recognised(before, acronym, phrase):
    assert acronym_from(before, acronym) == phrase


def test_both_wordings_get_the_same_keyword_terms():
    abortion = expansion_terms("precertification requirements for abortion", BUILTIN_GROUPS)
    termination = expansion_terms("precertification requirements for termination of pregnancy", BUILTIN_GROUPS)
    assert "termination of pregnancy" in abortion and "abortion" in termination
    assert "prior authorization" in abortion and "prior authorization" in termination


def test_acronyms_expand_both_ways_and_short_ones_need_capitals():
    groups = [("cob", "coordination of benefits"), ("it", "information technology")]
    assert expansion_terms("What is COB?", groups) == ["coordination of benefits"]
    assert expansion_terms("explain coordination of benefits", groups) == ["cob"]
    assert expansion_terms("what is cob", groups) == ["coordination of benefits"]  # 3+ letters: any case
    assert expansion_terms("is it covered?", groups) == []                          # "it" is not "IT"
    assert expansion_terms("IT security rules", groups) == ["information technology"]


def test_synonyms_file_groups(tmp_path):
    path = tmp_path / "synonyms.json"
    path.write_text(json.dumps([["HRP", "health rules payer"], ["single"]]))
    assert load_file_groups(path) == [("HRP", "health rules payer")]
    path.write_text("not json")
    assert load_file_groups(path) == [] and load_file_groups(None) == []
