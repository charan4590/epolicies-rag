"""Stage 1-2: query understanding and retrieval planning.

A deterministic analyser always runs (identifiers, quoted phrases, acronyms,
complexity). Gemini is consulted only when it adds value: when there is
conversation history to resolve, or the question looks complex/ambiguous
(PLANNER_MODE=auto), or always/never as configured.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.prompts import PLANNER_SYSTEM, format_history
from app.llm.gemini import GeminiError, GeminiLLM, LLMUsage

logger = logging.getLogger(__name__)

_IDENTIFIER_PATTERNS = [
    re.compile(r"\b[A-Z][A-Z0-9]{1,9}(?:[-_/.][A-Z0-9]{1,10}){1,4}\b"),  # HR-POL-012, IT-SEC-004, ISO/IEC
    re.compile(r"\b(?:section|sec\.?|clause|article|appendix|annex|form|policy)\s+[A-Z]?\d+(?:\.\d+)*[a-z]?\b",
               re.IGNORECASE),
    re.compile(r"§\s*\d+(?:\.\d+)*"),
    re.compile(r"\b\d{3}\([a-z]\)"),  # 401(k)
    re.compile(r"\b[A-Z]{1,6}\d{2,}[A-Z0-9]*\b"),  # GDPR2016, ISO27001
]
_ACRONYM = re.compile(r"\b[A-Z]{2,6}s?\b")
_QUOTED = re.compile(r"[\"“”']([^\"“”']{3,80})[\"“”']")
_COMPLEX_MARKERS = re.compile(
    r"\b(compare|comparison|difference|differ|versus|vs\.?|both|all (?:the )?polic|across|each|every|"
    r"conflict|contradict|summari[sz]e|list all|which documents|relationship|and also|as well as|"
    r"overview|exceptions?)\b",
    re.IGNORECASE,
)
_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening)|ok(?:ay)?|cheers)[\s!.?]*$",
    re.IGNORECASE,
)
_ACRONYM_STOP = {"I", "A", "OK", "US", "AM", "PM", "IT"}  # IT is ambiguous as a word ("is it")


class LLMQueryPlan(BaseModel):
    standalone_question: str
    is_policy_question: bool
    intent: str = Field(description="One short sentence describing what the user wants")
    key_terms: list[str]
    identifiers: list[str]
    exact_phrases: list[str]
    search_queries: list[str]
    retrieval_emphasis: Literal["semantic", "keyword", "balanced"]
    needs_multiple_documents: bool
    ambiguity: str


@dataclass
class QueryPlan:
    original_question: str
    standalone_question: str
    is_policy_question: bool = True
    intent: str = ""
    key_terms: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    exact_phrases: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    retrieval_emphasis: str = "balanced"
    needs_multiple_documents: bool = False
    ambiguity: str = ""
    planner: str = "deterministic"  # deterministic | llm | deterministic_fallback
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def extract_identifiers(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if value not in found:
                found.append(value)
    return found


def extract_acronyms(text: str) -> list[str]:
    acronyms: list[str] = []
    for match in _ACRONYM.finditer(text):
        word = match.group(0)
        base = word[:-1] if word.endswith("s") and word[:-1].isupper() else word
        if base not in _ACRONYM_STOP and base not in acronyms:
            acronyms.append(base)
    return acronyms


def extract_quoted_phrases(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(1).strip() for m in _QUOTED.finditer(text)))


def is_small_talk(text: str) -> bool:
    return bool(_SMALL_TALK.match(text))


def is_complex(question: str) -> bool:
    words = len(question.split())
    return (
        words > 22
        or question.count("?") > 1
        or bool(_COMPLEX_MARKERS.search(question))
        or len(re.findall(r"\band\b", question, re.IGNORECASE)) >= 2
    )


def deterministic_plan(question: str, history_hint: str | None = None) -> QueryPlan:
    identifiers = extract_identifiers(question)
    letters = [ch for ch in question if ch.isalpha()]
    shouting = bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) > 0.5
    acronyms = [] if shouting else [a for a in extract_acronyms(question) if not any(a in i for i in identifiers)]
    phrases = extract_quoted_phrases(question)
    queries = [question]
    if history_hint:
        # Without an LLM rewrite, add the previous user question as context for retrieval.
        queries.append(f"{history_hint} {question}")
    return QueryPlan(
        original_question=question,
        standalone_question=question,
        is_policy_question=not is_small_talk(question),
        key_terms=acronyms,
        identifiers=identifiers + acronyms,
        exact_phrases=phrases,
        search_queries=queries,
        retrieval_emphasis="keyword" if identifiers else "balanced",
        needs_multiple_documents=is_complex(question),
    )


def plan_query(
    question: str,
    history: Sequence[dict[str, str]],
    *,
    llm: GeminiLLM | None,
    mode: str,
    usage: LLMUsage,
    max_queries: int,
) -> QueryPlan:
    question = question.strip()
    previous_user = next((t["content"] for t in reversed(history) if t.get("role") == "user"), None)
    base = deterministic_plan(question)
    if base.is_policy_question is False and not history:
        return base

    use_llm = mode == "always" or (mode == "auto" and (bool(history) or is_complex(question)))
    if not use_llm or llm is None:
        plan = deterministic_plan(question, history_hint=previous_user if history else None)
        if history and llm is None:
            plan.notes.append("No LLM available to resolve the follow-up; used previous question as context.")
        return plan

    prompt = (
        f"Conversation history (most recent last):\n{format_history(history)}\n\n"
        f"Latest user question:\n{question}\n\n"
        f"Deterministically detected identifiers: {base.identifiers or 'none'}\n"
        f"Return at most {max_queries} search queries."
    )
    try:
        result = llm.generate_json(stage="planning", system=PLANNER_SYSTEM, prompt=prompt, schema=LLMQueryPlan,
                                   usage=usage, max_output_tokens=1024)
    except GeminiError as exc:
        plan = deterministic_plan(question, history_hint=previous_user if history else None)
        plan.planner = "deterministic_fallback"
        plan.notes.append(f"LLM planning unavailable ({exc.kind}); used deterministic analysis.")
        logger.warning("planner_fallback", extra={"kind": exc.kind})
        return plan

    standalone = result.standalone_question.strip() or question
    queries = [q.strip() for q in result.search_queries if q and q.strip()][:max_queries]
    if standalone not in queries:
        queries.insert(0, standalone)
    queries = queries[: max(1, max_queries)]
    # Union of LLM- and regex-detected identifiers; regex results are always kept.
    identifiers = list(dict.fromkeys(base.identifiers + [i.strip() for i in result.identifiers if i.strip()]))
    return QueryPlan(
        original_question=question,
        standalone_question=standalone,
        is_policy_question=result.is_policy_question or bool(base.identifiers),
        intent=result.intent,
        key_terms=result.key_terms[:12],
        identifiers=identifiers[:10],
        exact_phrases=list(dict.fromkeys(base.exact_phrases + result.exact_phrases))[:5],
        search_queries=queries,
        retrieval_emphasis=result.retrieval_emphasis,
        needs_multiple_documents=result.needs_multiple_documents,
        ambiguity=result.ambiguity.strip(),
        planner="llm",
    )
