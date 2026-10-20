"""Stage 5: grounded answer generation (structured output with claims)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from app.agents.prompts import ANSWER_SYSTEM, format_history, format_sources
from app.llm.gemini import GeminiLLM, LLMUsage
from app.retrieval.types import RetrievedChunk


class Claim(BaseModel):
    statement: str
    source_ids: list[str]
    support: Literal["explicit", "interpretation"]


class Conflict(BaseModel):
    description: str
    source_ids: list[str]


class GeneratedAnswer(BaseModel):
    answer_markdown: str
    claims: list[Claim]
    insufficient_evidence: bool
    missing_information: list[str]
    conflicts: list[Conflict]


def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    labels: Sequence[str],
    *,
    llm: GeminiLLM,
    usage: LLMUsage,
    history: Sequence[dict[str, str]] = (),
    missing_hint: Sequence[str] = (),
    revision_feedback: str | None = None,
    previous_answer: str | None = None,
    max_chars_each: int = 3000,
) -> GeneratedAnswer:
    sections = []
    if history:
        sections.append(
            "Conversation so far (for resolving references only; previous answers are NOT sources):\n"
            + format_history(history)
        )
    sections.append(f"Question:\n{question}")
    sections.append("Sources:\n" + format_sources(chunks, labels, max_chars_each=max_chars_each))
    if missing_hint:
        sections.append("The retrieval step could not find: " + "; ".join(missing_hint))
    if revision_feedback and previous_answer is not None:
        sections.append(
            "Your previous draft answer:\n" + previous_answer
            + "\n\nCitation verification found these problems. Rewrite the answer so that every statement "
              "is supported by the cited sources, remove unsupported statements, and restore omitted "
              "qualifications:\n" + revision_feedback
        )
    return llm.generate_json(
        stage="revision" if revision_feedback else "answer",
        system=ANSWER_SYSTEM,
        prompt="\n\n".join(sections),
        schema=GeneratedAnswer,
        usage=usage,
        max_output_tokens=8192,
    )
