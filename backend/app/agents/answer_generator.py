"""Stage 5: grounded answer generation (structured output with claims).

If the model fails to return valid structured JSON (e.g. it loops until the
output limit), the answer is requested once more as plain Markdown with [S#]
markers, and claims are derived from the cited sentences. Citation
verification and strict grounding apply to both paths.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from app.agents.prompts import ANSWER_SYSTEM, format_history, format_sources
from app.llm.gemini import GeminiLLM, GeminiResponseError, LLMUsage
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
    prompt = "\n\n".join(sections)
    try:
        return llm.generate_json(
            stage="revision" if revision_feedback else "answer",
            system=ANSWER_SYSTEM,
            prompt=prompt,
            schema=GeneratedAnswer,
            usage=usage,
            max_output_tokens=8192,
        )
    except GeminiResponseError:
        text = llm.generate_text(
            stage="answer_text",
            system=ANSWER_SYSTEM + TEXT_MODE_RULES,
            prompt=prompt,
            usage=usage,
            max_output_tokens=8192,
        )
        return answer_from_markdown(text, set(labels))


TEXT_MODE_RULES = """

Output format for this request: reply with the answer as GitHub-flavoured Markdown only (no JSON, no code
fences around the whole answer). Put [S#] citation markers at the end of every factual sentence.
If the sources do not answer the question, start your reply with the line "INSUFFICIENT EVIDENCE:" followed by
what is missing. If sources conflict, add a final line starting with "CONFLICT:" describing it, with markers."""

_MARKER = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def answer_from_markdown(text: str, valid_labels: set[str]) -> GeneratedAnswer:
    """Build a GeneratedAnswer from a plain Markdown reply: claims = cited sentences."""
    body = text.strip()
    insufficient = body.upper().startswith("INSUFFICIENT EVIDENCE")
    missing: list[str] = []
    if insufficient:
        first, _, rest = body.partition("\n")
        missing = [first.split(":", 1)[1].strip()] if ":" in first and first.split(":", 1)[1].strip() else []
        body = rest.strip() or first
    conflicts: list[Conflict] = []
    lines = []
    for line in body.splitlines():
        if line.strip().upper().startswith("CONFLICT:"):
            labels = [lbl for g in _MARKER.findall(line) for lbl in re.findall(r"S\d+", g) if lbl in valid_labels]
            conflicts.append(Conflict(description=_MARKER.sub("", line.split(":", 1)[1]).strip(), source_ids=labels))
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    claims: list[Claim] = []
    for piece in _SENTENCE_SPLIT.split(body):
        labels = [lbl for g in _MARKER.findall(piece) for lbl in re.findall(r"S\d+", g) if lbl in valid_labels]
        statement = re.sub(r"\s+([.,;:!?])", r"\1", _MARKER.sub("", piece)).strip(" -*|#>\t")
        if labels and len(statement.split()) >= 3:
            claims.append(Claim(statement=statement, source_ids=list(dict.fromkeys(labels)), support="explicit"))
    return GeneratedAnswer(answer_markdown=body, claims=claims, insufficient_evidence=insufficient and not claims,
                           missing_information=missing, conflicts=conflicts)
