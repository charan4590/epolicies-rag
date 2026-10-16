"""Stage 4: evidence sufficiency evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from app.agents.prompts import EVIDENCE_SYSTEM, format_sources
from app.llm.gemini import GeminiLLM, LLMUsage
from app.retrieval.types import RetrievedChunk


class LLMEvidenceAssessment(BaseModel):
    sufficient: bool
    confidence: Literal["high", "medium", "low"]
    relevant_source_ids: list[str]
    missing_information: list[str]
    follow_up_queries: list[str]


@dataclass
class EvidenceAssessment:
    sufficient: bool
    confidence: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)
    method: str = "llm"  # llm | deterministic

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def assess_evidence(
    question: str,
    chunks: Sequence[RetrievedChunk],
    *,
    llm: GeminiLLM,
    usage: LLMUsage,
    tried_queries: Sequence[str],
    max_follow_ups: int,
    max_chars_each: int = 1500,
) -> EvidenceAssessment:
    if not chunks:
        return EvidenceAssessment(
            sufficient=False, confidence="low", missing_information=["No matching policy passages were found."],
            method="deterministic",
        )
    labels = [f"S{i}" for i in range(1, len(chunks) + 1)]
    prompt = (
        f"Question:\n{question}\n\n"
        f"Queries already tried: {list(tried_queries)}\n\n"
        f"Retrieved sources:\n{format_sources(chunks, labels, max_chars_each=max_chars_each)}"
    )
    result = llm.generate_json(stage="evidence_check", system=EVIDENCE_SYSTEM, prompt=prompt,
                               schema=LLMEvidenceAssessment, usage=usage, max_output_tokens=1024)
    by_label = dict(zip(labels, chunks))
    relevant = [by_label[s].chunk_id for s in result.relevant_source_ids if s in by_label]
    tried = {q.strip().lower() for q in tried_queries}
    follow_ups = [q.strip() for q in result.follow_up_queries if q.strip() and q.strip().lower() not in tried]
    return EvidenceAssessment(
        sufficient=result.sufficient,
        confidence=result.confidence,
        relevant_chunk_ids=list(dict.fromkeys(relevant)),
        missing_information=[m for m in result.missing_information if m.strip()],
        follow_up_queries=follow_ups[:max_follow_ups],
    )
