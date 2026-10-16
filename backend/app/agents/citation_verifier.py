"""Stage 6: citation verification.

Deterministic checks (always):
  * every inline marker refers to a source label that was actually provided;
  * the label maps to a chunk that was retrieved in this request;
  * that chunk still exists in PostgreSQL and belongs to the cited document;
  * each claim cites at least one valid source;
  * lexical support: the claim's content words — and in particular every
    number, amount and identifier — appear in the cited text.

Model-based check (optional, only for claims the deterministic check marks
weak): Gemini judges supported / partially supported / unsupported and lists
omitted qualifications. The orchestrator then performs at most one revision.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from app.agents.answer_generator import GeneratedAnswer
from app.agents.prompts import VERIFIER_SYSTEM, neutralise
from app.llm.gemini import GeminiLLM, LLMUsage
from app.retrieval.types import RetrievedChunk

MARKER_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")
_LABEL_RE = re.compile(r"S\d+")
_WORD_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_NUMBER_RE = re.compile(r"\$?\d+(?:[.,]\d+)*%?")
_STOP = set(
    "a an the and or but if then than of to in on at by for with from as is are was were be been being this that "
    "these those it its their there here which who whom whose what when where why how can could should would will "
    "shall may might must not no do does did done have has had having such any all each every some other more most "
    "less least very also only just into onto over under about after before within without per via up down out "
    "your you employees employee policy policies company staff they them he she we us our".split()
)
SUPPORT_THRESHOLD = 0.6


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 1}


def numbers_in(text: str) -> set[str]:
    return {n.strip("$%").replace(",", "") for n in _NUMBER_RE.findall(text)}


def lexical_support(claim: str, evidence: str) -> tuple[float, list[str]]:
    """Return (coverage of claim content words by evidence, numbers missing from evidence)."""
    claim_words = content_words(claim)
    evidence_words = content_words(evidence)
    coverage = 1.0 if not claim_words else len(claim_words & evidence_words) / len(claim_words)
    evidence_numbers = numbers_in(evidence)
    missing_numbers = sorted(n for n in numbers_in(claim) if n not in evidence_numbers)
    return coverage, missing_numbers


class ClaimVerdict(BaseModel):
    claim_index: int
    verdict: Literal["supported", "partially_supported", "unsupported"]
    missing_qualifications: list[str]
    explanation: str


class LLMVerification(BaseModel):
    verdicts: list[ClaimVerdict]


@dataclass
class ClaimCheck:
    statement: str
    labels: list[str]
    support_type: str
    lexical_coverage: float
    missing_numbers: list[str]
    status: str = "supported"  # supported | weak | unsupported | partially_supported | uncited
    missing_qualifications: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class VerificationReport:
    valid_labels: list[str] = field(default_factory=list)
    removed_labels: list[str] = field(default_factory=list)
    label_errors: list[str] = field(default_factory=list)
    claims: list[ClaimCheck] = field(default_factory=list)
    llm_checked: bool = False
    uncited_sentences: int = 0

    @property
    def problems(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.status in {"unsupported", "partially_supported", "uncited"}]

    @property
    def weak(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.status == "weak"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid_labels": self.valid_labels,
            "removed_labels": self.removed_labels,
            "label_errors": self.label_errors,
            "llm_checked": self.llm_checked,
            "uncited_sentences": self.uncited_sentences,
            "claims": [c.__dict__ for c in self.claims],
        }


def parse_marker_labels(text: str) -> list[str]:
    labels: list[str] = []
    for match in MARKER_RE.finditer(text):
        for label in _LABEL_RE.findall(match.group(1)):
            if label not in labels:
                labels.append(label)
    return labels


def strip_invalid_markers(text: str, valid: set[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        kept = [lbl for lbl in _LABEL_RE.findall(match.group(1)) if lbl in valid]
        return "".join(f"[{lbl}]" for lbl in kept)

    return MARKER_RE.sub(repl, text)


def count_uncited_sentences(text: str) -> int:
    count = 0
    for line in text.split("\n"):
        stripped = line.strip().lstrip("-*0123456789.) ").strip()
        if not stripped or stripped.startswith(("|", "#")) or len(stripped.split()) < 6:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            if len(sentence.split()) >= 6 and not MARKER_RE.search(sentence) and numbers_in(sentence):
                count += 1
    return count


def verify_deterministic(
    answer: GeneratedAnswer,
    label_map: Mapping[str, RetrievedChunk],
    retrieved_ids: set[str],
    lookup_chunks: Callable[[Sequence[str]], Mapping[str, Mapping[str, Any]]],
) -> tuple[str, VerificationReport]:
    """Return (cleaned_answer_markdown, report)."""
    report = VerificationReport()
    cited = parse_marker_labels(answer.answer_markdown)
    for claim in answer.claims:
        for label in claim.source_ids:
            if label not in cited:
                cited.append(label)

    candidate_ids = [label_map[lbl].chunk_id for lbl in cited if lbl in label_map]
    db_rows = lookup_chunks(candidate_ids) if candidate_ids else {}
    valid: set[str] = set()
    for label in cited:
        chunk = label_map.get(label)
        if chunk is None:
            report.removed_labels.append(label)
            report.label_errors.append(f"{label}: not one of the provided sources")
            continue
        if chunk.chunk_id not in retrieved_ids:
            report.removed_labels.append(label)
            report.label_errors.append(f"{label}: chunk was not retrieved for this question")
            continue
        row = db_rows.get(chunk.chunk_id)
        if row is None:
            report.removed_labels.append(label)
            report.label_errors.append(f"{label}: chunk no longer exists in the index")
            continue
        if row["document_id"] != chunk.document_id or row.get("status") == "deleted":
            report.removed_labels.append(label)
            report.label_errors.append(f"{label}: chunk does not belong to an active cited document")
            continue
        valid.add(label)
    report.valid_labels = [lbl for lbl in cited if lbl in valid]
    cleaned = strip_invalid_markers(answer.answer_markdown, valid)

    for claim in answer.claims:
        labels = [lbl for lbl in claim.source_ids if lbl in valid]
        evidence = "\n".join(label_map[lbl].text + " " + label_map[lbl].title + " " +
                             " ".join(label_map[lbl].heading_path) for lbl in labels)
        coverage, missing_numbers = lexical_support(claim.statement, evidence) if labels else (0.0, [])
        check = ClaimCheck(
            statement=claim.statement,
            labels=labels,
            support_type=claim.support,
            lexical_coverage=round(coverage, 3),
            missing_numbers=missing_numbers,
        )
        if not labels:
            check.status = "uncited"
        elif missing_numbers or coverage < SUPPORT_THRESHOLD:
            check.status = "weak"
        report.claims.append(check)
    report.uncited_sentences = count_uncited_sentences(cleaned)
    return cleaned, report


def verify_with_llm(
    report: VerificationReport,
    label_map: Mapping[str, RetrievedChunk],
    *,
    llm: GeminiLLM,
    usage: LLMUsage,
    max_chars_each: int = 2500,
) -> None:
    """Ask Gemini to adjudicate claims the deterministic check marked weak."""
    weak = [(i, c) for i, c in enumerate(report.claims) if c.status == "weak"]
    if not weak:
        return
    blocks = []
    for index, claim in weak:
        evidence = "\n\n".join(
            f'<source id="{lbl}">\n{neutralise(label_map[lbl].text[:max_chars_each])}\n</source>' for lbl in claim.labels
        )
        blocks.append(f"Claim {index}: {claim.statement}\nCited sources:\n{evidence}")
    result = llm.generate_json(stage="citation_verification", system=VERIFIER_SYSTEM,
                               prompt="\n\n---\n\n".join(blocks), schema=LLMVerification, usage=usage,
                               max_output_tokens=2048)
    report.llm_checked = True
    by_index = {v.claim_index: v for v in result.verdicts}
    for index, claim in weak:
        verdict = by_index.get(index)
        if verdict is None:
            continue  # stays "weak" (unverified)
        claim.status = verdict.verdict
        claim.missing_qualifications = [q for q in verdict.missing_qualifications if q.strip()]
        claim.explanation = verdict.explanation[:500]
        if claim.status == "partially_supported" and not claim.missing_qualifications:
            claim.status = "supported"


def revision_feedback(report: VerificationReport) -> str:
    lines = []
    for claim in report.problems:
        if claim.status == "uncited":
            lines.append(f"- Uncited statement (add a valid citation or remove it): {claim.statement}")
        elif claim.status == "unsupported":
            lines.append(f"- Not supported by {', '.join(claim.labels)}: {claim.statement}")
        elif claim.status == "partially_supported":
            lines.append(
                f"- Missing qualifications for '{claim.statement}': {'; '.join(claim.missing_qualifications)}"
            )
    for error in report.label_errors:
        lines.append(f"- Invalid citation {error}")
    return "\n".join(lines)
