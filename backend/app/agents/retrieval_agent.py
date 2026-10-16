"""Stage 3-4 loop: hybrid retrieval → evidence check → targeted re-retrieval.

The loop is bounded by MAX_RETRIEVAL_ROUNDS and stops early when the evidence
is sufficient, when the checker proposes no new queries, or when a follow-up
search surfaces no chunks that were not already seen.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.agents.evidence_checker import EvidenceAssessment, assess_evidence
from app.agents.query_planner import QueryPlan
from app.config import Settings
from app.database.connection import connection
from app.ingestion.embedder import Embedder
from app.llm.gemini import GeminiLLM, LLMUsage
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.keyword_search import keyword_terms
from app.retrieval.types import RetrievedChunk

logger = logging.getLogger(__name__)
EventFn = Callable[[dict[str, Any]], None]


@dataclass
class RetrievalRound:
    round: int
    queries: list[str]
    retrieved: int
    new_chunks: int
    search: dict[str, Any]
    assessment: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RetrievalOutcome:
    chunks: list[RetrievedChunk]  # ordered best-first, bounded by MAX_CONTEXT_CHUNKS
    all_retrieved_ids: set[str]
    rounds: list[RetrievalRound] = field(default_factory=list)
    stop_reason: str = ""
    assessment: EvidenceAssessment | None = None

    @property
    def missing_information(self) -> list[str]:
        if self.assessment and not self.assessment.sufficient:
            return self.assessment.missing_information
        return []


class RetrievalAgent:
    def __init__(self, settings: Settings, embedder: Embedder, llm: GeminiLLM | None) -> None:
        self.settings = settings
        self.embedder = embedder
        self.llm = llm

    def _search(self, queries: list[str], plan: QueryPlan, *, first_round: bool):
        with connection() as conn:
            result = hybrid_search(
                conn,
                self.embedder,
                queries,
                settings=self.settings,
                exact_terms=plan.identifiers if first_round else (),
                phrases=plan.exact_phrases if first_round else (),
                emphasis=plan.retrieval_emphasis,
            )
            conn.rollback()
        return result

    def run(self, plan: QueryPlan, *, usage: LLMUsage, emit: EventFn, check_cancel: Callable[[], None]) -> RetrievalOutcome:
        settings = self.settings
        pool: dict[str, RetrievedChunk] = {}
        order: list[str] = []  # discovery order (round, then fused rank)
        relevant: list[str] = []
        tried: list[str] = []
        outcome = RetrievalOutcome(chunks=[], all_retrieved_ids=set())
        queries = plan.search_queries[: settings.max_queries_per_round] or [plan.standalone_question]

        for round_no in range(1, settings.max_retrieval_rounds + 1):
            check_cancel()
            emit({"type": "stage", "stage": "retrieval", "round": round_no,
                  "message": f"Searching policies (round {round_no})"})
            result = self._search(queries, plan, first_round=(round_no == 1))
            tried.extend(queries)
            new_ids = [c.chunk_id for c in result.chunks if c.chunk_id not in pool]
            for chunk in result.chunks:
                if chunk.chunk_id not in pool:
                    pool[chunk.chunk_id] = chunk
                    order.append(chunk.chunk_id)
                elif chunk.rrf_score > pool[chunk.chunk_id].rrf_score:
                    pool[chunk.chunk_id].rrf_score = chunk.rrf_score
            record = RetrievalRound(round_no, list(queries), len(result.chunks), len(new_ids), result.diagnostics())
            outcome.rounds.append(record)

            if round_no == 1 and not pool:
                # Nothing at all: one deterministic broadening using the key terms before giving up.
                broad = " ".join(dict.fromkeys(plan.key_terms + keyword_terms(plan.standalone_question)))
                if broad and broad not in tried and settings.max_retrieval_rounds > 1:
                    queries = [broad]
                    continue
                outcome.stop_reason = "no_results"
                break
            if round_no > 1 and not new_ids:
                outcome.stop_reason = "no_new_evidence"
                break
            if settings.max_retrieval_rounds == 1 or self.llm is None:
                outcome.stop_reason = "single_round"
                break

            check_cancel()
            emit({"type": "stage", "stage": "evidence_check", "round": round_no,
                  "message": "Checking whether the evidence is sufficient"})
            candidates = self._ordered(pool, order, relevant)[: settings.max_context_chunks]
            assessment = assess_evidence(
                plan.standalone_question, candidates, llm=self.llm, usage=usage, tried_queries=tried,
                max_follow_ups=settings.max_queries_per_round,
            )
            outcome.assessment = assessment
            record.assessment = assessment.as_dict()
            relevant = list(dict.fromkeys(relevant + assessment.relevant_chunk_ids))
            if assessment.sufficient:
                outcome.stop_reason = "sufficient_evidence"
                break
            if round_no == settings.max_retrieval_rounds:
                outcome.stop_reason = "max_rounds_reached"
                break
            if not assessment.follow_up_queries:
                outcome.stop_reason = "no_follow_up_queries"
                break
            queries = assessment.follow_up_queries
        else:  # pragma: no cover - loop always breaks or exhausts
            outcome.stop_reason = outcome.stop_reason or "max_rounds_reached"
        if not outcome.stop_reason:
            outcome.stop_reason = "max_rounds_reached"

        outcome.all_retrieved_ids = set(pool)
        outcome.chunks = self._select_context(pool, order, relevant)
        return outcome

    @staticmethod
    def _ordered(pool: dict[str, RetrievedChunk], order: list[str], relevant: list[str]) -> list[RetrievedChunk]:
        rest = [cid for cid in order if cid not in relevant]
        return [pool[cid] for cid in relevant if cid in pool] + [pool[cid] for cid in rest]

    def _select_context(self, pool: dict[str, RetrievedChunk], order: list[str],
                        relevant: list[str]) -> list[RetrievedChunk]:
        """Relevant chunks first, then the remainder by discovery order, bounded by
        MAX_CONTEXT_CHUNKS and MAX_CONTEXT_CHARS."""
        selected: list[RetrievedChunk] = []
        used = 0
        for chunk in self._ordered(pool, order, relevant):
            if len(selected) >= self.settings.max_context_chunks:
                break
            if selected and used + len(chunk.text) > self.settings.max_context_chars:
                continue
            selected.append(chunk)
            used += len(chunk.text)
        return selected
