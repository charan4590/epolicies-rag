"""Agentic RAG orchestration (Stages 1-7).

    plan ─▶ retrieve ⇄ check evidence (bounded) ─▶ generate ─▶ verify ─▶ (revise once) ─▶ respond

Retrieved chunks are the only evidence. Previous answers in the conversation
are used solely to resolve references in follow-up questions.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from app.agents.answer_generator import GeneratedAnswer, generate_answer
from app.agents.catalog import build_catalog_answer, parse_catalog_question
from app.agents.citation_verifier import (
    MARKER_RE,
    VerificationReport,
    revision_feedback,
    verify_deterministic,
    verify_with_llm,
)
from app.agents.query_planner import QueryPlan, deterministic_plan, plan_query
from app.agents.retrieval_agent import RetrievalAgent, RetrievalOutcome
from app.config import Settings, get_settings
from app.database import repository as repo
from app.database.connection import connection
from app.ingestion.embedder import Embedder, GeminiEmbedder
from app.retrieval.keyword_search import unknown_terms
from app.llm.gemini import GeminiError, GeminiLLM, LLMUsage
from app.retrieval.types import RetrievedChunk
from app.schemas.api_models import (
    ChatRequest,
    ChatResponse,
    Citation,
    ClaimInfo,
    ConflictInfo,
    VerificationInfo,
)

logger = logging.getLogger(__name__)
EventFn = Callable[[dict[str, Any]], None]

SMALL_TALK_REPLY = (
    "Hello! I'm the ePolicies AI Assistant. Ask me about company policies — for example leave, remote work, "
    "IT security or expenses — and I'll answer from the indexed policy documents with citations."
)
NO_EVIDENCE_REPLY = (
    "I could not find information about this in the indexed policy documents, so I can't answer it reliably. "
    "Try rephrasing with the policy name or specific terms, or check that the relevant document has been indexed."
)


class ChatCancelled(RuntimeError):
    pass


class ChatTimeout(RuntimeError):
    pass


class RAGService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        llm: GeminiLLM | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._llm = llm

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = GeminiEmbedder(self.settings)
        return self._embedder

    @property
    def llm(self) -> GeminiLLM:
        if self._llm is None:
            self._llm = GeminiLLM(self.settings)
        return self._llm

    # ------------------------------------------------------------------ history
    def _load_history(self, request: ChatRequest) -> list[dict[str, str]]:
        limit = self.settings.max_history_turns * 2
        if limit == 0:
            return []
        if request.conversation_id is not None:
            with connection() as conn:
                rows = repo.recent_messages(conn, request.conversation_id, limit)
            if rows:
                return [
                    {"role": r["role"], "content": r["standalone_question"] or r["content"]
                     if r["role"] == "user" else r["content"]}
                    for r in rows
                ]
        return [{"role": m.role, "content": m.content} for m in request.history[-limit:]]

    # ------------------------------------------------------------------ main entry
    def answer(
        self,
        request: ChatRequest,
        *,
        on_event: EventFn | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> ChatResponse:
        started = time.monotonic()
        emit = on_event or (lambda event: None)
        timings: dict[str, float] = {}
        usage = LLMUsage()

        def check() -> None:
            if is_cancelled and is_cancelled():
                raise ChatCancelled("Request cancelled by client")
            if time.monotonic() - started > self.settings.chat_timeout_seconds:
                raise ChatTimeout("The request took too long to complete.")

        def mark(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = round((now - since) * 1000, 1)
            return now

        history = self._load_history(request)
        catalog = self._catalog_answer(request, usage, timings, started)
        if catalog is not None:
            return catalog
        t = time.perf_counter()
        emit({"type": "stage", "stage": "planning", "message": "Understanding the question"})
        mode, misspelt = self.settings.planner_mode, []
        if mode == "auto":
            # Words found in no policy are usually typos; Gemini rewrites the question so that keyword
            # search uses the corrected words (vector search tolerates typos, keyword search does not).
            try:
                with connection() as conn:
                    misspelt = unknown_terms(conn, request.question, ts_config=self.settings.text_search_config)
            except Exception:  # noqa: BLE001 - a spelling hint is optional; retrieval reports DB errors
                logger.warning("unknown_terms_check_failed", exc_info=True)
            mode = "always" if misspelt else mode
        plan = plan_query(request.question, history, llm=self.llm, mode=mode, usage=usage,
                          max_queries=self.settings.max_queries_per_round)
        if misspelt:
            plan.notes.append("Words not found in any policy: " + ", ".join(misspelt)
                              + " – Gemini corrected the question before searching.")
        t = mark("planning", t)
        check()

        if not plan.is_policy_question:
            response = ChatResponse(
                answer=SMALL_TALK_REPLY, citations=[], insufficient_evidence=False,
                verification=VerificationInfo(status="not_applicable"), standalone_question=plan.standalone_question,
            )
            return self._finish(request, plan, response, None, None, usage, timings, started)

        agent = RetrievalAgent(self.settings, self.embedder, self.llm if self.settings.max_retrieval_rounds > 1 else None)
        outcome = agent.run(plan, usage=usage, emit=emit, check_cancel=check)
        t = mark("retrieval", t)
        check()

        if not outcome.chunks:
            response = ChatResponse(
                answer=NO_EVIDENCE_REPLY, citations=[], insufficient_evidence=True,
                missing_information=outcome.missing_information or ["No relevant passages were retrieved."],
                verification=VerificationInfo(status="not_applicable"), standalone_question=plan.standalone_question,
            )
            return self._finish(request, plan, response, outcome, None, usage, timings, started)

        labels = [f"S{i}" for i in range(1, len(outcome.chunks) + 1)]
        label_map = dict(zip(labels, outcome.chunks))
        emit({"type": "stage", "stage": "answer", "message": "Writing a grounded answer"})
        generated = generate_answer(plan.standalone_question, outcome.chunks, labels, llm=self.llm, usage=usage,
                                    history=history, missing_hint=outcome.missing_information)
        t = mark("answer", t)
        check()

        emit({"type": "stage", "stage": "verification", "message": "Verifying citations"})
        final_answer, report, revised = self._verify(plan, generated, outcome, label_map, labels, history, usage,
                                                     check)
        mark("verification", t)
        response = self._build_response(plan, final_answer, generated, report, revised, label_map)
        if self.settings.strict_grounding and not response.citations:
            # Strict grounding: an answer that cites none of the indexed documents could only have come
            # from the model's general knowledge, so it is withheld.
            response.answer = NO_EVIDENCE_REPLY
            response.insufficient_evidence = True
            response.missing_information = response.missing_information or [
                "No statement in the draft answer could be tied to a passage in the indexed documents."
            ]
            response.conflicts = []
            response.verification = VerificationInfo(
                status="not_applicable", revised=response.verification.revised,
                llm_checked=response.verification.llm_checked,
                removed_citations=response.verification.removed_citations,
                notes=["The draft answer was withheld because it was not supported by any indexed document."],
            )
        return self._finish(request, plan, response, outcome, report, usage, timings, started)

    # ------------------------------------------------------------------ catalog questions
    def _catalog_answer(self, request: ChatRequest, usage: LLMUsage, timings: dict[str, float],
                        started: float) -> ChatResponse | None:
        """Answer "how many policies / list all policies" from the document index (no Gemini call)."""
        parsed = parse_catalog_question(request.question)
        if parsed is None:
            return None
        with connection() as conn:
            titles = repo.indexed_document_titles(conn)
        answer = build_catalog_answer(parsed, titles)
        if answer is None:
            return None
        response = ChatResponse(
            answer=answer, citations=[], insufficient_evidence=False,
            verification=VerificationInfo(status="not_applicable", notes=[
                "Answered from the list of indexed documents, not from policy text."
            ]),
            standalone_question=request.question,
        )
        return self._finish(request, deterministic_plan(request.question), response, None, None, usage, timings,
                            started)

    # ------------------------------------------------------------------ verification
    def _lookup(self, chunk_ids):
        with connection() as conn:
            return repo.get_chunks_by_ids(conn, chunk_ids)

    def _verify(self, plan: QueryPlan, generated: GeneratedAnswer, outcome: RetrievalOutcome,
                label_map: dict[str, RetrievedChunk], labels: list[str], history: list[dict[str, str]],
                usage: LLMUsage, check: Callable[[], None]) -> tuple[str, VerificationReport, bool]:
        cleaned, report = verify_deterministic(generated, label_map, outcome.all_retrieved_ids, self._lookup)
        if self.settings.llm_citation_verification and report.weak:
            try:
                verify_with_llm(report, label_map, llm=self.llm, usage=usage)
            except GeminiError as exc:
                logger.warning("llm_verification_unavailable", extra={"kind": exc.kind})
        revised = False
        if report.problems or report.label_errors:
            check()
            feedback = revision_feedback(report)
            try:
                revision = generate_answer(
                    plan.standalone_question, outcome.chunks, labels, llm=self.llm, usage=usage, history=history,
                    missing_hint=outcome.missing_information, revision_feedback=feedback,
                    previous_answer=generated.answer_markdown,
                )
            except GeminiError as exc:
                logger.warning("revision_unavailable", extra={"kind": exc.kind})
            else:
                revised = True
                generated.answer_markdown = revision.answer_markdown
                generated.claims = revision.claims
                generated.conflicts = revision.conflicts
                generated.insufficient_evidence = revision.insufficient_evidence
                generated.missing_information = revision.missing_information
                cleaned, second = verify_deterministic(revision, label_map, outcome.all_retrieved_ids, self._lookup)
                second.llm_checked = report.llm_checked
                report = second
        return cleaned, report, revised

    # ------------------------------------------------------------------ response building
    def _build_response(self, plan: QueryPlan, answer_text: str, generated: GeneratedAnswer,
                        report: VerificationReport, revised: bool,
                        label_map: dict[str, RetrievedChunk]) -> ChatResponse:
        valid = [lbl for lbl in report.valid_labels if lbl in label_map]
        renumber = {old: f"S{i}" for i, old in enumerate(valid, start=1)}
        answer_text = renumber_markers(answer_text, renumber)

        claim_status: dict[str, set[str]] = {}
        for claim in report.claims:
            for lbl in claim.labels:
                claim_status.setdefault(lbl, set()).add(claim.status)
        citations: list[Citation] = []
        for old in valid:
            chunk = label_map[old]
            statuses = claim_status.get(old, set())
            if statuses and statuses <= {"supported"}:
                state = "supported"
            elif statuses & {"weak", "partially_supported", "unsupported"}:
                state = "weak"
            else:
                state = "unverified"
            citations.append(Citation(
                label=renumber[old], chunk_id=chunk.chunk_id, document_id=chunk.document_id,
                document_title=chunk.title, relative_path=chunk.relative_path, heading_path=chunk.heading_path,
                section_paths=chunk.metadata.get("section_paths", []) if isinstance(chunk.metadata, dict) else [],
                passage=chunk.text, source_urls=chunk.source_urls, document_source_url=chunk.document_source_url,
                verification=state,
            ))

        remaining = [c for c in report.claims if c.status in {"weak", "unsupported", "partially_supported", "uncited"}]
        notes: list[str] = []
        if remaining:
            notes.append(
                f"{len(remaining)} statement(s) could not be fully verified against the cited passages. "
                "Check the highlighted sources before relying on them."
            )
        if report.removed_labels:
            notes.append("Citations to sources that were not retrieved were removed.")
        if not citations and not generated.insufficient_evidence:
            notes.append("The answer does not cite any policy source.")
        if not citations:
            status = "not_applicable" if generated.insufficient_evidence else "partially_verified"
        elif remaining:
            status = "partially_verified"
        elif revised:
            status = "revised"
        else:
            status = "verified"

        conflicts = []
        for conflict in generated.conflicts:
            lbls = [renumber[lbl] for lbl in conflict.source_ids if lbl in renumber]
            if conflict.description.strip():
                conflicts.append(ConflictInfo(description=conflict.description.strip(), labels=lbls))

        return ChatResponse(
            answer=answer_text.strip(),
            citations=citations,
            insufficient_evidence=generated.insufficient_evidence,
            missing_information=[m for m in generated.missing_information if m.strip()],
            conflicts=conflicts,
            verification=VerificationInfo(
                status=status, revised=revised, llm_checked=report.llm_checked,
                removed_citations=report.removed_labels,
                claims=[ClaimInfo(statement=c.statement, labels=[renumber.get(lbl, lbl) for lbl in c.labels],
                                  support_type=c.support_type, status=c.status,
                                  missing_qualifications=c.missing_qualifications) for c in report.claims],
                notes=notes,
            ),
            standalone_question=plan.standalone_question,
        )

    def _finish(self, request: ChatRequest, plan: QueryPlan, response: ChatResponse,
                outcome: RetrievalOutcome | None, report: VerificationReport | None, usage: LLMUsage,
                timings: dict[str, float], started: float) -> ChatResponse:
        timings["total"] = round((time.monotonic() - started) * 1000, 1)
        if request.include_diagnostics and self.settings.expose_diagnostics:
            response.diagnostics = {
                "plan": plan.as_dict(),
                "rounds": [r.as_dict() for r in outcome.rounds] if outcome else [],
                "retrieval_rounds": len(outcome.rounds) if outcome else 0,
                "stop_reason": outcome.stop_reason if outcome else "not_applicable",
                "retrieved_chunks": len(outcome.all_retrieved_ids) if outcome else 0,
                "context_chunks": [c.brief() for c in outcome.chunks] if outcome else [],
                "relevant_chunk_ids": outcome.relevant_ids if outcome else [],
                "verification": report.as_dict() if report else None,
                "models": {
                    "generation": self.settings.gemini_model,
                    "embedding": self.settings.embedding_model,
                    "embedding_dimensions": self.settings.embedding_dimensions,
                },
                "llm_usage": usage.as_dict(),
                "timings_ms": timings,
            }
        if request.persist:
            try:
                with connection() as conn:
                    with conn.transaction():
                        conversation_id = repo.ensure_conversation(conn, request.conversation_id,
                                                                   _title_from(request.question))
                        repo.add_message(conn, conversation_id, role="user", content=request.question,
                                         standalone_question=plan.standalone_question)
                        message_id = repo.add_message(
                            conn, conversation_id, role="assistant", content=response.answer,
                            payload=response.model_dump(mode="json", exclude={"diagnostics"}),
                        )
                response.conversation_id = conversation_id
                response.message_id = message_id
            except Exception:  # noqa: BLE001 - history is optional; the answer is still returned
                logger.exception("could not persist conversation")
        else:
            response.conversation_id = request.conversation_id
        logger.info(
            "chat_completed",
            extra={"llm_calls": usage.calls, "rounds": len(outcome.rounds) if outcome else 0,
                   "citations": len(response.citations), "total_ms": timings["total"],
                   "verification": response.verification.status},
        )
        return response


def renumber_markers(text: str, mapping: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        labels = re.findall(r"S\d+", match.group(1))
        out = []
        for lbl in labels:
            new = mapping.get(lbl)
            if new and new not in out:
                out.append(new)
        return "".join(f"[{lbl}]" for lbl in out)

    return MARKER_RE.sub(repl, text)


def _title_from(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 80 else title[:77] + "…"


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
