"""RAG evaluation.

Dataset: JSONL, one case per line::

    {"id": "leave-01", "question": "...", "category": "direct_fact",
     "expected_sources": ["hr/leave-policy.md"],          # relative paths judged relevant
     "expected_insufficient": false,                       # true for out-of-scope questions
     "reference_answer": "...",                            # optional, written from the source docs
     "human_reviewed": true}

Metrics
-------
Automated retrieval metrics (embedding API only, no generation):
  * Recall@K – fraction of expected source documents present in the top-K
    fused results (document level).
  * MRR – reciprocal rank of the first relevant document.
Automated answer metrics (``--with-answers``, uses Gemini):
  * citation precision – share of cited documents that are expected sources;
  * citation recall – share of expected sources that were cited;
  * insufficient-evidence accuracy – for cases with ``expected_insufficient``;
  * faithfulness (proxy) – share of claims the citation verifier marked supported.
  * answer relevance (``--judge``) – Gemini-judged 1-5 score. This is an
    automated LLM judgement, NOT a human review.
Human review status is reported per case from the dataset's ``human_reviewed`` flag.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.agents.query_planner import deterministic_plan
from app.config import BACKEND_DIR, get_settings
from app.database.connection import connection
from app.ingestion.embedder import GeminiEmbedder
from app.llm.gemini import GeminiLLM, LLMUsage
from app.retrieval.hybrid_search import hybrid_search
from app.schemas.api_models import ChatRequest
from app.services.rag_service import RAGService


class RelevanceJudgement(BaseModel):
    score: int
    rationale: str


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            case = json.loads(line)
            if "question" not in case:
                raise ValueError(f"{path}:{line_no}: missing 'question'")
            case.setdefault("id", f"case-{line_no}")
            case.setdefault("expected_sources", [])
            cases.append(case)
    return cases


def recall_at_k(ranked_docs: list[str], expected: list[str], k: int) -> float | None:
    if not expected:
        return None
    top = set(ranked_docs[:k])
    return sum(1 for e in expected if e in top) / len(expected)


def reciprocal_rank(ranked_docs: list[str], expected: list[str]) -> float | None:
    if not expected:
        return None
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in expected:
            return 1.0 / rank
    return 0.0


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 4) if clean else None


def run_evaluation(dataset: Path, *, k: int = 5, with_answers: bool = False, judge: bool = False,
                   output: Path | None = None) -> dict[str, Any]:
    settings = get_settings()
    if not dataset.is_absolute():
        dataset = (BACKEND_DIR / dataset).resolve() if not dataset.exists() else dataset.resolve()
    cases = load_dataset(dataset)
    embedder = GeminiEmbedder(settings)
    service = RAGService(settings, embedder=embedder) if with_answers else None
    judge_llm = GeminiLLM(settings) if judge else None
    results: list[dict[str, Any]] = []

    for case in cases:
        plan = deterministic_plan(case["question"])
        started = time.perf_counter()
        with connection() as conn:
            search = hybrid_search(conn, embedder, plan.search_queries, settings=settings,
                                   exact_terms=plan.identifiers, phrases=plan.exact_phrases,
                                   emphasis=plan.retrieval_emphasis, top_k_final=max(k, settings.top_k_final))
        ranked_docs = list(dict.fromkeys(c.relative_path for c in search.chunks))
        expected = case["expected_sources"]
        row: dict[str, Any] = {
            "id": case["id"],
            "category": case.get("category"),
            "question": case["question"],
            "human_reviewed": bool(case.get("human_reviewed", False)),
            "expected_sources": expected,
            "retrieved_documents": ranked_docs[:k],
            "recall_at_k": recall_at_k(ranked_docs, expected, k),
            "reciprocal_rank": reciprocal_rank(ranked_docs, expected),
            "retrieval_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if service is not None:
            response = service.answer(ChatRequest(question=case["question"], persist=False,
                                                  include_diagnostics=True))
            cited_docs = list(dict.fromkeys(c.relative_path for c in response.citations))
            claims = response.verification.claims
            row.update({
                "answer": response.answer,
                "cited_documents": cited_docs,
                "citation_precision": (sum(1 for d in cited_docs if d in expected) / len(cited_docs))
                if cited_docs and expected else None,
                "citation_recall": (sum(1 for e in expected if e in cited_docs) / len(expected)) if expected else None,
                "insufficient_evidence": response.insufficient_evidence,
                "insufficient_correct": (response.insufficient_evidence == case["expected_insufficient"])
                if "expected_insufficient" in case else None,
                "faithfulness_proxy": (sum(1 for c in claims if c.status == "supported") / len(claims))
                if claims else None,
                "verification": response.verification.status,
                "llm_calls": (response.diagnostics or {}).get("llm_usage", {}).get("calls"),
                "retrieval_rounds": (response.diagnostics or {}).get("retrieval_rounds"),
            })
            if judge_llm is not None:
                verdict = judge_llm.generate_json(
                    stage="eval_judge",
                    system="Rate how well the answer addresses the question on a 1-5 scale "
                           "(5 = fully answers it, 1 = irrelevant). Judge relevance only, not correctness.",
                    prompt=f"Question: {case['question']}\n\nAnswer:\n{response.answer}",
                    schema=RelevanceJudgement, usage=LLMUsage(), max_output_tokens=512,
                )
                row["answer_relevance_llm_judge"] = max(1, min(5, verdict.score))
        results.append(row)

    summary: dict[str, Any] = {
        "dataset": str(dataset),
        "cases": len(results),
        "human_reviewed_cases": sum(1 for r in results if r["human_reviewed"]),
        "k": k,
        f"recall@{k}": _mean([r["recall_at_k"] for r in results]),
        "mrr": _mean([r["reciprocal_rank"] for r in results]),
        "metric_type": "automated",
    }
    if with_answers:
        summary.update({
            "citation_precision": _mean([r.get("citation_precision") for r in results]),
            "citation_recall": _mean([r.get("citation_recall") for r in results]),
            "insufficient_evidence_accuracy": _mean(
                [float(r["insufficient_correct"]) for r in results if r.get("insufficient_correct") is not None]),
            "faithfulness_proxy": _mean([r.get("faithfulness_proxy") for r in results]),
        })
        if judge:
            summary["answer_relevance_llm_judge"] = _mean([r.get("answer_relevance_llm_judge") for r in results])
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_category.setdefault(r.get("category") or "uncategorised", []).append(r)
    summary["by_category"] = {
        cat: {f"recall@{k}": _mean([r["recall_at_k"] for r in rows]),
              "mrr": _mean([r["reciprocal_rank"] for r in rows]), "cases": len(rows)}
        for cat, rows in by_category.items()
    }

    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = BACKEND_DIR / "eval" / "results" / f"{dataset.stem}-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str), encoding="utf-8")
    return {"summary": summary, "results": results, "output_path": str(output)}
