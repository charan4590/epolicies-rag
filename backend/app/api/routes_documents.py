"""GET /documents, GET /documents/{id}, GET /search, GET /health."""

from __future__ import annotations

import re
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app import __version__
from app.agents.query_planner import deterministic_plan
from app.api.deps import get_rag_service, settings_dep
from app.config import Settings
from app.database import repository as repo
from app.database.connection import connection
from app.database.schema import check_schema
from app.retrieval.hybrid_search import hybrid_search
from app.schemas.api_models import (
    ChunkSummary,
    DocumentDetail,
    DocumentList,
    DocumentSummary,
    HealthResponse,
    SearchHit,
    SearchResponse,
)
from app.services.rag_service import RAGService

router = APIRouter()
DOCUMENT_ID_RE = re.compile(r"^doc_[0-9a-f]{24}$")


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: Settings = Depends(settings_dep)):
    """Database connectivity, pgvector/schema state and configuration (no secrets)."""
    gemini = {
        "configured": settings.gemini_configured,
        "backend": settings.gemini_backend,
        "generation_model": settings.gemini_model or None,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
    }
    started = time.perf_counter()
    try:
        with connection() as conn:
            conn.execute("SELECT 1")
            schema = check_schema(conn, settings)
            index = repo.index_stats(conn) if schema["schema_initialized"] else None
        database = {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1), **schema}
    except Exception as exc:  # noqa: BLE001 - reported as status, never raised
        from app.api.errors import classify_exception

        _, code, message = classify_exception(exc)
        body = HealthResponse(status="error", version=__version__, database={"ok": False, "error": message},
                              gemini=gemini, config=settings.public_summary())
        return JSONResponse(body.model_dump(mode="json"), status_code=503)
    healthy = schema["schema_initialized"] and schema["dimensions_match"] and gemini["configured"] \
        and bool(settings.gemini_model)
    return HealthResponse(status="ok" if healthy else "degraded", version=__version__, database=database,
                          index=index, gemini=gemini, config=settings.public_summary())


@router.get("/documents", response_model=DocumentList, tags=["documents"])
def list_documents(
    page: int = Query(1, ge=1, le=10_000),
    page_size: int = Query(50, ge=1, le=200),
    q: str | None = Query(None, max_length=200, description="Filter by title or path"),
    status: Literal["pending", "indexed", "failed", "deleted"] | None = None,
) -> DocumentList:
    with connection() as conn:
        rows, total = repo.list_documents(conn, page=page, page_size=page_size, query=q, status=status)
    return DocumentList(items=[DocumentSummary(**r) for r in rows], total=total, page=page, page_size=page_size)


@router.get("/documents/{document_id}", response_model=DocumentDetail, tags=["documents"])
def get_document(document_id: str) -> DocumentDetail:
    if not DOCUMENT_ID_RE.match(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    with connection() as conn:
        doc = repo.get_document(conn, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = [ChunkSummary(**c) for c in doc.pop("chunks")]
    metadata = doc.get("metadata") or {}
    doc["is_empty"] = bool(metadata.get("empty"))
    return DocumentDetail(**doc, chunks=chunks)


@router.get("/search", response_model=SearchResponse, tags=["retrieval"])
def search(
    q: str = Query(..., min_length=1, max_length=500),
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    top_k: int = Query(10, ge=1, le=50),
    settings: Settings = Depends(settings_dep),
    service: RAGService = Depends(get_rag_service),
) -> SearchResponse:
    """Retrieval debugging: inspect semantic, keyword and fused rankings for a query."""
    if not settings.enable_search_debug:
        raise HTTPException(status_code=404, detail="Search debugging is disabled")
    plan = deterministic_plan(q)
    identifiers, phrases = plan.identifiers, plan.exact_phrases
    embedder = service.embedder if mode in {"hybrid", "vector"} else None
    with connection() as conn:
        result = hybrid_search(conn, embedder, [q], settings=settings, exact_terms=identifiers, phrases=phrases,
                               mode=mode, top_k_final=top_k)
    hits = [
        SearchHit(rank=i, chunk_id=c.chunk_id, document_id=c.document_id, document_title=c.title,
                  relative_path=c.relative_path, heading_path=c.heading_path, text=c.text,
                  source_urls=c.source_urls, rrf_score=round(c.rrf_score, 6), vector_rank=c.vector_rank,
                  vector_similarity=c.vector_similarity, keyword_rank=c.keyword_rank,
                  keyword_score=c.keyword_score, exact_rank=c.exact_rank)
        for i, c in enumerate(result.chunks, start=1)
    ]
    return SearchResponse(query=q, mode=mode, identifiers=identifiers, hits=hits, diagnostics=result.diagnostics())
