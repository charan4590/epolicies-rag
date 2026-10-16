"""FastAPI application entry point.

Run:  uvicorn app.main:app --reload --port 8000   (from epolicies-rag/backend)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import routes_chat, routes_documents, routes_ingestion
from app.api.errors import classify_exception, error_payload, install_error_handlers
from app.config import get_settings
from app.database.connection import close_pool
from app.logging_config import configure_logging, request_id_var

MAX_BODY_BYTES = 64 * 1024

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("startup", extra={"config": settings.public_summary()})
    yield
    close_pool()


app = FastAPI(
    title="ePolicies AI Assistant API",
    version=__version__,
    description="Agentic RAG over the ePolicies Markdown knowledge base "
                "(Gemini + PostgreSQL/pgvector hybrid search).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Admin-Token", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if incoming.isalnum() and len(incoming) <= 64 else uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse(error_payload("payload_too_large", "Request body is too large."), status_code=413)
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - unhandled errors become safe JSON with a request id
            status, code, message = classify_exception(exc)
            if status >= 500:
                logger.error("request_failed", exc_info=exc, extra={"code": code, "path": request.url.path})
            response = JSONResponse(error_payload(code, message), status_code=status)
        response.headers["X-Request-ID"] = request_id
        logger.info("request", extra={"method": request.method, "path": request.url.path,
                                      "status": response.status_code,
                                      "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
        return response
    finally:
        request_id_var.reset(token)


install_error_handlers(app)
app.include_router(routes_documents.router)
app.include_router(routes_chat.router)
app.include_router(routes_ingestion.router)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"name": "ePolicies AI Assistant API", "docs": "/docs", "health": "/health"}
