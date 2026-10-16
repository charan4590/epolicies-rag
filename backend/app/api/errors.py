"""Consistent error responses. Internal details and stack traces are logged,
never returned to clients."""

from __future__ import annotations

import logging

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import ConfigurationError
from app.database.connection import DatabaseUnavailableError
from app.database.schema import SchemaMismatchError
from app.ingestion.pipeline import IngestionInProgressError
from app.llm.gemini import GeminiAuthError, GeminiError, GeminiQuotaError
from app.logging_config import request_id_var
from app.services.rag_service import ChatCancelled, ChatTimeout

logger = logging.getLogger(__name__)


def error_payload(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}


def classify_exception(exc: BaseException) -> tuple[int, str, str]:
    """Map an exception to (http_status, code, user-safe message)."""
    if isinstance(exc, ConfigurationError):
        return 503, "configuration_error", str(exc)
    if isinstance(exc, (DatabaseUnavailableError, SchemaMismatchError)):
        return 503, "database_unavailable", str(exc)
    if isinstance(exc, psycopg.OperationalError):
        return 503, "database_unavailable", "The database is unavailable. Please try again shortly."
    if isinstance(exc, psycopg.Error):
        return 500, "database_error", "A database error occurred."
    if isinstance(exc, GeminiQuotaError):
        return 429, exc.kind, "The AI service rate limit was reached. Please wait a moment and retry."
    if isinstance(exc, GeminiAuthError):
        return 503, exc.kind, "The AI service rejected the server's credentials. An administrator must check GOOGLE_API_KEY."
    if isinstance(exc, GeminiError):
        return 502, exc.kind, str(exc)[:300]
    if isinstance(exc, ChatTimeout):
        return 504, "timeout", str(exc)
    if isinstance(exc, ChatCancelled):
        return 499, "cancelled", "Request cancelled."
    if isinstance(exc, IngestionInProgressError):
        return 409, "ingestion_in_progress", str(exc)
    return 500, "internal_error", "An unexpected error occurred."


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        code = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "conflict",
                413: "payload_too_large", 429: "rate_limited"}.get(exc.status_code, "http_error")
        return JSONResponse(error_payload(code, str(exc.detail)), status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = []
        for err in exc.errors()[:10]:
            location = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            problems.append(f"{location}: {err.get('msg')}" if location else str(err.get("msg")))
        return JSONResponse(error_payload("validation_error", "; ".join(problems) or "Invalid request"),
                            status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        status, code, message = classify_exception(exc)
        if status >= 500:
            logger.error("request_failed", exc_info=exc, extra={"code": code})
        else:
            logger.warning("request_failed", extra={"code": code, "detail": message})
        return JSONResponse(error_payload(code, message), status_code=status)
