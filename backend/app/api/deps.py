"""Shared FastAPI dependencies."""

from __future__ import annotations

import hmac
import threading

from fastapi import Depends, Header, HTTPException

from app.config import Settings, get_settings
from app.services.rag_service import RAGService

_service: RAGService | None = None
_service_lock = threading.Lock()


def settings_dep() -> Settings:
    return get_settings()


def get_rag_service() -> RAGService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = RAGService(get_settings())
    return _service


def require_admin(
    x_admin_token: str | None = Header(default=None),
    settings: Settings = Depends(settings_dep),
) -> None:
    """Admin endpoints require ``X-Admin-Token`` matching ADMIN_API_TOKEN. When no
    token is configured these endpoints are disabled (use the CLI instead)."""
    configured = settings.admin_api_token.get_secret_value() if settings.admin_api_token else ""
    if not configured:
        raise HTTPException(status_code=403, detail="Administrative API is disabled. Set ADMIN_API_TOKEN or use the CLI.")
    if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), configured.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid admin token.")
