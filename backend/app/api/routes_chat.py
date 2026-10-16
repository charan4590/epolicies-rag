"""POST /chat, POST /chat/stream and conversation history endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.api.deps import get_rag_service
from app.api.errors import classify_exception
from app.database import repository as repo
from app.database.connection import connection
from app.logging_config import request_id_var
from app.schemas.api_models import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    ErrorResponse,
)
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
    503: {"model": ErrorResponse}, 504: {"model": ErrorResponse},
}


@router.post("/chat", response_model=ChatResponse, responses=ERROR_RESPONSES)
async def chat(body: ChatRequest, service: RAGService = Depends(get_rag_service)) -> ChatResponse:
    """Answer a policy question with verified citations (non-streaming)."""
    return await run_in_threadpool(service.answer, body)


@router.post("/chat/stream", responses={200: {"content": {"text/event-stream": {}}}, **ERROR_RESPONSES})
async def chat_stream(body: ChatRequest, request: Request, service: RAGService = Depends(get_rag_service)):
    """Server-Sent Events: ``stage`` progress events, then one ``final`` event with the
    complete ChatResponse (or an ``error`` event). Closing the connection cancels
    the remaining workflow at the next stage boundary."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    cancelled = threading.Event()
    request_id = request_id_var.get()

    def push(event: dict[str, Any] | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def worker() -> None:
        request_id_var.set(request_id)
        try:
            response = service.answer(body, on_event=push, is_cancelled=cancelled.is_set)
            push({"type": "final", "response": response.model_dump(mode="json")})
        except Exception as exc:  # noqa: BLE001 - converted to a safe SSE error event
            status, code, message = classify_exception(exc)
            if status >= 500:
                logger.error("stream_failed", exc_info=exc, extra={"code": code})
            push({"type": "error", "status": status,
                  "error": {"code": code, "message": message, "request_id": request_id}})
        finally:
            push(None)

    threading.Thread(target=worker, name="chat-stream", daemon=True).start()

    async def events():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    break
                yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
                if await request.is_disconnected():
                    break
        finally:
            cancelled.set()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/conversations", response_model=list[ConversationSummary], tags=["conversations"])
def list_conversations(limit: int = 30) -> list[ConversationSummary]:
    limit = max(1, min(limit, 100))
    with connection() as conn:
        return [ConversationSummary(**row) for row in repo.list_conversations(conn, limit)]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail, tags=["conversations"])
def get_conversation(conversation_id: uuid.UUID) -> ConversationDetail:
    with connection() as conn:
        conv = repo.get_conversation(conn, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(**conv)


@router.delete("/conversations/{conversation_id}", status_code=204, tags=["conversations"])
def delete_conversation(conversation_id: uuid.UUID) -> None:
    with connection() as conn:
        deleted = repo.delete_conversation(conn, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
