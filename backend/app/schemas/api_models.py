"""Pydantic request/response models. ``frontend/lib/types.ts`` mirrors these."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_MESSAGE_CHARS = 4000


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# --------------------------------------------------------------------------- chat
class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_HISTORY_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: uuid.UUID | None = None
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_MESSAGES,
        description="Optional client-side context, used only when conversation_id has no stored history.",
    )
    include_diagnostics: bool = False
    persist: bool = True

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class Citation(BaseModel):
    label: str = Field(description="Marker used in the answer, e.g. S1")
    chunk_id: str
    document_id: str
    document_title: str
    relative_path: str
    heading_path: list[str]
    section_paths: list[list[str]] = Field(default_factory=list)
    passage: str
    source_urls: list[str]
    document_source_url: str | None = None
    verification: Literal["supported", "weak", "unverified"] = "unverified"


class ConflictInfo(BaseModel):
    description: str
    labels: list[str]


class ClaimInfo(BaseModel):
    statement: str
    labels: list[str]
    support_type: str
    status: str
    missing_qualifications: list[str] = Field(default_factory=list)


class VerificationInfo(BaseModel):
    status: Literal["verified", "revised", "partially_verified", "not_applicable"]
    revised: bool = False
    llm_checked: bool = False
    removed_citations: list[str] = Field(default_factory=list)
    claims: list[ClaimInfo] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool
    missing_information: list[str] = Field(default_factory=list)
    conflicts: list[ConflictInfo] = Field(default_factory=list)
    verification: VerificationInfo
    standalone_question: str
    conversation_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None
    diagnostics: dict[str, Any] | None = None


# --------------------------------------------------------------------------- conversations
class ConversationSummary(BaseModel):
    conversation_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationMessage(BaseModel):
    message_id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    payload: dict[str, Any]
    created_at: datetime


class ConversationDetail(BaseModel):
    conversation_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessage]


# --------------------------------------------------------------------------- documents
DocumentStatus = Literal["pending", "indexed", "failed", "deleted"]


class DocumentSummary(BaseModel):
    document_id: str
    relative_path: str
    filename: str
    title: str
    status: DocumentStatus
    chunk_count: int
    source_url: str | None = None
    size_bytes: int
    is_empty: bool | None = None
    last_error: str | None = None
    last_indexed_at: datetime | None = None
    file_modified_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int
    page: int
    page_size: int


class ChunkSummary(BaseModel):
    chunk_id: str
    chunk_index: int
    heading_path: list[str]
    token_count: int
    char_count: int
    source_urls: list[str]
    preview: str


class DocumentDetail(DocumentSummary):
    encoding: str | None = None
    content_hash: str | None = None
    source_hash: str | None = None
    embedding_config: str | None = None
    failure_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    deleted_at: datetime | None = None
    chunks: list[ChunkSummary]


class SourceStatus(BaseModel):
    """Files on disk compared with the index (nothing is processed)."""

    source_dir_configured: bool
    source_dir_exists: bool
    discovered: int
    new: list[str]
    modified: list[str]
    unchanged: int
    deleted: list[str]
    unreadable: list[dict[str, str]]
    error: str | None = None


# --------------------------------------------------------------------------- search / health / ingest
class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    document_title: str
    relative_path: str
    heading_path: list[str]
    text: str
    source_urls: list[str]
    rrf_score: float
    vector_rank: int | None = None
    vector_similarity: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    exact_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    mode: Literal["hybrid", "vector", "keyword"]
    identifiers: list[str]
    hits: list[SearchHit]
    diagnostics: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    database: dict[str, Any]
    index: dict[str, Any] | None = None
    gemini: dict[str, Any]
    config: dict[str, Any]


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["incremental", "force"] = "incremental"
    retry_failed_only: bool = False
    dry_run: bool = False
    limit: int | None = Field(default=None, ge=1, le=10_000)


class IngestAccepted(BaseModel):
    run_id: uuid.UUID | None
    status: str
    message: str


class IngestionRun(BaseModel):
    run_id: uuid.UUID
    mode: str
    status: str
    triggered_by: str
    embedding_config: str
    report: dict[str, Any]
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    heartbeat_at: datetime | None = None
    failures: list[dict[str, Any]] = Field(default_factory=list)
