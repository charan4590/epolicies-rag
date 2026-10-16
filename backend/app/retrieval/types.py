"""Shared retrieval data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CHUNK_COLUMNS = """
    c.chunk_id, c.document_id, c.chunk_index, c.text, c.heading_path, c.source_urls, c.content_hash,
    c.metadata AS chunk_metadata, d.title, d.relative_path, d.source_url AS document_source_url
"""


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    title: str
    relative_path: str
    heading_path: list[str]
    source_urls: list[str]
    document_source_url: str | None
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Ranking diagnostics (1-based ranks within each list; None = not present)
    vector_rank: int | None = None
    vector_similarity: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    exact_rank: int | None = None
    exact_hits: int | None = None
    rrf_score: float = 0.0
    matched_queries: list[str] = field(default_factory=list)
    duplicate_of: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "RetrievedChunk":
        return cls(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            title=row["title"],
            relative_path=row["relative_path"],
            heading_path=list(row["heading_path"] or []),
            source_urls=list(row["source_urls"] or []),
            document_source_url=row.get("document_source_url"),
            content_hash=row["content_hash"],
            metadata=row.get("chunk_metadata") or {},
        )

    def brief(self) -> dict[str, Any]:
        """Compact form for diagnostics."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "relative_path": self.relative_path,
            "heading_path": self.heading_path,
            "vector_rank": self.vector_rank,
            "vector_similarity": round(self.vector_similarity, 4) if self.vector_similarity is not None else None,
            "keyword_rank": self.keyword_rank,
            "keyword_score": round(self.keyword_score, 4) if self.keyword_score is not None else None,
            "exact_rank": self.exact_rank,
            "rrf_score": round(self.rrf_score, 6),
        }
