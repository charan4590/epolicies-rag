"""gemini-embedding-001 document/query embeddings with validation.

* Documents use task type RETRIEVAL_DOCUMENT; queries use RETRIEVAL_QUERY.
* ``output_dimensionality`` is set explicitly and every returned vector is
  checked for length and finiteness. Vectors are stored exactly as returned —
  never truncated, padded or rescaled. (Reduced-dimension outputs are not unit
  length; retrieval therefore uses cosine distance, which is scale-invariant.)
* Batching, request pacing and bounded retries for transient errors only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from google.genai import types

from app.config import REQUIRED_EMBEDDING_MODEL, Settings, get_settings
from app.llm.gemini import GeminiError, GeminiInvalidRequestError, RateLimiter, call_with_retries, make_client

logger = logging.getLogger(__name__)

DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
EMBEDDING_INPUT_FORMAT = "v1"  # bump when build_embedding_input() changes


class EmbeddingDimensionError(GeminiError):
    kind = "embedding_dimension_mismatch"


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    dimensions: int
    task_type: str = DOCUMENT_TASK_TYPE
    input_format: str = EMBEDDING_INPUT_FORMAT

    @property
    def config_id(self) -> str:
        """Identifies vectors that are interchangeable. Cached embeddings are only
        reused when this matches exactly."""
        return f"{self.model}|dim={self.dimensions}|task={self.task_type}|fmt={self.input_format}"

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "task_type": self.task_type,
            "input_format": self.input_format,
            "config_id": self.config_id,
        }


def document_config(settings: Settings | None = None) -> EmbeddingConfig:
    settings = settings or get_settings()
    return EmbeddingConfig(settings.embedding_model, settings.embedding_dimensions, DOCUMENT_TASK_TYPE)


@dataclass
class EmbeddingStats:
    requests: int = 0
    texts_embedded: int = 0
    characters_embedded: int = 0
    retries: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class Embedder(Protocol):
    config: EmbeddingConfig
    stats: EmbeddingStats

    def embed_documents(
        self, texts: Sequence[str], on_batch: Callable[[int, list[list[float]]], None] | None = None
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def validate_vectors(vectors: Sequence[Sequence[float]], expected_count: int, dimensions: int) -> None:
    if len(vectors) != expected_count:
        raise EmbeddingDimensionError(
            f"Embedding API returned {len(vectors)} vectors for {expected_count} inputs."
        )
    for idx, vector in enumerate(vectors):
        if len(vector) != dimensions:
            raise EmbeddingDimensionError(
                f"Vector {idx} has {len(vector)} dimensions; expected {dimensions} "
                "(EMBEDDING_DIMENSIONS). Vectors are never truncated or padded."
            )
        norm_sq = 0.0
        for value in vector:
            if not math.isfinite(value):
                raise EmbeddingDimensionError(f"Vector {idx} contains a non-finite value.")
            norm_sq += value * value
        if norm_sq == 0.0:
            raise EmbeddingDimensionError(f"Vector {idx} is all zeros.")


class GeminiEmbedder:
    def __init__(self, settings: Settings | None = None, client=None) -> None:
        self.settings = settings or get_settings()
        if self.settings.embedding_model != REQUIRED_EMBEDDING_MODEL:  # defence in depth
            raise ValueError(f"Only {REQUIRED_EMBEDDING_MODEL} is supported")
        self.config = document_config(self.settings)
        self.query_config = EmbeddingConfig(
            self.settings.embedding_model, self.settings.embedding_dimensions, QUERY_TASK_TYPE
        )
        self.stats = EmbeddingStats()
        self._client = client or make_client(self.settings)
        self._limiter = RateLimiter(self.settings.embedding_requests_per_minute)
        # Vertex AI may accept only one input per gemini-embedding-001 request; switch to single-text
        # requests automatically if a batched request is rejected.
        self._single_input = False

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.settings.embedding_dimensions,
        )

        def _call():
            self._limiter.wait()
            self.stats.requests += 1
            return self._client.models.embed_content(
                model=self.settings.embedding_model, contents=texts, config=config
            )

        def _on_retry(attempt: int, error: GeminiError) -> None:
            self.stats.retries += 1

        response = call_with_retries(_call, max_retries=self.settings.embedding_max_retries, on_retry=_on_retry)
        vectors = [list(e.values or []) for e in (response.embeddings or [])]
        validate_vectors(vectors, len(texts), self.settings.embedding_dimensions)
        self.stats.texts_embedded += len(texts)
        self.stats.characters_embedded += sum(len(t) for t in texts)
        return vectors

    def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        if self._single_input or len(batch) == 1:
            return [self._embed([text], task_type)[0] for text in batch]
        try:
            return self._embed(batch, task_type)
        except GeminiInvalidRequestError:
            if not self.settings.google_genai_use_vertexai:
                raise
            logger.warning("Vertex AI rejected a batched embedding request; using one text per request.")
            self._single_input = True
            return [self._embed([text], task_type)[0] for text in batch]

    def embed_documents(
        self, texts: Sequence[str], on_batch: Callable[[int, list[list[float]]], None] | None = None
    ) -> list[list[float]]:
        """Embed in batches of EMBEDDING_BATCH_SIZE. ``on_batch(offset, vectors)``
        is invoked after each successful batch so callers can persist progress."""
        out: list[list[float]] = []
        size = self.settings.embedding_batch_size
        for offset in range(0, len(texts), size):
            batch = list(texts[offset : offset + size])
            vectors = self._embed_batch(batch, DOCUMENT_TASK_TYPE)
            if on_batch:
                on_batch(offset, vectors)
            out.extend(vectors)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], QUERY_TASK_TYPE)[0]
