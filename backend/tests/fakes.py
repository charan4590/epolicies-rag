"""Test doubles for Gemini. No test in the default suite calls a paid API."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from typing import Any

from app.config import get_settings
from app.ingestion.embedder import EmbeddingConfig, EmbeddingStats, document_config
from app.llm.gemini import GeminiQuotaError, LLMUsage

_WORD = re.compile(r"[a-z0-9]+")


def hashed_vector(text: str, dims: int) -> list[float]:
    """Deterministic bag-of-words feature hashing: texts sharing words are close."""
    vector = [0.0] * dims
    for word in _WORD.findall(text.lower()):
        if len(word) < 3:
            continue
        stem = word[:6]
        h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
        vector[h % dims] += 1.0 if (h >> 64) & 1 else -1.0
    vector[0] += 0.01  # never all-zero
    return vector


class FakeEmbedder:
    def __init__(self, dims: int | None = None, fail_after_requests: int | None = None,
                 batch_size: int = 4) -> None:
        settings = get_settings()
        self.config: EmbeddingConfig = document_config(settings)
        self.dims = dims or settings.embedding_dimensions
        self.stats = EmbeddingStats()
        self.fail_after_requests = fail_after_requests
        self.batch_size = batch_size
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def _maybe_fail(self) -> None:
        if self.fail_after_requests is not None and self.stats.requests >= self.fail_after_requests:
            raise GeminiQuotaError("simulated quota exhaustion")

    def embed_documents(self, texts: Sequence[str],
                        on_batch: Callable[[int, list[list[float]]], None] | None = None) -> list[list[float]]:
        out: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            self._maybe_fail()
            batch = list(texts[offset: offset + self.batch_size])
            self.stats.requests += 1
            self.document_calls.append(batch)
            vectors = [hashed_vector(t, self.dims) for t in batch]
            self.stats.texts_embedded += len(batch)
            if on_batch:
                on_batch(offset, vectors)
            out.extend(vectors)
        return out

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return hashed_vector(text, self.dims)


class FakeLLM:
    """Scripted structured-output LLM. ``handlers`` maps a stage name to a callable
    (prompt) -> dict, or to a dict returned as-is."""

    def __init__(self, handlers: dict[str, Any] | None = None) -> None:
        self.model = "fake-model"
        self.handlers = handlers or {}
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, *, stage: str, system: str, prompt: str, schema, usage: LLMUsage | None = None,
                      max_output_tokens: int = 4096):
        self.calls.append({"stage": stage, "prompt": prompt, "system": system})
        if usage is not None:
            usage.record(stage, None)
        handler = self.handlers.get(stage)
        if handler is None:
            raise AssertionError(f"FakeLLM has no handler for stage '{stage}'")
        if isinstance(handler, Exception):
            raise handler
        data = handler(prompt) if callable(handler) else handler
        return schema.model_validate(data)

    def stages(self) -> list[str]:
        return [c["stage"] for c in self.calls]


def first_source_label(prompt: str, contains: str) -> str | None:
    """Find the label of the first <source> whose body contains ``contains``."""
    for match in re.finditer(r'<source id="(S\d+)"[^\n]*>\n(.*?)\n</source>', prompt, re.S):
        if contains.lower() in match.group(2).lower():
            return match.group(1)
    return None
