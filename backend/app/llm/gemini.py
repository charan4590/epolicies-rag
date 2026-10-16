"""Google Gen AI SDK wrapper: client creation, error classification, bounded
retries and JSON-structured generation.

Only the official ``google-genai`` SDK is used. Model names come from settings;
nothing here falls back to a different model when a call fails.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class GeminiError(RuntimeError):
    """Base error for Gemini failures. ``kind`` is safe to show to users."""

    kind = "gemini_error"
    retryable = False


class GeminiAuthError(GeminiError):
    kind = "gemini_auth"


class GeminiInvalidRequestError(GeminiError):
    kind = "gemini_invalid_request"


class GeminiQuotaError(GeminiError):
    kind = "gemini_quota"
    retryable = True


class GeminiUnavailableError(GeminiError):
    kind = "gemini_unavailable"
    retryable = True


class GeminiResponseError(GeminiError):
    """The model answered but the output was empty, blocked or unparsable."""

    kind = "gemini_bad_response"


def classify_error(exc: BaseException) -> GeminiError:
    """Map SDK / transport exceptions to actionable, non-secret errors."""
    if isinstance(exc, GeminiError):
        return exc
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None) or 0
        status = str(getattr(exc, "status", "") or "")
        message = str(getattr(exc, "message", "") or "")
        lowered = message.lower()
        if code in (401, 403) or "api key not valid" in lowered or ("permission" in lowered and code == 400):
            return GeminiAuthError(
                "Gemini rejected the credentials (check GOOGLE_API_KEY and that the Generative "
                f"Language API is enabled). HTTP {code} {status}."
            )
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            return GeminiQuotaError(f"Gemini rate limit or quota exceeded (HTTP 429). {message[:200]}")
        if code == 404:
            return GeminiInvalidRequestError(
                f"Gemini model or endpoint not found (HTTP 404). Check GEMINI_MODEL / EMBEDDING_MODEL. {message[:200]}"
            )
        if 400 <= code < 500:
            return GeminiInvalidRequestError(f"Gemini rejected the request (HTTP {code} {status}). {message[:300]}")
        if code >= 500:
            return GeminiUnavailableError(f"Gemini service error (HTTP {code} {status}).")
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return GeminiUnavailableError("Gemini request timed out.")
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return GeminiUnavailableError("Could not reach the Gemini API (network error).")
    return GeminiError(f"Unexpected Gemini error: {type(exc).__name__}")


def call_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    on_retry: Callable[[int, GeminiError], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``fn`` retrying only transient failures (429/5xx/timeouts), bounded."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            error = classify_error(exc)
            if not error.retryable or attempt >= max_retries:
                if error is not exc:
                    raise error from exc
                raise
            attempt += 1
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * (0.75 + random.random() * 0.5)
            if on_retry:
                on_retry(attempt, error)
            logger.warning("gemini_retry", extra={"attempt": attempt, "kind": error.kind, "delay_s": round(delay, 2)})
            sleep(delay)


def make_client(settings: Settings | None = None) -> genai.Client:
    """SDK-level retries stay disabled (no ``retry_options``); :func:`call_with_retries`
    is the single, bounded retry policy."""
    settings = settings or get_settings()
    return genai.Client(
        api_key=settings.require_gemini_key(),
        http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_seconds * 1000)),
    )


class RateLimiter:
    """Simple process-wide pacing: at most ``per_minute`` calls per minute."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.interval = 60.0 / max(1, per_minute)
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if now < self._next:
                self._sleep(self._next - now)
                now = self._clock()
            self._next = now + self.interval


# --------------------------------------------------------------------------- generation
@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)

    def record(self, stage: str, usage: Any | None) -> None:
        self.calls += 1
        self.by_stage[stage] = self.by_stage.get(stage, 0) + 1
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
        self.output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
        self.total_tokens += int(getattr(usage, "total_token_count", 0) or 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "by_stage": dict(self.by_stage),
        }


class GeminiLLM:
    """Structured (JSON-schema constrained) generation with Gemini."""

    def __init__(self, settings: Settings | None = None, client: genai.Client | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.require_generation_model()
        self._client = client or make_client(self.settings)

    def generate_json(
        self,
        *,
        stage: str,
        system: str,
        prompt: str,
        schema: type[M],
        usage: LLMUsage | None = None,
        max_output_tokens: int = 4096,
    ) -> M:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": self.settings.gemini_temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
            "max_output_tokens": max_output_tokens,
        }
        if self.settings.gemini_thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.settings.gemini_thinking_level)
        elif self.settings.gemini_thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.settings.gemini_thinking_budget
            )
        config = types.GenerateContentConfig(**config_kwargs)

        def _call() -> types.GenerateContentResponse:
            return self._client.models.generate_content(model=self.model, contents=prompt, config=config)

        response = call_with_retries(_call, max_retries=self.settings.gemini_max_retries)
        if usage is not None:
            usage.record(stage, getattr(response, "usage_metadata", None))
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        text = getattr(response, "text", None)
        if not text:
            reason = _finish_reason(response)
            raise GeminiResponseError(f"Gemini returned no content for stage '{stage}' (finish reason: {reason}).")
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise GeminiResponseError(f"Gemini returned malformed JSON for stage '{stage}'.") from exc


def _finish_reason(response: Any) -> str:
    try:
        return str(response.candidates[0].finish_reason)
    except Exception:  # noqa: BLE001
        return "unknown"
