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

from app.config import ConfigurationError, Settings, get_settings

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
            hint = ""
            if "oauth" in lowered or "api keys are not supported" in lowered:
                hint = (" This looks like a Vertex AI key or project: set GOOGLE_GENAI_USE_VERTEXAI=true "
                        "in .env (with GOOGLE_CLOUD_PROJECT, or the key alone for Vertex AI express mode).")
            return GeminiAuthError(
                "Gemini rejected the credentials (check GOOGLE_API_KEY and that the Generative "
                f"Language API is enabled). HTTP {code} {status}. Google says: {message[:300]}{hint}"
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
    try:
        from google.auth import exceptions as auth_exceptions

        if isinstance(exc, (auth_exceptions.DefaultCredentialsError, auth_exceptions.RefreshError)):
            return GeminiAuthError(
                "Google Cloud credentials were rejected or expired (Vertex AI). Check "
                "GOOGLE_APPLICATION_CREDENTIALS or rerun `gcloud auth application-default login`."
            )
        if isinstance(exc, auth_exceptions.TransportError):
            return GeminiUnavailableError("Could not reach Google to refresh Vertex AI credentials (network error).")
    except ImportError:  # pragma: no cover - google-auth is a dependency of google-genai
        pass
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


VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _vertex_credentials(settings: Settings):
    """Service-account JSON if GOOGLE_APPLICATION_CREDENTIALS is set, else Application Default
    Credentials (`gcloud auth application-default login`, or the machine's attached identity)."""
    import google.auth
    from google.auth import exceptions as auth_exceptions
    from google.oauth2 import service_account

    path = settings.google_application_credentials
    if path:
        path = path.expanduser()
        if not path.is_file():
            raise ConfigurationError(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {path}")
        return service_account.Credentials.from_service_account_file(str(path), scopes=VERTEX_SCOPES)
    try:
        credentials, _ = google.auth.default(scopes=VERTEX_SCOPES)
    except auth_exceptions.DefaultCredentialsError as exc:
        raise ConfigurationError(
            "Vertex AI credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS to your service-account "
            "JSON file, or run `gcloud auth application-default login`."
        ) from exc
    return credentials


def make_client(settings: Settings | None = None) -> genai.Client:
    """Create a google-genai client for the Gemini Developer API (API key) or Vertex AI.

    SDK-level retries stay disabled (no ``retry_options``); :func:`call_with_retries`
    is the single, bounded retry policy."""
    settings = settings or get_settings()
    http_options = types.HttpOptions(timeout=int(settings.gemini_timeout_seconds * 1000))
    if settings.google_genai_use_vertexai and not settings.google_cloud_project and settings.google_api_key \
            and settings.google_api_key.get_secret_value().strip():
        # Vertex AI express mode: a Vertex API key without a project / service account.
        return genai.Client(vertexai=True, api_key=settings.google_api_key.get_secret_value().strip(),
                            http_options=http_options)
    if settings.google_genai_use_vertexai:
        return genai.Client(
            vertexai=True,
            project=settings.require_vertex_project(),
            location=settings.google_cloud_location,
            credentials=_vertex_credentials(settings),
            http_options=http_options,
        )
    return genai.Client(api_key=settings.require_gemini_key(), http_options=http_options)


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
        # Thinking models count their reasoning tokens against max_output_tokens, which can cut the JSON
        # answer off. If that happens the call is retried (bounded) with a larger output allowance.
        limit = max_output_tokens
        last_error: GeminiResponseError | None = None
        for attempt in range(3):
            response = self._generate(prompt=prompt, system=system, schema=schema, max_output_tokens=limit)
            if usage is not None:
                usage.record(stage, getattr(response, "usage_metadata", None))
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema):
                return parsed
            reason = _finish_reason(response)
            text = getattr(response, "text", None)
            if text:
                try:
                    return schema.model_validate_json(text)
                except ValidationError:
                    logger.warning("gemini_malformed_json", extra={"stage": stage, "finish_reason": reason,
                                                                  "chars": len(text), "tail": text[-160:]})
                    last_error = GeminiResponseError(
                        f"Gemini returned malformed JSON for stage '{stage}' (finish reason: {reason})."
                    )
            else:
                last_error = GeminiResponseError(
                    f"Gemini returned no content for stage '{stage}' (finish reason: {reason})."
                )
            truncated = "MAX_TOKENS" in reason or not text
            if not truncated or limit >= 32768:
                break
            logger.warning("gemini_output_truncated", extra={"stage": stage, "attempt": attempt + 1, "limit": limit})
            limit = min(limit * 2, 32768)
        assert last_error is not None
        raise last_error

    def generate_text(self, *, stage: str, system: str, prompt: str, usage: LLMUsage | None = None,
                      max_output_tokens: int = 8192) -> str:
        """Plain-text generation (no JSON schema). Retries once with more room if cut off."""
        limit = max_output_tokens
        text = ""
        for _ in range(2):
            response = self._generate(prompt=prompt, system=system, schema=None, max_output_tokens=limit)
            if usage is not None:
                usage.record(stage, getattr(response, "usage_metadata", None))
            text = (getattr(response, "text", None) or "").strip()
            if text and "MAX_TOKENS" not in _finish_reason(response):
                return text
            limit = min(limit * 2, 32768)
        if not text:
            raise GeminiResponseError(f"Gemini returned no content for stage '{stage}'.")
        return text

    def _generate(self, *, prompt: str, system: str, schema: type[BaseModel] | None, max_output_tokens: int):
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": self.settings.gemini_temperature,
            "max_output_tokens": max_output_tokens,
        }
        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema
        if self.settings.gemini_thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.settings.gemini_thinking_level)
        elif self.settings.gemini_thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.settings.gemini_thinking_budget
            )
        config = types.GenerateContentConfig(**config_kwargs)

        def _call() -> types.GenerateContentResponse:
            return self._client.models.generate_content(model=self.model, contents=prompt, config=config)

        return call_with_retries(_call, max_retries=self.settings.gemini_max_retries)


def _finish_reason(response: Any) -> str:
    try:
        return str(response.candidates[0].finish_reason)
    except Exception:  # noqa: BLE001
        return "unknown"
