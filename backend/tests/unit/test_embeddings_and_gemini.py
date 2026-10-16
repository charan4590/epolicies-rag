"""Embedding validation, Gemini error classification and bounded retries (mocked)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors

from app.config import Settings, get_settings
from app.ingestion.embedder import (
    DOCUMENT_TASK_TYPE,
    QUERY_TASK_TYPE,
    EmbeddingDimensionError,
    GeminiEmbedder,
    validate_vectors,
)
from app.llm.gemini import (
    GeminiAuthError,
    GeminiInvalidRequestError,
    GeminiQuotaError,
    GeminiUnavailableError,
    RateLimiter,
    call_with_retries,
    classify_error,
)


def api_error(code: int, status: str = "", message: str = "boom") -> genai_errors.APIError:
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "status": status, "message": message}})


class FakeModels:
    def __init__(self, dims: int, script: list | None = None) -> None:
        self.dims = dims
        self.script = list(script or [])
        self.calls: list[dict] = []

    def embed_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            if item == "wrong_dims":
                return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * (self.dims - 1)) for _ in contents])
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1 + i] * self.dims) for i, _ in enumerate(contents)])


def make_embedder(script=None, **overrides) -> tuple[GeminiEmbedder, FakeModels]:
    settings = Settings(_env_file=None, **{"embedding_batch_size": 2, "embedding_requests_per_minute": 100000,
                                           "embedding_max_retries": 2, **overrides})
    models = FakeModels(settings.embedding_dimensions, script)
    embedder = GeminiEmbedder(settings, client=SimpleNamespace(models=models))
    return embedder, models


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("app.llm.gemini.time.sleep", lambda s: None)


def test_validate_vectors():
    validate_vectors([[1.0, 0.0]], 1, 2)
    with pytest.raises(EmbeddingDimensionError, match="expected 3"):
        validate_vectors([[1.0, 0.0]], 1, 3)
    with pytest.raises(EmbeddingDimensionError, match="returned 1 vectors for 2"):
        validate_vectors([[1.0, 0.0]], 2, 2)
    with pytest.raises(EmbeddingDimensionError, match="non-finite"):
        validate_vectors([[float("nan"), 1.0]], 1, 2)
    with pytest.raises(EmbeddingDimensionError, match="all zeros"):
        validate_vectors([[0.0, 0.0]], 1, 2)


def test_embedder_batches_and_uses_task_types_and_dimensions():
    embedder, models = make_embedder()
    batches = []
    vectors = embedder.embed_documents(["a", "b", "c"], on_batch=lambda off, v: batches.append((off, len(v))))
    assert len(vectors) == 3 and batches == [(0, 2), (2, 1)]
    assert all(c["model"] == "gemini-embedding-001" for c in models.calls)
    assert models.calls[0]["config"].task_type == DOCUMENT_TASK_TYPE
    assert models.calls[0]["config"].output_dimensionality == 768
    embedder.embed_query("q")
    assert models.calls[-1]["config"].task_type == QUERY_TASK_TYPE
    assert embedder.stats.requests == 3 and embedder.stats.texts_embedded == 4


def test_embedder_retries_transient_errors_only():
    embedder, models = make_embedder([api_error(429, "RESOURCE_EXHAUSTED"), api_error(503)])
    assert len(embedder.embed_documents(["a"])) == 1
    assert embedder.stats.retries == 2 and len(models.calls) == 3

    embedder, models = make_embedder([api_error(400, "INVALID_ARGUMENT", "API key not valid")])
    with pytest.raises(GeminiAuthError):
        embedder.embed_documents(["a"])
    assert len(models.calls) == 1  # never retried

    embedder, models = make_embedder([api_error(429)] * 5)
    with pytest.raises(GeminiQuotaError):
        embedder.embed_documents(["a"])
    assert len(models.calls) == 3  # 1 + embedding_max_retries


def test_embedder_rejects_wrong_dimensions():
    embedder, _ = make_embedder(["wrong_dims"])
    with pytest.raises(EmbeddingDimensionError, match="767 dimensions; expected 768"):
        embedder.embed_documents(["a"])


def test_only_gemini_embedding_001_is_accepted():
    with pytest.raises(ValueError, match="gemini-embedding-001"):
        Settings(_env_file=None, embedding_model="text-embedding-004")
    with pytest.raises(ValueError, match="not a documented output size"):
        Settings(_env_file=None, embedding_dimensions=1000)


@pytest.mark.parametrize(
    "exc,expected",
    [
        (api_error(401), GeminiAuthError),
        (api_error(403), GeminiAuthError),
        (api_error(400, message="API key not valid. Please pass a valid API key."), GeminiAuthError),
        (api_error(429), GeminiQuotaError),
        (api_error(404), GeminiInvalidRequestError),
        (api_error(400), GeminiInvalidRequestError),
        (api_error(500), GeminiUnavailableError),
        (httpx.ReadTimeout("t"), GeminiUnavailableError),
        (httpx.ConnectError("c"), GeminiUnavailableError),
    ],
)
def test_classify_error(exc, expected):
    assert isinstance(classify_error(exc), expected)


def test_call_with_retries_is_bounded():
    attempts = []

    def flaky():
        attempts.append(1)
        raise api_error(503)

    with pytest.raises(GeminiUnavailableError):
        call_with_retries(flaky, max_retries=3, sleep=lambda s: None)
    assert len(attempts) == 4


def test_rate_limiter_paces_calls():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(60, clock=lambda: now[0], sleep=sleep)
    limiter.wait()
    limiter.wait()
    assert slept == [pytest.approx(1.0)]


def test_settings_resolve_relative_paths_against_project_root():
    from app.config import PROJECT_ROOT

    settings = get_settings()
    s = Settings(_env_file=None, markdown_dir="./data/markdown")
    assert s.resolved_markdown_dir == (PROJECT_ROOT / "data" / "markdown").resolve()
    assert settings.public_summary()["embedding_model"] == "gemini-embedding-001"
    assert "google_api_key" not in settings.public_summary()
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


# --------------------------------------------------------------------------- Vertex AI
def test_vertex_settings_and_client(monkeypatch, tmp_path):
    from app.llm import gemini as gemini_module

    blank = Settings(_env_file=None, google_application_credentials="", google_cloud_project="  ")
    assert blank.google_application_credentials is None and blank.google_cloud_project is None
    assert blank.gemini_backend == "api_key"

    captured = {}
    monkeypatch.setattr(gemini_module.genai, "Client", lambda **kw: captured.update(kw) or SimpleNamespace())
    monkeypatch.setattr(gemini_module, "_vertex_credentials", lambda settings: "CREDS")
    vertex = Settings(_env_file=None, google_genai_use_vertexai=True, google_cloud_project="acme-prj",
                      google_cloud_location="europe-west4", google_api_key=None)
    assert vertex.gemini_configured and vertex.gemini_backend == "vertex"
    gemini_module.make_client(vertex)
    assert captured["vertexai"] is True and captured["project"] == "acme-prj"
    assert captured["location"] == "europe-west4" and captured["credentials"] == "CREDS"
    assert "api_key" not in captured

    from app.config import ConfigurationError
    with pytest.raises(ConfigurationError, match="GOOGLE_CLOUD_PROJECT"):
        gemini_module.make_client(Settings(_env_file=None, google_genai_use_vertexai=True))
    missing = Settings(_env_file=None, google_genai_use_vertexai=True, google_cloud_project="p",
                       google_application_credentials=str(tmp_path / "nope.json"))
    monkeypatch.undo()
    with pytest.raises(ConfigurationError, match="file not found"):
        gemini_module.make_client(missing)


def test_vertex_embedding_falls_back_to_single_inputs():
    embedder, models = make_embedder([api_error(400, "INVALID_ARGUMENT", "only one instance allowed")],
                                     google_genai_use_vertexai=True, google_cloud_project="p")
    vectors = embedder.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    sizes = [len(c["contents"]) for c in models.calls]
    assert sizes[0] == 2 and all(n == 1 for n in sizes[1:])  # rejected batch, then one text per request
    # API-key mode does not hide invalid requests
    embedder, _ = make_embedder([api_error(400, "INVALID_ARGUMENT")])
    with pytest.raises(GeminiInvalidRequestError):
        embedder.embed_documents(["a", "b"])


def test_google_auth_errors_are_auth_errors():
    from google.auth import exceptions as auth_exceptions

    assert isinstance(classify_error(auth_exceptions.RefreshError("expired")), GeminiAuthError)
    assert isinstance(classify_error(auth_exceptions.DefaultCredentialsError("none")), GeminiAuthError)
