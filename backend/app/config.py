"""Centralised, validated configuration.

All settings come from environment variables (optionally loaded from
``epolicies-rag/.env`` or ``backend/.env``).  Nothing in this module logs or
returns secret values; use :meth:`Settings.public_summary` for diagnostics.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# The only embedding model this application is permitted to use.
REQUIRED_EMBEDDING_MODEL = "gemini-embedding-001"
# Output sizes Google documents for gemini-embedding-001 (MRL truncation).
SUPPORTED_EMBEDDING_DIMENSIONS = {128, 256, 512, 768, 1536, 3072}
# pgvector HNSW/IVFFlat indexes support at most 2000 dimensions for `vector`.
PGVECTOR_MAX_INDEXED_DIMENSIONS = 2000


class ConfigurationError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"), str(BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Environment -------------------------------------------------------
    app_env: str = Field(default="development", description="development | production | test")
    log_level: str = "INFO"
    log_format: str = Field(default="json", description="json | text")

    # --- Gemini -------------------------------------------------------------
    # Two ways to reach Gemini through the same google-genai SDK:
    #   1. Gemini Developer API: GOOGLE_API_KEY
    #   2. Vertex AI (Google Cloud): GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT +
    #      GOOGLE_CLOUD_LOCATION, authenticated with a service-account JSON
    #      (GOOGLE_APPLICATION_CREDENTIALS) or `gcloud auth application-default login`.
    google_api_key: SecretStr | None = None
    google_genai_use_vertexai: bool = False
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    google_application_credentials: Path | None = None
    gemini_model: str = Field(default="", description="Generation model, e.g. gemini-3.5-flash")
    gemini_timeout_seconds: float = 60.0
    gemini_max_retries: int = Field(default=4, ge=0, le=10)
    gemini_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    gemini_thinking_budget: int | None = Field(
        default=None, description="Optional thinking token budget (Gemini 2.5-series models)."
    )
    gemini_thinking_level: str | None = Field(
        default=None, description="Optional thinking level for Gemini 3-series models: minimal|low|medium|high."
    )

    embedding_model: str = REQUIRED_EMBEDDING_MODEL
    embedding_dimensions: int = 768
    embedding_batch_size: int = Field(default=50, ge=1, le=100)
    embedding_requests_per_minute: int = Field(default=60, ge=1)
    embedding_max_retries: int = Field(default=5, ge=0, le=10)
    embedding_max_documents_per_run: int | None = Field(
        default=None, ge=1, description="Optional cap on documents processed per ingestion run."
    )

    # --- PostgreSQL ---------------------------------------------------------
    database_url: SecretStr = SecretStr("postgresql://postgres:postgres@localhost:5432/epolicies")
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    db_statement_timeout_ms: int = 30_000
    # auto = use pgvector when the extension is installed on the server, otherwise store embeddings as
    # real[] and compute cosine similarity in the app (fine for a few thousand chunks).
    vector_backend: str = Field(default="auto", description="auto | pgvector | array")

    # --- Paths ----------------------------------------------------------------
    markdown_dir: Path = Path("./data/markdown")
    embedding_jsonl_path: Path = Path("./artifacts/embedding.jsonl")
    max_markdown_file_bytes: int = 5 * 1024 * 1024

    # --- Chunking -------------------------------------------------------------
    chunk_size: int = Field(default=512, ge=64, le=1800, description="Target tokens per chunk")
    chunk_overlap: int = Field(default=64, ge=0, le=400)
    min_chunk_tokens: int = Field(default=25, ge=0)

    # --- Retrieval ------------------------------------------------------------
    top_k_vector: int = Field(default=20, ge=1, le=200)
    top_k_keyword: int = Field(default=20, ge=1, le=200)
    top_k_exact: int = Field(default=10, ge=0, le=100)
    top_k_final: int = Field(default=12, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1)
    rrf_max_rank: int = Field(default=10, ge=0, le=200,
                              description="Only ranks 1..N of each list count in RRF (0 = all ranks)")
    text_search_config: str = "english"

    # --- Agentic workflow -----------------------------------------------------
    max_retrieval_rounds: int = Field(default=3, ge=1, le=6)
    max_queries_per_round: int = Field(default=3, ge=1, le=6)
    max_context_chunks: int = Field(default=16, ge=1, le=40)
    max_context_chars: int = Field(default=40_000, ge=2_000)
    planner_mode: str = Field(default="auto", description="auto | always | never")
    synonym_expansion: bool = Field(
        default=True, description="Add known synonyms/acronyms (built-in, SYNONYMS_FILE, learned) to keyword search")
    synonyms_file: Path | None = Field(default=None, description="Optional JSON list of synonym groups")
    llm_citation_verification: bool = True
    strict_grounding: bool = Field(
        default=True,
        description="Never show an answer that has no verified citation to the indexed documents.",
    )
    max_history_turns: int = Field(default=3, ge=0, le=10)
    max_question_chars: int = Field(default=2000, ge=10, le=10_000)

    # --- API ------------------------------------------------------------------
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    admin_api_token: SecretStr | None = None
    enable_search_debug: bool = True
    expose_diagnostics: bool = True
    chat_timeout_seconds: float = 180.0

    @field_validator("embedding_model")
    @classmethod
    def _require_embedding_model(cls, value: str) -> str:
        if value != REQUIRED_EMBEDDING_MODEL:
            raise ValueError(
                f"EMBEDDING_MODEL must be '{REQUIRED_EMBEDDING_MODEL}' (got '{value}'). "
                "This application does not substitute embedding models."
            )
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def _check_dimensions(cls, value: int) -> int:
        if value not in SUPPORTED_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"EMBEDDING_DIMENSIONS={value} is not a documented output size for "
                f"{REQUIRED_EMBEDDING_MODEL}; choose one of {sorted(SUPPORTED_EMBEDDING_DIMENSIONS)}."
            )
        return value

    @field_validator("google_application_credentials", "google_cloud_project", "synonyms_file", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        # An empty `KEY=` line in .env means "not set" (Path("") would otherwise become ".").
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("gemini_thinking_level")
    @classmethod
    def _check_thinking_level(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip().upper()
        if value not in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}:
            raise ValueError("GEMINI_THINKING_LEVEL must be one of: minimal, low, medium, high")
        return value

    @field_validator("vector_backend")
    @classmethod
    def _check_vector_backend(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"auto", "pgvector", "array"}:
            raise ValueError("VECTOR_BACKEND must be one of: auto, pgvector, array")
        return value

    @field_validator("planner_mode")
    @classmethod
    def _check_planner_mode(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"auto", "always", "never"}:
            raise ValueError("PLANNER_MODE must be one of: auto, always, never")
        return value

    @field_validator("text_search_config")
    @classmethod
    def _check_ts_config(cls, value: str) -> str:
        # The value is interpolated as a regconfig literal; restrict to identifiers.
        if not value.replace("_", "").isalnum():
            raise ValueError("TEXT_SEARCH_CONFIG must be a simple identifier such as 'english'")
        return value

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    # --- Derived values -------------------------------------------------------
    @property
    def resolved_markdown_dir(self) -> Path:
        return _resolve(self.markdown_dir)

    @property
    def resolved_jsonl_path(self) -> Path:
        return _resolve(self.embedding_jsonl_path)

    @property
    def resolved_synonyms_path(self) -> Path | None:
        return _resolve(self.synonyms_file) if self.synonyms_file else None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def vector_index_supported(self) -> bool:
        return self.embedding_dimensions <= PGVECTOR_MAX_INDEXED_DIMENSIONS

    @property
    def gemini_backend(self) -> str:
        return "vertex" if self.google_genai_use_vertexai else "api_key"

    @property
    def gemini_configured(self) -> bool:
        if self.google_genai_use_vertexai:
            has_key = self.google_api_key is not None and bool(self.google_api_key.get_secret_value().strip())
            return bool(self.google_cloud_project and self.google_cloud_project.strip()) or has_key
        return self.google_api_key is not None and bool(self.google_api_key.get_secret_value().strip())

    def require_gemini_key(self) -> str:
        if self.google_api_key is None or not self.google_api_key.get_secret_value().strip():
            raise ConfigurationError(
                "GOOGLE_API_KEY is not set. Add it to epolicies-rag/.env (never to frontend env files), "
                "or set GOOGLE_GENAI_USE_VERTEXAI=true with GOOGLE_CLOUD_PROJECT to use Vertex AI."
            )
        return self.google_api_key.get_secret_value().strip()

    def require_vertex_project(self) -> str:
        if not self.google_cloud_project or not self.google_cloud_project.strip():
            raise ConfigurationError(
                "GOOGLE_GENAI_USE_VERTEXAI=true but GOOGLE_CLOUD_PROJECT is not set. Add your Google Cloud "
                "project ID (and GOOGLE_CLOUD_LOCATION) to epolicies-rag/.env."
            )
        return self.google_cloud_project.strip()

    def require_generation_model(self) -> str:
        if not self.gemini_model.strip():
            raise ConfigurationError(
                "GEMINI_MODEL is not set. Set it in epolicies-rag/.env, e.g. GEMINI_MODEL=gemini-3.5-flash."
            )
        return self.gemini_model.strip()

    def public_summary(self) -> dict[str, object]:
        """Non-secret settings that are safe to expose in diagnostics."""
        return {
            "app_env": self.app_env,
            "gemini_model": self.gemini_model or None,
            "gemini_backend": self.gemini_backend,
            "gemini_configured": self.gemini_configured,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k_vector": self.top_k_vector,
            "top_k_keyword": self.top_k_keyword,
            "top_k_exact": self.top_k_exact,
            "top_k_final": self.top_k_final,
            "rrf_k": self.rrf_k,
            "rrf_max_rank": self.rrf_max_rank,
            "max_retrieval_rounds": self.max_retrieval_rounds,
            "planner_mode": self.planner_mode,
            "llm_citation_verification": self.llm_citation_verification,
            "strict_grounding": self.strict_grounding,
        }


def _resolve(path: Path) -> Path:
    """Resolve relative paths against the project root, not the process CWD."""
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Tests are isolated from any developer .env file.
    if os.environ.get("APP_ENV") == "test":
        return Settings(_env_file=None)
    return Settings()
