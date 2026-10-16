-- ePolicies PostgreSQL schema (idempotent).
--
-- `__EMBEDDING_DIMENSIONS__` is substituted by app/database/schema.py with the
-- validated EMBEDDING_DIMENSIONS setting.  Changing the dimension of an
-- existing database requires `python -m app.cli rebuild --confirm`.
--
-- pgvector must already be installed on the PostgreSQL *server*; this script
-- only enables the extension inside the current database.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_meta (
    key         text PRIMARY KEY,
    value       text NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per Markdown source file (identified by its normalised relative path).
CREATE TABLE IF NOT EXISTS documents (
    document_id       text PRIMARY KEY,
    relative_path     text NOT NULL UNIQUE,
    filename          text NOT NULL,
    title             text NOT NULL,
    content_hash      text,                -- hash of the version currently indexed (NULL = never indexed)
    source_hash       text NOT NULL,       -- hash of the most recently seen file content
    source_url        text,
    size_bytes        bigint NOT NULL DEFAULT 0,
    encoding          text,
    status            text NOT NULL CHECK (status IN ('pending', 'indexed', 'failed', 'deleted')),
    chunk_count       integer NOT NULL DEFAULT 0,
    embedding_config  text,
    metadata          jsonb NOT NULL DEFAULT '{}'::jsonb,
    last_error        text,
    failure_count     integer NOT NULL DEFAULT 0,
    file_modified_at  timestamptz,
    last_indexed_at   timestamptz,
    deleted_at        timestamptz,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);
CREATE INDEX IF NOT EXISTS documents_source_hash_idx ON documents (source_hash);

-- Retrieval units.  Rows for a document are replaced atomically in one transaction.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          text PRIMARY KEY,
    document_id       text NOT NULL REFERENCES documents (document_id) ON DELETE CASCADE,
    chunk_index       integer NOT NULL CHECK (chunk_index >= 0),
    text              text NOT NULL CHECK (length(btrim(text)) > 0),
    heading_path      text[] NOT NULL DEFAULT '{}',
    source_urls       text[] NOT NULL DEFAULT '{}',
    metadata          jsonb NOT NULL DEFAULT '{}'::jsonb,
    content_hash      text NOT NULL,       -- sha256 of the exact text sent to the embedding model
    token_count       integer NOT NULL CHECK (token_count > 0),
    char_count        integer NOT NULL CHECK (char_count > 0),
    embedding         vector(__EMBEDDING_DIMENSIONS__) NOT NULL,
    embedding_model   text NOT NULL,
    embedding_config  text NOT NULL,
    search_tsv        tsvector NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT chunks_document_order_uniq UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_content_hash_idx ON chunks (content_hash);
CREATE INDEX IF NOT EXISTS chunks_search_tsv_gin_idx ON chunks USING gin (search_tsv);

-- Embeddings keyed by input hash + embedding configuration.  Survives document
-- deletion/renames so unchanged text is never re-embedded, and makes interrupted
-- ingestion resumable (each API batch is committed here immediately).
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash      text NOT NULL,
    embedding_config  text NOT NULL,
    embedding_model   text NOT NULL,
    dimensions        integer NOT NULL CHECK (dimensions = __EMBEDDING_DIMENSIONS__),
    embedding         vector(__EMBEDDING_DIMENSIONS__) NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    last_used_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (content_hash, embedding_config)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id            uuid PRIMARY KEY,
    mode              text NOT NULL,
    status            text NOT NULL CHECK (status IN ('running', 'completed', 'completed_with_errors', 'failed', 'interrupted')),
    triggered_by      text NOT NULL,
    embedding_config  text NOT NULL,
    report            jsonb NOT NULL DEFAULT '{}'::jsonb,
    error             text,
    started_at        timestamptz NOT NULL DEFAULT now(),
    heartbeat_at      timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz
);
CREATE INDEX IF NOT EXISTS ingestion_runs_started_idx ON ingestion_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS ingestion_events (
    event_id              bigserial PRIMARY KEY,
    run_id                uuid NOT NULL REFERENCES ingestion_runs (run_id) ON DELETE CASCADE,
    document_id           text,
    relative_path         text,
    stage                 text NOT NULL CHECK (stage IN ('discover', 'parse', 'chunk', 'embed', 'store', 'delete', 'export')),
    status                text NOT NULL CHECK (status IN ('succeeded', 'failed', 'skipped')),
    error                 text,
    retry_count           integer NOT NULL DEFAULT 0,
    chunk_count           integer,
    embeddings_generated  integer,
    embeddings_reused     integer,
    embedding_config      text,
    started_at            timestamptz NOT NULL DEFAULT now(),
    finished_at           timestamptz
);
CREATE INDEX IF NOT EXISTS ingestion_events_run_idx ON ingestion_events (run_id);
CREATE INDEX IF NOT EXISTS ingestion_events_document_idx ON ingestion_events (document_id);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  uuid PRIMARY KEY,
    title            text NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_updated_idx ON conversations (updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    message_id           uuid PRIMARY KEY,
    conversation_id      uuid NOT NULL REFERENCES conversations (conversation_id) ON DELETE CASCADE,
    role                 text NOT NULL CHECK (role IN ('user', 'assistant')),
    content              text NOT NULL,
    standalone_question  text,
    payload              jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at           timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, created_at);
