# ePolicies AI Assistant — Agentic RAG over Markdown policies

Ask questions about your ePolicies Markdown knowledge base and get grounded answers with **verified
citations**. Retrieval is **hybrid** (pgvector semantic search + PostgreSQL full-text search + exact
identifier matching, fused with Reciprocal Rank Fusion) and the answer workflow is **agentic**: it plans
the query, checks whether the evidence is sufficient, re-searches when something is missing, and verifies
every citation before showing the answer.

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 (App Router) · React 19 · TypeScript (strict) · Tailwind CSS v4 · shadcn/ui · lucide-react · react-markdown + remark-gfm |
| Backend | Python 3.11+ (tested on 3.12) · FastAPI · Pydantic v2 · psycopg 3 · official `google-genai` SDK |
| LLM | Gemini (model set by `GEMINI_MODEL`) |
| Embeddings | **`gemini-embedding-001`** only (768 dimensions by default, configurable) |
| Storage | **PostgreSQL only** (documents, chunks, vectors, ingestion tracking, conversations) + pgvector |
| Artifact | `artifacts/embedding.jsonl` (+ manifest) — portable snapshot, not used at query time |

No Streamlit, ChromaDB, SQLite, Redis or any other database is used.

---

## Contents

1. [How it works](#how-it-works)
2. [Prerequisites](#a-prerequisites)
3. [Backend installation](#b-backend-installation)
4. [PostgreSQL + pgvector setup](#c-postgresql-and-pgvector-setup)
5. [Frontend installation](#d-frontend-installation)
6. [Importing your Markdown files](#e-importing-your-markdown-files)
7. [Embeddings, `embedding.jsonl` and loading PostgreSQL](#f-embeddings-embeddingjsonl-and-loading-postgresql)
8. [Running the application](#g-running-the-complete-application)
9. [Testing and evaluation](#h-testing-and-evaluation)
10. [Updating documents](#i-updating-documents)
11. [Troubleshooting](#j-troubleshooting)
12. [Example usage](#k-example-usage)
13. [Configuration reference](#configuration-reference)
14. [Project structure](#project-structure)
15. [Design notes and limitations](#design-notes-and-limitations)

---

## How it works

```
Markdown files ──► discover ─► parse (markdown-it) ─► structure-aware chunks ─► embed (gemini-embedding-001, cached)
                                                                                     │
                       artifacts/embedding.jsonl ◄── atomic export ◄── PostgreSQL ◄──┘  (one transaction per document)

Question ─► 1 plan (deterministic, + Gemini when needed) ─► 2/3 hybrid retrieval ─► 4 evidence check ─┐
              ▲                                                                                       │ insufficient?
              └──────────────── targeted follow-up queries (bounded by MAX_RETRIEVAL_ROUNDS) ◄────────┘
           ─► 5 grounded answer with [S#] markers ─► 6 citation verification (code + optional Gemini) ─► one revision if needed ─► 7 response
```

**Hybrid retrieval** (`backend/app/retrieval/`)

* `vector_search.py` — cosine distance (`<=>`) over an HNSW index. Cosine is used because
  `gemini-embedding-001` vectors at reduced dimensionality are *not* unit-normalised (we measured a norm of
  ≈0.59 at 768d). Vectors are stored exactly as returned.
* `keyword_search.py` — PostgreSQL full-text search over a weighted `tsvector` (title A, heading path B,
  body C), OR-combined terms ranked by `ts_rank_cd`; plus an **exact channel** that matches identifiers
  (`HR-POL-012`, `Section 4.2`, `401(k)`, acronyms) as whole words, because stemming splits them apart.
* `synonyms.py` — before a keyword search, equivalent terms found in the query are added: a small built-in
  list ("abortion" ↔ "termination of pregnancy", "precertification" ↔ "prior authorization"), optional
  groups from `SYNONYMS_FILE`, and acronyms **learned from the policies** ("Coordination Of Benefits (COB)").
  Vector search already matches meaning; this makes two wordings of a question retrieve the same passages.
* `ranking.py` — weighted RRF: `score(d) = Σ wᵢ / (k + rankᵢ(d))`, k = `RRF_K` (60). Results are
  de-duplicated by chunk id and then by content hash (duplicate files don't crowd the top-K).

**Agentic workflow** (`backend/app/agents/`, orchestrated by `services/rag_service.py`)

1. **Query understanding/planning** — regex extraction of identifiers, acronyms and quoted phrases always
   runs. Gemini (structured JSON output) is called only for follow-ups (to resolve "what about the
   exception?"), complex/multi-part questions, or questions containing words that occur in no policy —
   usually spelling mistakes, which Gemini corrects before the keyword search (`PLANNER_MODE=auto`).
   Only greetings/thanks skip retrieval. Questions about the collection itself ("how many policies are
   there?", "list all policies about Medicare") are answered from the document index, without Gemini.
2. **Hybrid retrieval** for 1–3 planned queries.
3. **Evidence sufficiency** — Gemini judges whether the retrieved passages answer every part of the
   question; if not it names what is missing and proposes targeted queries.
4. **Re-retrieval loop** — bounded by `MAX_RETRIEVAL_ROUNDS`; policies that relevant passages refer to
   (`refer to the "…" policy`, `[Policy name](url)`) are searched next, then the checker's queries. Stops
   early on sufficient evidence, no follow-up queries, or when a follow-up search finds no new chunks.
   Passages the checker marks relevant are placed first in the answer context, then the rest by RRF.
5. **Answer generation** — sources are labelled `S1…Sn` in code and wrapped as untrusted
   `<source>` data; the model must cite every statement, preserve exceptions, surface conflicts and
   report insufficient evidence.
6. **Citation verification** — deterministic checks (label exists, chunk was retrieved, chunk exists in
   PostgreSQL and belongs to the cited document, claim words and *every number* appear in the cited
   text). Only weak claims go to a Gemini verifier; problems trigger **one** revision.
7. **Response** — answer, citations (document, heading path, passage, URLs, verification state),
   conflicts, insufficient-evidence details, and optional diagnostics (rankings, rounds, models,
   token usage — never model reasoning).

---

## A. Prerequisites

| Requirement | Version / notes |
|---|---|
| Python | 3.11 or newer (developed with 3.12). macOS system Python 3.9 is too old. |
| Node.js / npm | Node 20.9+ (developed with Node 26, npm 11) |
| PostgreSQL | 15+ (developed with 17.11) |
| pgvector | 0.7+ **installed on the PostgreSQL server** (developed with 0.8.7) |
| Gemini API key | From Google AI Studio, with access to `gemini-embedding-001` and your chosen `GEMINI_MODEL` |
| Tools | `git`, a shell; optionally Docker for the bundled `docker-compose.yml` |

## B. Backend installation

macOS (Homebrew) and Linux:

```bash
cd epolicies-rag
python3.12 -m venv backend/.venv          # Linux: install python3.12/python3.11 + python3-venv first
source backend/.venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements-dev.txt   # runtime deps + pytest/httpx
cp .env.example .env                          # then edit .env (see below)
```

Minimum `.env` values:

```dotenv
GOOGLE_API_KEY=your-key
GEMINI_MODEL=gemini-3.5-flash
DATABASE_URL=postgresql://postgres:password@localhost:5432/epolicies
MARKDOWN_DIR=./data/markdown        # or an absolute path to your existing folder
```

**Vertex AI instead of an API key** (common for company Google Cloud accounts): the same
`google-genai` SDK is used, only the credentials change:

```dotenv
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=C:/keys/epolicies-sa.json   # or leave empty and run: gcloud auth application-default login
```

The service account needs the **Vertex AI User** role (`roles/aiplatform.user`) in that project. If Vertex
rejects batched embedding requests, the app automatically switches to one text per request.
(Vertex AI support is covered by unit tests with a mocked client; it has not been run against a real
Google Cloud project yet.)

`.env` is read from `epolicies-rag/.env` (or `backend/.env`); relative paths resolve against
`epolicies-rag/`. `.env` files are git-ignored. **Never** put the API key or database URL in any
frontend env file.

> **Model availability (checked 2026-10-16):** `gemini-2.5-flash` returned *"no longer available to new
> users"*; `gemini-3.5-flash` and `gemini-3.5-flash-lite` worked. Quotas differ per model — if one returns
> HTTP 429 the app reports it; switch `GEMINI_MODEL` rather than expecting a silent fallback. List the
> models your key can use with:
> `python -c "from google import genai; [print(m.name) for m in genai.Client().models.list()]"`
> (reads `GOOGLE_API_KEY` / `GEMINI_API_KEY` from the environment).

## C. PostgreSQL and pgvector setup

pgvector is a server extension: it must be **installed on the PostgreSQL server** before
`CREATE EXTENSION vector` can succeed. The app's `init-db` command enables it in your database and
creates the schema, but it cannot install it.

**macOS (Homebrew)**

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17            # or run postgres manually without a service
/opt/homebrew/opt/postgresql@17/bin/createdb epolicies
```

**Debian/Ubuntu** (PGDG repository; adjust `17` to your server version)

```bash
sudo apt install postgresql-17 postgresql-17-pgvector
sudo -u postgres createdb epolicies
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'password';"
```

**Fedora/RHEL**: `sudo dnf install postgresql17-server pgvector_17` (PGDG repo), then `initdb`/start the service.

**Docker (optional)**: `docker compose up -d db` starts `pgvector/pgvector:pg17` on port 5432 with
databases `epolicies` and `epolicies_test` (user `postgres`, password `postgres` unless
`POSTGRES_PASSWORD` is set).

Then, from `epolicies-rag/backend` with the venv active:

```bash
python -m app.cli init-db        # enables pgvector, creates tables + indexes (idempotent)
psql "postgresql://postgres:password@localhost:5432/epolicies" -c "\dx vector"   # verify extension
curl http://localhost:8000/health   # after starting the API: database.ok / pgvector_enabled / dimensions_match
```

If you are not a superuser, ask an administrator to run `CREATE EXTENSION vector;` once in the database.

**No pgvector on the server?** With `VECTOR_BACKEND=auto` (default) `init-db` falls back to storing
embeddings as `real[]` in PostgreSQL and computing cosine similarity in the app. Everything else
(full-text search, RRF, agent loop, citations, JSONL) is unchanged, and it is fast enough for a few
thousand chunks. When pgvector is installed later, run `python -m app.cli rebuild --confirm` to switch.

## D. Frontend installation

```bash
cd epolicies-rag/frontend
npm ci                                    # or: npm install
cp .env.example .env.local                # contains only NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev                               # http://localhost:3000
npm run typecheck                         # next typegen && tsc --noEmit
npm run lint                              # eslint, zero warnings allowed
npm run build && npm start                # production build / server
```

## E. Importing your Markdown files

You **do not** upload files through the chat. Point the backend at a folder:

* **Copy** files into `epolicies-rag/data/markdown/` (any nesting), **or**
* **Keep them where they are** and set `MARKDOWN_DIR=/absolute/path/to/ePolicies` in `.env`
  (absolute, relative and `~` paths work; files are only read, never modified).

Discovery is recursive, picks up `.md` and `.markdown`, skips hidden folders/files and `__MACOSX`, and
ignores symlinks that point outside `MARKDOWN_DIR`. Identically named files in different folders are
separate documents (IDs derive from the relative path).

Check what will be indexed:

```bash
cd epolicies-rag/backend
python -m app.cli discover            # prints the count (expect ≈235 for your corpus)
python -m app.cli discover --list     # every relative path
python -m app.cli ingest --dry-run    # new / modified / unchanged / deleted, without writing anything
python -m app.cli check-files         # offline: parse + chunk every file (no database or key needed)
```

**ZIP import** (safe extraction into `MARKDOWN_DIR/<subdir>`): rejects absolute paths, `..` traversal,
symlinks and oversized members; preserves folders; never overwrites a different existing file unless
`--overwrite` is given.

```bash
python -m app.cli import-zip ~/Downloads/epolicies.zip --subdir epolicies
python -m app.cli ingest
```

## F. Embeddings, `embedding.jsonl` and loading PostgreSQL

```bash
cd epolicies-rag/backend
python -m app.cli init-db
python -m app.cli ingest              # parse → chunk → embed with gemini-embedding-001 → PostgreSQL → embedding.jsonl
python -m app.cli status              # index counts, failed documents, JSONL validity, recent runs
```

`ingest` prints an import report: files discovered, readable, empty, duplicate-content, new, modified,
unchanged, deleted, renamed, deferred, failed (with stage + reason), chunks created, embeddings
generated vs. reused, and the JSONL record count. Failed files are listed, never silently skipped, and
the exit code is `2` when any document failed.

How embeddings are generated and reused:

* Batches of `EMBEDDING_BATCH_SIZE` texts, paced to `EMBEDDING_REQUESTS_PER_MINUTE`, task type
  `RETRIEVAL_DOCUMENT` (queries use `RETRIEVAL_QUERY`), `output_dimensionality=EMBEDDING_DIMENSIONS`.
* Every vector is validated (count, dimension, finite, non-zero). Nothing is truncated or padded.
* Each successful batch is committed to the `embedding_cache` table immediately, keyed by
  `sha256(embedded text)` + embedding configuration (`model|dim|task|input format`). Unchanged,
  renamed or duplicated text is never re-embedded; an interrupted run resumes with `python -m app.cli resume`.
* Retries: only HTTP 429/5xx/timeouts, with exponential backoff, at most `EMBEDDING_MAX_RETRIES`.
  Authentication and invalid-request errors stop immediately with an actionable message.

`embedding.jsonl` (path: `EMBEDDING_JSONL_PATH`) is exported from PostgreSQL after each run: written to
a temp file, fsynced, atomically renamed, then a manifest `embedding.jsonl.manifest.json` (SHA-256,
record count, model, dimensions) is written the same way. A file without a matching manifest is
treated as incomplete. One JSON object per line:

```json
{"document_id": "doc_…", "chunk_id": "chk_…", "source_file": "hr/leave-policy.md", "text": "…",
 "embedding_model": "gemini-embedding-001", "embedding_dimensions": 768, "embedding": [ …768 floats… ],
 "metadata": {"title": "…", "heading_path": ["…"], "source_urls": [], "chunk_index": 0, "content_hash": "…",
              "token_count": 120, "char_count": 480, "embedding_config": "gemini-embedding-001|dim=768|task=RETRIEVAL_DOCUMENT|fmt=v1",
              "document_content_hash": "…", "document_source_url": null, "filename": "…", "chunk_metadata": {…}}}
```

JSONL commands (PostgreSQL stays authoritative at query time):

```bash
python -m app.cli validate-jsonl        # JSON, required fields, dimensions, duplicates, manifest
python -m app.cli export-jsonl          # regenerate from PostgreSQL
python -m app.cli load-jsonl            # rebuild PostgreSQL from the file — no API calls; safe to re-run
python -m app.cli load-jsonl --prune    # …and mark documents absent from the file as deleted
python -m app.cli reconcile             # compare file vs database and print the fix
```

## G. Running the complete application

Terminal 1 — API (http://localhost:8000, interactive docs at http://localhost:8000/docs):

```bash
cd epolicies-rag/backend && source .venv/bin/activate
uvicorn app.main:app --reload --port 8000        # or: python -m app.cli serve --reload
```

Terminal 2 — web app (http://localhost:3000):

```bash
cd epolicies-rag/frontend && npm run dev
```

The browser calls FastAPI directly over HTTP at `NEXT_PUBLIC_API_BASE_URL` (CORS allows the origins in
`CORS_ORIGINS`). The browser never talks to PostgreSQL or Gemini.

API endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | DB connectivity, pgvector/schema/dimension checks, index counts, model names (no secrets) |
| POST | `/chat` | Full agentic answer (`question`, optional `conversation_id`, `history`, `include_diagnostics`) |
| POST | `/chat/stream` | Same, as Server-Sent Events: `stage` progress events then one `final` (or `error`) event; closing the connection cancels the workflow |
| GET | `/conversations`, `/conversations/{id}` · DELETE `/conversations/{id}` | Stored history (PostgreSQL) |
| GET | `/documents?page&page_size&q&status` | Paginated indexed documents |
| GET | `/documents/{document_id}` | Metadata, ingestion info, chunk list |
| GET | `/search?q&mode=hybrid\|vector\|keyword&top_k` | Retrieval debugging (disable with `ENABLE_SEARCH_DEBUG=false`) |
| GET | `/ingest/source-status` | Files on disk vs. index (nothing is processed) |
| POST | `/ingest` | **Admin**: start background ingestion (`mode`, `retry_failed_only`, `dry_run`, `limit`) — header `X-Admin-Token: $ADMIN_API_TOKEN`; disabled when the token is unset |
| GET | `/ingest/runs`, `/ingest/runs/{id}` | **Admin**: persisted run status and failures |

## H. Testing and evaluation

```bash
cd epolicies-rag/backend && source .venv/bin/activate
pytest tests/unit tests/api                       # no network, no database (Gemini is mocked)

createdb epolicies_test                           # dedicated, disposable database (name must contain "test")
TEST_DATABASE_URL=postgresql://postgres:password@localhost:5432/epolicies_test pytest   # everything incl. PostgreSQL

cd ../frontend
npm run check                                     # typecheck + lint
npm run build                                     # production build
```

Integration tests wipe and recreate the schema in `TEST_DATABASE_URL`; they refuse to run against a
database whose name does not contain `test`. They use a deterministic fake embedder — no API cost.

**RAG evaluation** (`app/evaluation.py`): JSONL datasets with `question`, `category`,
`expected_sources` (relative paths), optional `expected_insufficient`, `reference_answer`,
`human_reviewed`.

```bash
cp eval/epolicies_eval.template.jsonl eval/epolicies_eval.jsonl   # then write real cases from your documents
python -m app.cli evaluate --dataset eval/epolicies_eval.jsonl                 # Recall@K, MRR (embeddings only)
python -m app.cli evaluate --dataset eval/epolicies_eval.jsonl --with-answers  # + citation precision/recall, insufficient-evidence accuracy, faithfulness proxy
python -m app.cli evaluate --dataset eval/epolicies_eval.jsonl --with-answers --judge   # + LLM-judged relevance (automated, not human)
```

Results are written to `backend/eval/results/` (git-ignored). `eval/fixtures_eval.jsonl` evaluates the
synthetic test fixtures only and says nothing about accuracy on your real corpus.

Make targets wrap the common commands: `make db-init ingest status api web test-unit frontend-check …`
(`make help`).

## I. Updating documents

| Change | What to do | What happens |
|---|---|---|
| Add files | drop them into `MARKDOWN_DIR`, run `python -m app.cli ingest` | new documents indexed |
| Modify files | run `ingest` | re-chunked; only changed chunks are embedded; obsolete chunks removed in the same transaction |
| Delete files | run `ingest` | chunks removed; the document row is kept with status `deleted` for audit |
| Rename/move | run `ingest` | reported as a rename; no new embeddings |
| A run was interrupted | `python -m app.cli resume` | finished batches come from the cache |
| Some documents failed | `python -m app.cli ingest --retry-failed` | only failed documents are retried |
| Re-chunk everything (e.g. new `CHUNK_SIZE`) | `python -m app.cli ingest --force` | all documents re-processed; identical text reuses cached embeddings |
| Change `EMBEDDING_DIMENSIONS` or force fresh embeddings | `python -m app.cli rebuild --confirm` | **destructive**: drops chunks + cache, re-embeds everything (API cost) |

Safety: if *every* indexed document is missing on disk (wrong path, unmounted drive), ingestion refuses
to delete the index unless you pass `--allow-delete-all`. Only one ingestion runs at a time (PostgreSQL
advisory lock, shared by CLI and API). If a modified document fails to index, its previous version stays
searchable and the document is marked `failed`.

The Documents page (`/documents`) shows indexed documents and a separate list of files on disk that are
new, modified, deleted or unreadable and have not been processed yet.

## J. Troubleshooting

| Symptom | Fix |
|---|---|
| `MARKDOWN_DIR does not exist` / `Discovered 0 Markdown file(s)` | Check the path in `.env` (relative paths are relative to `epolicies-rag/`). Use `discover --list`. Files must end in `.md`/`.markdown`; hidden folders are skipped. |
| Empty directory warning | Expected until files are added; the API still starts. |
| `Cannot connect to PostgreSQL: connection refused` | Start PostgreSQL; check host/port in `DATABASE_URL`. `pg_isready -h localhost -p 5432`. |
| `database does not exist` | `createdb epolicies` |
| `pgvector is not installed on this PostgreSQL server` | Install the OS package for **your server's major version** (see §C), restart PostgreSQL, rerun `init-db`. |
| `Permission denied while enabling pgvector` | A superuser runs `CREATE EXTENSION vector;` once in the database. |
| `GOOGLE_API_KEY is not set` / `gemini_auth` | Put the key in `epolicies-rag/.env`; make sure the Generative Language API is enabled for it. |
| `Gemini model or endpoint not found (HTTP 404)` | The model name is wrong or retired — pick one from the model list (§B). |
| `gemini_quota` (HTTP 429) / rate limits | Wait or raise quota; lower `EMBEDDING_REQUESTS_PER_MINUTE` / `EMBEDDING_BATCH_SIZE`; cap work with `ingest --limit N`. Ingestion is resumable. |
| `gemini_unavailable` (HTTP 503) | Temporary Gemini overload; retries are automatic and bounded — try again later. |
| `The database stores 768-dimensional embeddings but EMBEDDING_DIMENSIONS=…` | Restore the old value or run `rebuild --confirm`. |
| `embedding.jsonl failed validation` / manifest mismatch | The file was edited or partially written: `export-jsonl` (or `load-jsonl --allow-unverified` if you trust it). |
| JSONL and database disagree | `python -m app.cli reconcile` prints the exact command to run. |
| Empty search results | `status` (are documents indexed?), `search "your query" --mode keyword` / `--mode vector` to see which channel fails. |
| Citation shows "Check source" / "Partially verified" | The verifier could not confirm a statement (often a number). Read the passage in the source card. |
| Python dependency conflicts | Use a fresh venv with Python ≥ 3.11: `rm -rf backend/.venv` and reinstall. |
| Node dependency conflicts | `rm -rf frontend/node_modules frontend/.next && npm ci` with Node ≥ 20.9. |
| Port already in use | `uvicorn … --port 8001` (then set `NEXT_PUBLIC_API_BASE_URL` and restart Next.js) / `npm run dev -- --port 3001` (add the origin to `CORS_ORIGINS`). |
| CORS error in the browser console | Add the exact frontend origin (scheme+host+port) to `CORS_ORIGINS` and restart the API. |
| Frontend says "API offline" | Backend not running or wrong `NEXT_PUBLIC_API_BASE_URL`; `NEXT_PUBLIC_*` values are baked in at build time, so rebuild after changing it. |

## K. Example usage

Your real documents were not available while this was built, so no questions about them are listed
here. The welcome screen shows no suggested questions; users type their own.

With the synthetic fixtures (`MARKDOWN_DIR=backend/tests/fixtures/markdown`) these were run end-to-end
against the real Gemini API:

* *"How many PTO days can I carry over, and is there any exception?"* → 5 days, plus the 10-day
  exception with its written-approval and Dec-31 conditions; 2 verified citations.
* *"How many days per week can employees in the IT department work remotely?"* → cites both the IT (2
  days) and HR (3 days) policies and reports the conflict.
* *"What is the policy on bringing pet dogs to the office?"* → follow-up searches find nothing new; the
  assistant says the documents do not cover it and shows no citations.

CLI equivalents: `python -m app.cli ask "…" --diagnostics` and `python -m app.cli search "…"`.

---

## Configuration reference

All backend settings live in `epolicies-rag/.env` (see `.env.example`); they are validated at startup.

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key (server only) |
| `GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` / `GOOGLE_APPLICATION_CREDENTIALS` | `false` / — / `us-central1` / — | Use Vertex AI instead of an API key |
| `GEMINI_MODEL` | — (required for chat) | Generation model for planning, evidence checks, answers, verification |
| `GEMINI_TIMEOUT_SECONDS` / `GEMINI_MAX_RETRIES` / `GEMINI_TEMPERATURE` | 60 / 4 / 0.1 | Generation call limits |
| `GEMINI_THINKING_LEVEL` / `GEMINI_THINKING_BUDGET` | unset | Optional reasoning controls (Gemini 3 / 2.5 models). For `gemini-2.5-flash` use `GEMINI_THINKING_BUDGET=1024`: 0 weakens answers, unset can cut answers off |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Fixed; any other value is rejected |
| `EMBEDDING_DIMENSIONS` | 768 | 128/256/512/768/1536/3072; must match the database |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_REQUESTS_PER_MINUTE` / `EMBEDDING_MAX_RETRIES` | 50 / 60 / 5 | Embedding throughput and retry limits |
| `EMBEDDING_MAX_DOCUMENTS_PER_RUN` | unset | Optional cost cap per ingestion run |
| `DATABASE_URL` / `DB_STATEMENT_TIMEOUT_MS` | — / 30000 | PostgreSQL connection |
| `VECTOR_BACKEND` | `auto` | `pgvector`, `array` (no extension needed), or `auto` (pgvector if installed) |
| `MARKDOWN_DIR` / `EMBEDDING_JSONL_PATH` / `MAX_MARKDOWN_FILE_BYTES` | `./data/markdown` / `./artifacts/embedding.jsonl` / 5 MB | Paths and file limit |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `MIN_CHUNK_TOKENS` | 512 / 64 / 25 | Approximate tokens (≈4 chars/token) |
| `TOP_K_VECTOR` / `TOP_K_KEYWORD` / `TOP_K_EXACT` / `TOP_K_FINAL` / `RRF_K` | 20 / 20 / 10 / 12 / 60 | Retrieval sizes and RRF constant |
| `SYNONYM_EXPANSION` / `SYNONYMS_FILE` | `true` / — | Synonyms and learned acronyms for keyword search; optional JSON list of groups |
| `TEXT_SEARCH_CONFIG` | `english` | PostgreSQL text-search configuration |
| `MAX_RETRIEVAL_ROUNDS` / `MAX_QUERIES_PER_ROUND` | 3 / 3 | Agent loop bounds |
| `MAX_CONTEXT_CHUNKS` / `MAX_CONTEXT_CHARS` | 16 / 40000 | Context sent to the answer model |
| `PLANNER_MODE` | `auto` | `auto` (LLM planner only when useful), `always`, `never` |
| `LLM_CITATION_VERIFICATION` | `true` | Gemini check of weakly supported claims |
| `STRICT_GROUNDING` | `true` | Withhold any answer that has no verified citation to your documents |
| `MAX_HISTORY_TURNS` / `CHAT_TIMEOUT_SECONDS` | 3 / 180 | Conversation context bound / request deadline |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed browser origins |
| `ADMIN_API_TOKEN` | unset (admin API disabled) | Required header value for `/ingest*` |
| `ENABLE_SEARCH_DEBUG` / `EXPOSE_DIAGNOSTICS` | `true` / `true` | Debug endpoint / diagnostics in chat responses |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | Structured logging |

Frontend (`frontend/.env.local`): `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`).

## Project structure

```
epolicies-rag/
├── .env.example · Makefile · docker-compose.yml (optional) · README.md
├── data/markdown/            # put Markdown here (or set MARKDOWN_DIR) — git-ignored
├── artifacts/                # embedding.jsonl + manifest — git-ignored
├── backend/
│   ├── app/
│   │   ├── main.py · config.py · cli.py · evaluation.py · logging_config.py
│   │   ├── api/              # routes_chat, routes_documents (+health/search), routes_ingestion, deps, errors
│   │   ├── ingestion/        # markdown_loader (discovery, ZIP), markdown_parser, chunker, embedder, jsonl_store, pipeline
│   │   ├── database/         # schema.sql, schema.py, connection.py, repository.py
│   │   ├── retrieval/        # vector_search, keyword_search (+exact), hybrid_search, ranking (RRF), types
│   │   ├── agents/           # prompts, query_planner, retrieval_agent, evidence_checker, answer_generator, citation_verifier
│   │   ├── llm/gemini.py     # client, error classification, bounded retries, structured output
│   │   ├── services/rag_service.py
│   │   └── schemas/api_models.py
│   ├── scripts/              # thin wrappers around app.cli (ingest_documents.py, load_embeddings.py, …)
│   ├── eval/                 # fixtures_eval.jsonl, epolicies_eval.template.jsonl
│   └── tests/                # unit/, api/, integration/ (PostgreSQL), fixtures/markdown (SYNTHETIC)
└── frontend/
    ├── app/                  # layout, page (chat), chat/[conversationId], documents, documents/[documentId]
    ├── components/           # chat/, citations/, documents/, layout/, ui/ (shadcn)
    └── lib/                  # api.ts (HTTP + SSE client), types.ts (mirrors Pydantic), safe-url.ts
```

PostgreSQL schema (`backend/app/database/schema.sql`):

* `documents` — id (hash of relative path), path, filename, title, content/source hashes, source URL,
  status (`pending|indexed|failed|deleted`), chunk count, embedding config, metadata (headings, URLs,
  front matter), errors, timestamps.
* `chunks` — id, document FK (cascade), order (unique per document), text, heading path, URLs, JSONB
  metadata, content hash, token/char counts, `vector(768)` embedding, model, config, weighted `tsvector`.
  Indexes: HNSW (cosine), GIN (tsvector), B-tree on document/content hash.
* `embedding_cache` — (content hash, config) → vector.
* `ingestion_runs`, `ingestion_events` — per-run report and per-document stage/status/error/retries.
* `conversations`, `messages` — optional chat history.
* `schema_meta` — schema version, embedding model and dimensions.

## Your documents are the only knowledge source

* **No training.** RAG does not train or fine-tune any model. Your Markdown files are indexed in your own
  PostgreSQL database and, for each question, the most relevant passages are retrieved and handed to Gemini
  with instructions to answer **only** from them.
* **No outside knowledge in answers.** Every statement must cite a retrieved passage; citations are checked
  in code, unsupported statements are removed in a revision step, and with `STRICT_GROUNDING=true`
  (default) an answer that cites none of your documents is replaced by *"I could not find information about
  this in the indexed policy documents…"*.
* **No web access.** The app never searches the web, never uses Gemini's Google Search grounding tool, and
  never opens the URLs found inside your documents (they are only shown as links). The frontend uses
  system fonts and loads nothing from third-party sites.
* **What does leave the computer:** only calls to the Google Gemini API — chunk text during ingestion (to
  create embeddings) and, per question, the question plus the retrieved passages. Use a key issued under
  your company's agreement with Google: Google's Gemini API terms say paid-tier prompts are not used to
  improve Google's models, while free-tier data may be. Confirm with your company that sending the policy
  text to Gemini is approved.
* Installing dependencies (`pip`, `npm`) needs internet once, or your company's package mirror.

## Design notes and limitations

* **Deeply nested lists**: ePolicies step tables become lists nested ~20 levels deep. The Markdown parser's
  nesting limit is raised to 100 (markdown-it's default of 20 made every later heading part of that list).
  Documents indexed by an older parser (`CHUNKER_VERSION`) are re-chunked automatically by the next `ingest`;
  unchanged chunk texts reuse cached embeddings.
* **Split step lists keep their conditions**: when one list item (an ePolicies step tree: Step → If… →
  Then…) is too long for a chunk, it is split between lines and every later part starts with its parent
  steps (link text only, no URLs), like split tables repeat their header. Overlap between chunks uses whole
  sentences only and is not used between list items, so a chunk never starts mid-sentence.
* **Numbers in ranges**: citation verification accepts a number inside a range stated in the cited passage
  ("20 weeks" ⊂ "17 to 24 weeks", code 99203 ⊂ "99202 - 99205"; "N … or greater" only with the same unit).
  When the claim names a code (S2260), only ranges on that code's line count.
* **Referenced policies**: only references whose name shares a topic word with the question or the missing
  information are searched (max 2 per round). If round 1 finds nothing relevant, one more search is made and
  the assistant answers "not found" without an answer-writing call. `python -m app.cli missing-references`
  lists referenced policies/pages that are not indexed.
* **Spelling correction** runs only for words that look like misspellings of a policy word ("subrogaton");
  normal words that the policies do not use ("parking") do not cost a Gemini call.
* **Keyword search extras** (stored text unchanged): slash-joined words are also indexed separately
  (`Rawlings/Machinify`), every code of a 5-digit code range (`99202 - 99205` → 99203, 99204), `RU486`/`RU-486`
  variants; 5-digit codes and dates (`06/03/25`) are exact identifiers; a match on the chunk's own section
  heading boosts keyword ranking. Embedded text contains link text only, no URLs.
* **Collection questions** accept typos ("how mny policies"), "which policies mention X", "… in X", and fall
  back to section headings when no title matches.
* **Ambiguous acronyms are not expanded**: an acronym learned with two meanings, or a learned acronym that is
  also used as a US state code ("Massachusetts (MA)" vs "MA" = Medicare Advantage), is left as typed.
* **Token counts are approximate** (≈4 characters per token with a words×1.3 floor) because no offline
  Gemini tokenizer is available; `CHUNK_SIZE` is a target, kept far below the 2048-token embedding limit.
* **Index choice**: an HNSW index is created (≤2000 dims). For ~235 documents an exact scan would also be
  fast; HNSW is approximate, so `hnsw.ef_search` is raised per query to at least 2×top-K.
* **Keyword search** uses an OR of terms with English stemming; exact identifiers are matched by a
  separate whole-word regex channel (sequential scan — fine at this scale; add `pg_trgm` for much larger
  corpora).
* **Citation verification** is lexical + optional LLM. It catches unsupported numbers and missing
  claims well, but it is not a proof of semantic entailment; the UI marks anything it cannot confirm.
* **Streaming** sends workflow progress events and the final answer; the answer text itself is not
  token-streamed (it is produced as validated structured JSON).
* **Authentication**: the chat, documents and history endpoints have no user authentication —
  conversations are shared by everyone who can reach the API. Put the app behind your SSO/reverse proxy
  for production; only ingestion endpoints are token-protected.
* **External URLs** in documents are recorded and shown as links but never crawled.
* Images are ignored (none are expected in the corpus); raw HTML in Markdown is kept as inert text and
  never rendered.
