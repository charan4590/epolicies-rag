"""Guided setup (`python -m app.cli setup`) and environment checks (`python -m app.cli doctor`).

Both are meant for a fresh machine: they explain each problem in plain language and
print the fix. Secrets are never printed (the API key is shown only as its length,
the database password is masked).
"""

from __future__ import annotations

import getpass
import os
import platform
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from app.config import PROJECT_ROOT, get_settings

ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def mask_database_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.password is None:
        return url
    netloc = parts.netloc.replace(f":{parts.password}@", ":****@", 1)
    return urlunsplit(parts._replace(netloc=netloc))


def clean_path_input(raw: str) -> Path:
    """Accept paths pasted from Explorer/Finder ("Copy as path" adds quotes; drag-and-drop may escape spaces)."""
    value = raw.strip().strip('"').strip("'").replace("\\ ", " ")
    return Path(os.path.expandvars(value)).expanduser()


def count_markdown(folder: Path) -> tuple[int, list[str]]:
    from app.ingestion.markdown_loader import discover_markdown_files

    files = discover_markdown_files(folder)
    return len(files), [f.relative_path for f in files[:5]]


def build_database_url(host: str, port: str, database: str, user: str, password: str) -> str:
    auth = quote(user, safe="") + (f":{quote(password, safe='')}" if password else "")
    return f"postgresql://{auth}@{host}:{port}/{quote(database, safe='')}"


def write_env(values: dict[str, str], path: Path = ENV_PATH) -> Path | None:
    """Write `.env` from `.env.example`, replacing the given keys. Backs up an existing file."""
    template = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines() if ENV_EXAMPLE.exists() else []
    backup = None
    if path.exists():
        backup = path.with_name(".env.bak")
        shutil.copy2(path, backup)
    remaining = dict(values)
    lines: list[str] = []
    for line in template:
        key = line.split("=", 1)[0].strip().lstrip("# ").strip() if "=" in line else ""
        if key in remaining:
            lines.append(f"{key}={remaining.pop(key)}")
        else:
            lines.append(line)
    lines.extend(f"{k}={v}" for k, v in remaining.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # owner-only on macOS/Linux; ignored on Windows
    except OSError:
        pass
    return backup


# --------------------------------------------------------------------------- doctor
class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"{OK} {message}")

    def fail(self, message: str, fix: str) -> None:
        self.failures += 1
        print(f"{FAIL} {message}\n       Fix: {fix}")

    def warn(self, message: str, fix: str = "") -> None:
        self.warnings += 1
        print(f"{WARN} {message}" + (f"\n       Hint: {fix}" if fix else ""))


def check_database(report: Report) -> bool:
    import psycopg

    from app.database.connection import DatabaseUnavailableError, raw_connect
    from app.database.schema import existing_vector_dimensions, pgvector_available

    settings = get_settings()
    print(f"       DATABASE_URL = {mask_database_url(settings.database_url.get_secret_value())}")
    try:
        with raw_connect(settings) as conn:
            version = conn.execute("SHOW server_version").fetchone()["server_version"]
            report.ok(f"Connected to PostgreSQL {version}")
            if not pgvector_available(conn):
                report.fail("pgvector is NOT installed on this PostgreSQL server",
                            "install pgvector for this server version (see START-HERE.md), or use Docker: "
                            "`docker compose up -d db`, or ask IT to install it")
                return False
            ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            if ext is None:
                report.warn("pgvector is installed but not enabled in this database yet",
                            "run `python -m app.cli init-db`")
                return True
            report.ok(f"pgvector {ext['extversion']} enabled")
            dims = existing_vector_dimensions(conn)
            if dims is None:
                report.warn("Tables not created yet", "run `python -m app.cli init-db`")
            elif dims != settings.embedding_dimensions:
                report.fail(f"Database stores {dims}-dimension vectors but EMBEDDING_DIMENSIONS={settings.embedding_dimensions}",
                            f"set EMBEDDING_DIMENSIONS={dims} in .env, or run `python -m app.cli rebuild --confirm`")
            else:
                row = conn.execute("""SELECT (SELECT count(*) FROM documents WHERE status = 'indexed') AS docs,
                                             (SELECT count(*) FROM chunks) AS chunks""").fetchone()
                report.ok(f"Tables ready ({dims}-dim vectors): {row['docs']} documents, {row['chunks']} chunks indexed")
                if row["docs"] == 0:
                    report.warn("Nothing indexed yet", "run `python -m app.cli ingest`")
            return True
    except DatabaseUnavailableError as exc:
        report.fail(str(exc), "check host/port/user/password/database in DATABASE_URL and that PostgreSQL is running")
    except psycopg.Error as exc:
        report.fail(f"Database error: {exc}", "check DATABASE_URL and database permissions")
    return False


def check_gemini(report: Report) -> None:
    from google.genai import types

    from app.llm.gemini import GeminiError, classify_error, make_client

    from app.config import ConfigurationError

    settings = get_settings()
    if settings.google_genai_use_vertexai:
        print(f"       Using Vertex AI: project={settings.google_cloud_project or '(not set)'} "
              f"location={settings.google_cloud_location}")
        if not settings.gemini_configured:
            report.fail("GOOGLE_CLOUD_PROJECT is not set", "add GOOGLE_CLOUD_PROJECT=<project id> to epolicies-rag/.env")
            return
        creds = settings.google_application_credentials
        if not settings.google_cloud_project:
            report.ok("Credentials: Vertex AI express-mode API key")
        else:
            report.ok("Credentials: " + (f"service-account file {creds}" if creds else
                                         "Application Default Credentials (gcloud login)"))
    else:
        key = settings.google_api_key.get_secret_value().strip() if settings.google_api_key else ""
        if not key:
            report.fail("GOOGLE_API_KEY is not set",
                        "add GOOGLE_API_KEY=<your key> to epolicies-rag/.env (or configure Vertex AI)")
            return
        report.ok(f"Using a Gemini API key ({len(key)} characters)")
    try:
        client = make_client(settings)
    except ConfigurationError as exc:
        report.fail(str(exc), "fix the Gemini settings in epolicies-rag/.env")
        return
    try:
        result = client.models.embed_content(
            model=settings.embedding_model, contents=["connection test"],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY",
                                            output_dimensionality=settings.embedding_dimensions),
        )
        dims = len(result.embeddings[0].values or [])
        if dims == settings.embedding_dimensions:
            report.ok(f"{settings.embedding_model} works ({dims} dimensions)")
        else:
            report.fail(f"{settings.embedding_model} returned {dims} dimensions", "check EMBEDDING_DIMENSIONS")
    except Exception as exc:  # noqa: BLE001
        error = classify_error(exc)
        report.fail(f"{settings.embedding_model} failed: {error}",
                    "confirm the key/project is valid and has access to gemini-embedding-001")
    if not settings.gemini_model:
        report.fail("GEMINI_MODEL is not set (needed for chat answers)",
                    "add GEMINI_MODEL=gemini-3.5-flash (or another model your key can use) to .env")
        return
    try:
        client.models.generate_content(model=settings.gemini_model, contents="Reply with OK.",
                                       config=types.GenerateContentConfig(max_output_tokens=16))
        report.ok(f"Generation model {settings.gemini_model} works")
    except Exception as exc:  # noqa: BLE001
        error: GeminiError = classify_error(exc)
        hint = {"gemini_quota": "quota exhausted for this model: wait, or choose another GEMINI_MODEL",
                "gemini_unavailable": "temporary Google overload: try again in a few minutes",
                "gemini_invalid_request": "model name not available to this key: choose another GEMINI_MODEL"}
        report.fail(f"Generation model {settings.gemini_model} failed: {error}",
                    hint.get(error.kind, "check the key and GEMINI_MODEL"))


def run_doctor(*, skip_gemini: bool = False) -> int:
    report = Report()
    print("== ePolicies doctor ==")
    print(f"       System: {platform.system()} {platform.release()} · Python {platform.python_version()}")
    if sys.version_info < (3, 11):
        report.fail("Python 3.11 or newer is required", "install Python 3.12 and recreate backend/.venv")
    else:
        report.ok("Python version")

    if ENV_PATH.exists():
        report.ok(f"Settings file found: {ENV_PATH}")
    elif (PROJECT_ROOT / "backend" / ".env").exists():
        report.ok("Settings file found: backend/.env")
    elif os.environ.get("DATABASE_URL") or os.environ.get("GOOGLE_API_KEY"):
        report.warn("No .env file; using settings from environment variables")
    else:
        report.fail("No .env settings file", "run `python -m app.cli setup`")

    settings = get_settings()
    folder = settings.resolved_markdown_dir
    print(f"\n-- Markdown files --\n       MARKDOWN_DIR = {folder}")
    try:
        count, sample = count_markdown(folder)
        if count == 0:
            report.fail("The folder exists but contains no .md files",
                        "point MARKDOWN_DIR at the folder that contains your policy .md files")
        else:
            report.ok(f"Found {count} Markdown file(s), e.g. {', '.join(sample[:3])}")
    except Exception as exc:  # noqa: BLE001 - SourceDirectoryError and permission errors
        report.fail(str(exc), "set MARKDOWN_DIR in .env to the full path of your policy folder")

    print("\n-- PostgreSQL --")
    check_database(report)

    print("\n-- Gemini --")
    if skip_gemini:
        report.warn("Gemini checks skipped (--skip-gemini)")
    else:
        check_gemini(report)

    print(f"\nSummary: {report.failures} problem(s), {report.warnings} warning(s).")
    if report.failures == 0:
        print("Next: `python -m app.cli ingest`, then start the API and the frontend (see START-HERE.md).")
    return 0 if report.failures == 0 else 1


# --------------------------------------------------------------------------- setup wizard
def _ask(prompt: str, default: str | None = None, *, secret: bool = False,
         reader: Callable[[str], str] = input) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = (getpass.getpass(f"{prompt}{suffix}: ") if secret else reader(f"{prompt}{suffix}: ")).strip()
        if value:
            return value
        if default is not None:
            return default
        print("  A value is required.")


def _yes(prompt: str, default: bool = True) -> bool:
    answer = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not answer else answer.startswith("y")


def _create_database_if_missing(url: str, database: str) -> bool:
    """Connect to the server's `postgres` database and create `database` if it does not exist."""
    import psycopg
    from psycopg import sql

    parts = urlsplit(url)
    admin_url = urlunsplit(parts._replace(path="/postgres"))
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=10) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,)).fetchone()
            if exists:
                return True
            if not _yes(f"  Database '{database}' does not exist. Create it now?"):
                return False
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
            print(f"  Created database '{database}'.")
            return True
    except psycopg.Error as exc:
        print(f"  Could not connect to PostgreSQL: {str(exc).strip().splitlines()[0]}")
        return False


def run_setup() -> int:
    from app.database.connection import close_pool

    print("== ePolicies setup ==", flush=True)
    print("This writes epolicies-rag/.env. Press Enter to accept the value in [brackets].\n")

    # 1. Markdown folder
    print("Step 1/4 — Your policy Markdown folder")
    print("  Tip: Windows Explorer → Shift+right-click the folder → 'Copy as path'; Mac Finder → Option+right-click → 'Copy as Pathname'.")
    while True:
        folder = clean_path_input(_ask("  Full path of the folder with your .md files"))
        try:
            count, sample = count_markdown(folder)
        except Exception as exc:  # noqa: BLE001
            print(f"  {exc}")
            continue
        if count == 0:
            print("  No .md files found there (sub-folders are searched too). Try another folder.")
            continue
        print(f"  Found {count} Markdown files, e.g. {', '.join(sample[:3])}")
        if _yes("  Use this folder?"):
            break

    # 2. Gemini
    print("\nStep 2/4 — Google Gemini")
    print("  How does your company give you Gemini?")
    print("    1 = an API key (Google AI Studio)")
    print("    2 = Vertex AI (a Google Cloud project)")
    choice = _ask("  Enter 1 or 2", "1")
    gemini: dict[str, str] = {}
    if choice.strip() == "2":
        gemini["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        gemini["GOOGLE_CLOUD_PROJECT"] = _ask("  Google Cloud project ID")
        gemini["GOOGLE_CLOUD_LOCATION"] = _ask("  Region (GOOGLE_CLOUD_LOCATION)", "us-central1")
        creds = input("  Path to the service-account JSON file (Enter if you use `gcloud auth "
                      "application-default login`): ").strip()
        if creds:
            gemini["GOOGLE_APPLICATION_CREDENTIALS"] = clean_path_input(creds).resolve().as_posix()
    else:
        gemini["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
        key = getpass.getpass("  GOOGLE_API_KEY (input is hidden; press Enter to skip if you don't have it yet): ")
        gemini["GOOGLE_API_KEY"] = key.strip()
        if not key.strip():
            print("  Skipped. Run `python -m app.cli setup` again when you have the key.")
    model = _ask("  Generation model for answers (GEMINI_MODEL)", "gemini-3.5-flash")

    # 3. PostgreSQL
    print("\nStep 3/4 — PostgreSQL connection")
    host = _ask("  Host", "localhost")
    port = _ask("  Port", "5432")
    database = _ask("  Database name", "epolicies")
    user = _ask("  User", "postgres")
    password = getpass.getpass("  Password (input is hidden, Enter for none): ")
    url = build_database_url(host, port, database, user, password)
    _create_database_if_missing(url, database)

    # 4. Write .env
    print("\nStep 4/4 — Saving settings")
    backup = write_env({
        **gemini,
        "GEMINI_MODEL": model,
        "EMBEDDING_MODEL": "gemini-embedding-001",
        "DATABASE_URL": url,
        "MARKDOWN_DIR": folder.resolve().as_posix(),
    })
    print(f"  Wrote {ENV_PATH}" + (f" (previous version saved as {backup.name})" if backup else ""))
    get_settings.cache_clear()
    close_pool()

    # Initialise the schema, then check everything.
    from app.database.connection import DatabaseUnavailableError
    from app.database.schema import init_schema

    try:
        info = init_schema()
        print(f"  Database ready: pgvector {info['pgvector_version']}, {info['embedding_dimensions']}-dim vectors.")
    except DatabaseUnavailableError as exc:
        print(f"  Database not ready yet: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"  Database not ready yet: {exc}")

    print()
    status = run_doctor()
    if status == 0 and _yes("\nIndex your documents now? (calls the Gemini embedding API)", default=False):
        from app.ingestion.pipeline import IngestionPipeline

        report = IngestionPipeline(progress=print).run()
        print(f"\nIndexed {report.documents_indexed} documents, {report.chunks_created} chunks; "
              f"{len(report.failed_documents)} failed.")
    return status
