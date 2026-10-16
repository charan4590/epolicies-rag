"""Unified command-line interface.

    python -m app.cli <command> [options]        (run from epolicies-rag/backend)

Commands: setup, doctor, init-db, discover, import-zip, ingest, resume, rebuild, status,
export-jsonl, validate-jsonl, load-jsonl, reconcile, search, ask, evaluate, serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.config import ConfigurationError, get_settings


def _print(data: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
        return
    if isinstance(data, dict):
        width = max((len(str(k)) for k in data), default=0)
        for key, value in data.items():
            if isinstance(value, (list, dict)) and value:
                print(f"{key}:")
                items = value.items() if isinstance(value, dict) else enumerate(value)
                for k, v in items:
                    print(f"    - {v}" if isinstance(value, list) else f"    {k}: {v}")
            else:
                print(f"{str(key).ljust(width)} : {value}")
    else:
        print(data)


def _fail(message: str, code: int = 1) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------- commands
def cmd_init_db(args: argparse.Namespace) -> int:
    from app.database.schema import init_schema

    _print(init_schema(), args.json)
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    from app.ingestion.markdown_loader import discover_markdown_files

    settings = get_settings()
    files = discover_markdown_files(settings.resolved_markdown_dir)
    print(f"MARKDOWN_DIR: {settings.resolved_markdown_dir}")
    print(f"Discovered {len(files)} Markdown file(s)")
    if args.list:
        for f in files:
            print(f"  {f.relative_path}  ({f.size_bytes} bytes)")
    return 0


def cmd_import_zip(args: argparse.Namespace) -> int:
    from app.ingestion.markdown_loader import import_zip

    settings = get_settings()
    report = import_zip(Path(args.zip_path).expanduser(), settings.resolved_markdown_dir,
                        target_subdir=args.subdir, overwrite=args.overwrite,
                        max_file_bytes=settings.max_markdown_file_bytes)
    _print({"destination": report.destination, "extracted": len(report.extracted),
            "unchanged": len(report.unchanged), "skipped_non_markdown": report.skipped_non_markdown,
            "conflicts (not overwritten)": report.conflicts, "rejected": report.rejected}, args.json)
    print("Next: run `python -m app.cli ingest` to index the imported files.")
    return 0 if not report.rejected else 2


def _run_ingest(args: argparse.Namespace, *, mode: str) -> int:
    from app.ingestion.pipeline import IngestionPipeline

    quiet = getattr(args, "json", False)
    pipeline = IngestionPipeline(progress=(lambda m: None) if quiet else print)
    report = pipeline.run(
        mode=mode,
        dry_run=getattr(args, "dry_run", False),
        retry_failed_only=getattr(args, "retry_failed", False),
        limit=getattr(args, "limit", None),
        allow_delete_all=getattr(args, "allow_delete_all", False),
        export=not getattr(args, "no_export", False),
    )
    if quiet:
        _print(report.as_dict(), True)
    else:
        print("\n=== Ingestion report ===")
        _print(report.summary(), False)
        for label, items in (("Empty files", report.empty_files), ("Duplicate content", report.duplicate_content_groups),
                             ("Renamed", report.renamed_documents), ("Deferred", report.deferred_documents),
                             ("Failed", report.failed_documents), ("Warnings", report.warnings)):
            if items:
                print(f"{label}:")
                for item in items:
                    print(f"    - {item}")
        if report.jsonl:
            print(f"embedding.jsonl: {report.jsonl['records']} records -> {report.jsonl['path']}")
    return 0 if not report.failed_documents else 2


def cmd_ingest(args: argparse.Namespace) -> int:
    return _run_ingest(args, mode="force" if args.force else "incremental")


def cmd_resume(args: argparse.Namespace) -> int:
    print("Resuming: unchanged documents are skipped and cached embeddings are reused.")
    return _run_ingest(args, mode="incremental")


def cmd_rebuild(args: argparse.Namespace) -> int:
    from app.database.schema import drop_index_data, init_schema

    if not args.confirm:
        return _fail("rebuild drops ALL indexed chunks and cached embeddings and re-embeds every document "
                     "(Gemini API cost). Re-run with --confirm to proceed.")
    drop_index_data()
    init_schema()
    args.allow_delete_all = True
    return _run_ingest(args, mode="force")


def cmd_status(args: argparse.Namespace) -> int:
    from app.database import repository as repo
    from app.database.connection import connection
    from app.ingestion import jsonl_store

    settings = get_settings()
    with connection() as conn:
        stats = repo.index_stats(conn)
        runs = repo.list_runs(conn, args.runs)
        failed, _ = repo.list_documents(conn, page=1, page_size=50, status="failed")
        running = repo.ingestion_lock_held(conn)
    jsonl = jsonl_store.validate_jsonl(settings.resolved_jsonl_path, settings.embedding_dimensions)
    data = {
        "index": stats,
        "ingestion_running": running,
        "failed_documents": [{"path": f["relative_path"], "error": f["last_error"]} for f in failed],
        "jsonl": {k: v for k, v in jsonl.as_dict().items() if k in ("path", "exists", "valid", "records",
                                                                  "manifest_ok", "error_count")},
        "recent_runs": [{"run_id": str(r["run_id"]), "mode": r["mode"], "status": r["status"],
                         "started_at": r["started_at"], "finished_at": r["finished_at"],
                         "summary": {k: (len(v) if isinstance(v, list) else v)
                                     for k, v in (r["report"] or {}).items()
                                     if k in ("total_files_discovered", "new_documents", "modified_documents",
                                              "unchanged_documents", "deleted_documents", "failed_documents",
                                              "chunks_created", "embeddings_generated", "embeddings_reused")}}
                        for r in runs],
    }
    _print(data, args.json)
    return 0


def cmd_export_jsonl(args: argparse.Namespace) -> int:
    from app.database.connection import connection
    from app.ingestion import jsonl_store
    from app.ingestion.embedder import document_config

    settings = get_settings()
    config = document_config(settings)
    path = Path(args.path).expanduser() if args.path else settings.resolved_jsonl_path
    with connection() as conn:
        manifest = jsonl_store.export_jsonl(conn, path, embedding_model=config.model, dimensions=config.dimensions,
                                            embedding_config=config.config_id)
    _print({"path": str(path), **manifest}, args.json)
    return 0


def cmd_validate_jsonl(args: argparse.Namespace) -> int:
    from app.ingestion import jsonl_store

    settings = get_settings()
    path = Path(args.path).expanduser() if args.path else settings.resolved_jsonl_path
    result = jsonl_store.validate_jsonl(path, settings.embedding_dimensions)
    _print(result.as_dict(), args.json)
    return 0 if result.valid else 2


def cmd_load_jsonl(args: argparse.Namespace) -> int:
    from app.database.connection import connection
    from app.ingestion import jsonl_store

    settings = get_settings()
    path = Path(args.path).expanduser() if args.path else settings.resolved_jsonl_path
    with connection() as conn:
        report = jsonl_store.load_jsonl(conn, path, embedding_model=settings.embedding_model,
                                        dimensions=settings.embedding_dimensions,
                                        text_search_config=settings.text_search_config, prune=args.prune,
                                        allow_unverified=args.allow_unverified)
    _print(report.as_dict(), args.json)
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    from app.database.connection import connection
    from app.ingestion import jsonl_store

    settings = get_settings()
    with connection() as conn:
        result = jsonl_store.reconcile(conn, settings.resolved_jsonl_path, settings.embedding_dimensions)
    _print(result, args.json)
    return 0 if result["consistent"] else 2


def cmd_search(args: argparse.Namespace) -> int:
    from app.agents.query_planner import deterministic_plan
    from app.database.connection import connection
    from app.ingestion.embedder import GeminiEmbedder
    from app.retrieval.hybrid_search import hybrid_search

    settings = get_settings()
    embedder = GeminiEmbedder(settings) if args.mode in {"hybrid", "vector"} else None
    plan = deterministic_plan(args.query)
    identifiers = plan.identifiers
    with connection() as conn:
        result = hybrid_search(conn, embedder, [args.query], settings=settings, exact_terms=identifiers,
                               phrases=plan.exact_phrases, mode=args.mode, top_k_final=args.top_k)
    if args.json:
        _print(result.diagnostics(), True)
        return 0
    print(f"Query: {args.query!r}  mode={args.mode}  identifiers={identifiers}")
    for name, items in result.lists.items():
        print(f"\n[{name}] {len(items)} result(s)")
        for item in items[:5]:
            print(f"  #{item['rank']} {item['title']} › {' > '.join(item['heading_path'])}")
    print("\n[fused (RRF)]")
    for i, c in enumerate(result.chunks, start=1):
        print(f"  {i:2}. rrf={c.rrf_score:.5f} v={c.vector_rank} k={c.keyword_rank} x={c.exact_rank} | "
              f"{c.relative_path} › {' > '.join(c.heading_path)}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from app.schemas.api_models import ChatRequest
    from app.services.rag_service import RAGService

    response = RAGService().answer(ChatRequest(question=args.question, include_diagnostics=args.diagnostics,
                                               persist=False))
    if args.json:
        _print(response.model_dump(mode="json"), True)
        return 0
    print(response.answer)
    print("\nSources:")
    for c in response.citations:
        print(f"  [{c.label}] {c.document_title} › {' > '.join(c.heading_path)} ({c.relative_path}) [{c.verification}]")
    print(f"\nVerification: {response.verification.status}; insufficient evidence: {response.insufficient_evidence}")
    for note in response.verification.notes:
        print(f"  note: {note}")
    if response.diagnostics:
        d = response.diagnostics
        print(f"Rounds: {d['retrieval_rounds']} ({d['stop_reason']}); LLM calls: {d['llm_usage']['calls']}; "
              f"timings: {d['timings_ms']}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from app.evaluation import run_evaluation

    report = run_evaluation(Path(args.dataset), k=args.k, with_answers=args.with_answers, judge=args.judge,
                            output=Path(args.output) if args.output else None)
    _print(report["summary"], args.json)
    if report.get("output_path"):
        print(f"\nFull results: {report['output_path']}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    from app.setup_tools import run_setup

    return run_setup()


def cmd_doctor(args: argparse.Namespace) -> int:
    from app.setup_tools import run_doctor

    return run_doctor(skip_gemini=args.skip_gemini)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="ePolicies RAG command-line tools")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--json", action="store_true", help="machine-readable output")
        p.set_defaults(func=func)
        return p

    add("setup", cmd_setup, "Guided first-time setup: Markdown folder, Gemini key, PostgreSQL, .env")
    add("doctor", cmd_doctor, "Check Python, .env, Markdown folder, PostgreSQL/pgvector and Gemini").add_argument(
        "--skip-gemini", action="store_true", help="do not make test calls to the Gemini API")
    add("init-db", cmd_init_db, "Enable pgvector and create tables/indexes (idempotent)")
    p = add("discover", cmd_discover, "Count (and optionally list) Markdown files in MARKDOWN_DIR")
    p.add_argument("--list", action="store_true")
    p = add("import-zip", cmd_import_zip, "Safely extract Markdown files from a ZIP into MARKDOWN_DIR/<subdir>")
    p.add_argument("zip_path")
    p.add_argument("--subdir", help="destination sub-directory name (default: ZIP file name)")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing files whose content differs")

    for name, func, text in (("ingest", cmd_ingest, "Incremental ingestion (new/modified/deleted documents)"),
                             ("resume", cmd_resume, "Resume an interrupted ingestion (same as incremental)"),
                             ("rebuild", cmd_rebuild, "DESTRUCTIVE: drop the index and re-embed everything")):
        p = add(name, func, text)
        if name == "ingest":
            p.add_argument("--force", action="store_true",
                           help="re-parse and re-chunk every document (cached embeddings are still reused)")
            p.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
            p.add_argument("--retry-failed", action="store_true", help="only retry documents that previously failed")
            p.add_argument("--allow-delete-all", action="store_true",
                           help="permit removing every indexed document when none are found on disk")
        if name == "rebuild":
            p.add_argument("--confirm", action="store_true", help="required: confirm the destructive rebuild")
        p.add_argument("--limit", type=int, help="process at most N new/modified documents this run")
        p.add_argument("--no-export", action="store_true", help="skip writing embedding.jsonl")

    add("status", cmd_status, "Show index, JSONL and recent ingestion status").add_argument("--runs", type=int, default=5)
    add("export-jsonl", cmd_export_jsonl, "Write embedding.jsonl from PostgreSQL (atomic)").add_argument("--path")
    add("validate-jsonl", cmd_validate_jsonl, "Validate embedding.jsonl and its manifest").add_argument("--path")
    p = add("load-jsonl", cmd_load_jsonl, "Load embedding.jsonl into PostgreSQL (no API calls)")
    p.add_argument("--path")
    p.add_argument("--prune", action="store_true", help="mark documents absent from the file as deleted")
    p.add_argument("--allow-unverified", action="store_true", help="accept a file without a matching manifest")
    add("reconcile", cmd_reconcile, "Compare embedding.jsonl with PostgreSQL")
    p = add("search", cmd_search, "Retrieval diagnostics for a query")
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    p.add_argument("--top-k", type=int, default=8)
    p = add("ask", cmd_ask, "Run the full agentic RAG workflow for one question")
    p.add_argument("question")
    p.add_argument("--diagnostics", action="store_true")
    p = add("evaluate", cmd_evaluate, "Evaluate retrieval (and optionally answers) on a JSONL dataset")
    p.add_argument("--dataset", default="eval/fixtures_eval.jsonl")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--with-answers", action="store_true", help="also run the full pipeline (uses Gemini)")
    p.add_argument("--judge", action="store_true", help="LLM-judged answer relevance (automated, not human)")
    p.add_argument("--output", help="where to write detailed results (default eval/results/<timestamp>.json)")
    p = add("serve", cmd_serve, "Start the FastAPI server with uvicorn")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    from app.logging_config import configure_logging

    args = build_parser().parse_args(argv)
    try:
        settings = get_settings()
    except Exception as exc:  # pydantic ValidationError on bad env values
        return _fail(f"invalid configuration: {exc}")
    configure_logging("WARNING" if args.command != "serve" else settings.log_level, "text")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return _fail("interrupted — re-run `python -m app.cli resume` to continue", 130)
    except ConfigurationError as exc:
        return _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 - CLI shows a concise, actionable message
        from app.api.errors import classify_exception

        _, code, message = classify_exception(exc)
        if code == "internal_error":
            message = f"{type(exc).__name__}: {exc}"
        return _fail(f"[{code}] {message}")


if __name__ == "__main__":
    sys.exit(main())
