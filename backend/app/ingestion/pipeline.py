"""Restartable, incremental ingestion.

Phases
------
1. **Discover & plan** – recursively find Markdown files, read and hash them,
   and classify each as new / modified / unchanged / deleted against
   PostgreSQL (renames are detected by matching content hashes).
2. **Parse & chunk** – structure-aware chunking for new/modified documents.
3. **Embed** – look up every chunk's embedding in ``embedding_cache``; only
   missing ones are sent to gemini-embedding-001, in batches *across*
   documents. Each successful batch is committed immediately, so an
   interrupted run resumes without re-embedding anything.
4. **Store** – each document (row + all chunks) is replaced in a single
   transaction, so a half-indexed document is never visible.
5. **Delete** – documents whose files disappeared lose their chunks; the row
   is kept with status ``deleted`` for auditing.
6. **Export** – ``embedding.jsonl`` is regenerated from PostgreSQL atomically.

Only one ingestion runs at a time (PostgreSQL advisory lock).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import psycopg

from app.config import Settings, get_settings
from app.database import repository as repo
from app.database.connection import connection, raw_connect
from app.ingestion import jsonl_store
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.embedder import Embedder, GeminiEmbedder, document_config
from app.ingestion.markdown_loader import (
    DocumentReadError,
    LoadedDocument,
    SourceFile,
    discover_markdown_files,
    load_document,
)
from app.ingestion.markdown_parser import parse_markdown
from app.llm.gemini import GeminiAuthError, GeminiError, GeminiInvalidRequestError

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


class IngestionInProgressError(RuntimeError):
    pass


class UnsafeDeletionError(RuntimeError):
    pass


@dataclass
class IngestionReport:
    mode: str = "incremental"
    dry_run: bool = False
    run_id: str | None = None
    source_dir: str = ""
    embedding_config: str = ""
    total_files_discovered: int = 0
    readable_files: int = 0
    empty_files: list[str] = field(default_factory=list)
    duplicate_content_groups: list[list[str]] = field(default_factory=list)
    new_documents: list[str] = field(default_factory=list)
    modified_documents: list[str] = field(default_factory=list)
    unchanged_documents: int = 0
    deleted_documents: list[str] = field(default_factory=list)
    renamed_documents: list[dict[str, str]] = field(default_factory=list)
    deferred_documents: list[str] = field(default_factory=list)
    failed_documents: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    documents_indexed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    embeddings_reused: int = 0
    embedding_requests: int = 0
    embedding_retries: int = 0
    jsonl: dict[str, Any] | None = None
    status: str = "running"
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "dry_run": self.dry_run,
            "total_files_discovered": self.total_files_discovered,
            "readable_files": self.readable_files,
            "empty_files": len(self.empty_files),
            "duplicate_content_files": sum(len(g) for g in self.duplicate_content_groups),
            "new_documents": len(self.new_documents),
            "modified_documents": len(self.modified_documents),
            "unchanged_documents": self.unchanged_documents,
            "deleted_documents": len(self.deleted_documents),
            "renamed_documents": len(self.renamed_documents),
            "deferred_documents": len(self.deferred_documents),
            "failed_documents": len(self.failed_documents),
            "documents_indexed": self.documents_indexed,
            "chunks_created": self.chunks_created,
            "embeddings_generated": self.embeddings_generated,
            "embeddings_reused": self.embeddings_reused,
        }


@dataclass
class _Work:
    loaded: LoadedDocument
    change: str  # new | modified
    title: str = ""
    source_url: str | None = None
    chunks: list[Chunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    failed: bool = False


@dataclass
class IngestionPlan:
    sources: list[SourceFile]
    loaded: dict[str, LoadedDocument]
    read_failures: dict[str, tuple[SourceFile, str]]
    new: list[str]
    modified: list[str]
    unchanged: list[str]
    deleted: list[str]  # document ids
    renamed: list[dict[str, str]]
    duplicates: list[list[str]]
    existing: dict[str, dict[str, Any]]

    def as_status(self) -> dict[str, Any]:
        by_id = {s.document_id: s.relative_path for s in self.sources}
        return {
            "discovered": len(self.sources),
            "new": [by_id[d] for d in self.new],
            "modified": [by_id[d] for d in self.modified],
            "unchanged": len(self.unchanged),
            "deleted": [self.existing[d]["relative_path"] for d in self.deleted],
            "unreadable": [{"path": s.relative_path, "error": e} for s, e in self.read_failures.values()],
            "renamed": self.renamed,
            "duplicate_content_groups": self.duplicates,
        }


def build_plan(settings: Settings, conn: psycopg.Connection, *, force: bool = False) -> IngestionPlan:
    config_id = document_config(settings).config_id
    sources = discover_markdown_files(settings.resolved_markdown_dir)
    existing = repo.get_all_documents(conn)
    loaded: dict[str, LoadedDocument] = {}
    failures: dict[str, tuple[SourceFile, str]] = {}
    for source in sources:
        try:
            loaded[source.document_id] = load_document(source, settings.max_markdown_file_bytes)
        except DocumentReadError as exc:
            failures[source.document_id] = (source, str(exc))

    new: list[str] = []
    modified: list[str] = []
    unchanged: list[str] = []
    for doc_id, doc in loaded.items():
        row = existing.get(doc_id)
        if row is None or row["status"] == "deleted":
            new.append(doc_id)
        elif (
            not force
            and row["status"] == "indexed"
            and row["content_hash"] == doc.content_hash
            and (row["embedding_config"] == config_id or row["chunk_count"] == 0)
        ):
            unchanged.append(doc_id)
        else:
            modified.append(doc_id)

    discovered_ids = {s.document_id for s in sources}
    deleted = [d for d, row in existing.items() if row["status"] != "deleted" and d not in discovered_ids]

    by_hash: dict[str, list[str]] = defaultdict(list)
    for doc in loaded.values():
        if not doc.is_empty:
            by_hash[doc.content_hash].append(doc.source.relative_path)
    duplicates = [sorted(paths) for paths in by_hash.values() if len(paths) > 1]

    deleted_hashes = {existing[d]["content_hash"]: existing[d]["relative_path"] for d in deleted
                      if existing[d]["content_hash"]}
    renamed = [
        {"from": deleted_hashes[loaded[d].content_hash], "to": loaded[d].source.relative_path}
        for d in new
        if loaded[d].content_hash in deleted_hashes
    ]
    return IngestionPlan(sources, loaded, failures, new, modified, unchanged, deleted, renamed, duplicates, existing)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder_factory: Callable[[], Embedder] | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedder_factory = embedder_factory or (lambda: GeminiEmbedder(self.settings))
        self._embedder: Embedder | None = None
        self.progress = progress or (lambda message: logger.info(message))
        self.config = document_config(self.settings)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._embedder_factory()
        return self._embedder

    # ------------------------------------------------------------------ public
    def run(
        self,
        *,
        mode: str = "incremental",
        dry_run: bool = False,
        retry_failed_only: bool = False,
        limit: int | None = None,
        allow_delete_all: bool = False,
        export: bool = True,
        triggered_by: str = "cli",
        run_id: uuid.UUID | None = None,
    ) -> IngestionReport:
        if mode not in {"incremental", "force"}:
            raise ValueError("mode must be 'incremental' or 'force'")
        started = time.monotonic()
        report = IngestionReport(
            mode=mode, dry_run=dry_run, source_dir=str(self.settings.resolved_markdown_dir),
            embedding_config=self.config.config_id,
        )
        lock_conn = raw_connect(self.settings, autocommit=True)
        try:
            if not repo.try_ingestion_lock(lock_conn):
                raise IngestionInProgressError("Another ingestion run is already in progress.")
            try:
                with connection() as conn:
                    stale = repo.mark_stale_runs_interrupted(conn, exclude=run_id)
                    if stale:
                        report.warnings.append(f"{stale} earlier run(s) were interrupted; resuming from cached work.")
                    if not dry_run:
                        if run_id is None:
                            run_id = repo.create_run(conn, mode=mode, triggered_by=triggered_by,
                                                     embedding_config=self.config.config_id)
                        else:
                            conn.execute(
                                "UPDATE ingestion_runs SET status = 'running', heartbeat_at = now() WHERE run_id = %s",
                                (run_id,),
                            )
                        report.run_id = str(run_id)
                    conn.commit()
                self._execute(report, run_id, mode=mode, dry_run=dry_run, retry_failed_only=retry_failed_only,
                              limit=limit, allow_delete_all=allow_delete_all, export=export)
                report.status = "completed_with_errors" if report.failed_documents else "completed"
            except Exception as exc:
                report.status = "failed"
                report.warnings.append(f"Run aborted: {exc}")
                if run_id is not None:
                    self._save_run(run_id, report, status="failed", error=str(exc))
                raise
            finally:
                report.duration_seconds = round(time.monotonic() - started, 2)
                if run_id is not None and report.status != "failed":
                    self._save_run(run_id, report, status=report.status)
                repo.release_ingestion_lock(lock_conn)
        finally:
            lock_conn.close()
        return report

    # ------------------------------------------------------------------ internals
    def _save_run(self, run_id: uuid.UUID, report: IngestionReport, *, status: str | None = None,
                  error: str | None = None) -> None:
        try:
            with connection() as conn:
                repo.update_run(conn, run_id, report=report.as_dict(), status=status, error=error)
                conn.commit()
        except Exception:  # noqa: BLE001 - never mask the original failure
            logger.exception("could not persist ingestion run status")

    def _event(self, run_id: uuid.UUID | None, **kwargs: Any) -> None:
        if run_id is None:
            return
        with connection() as conn:
            repo.add_event(conn, run_id, embedding_config=self.config.config_id, **kwargs)
            conn.commit()

    def _fail(self, report: IngestionReport, run_id: uuid.UUID | None, source: SourceFile, stage: str,
              error: str, *, source_hash: str = "", record: bool = True) -> None:
        report.failed_documents.append({"path": source.relative_path, "stage": stage, "error": error})
        self.progress(f"  FAILED [{stage}] {source.relative_path}: {error}")
        if run_id is None or not record:
            return
        with connection() as conn:
            repo.mark_document_failed(
                conn, document_id=source.document_id, relative_path=source.relative_path,
                filename=source.filename, source_hash=source_hash or "unreadable", size_bytes=source.size_bytes,
                error=f"[{stage}] {error}", file_modified_at=source.modified_at,
            )
            repo.add_event(conn, run_id, stage=stage, status="failed", document_id=source.document_id,
                           relative_path=source.relative_path, error=error,
                           embedding_config=self.config.config_id)
            conn.commit()

    def _execute(self, report: IngestionReport, run_id: uuid.UUID | None, *, mode: str, dry_run: bool,
                 retry_failed_only: bool, limit: int | None, allow_delete_all: bool, export: bool) -> None:
        settings = self.settings
        self.progress(f"Discovering Markdown files in {settings.resolved_markdown_dir}")
        with connection() as conn:
            plan = build_plan(settings, conn, force=(mode == "force"))
            conn.rollback()

        report.total_files_discovered = len(plan.sources)
        report.readable_files = len(plan.loaded)
        report.empty_files = sorted(d.source.relative_path for d in plan.loaded.values() if d.is_empty)
        report.duplicate_content_groups = plan.duplicates
        report.renamed_documents = plan.renamed
        report.unchanged_documents = len(plan.unchanged)
        for doc in plan.loaded.values():
            for warning in doc.warnings:
                report.warnings.append(f"{doc.source.relative_path}: {warning}")
        if not plan.sources:
            report.warnings.append(
                "No Markdown files were found. Check MARKDOWN_DIR (absolute paths are supported) "
                "and that files end in .md or .markdown."
            )

        indexed_before = [d for d, r in plan.existing.items() if r["status"] != "deleted"]
        if plan.deleted and len(plan.deleted) == len(indexed_before) and not allow_delete_all:
            raise UnsafeDeletionError(
                f"All {len(plan.deleted)} indexed documents are missing from {settings.resolved_markdown_dir}. "
                "This usually means MARKDOWN_DIR is wrong or a drive is not mounted. Refusing to delete the "
                "index; re-run with --allow-delete-all if this is intended."
            )

        todo = [(d, "new") for d in plan.new] + [(d, "modified") for d in plan.modified]
        if retry_failed_only:
            todo = [(d, c) for d, c in todo if plan.existing.get(d, {}).get("status") == "failed"]
        cap = limit or settings.embedding_max_documents_per_run
        if cap and len(todo) > cap:
            report.deferred_documents = [plan.loaded[d].source.relative_path for d, _ in todo[cap:]]
            todo = todo[:cap]
        report.new_documents = [plan.loaded[d].source.relative_path for d, c in todo if c == "new"]
        report.modified_documents = [plan.loaded[d].source.relative_path for d, c in todo if c == "modified"]
        report.deleted_documents = [plan.existing[d]["relative_path"] for d in plan.deleted] if not retry_failed_only else []

        self.progress(
            f"Discovered {report.total_files_discovered} files: {len(plan.new)} new, {len(plan.modified)} modified, "
            f"{len(plan.unchanged)} unchanged, {len(plan.deleted)} deleted, {len(plan.read_failures)} unreadable"
        )

        for source, error in plan.read_failures.values():
            self._fail(report, run_id, source, "parse", error, record=not dry_run)

        if dry_run:
            return

        # ---- parse & chunk ------------------------------------------------
        work: list[_Work] = []
        for doc_id, change in todo:
            loaded = plan.loaded[doc_id]
            item = _Work(loaded=loaded, change=change)
            try:
                parsed = parse_markdown(loaded.text, loaded.source.relative_path)
                item.title = parsed.title
                item.source_url = parsed.source_url
                item.chunks = chunk_document(
                    parsed, doc_id, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
                    min_chunk_tokens=settings.min_chunk_tokens,
                )
                item.metadata = {
                    "headings": [{"level": lvl, "text": text} for lvl, text in parsed.headings][:200],
                    "urls": parsed.urls[:200],
                    "internal_links": parsed.internal_links[:200],
                    "front_matter": {k: v for k, v in parsed.front_matter.items() if len(v) < 500},
                    "empty": not item.chunks,
                    "warnings": loaded.warnings,
                }
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                item.failed = True
                self._fail(report, run_id, loaded.source, "chunk", f"could not parse/chunk: {exc}",
                           source_hash=loaded.content_hash)
                continue
            work.append(item)

        # ---- embed (cache first, then API for the remainder) ---------------
        all_hashes = {c.content_hash: c.embedding_input for w in work for c in w.chunks}
        with connection() as conn:
            cached = repo.get_cached_embeddings(conn, all_hashes.keys(), self.config.config_id)
            conn.rollback()
        missing = [h for h in all_hashes if h not in cached]
        reused_hashes = set(cached)
        self.progress(f"Chunks: {sum(len(w.chunks) for w in work)} | cached embeddings: {len(cached)} | "
                      f"to embed: {len(missing)}")
        embed_error: GeminiError | None = None
        if missing:
            def persist(offset: int, vectors: list[list[float]]) -> None:
                batch_hashes = missing[offset : offset + len(vectors)]
                with connection() as conn:
                    repo.put_cached_embeddings(
                        conn, zip(batch_hashes, vectors), embedding_config=self.config.config_id,
                        embedding_model=self.config.model, dimensions=self.config.dimensions,
                    )
                    conn.commit()
                cached.update(zip(batch_hashes, vectors))
                report.embeddings_generated += len(vectors)
                self.progress(f"  embedded {min(offset + len(vectors), len(missing))}/{len(missing)}")
                if run_id is not None:
                    self._save_run(run_id, report)

            try:
                self.embedder.embed_documents([all_hashes[h] for h in missing], on_batch=persist)
            except GeminiError as exc:
                embed_error = exc
                report.warnings.append(f"Embedding stopped: {exc}")
                self.progress(f"  Embedding stopped: {exc}")
            finally:
                stats = getattr(self._embedder, "stats", None)
                if stats is not None:
                    report.embedding_requests = stats.requests
                    report.embedding_retries = stats.retries

        # ---- store each document atomically --------------------------------
        assigned_hashes: set[str] = set()
        for item in work:
            source = item.loaded.source
            needed = [c.content_hash for c in item.chunks]
            if any(h not in cached for h in needed):
                reason = str(embed_error) if embed_error else "embedding unavailable"
                hint = "" if isinstance(embed_error, (GeminiAuthError, GeminiInvalidRequestError)) else \
                    " Re-run ingestion to resume; completed embeddings are cached."
                self._fail(report, run_id, source, "embed", reason + hint, source_hash=item.loaded.content_hash)
                continue
            records = [
                repo.ChunkRecord(
                    chunk_id=c.chunk_id, chunk_index=c.chunk_index, text=c.text, heading_path=c.heading_path,
                    source_urls=c.source_urls, metadata=c.metadata, content_hash=c.content_hash,
                    token_count=c.token_count, char_count=c.char_count, embedding=cached[c.content_hash],
                    embedding_model=self.config.model, embedding_config=self.config.config_id,
                )
                for c in item.chunks
            ]
            doc = repo.DocumentRecord(
                document_id=item.loaded.document_id, relative_path=source.relative_path, filename=source.filename,
                title=item.title, content_hash=item.loaded.content_hash, source_url=item.source_url,
                size_bytes=source.size_bytes, encoding=item.loaded.encoding, file_modified_at=source.modified_at,
                metadata=item.metadata,
            )
            # A chunk "reuses" an embedding when it was cached before this run or was already
            # generated for an earlier chunk in this run (e.g. duplicate files).
            reused = 0
            for h in needed:
                if h in reused_hashes or h in assigned_hashes:
                    reused += 1
                assigned_hashes.add(h)
            try:
                with connection() as conn:
                    with conn.transaction():
                        repo.replace_document(conn, doc, records, embedding_config=self.config.config_id,
                                              text_search_config=settings.text_search_config)
                        repo.touch_cached_embeddings(conn, list(set(needed)), self.config.config_id)
                        if run_id is not None:
                            repo.add_event(conn, run_id, stage="store", status="succeeded",
                                           document_id=doc.document_id, relative_path=doc.relative_path,
                                           chunk_count=len(records),
                                           embeddings_generated=len(needed) - reused,
                                           embeddings_reused=reused, embedding_config=self.config.config_id)
            except psycopg.Error as exc:
                self._fail(report, run_id, source, "store", f"database error: {exc.__class__.__name__}: {exc}",
                           source_hash=item.loaded.content_hash)
                continue
            report.documents_indexed += 1
            report.chunks_created += len(records)
            report.embeddings_reused += reused
            self.progress(f"  indexed {source.relative_path} ({len(records)} chunks)")

        # ---- deletions -----------------------------------------------------
        if not retry_failed_only:
            for doc_id in plan.deleted:
                path = plan.existing[doc_id]["relative_path"]
                with connection() as conn:
                    with conn.transaction():
                        removed = repo.mark_document_deleted(conn, doc_id)
                        if run_id is not None:
                            repo.add_event(conn, run_id, stage="delete", status="succeeded", document_id=doc_id,
                                           relative_path=path, chunk_count=removed,
                                           embedding_config=self.config.config_id)
                self.progress(f"  removed deleted document {path} ({removed} chunks)")

        # ---- JSONL snapshot ------------------------------------------------
        if export:
            try:
                with connection() as conn:
                    manifest = jsonl_store.export_jsonl(
                        conn, settings.resolved_jsonl_path, embedding_model=self.config.model,
                        dimensions=self.config.dimensions, embedding_config=self.config.config_id,
                    )
                    conn.rollback()
                report.jsonl = {"path": str(settings.resolved_jsonl_path), **manifest}
                self.progress(f"Wrote {manifest['records']} records to {settings.resolved_jsonl_path}")
                self._event(run_id, stage="export", status="succeeded", chunk_count=manifest["records"])
            except OSError as exc:
                report.warnings.append(f"Could not write embedding.jsonl: {exc}")
                self._event(run_id, stage="export", status="failed", error=str(exc))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
