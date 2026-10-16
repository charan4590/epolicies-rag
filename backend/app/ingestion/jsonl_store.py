"""``embedding.jsonl`` artifact: atomic export, validation, loading and reconciliation.

PostgreSQL is authoritative at query time. The JSONL file is a portable
snapshot of every indexed chunk (text + real embedding vectors + metadata)
that can rebuild the database without calling the embedding API.

Completeness: the JSONL file is written to a temporary file, fsynced and
atomically renamed; then a manifest (``<name>.manifest.json``) recording the
file's SHA-256, record count and embedding configuration is written the same
way. A JSONL file whose manifest is missing or does not match is treated as
incomplete.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

from app.database import repository as repo

RECORD_VERSION = 1
REQUIRED_FIELDS = (
    "document_id",
    "chunk_id",
    "source_file",
    "text",
    "embedding_model",
    "embedding_dimensions",
    "embedding",
    "metadata",
)
REQUIRED_METADATA = ("title", "heading_path", "source_urls", "chunk_index", "content_hash")


def manifest_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_name(jsonl_path.name + ".manifest.json")


def _atomic_write_bytes(target: Path, writer) -> str:
    """Write via a temp file in the same directory, fsync, then os.replace.
    Returns the SHA-256 of the written content."""
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            def emit(data: bytes) -> None:
                digest.update(data)
                handle.write(data)

            writer(emit)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return digest.hexdigest()


def chunk_row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    embedding = repo.vector_to_list(row["embedding"])
    return {
        "document_id": row["document_id"],
        "chunk_id": row["chunk_id"],
        "source_file": row["relative_path"],
        "text": row["text"],
        "embedding_model": row["embedding_model"],
        "embedding_dimensions": len(embedding),
        "embedding": embedding,
        "metadata": {
            "title": row["title"],
            "heading_path": list(row["heading_path"] or []),
            "source_urls": list(row["source_urls"] or []),
            "chunk_index": row["chunk_index"],
            "content_hash": row["content_hash"],
            "token_count": row["token_count"],
            "char_count": row["char_count"],
            "embedding_config": row["embedding_config"],
            "document_content_hash": row["document_content_hash"],
            "document_source_url": row["document_source_url"],
            "filename": row["filename"],
            "chunk_metadata": row["chunk_metadata"] or {},
            "record_version": RECORD_VERSION,
        },
    }


def export_jsonl(conn: psycopg.Connection, path: Path, *, embedding_model: str, dimensions: int,
                 embedding_config: str) -> dict[str, Any]:
    """Snapshot every chunk of every non-deleted document into ``path`` atomically."""
    counts = {"records": 0, "documents": 0}
    documents: set[str] = set()

    def writer(emit) -> None:
        with conn.cursor(name="jsonl_export") as cur:  # server-side cursor: bounded memory
            cur.itersize = 500
            cur.execute(
                """
                SELECT c.chunk_id, c.document_id, c.chunk_index, c.text, c.heading_path, c.source_urls,
                       c.metadata AS chunk_metadata, c.content_hash, c.token_count, c.char_count, c.embedding,
                       c.embedding_model, c.embedding_config, d.relative_path, d.filename, d.title,
                       d.content_hash AS document_content_hash, d.source_url AS document_source_url
                FROM chunks c JOIN documents d ON d.document_id = c.document_id
                WHERE d.status <> 'deleted'
                ORDER BY d.relative_path, c.chunk_index
                """
            )
            for row in cur:
                record = chunk_row_to_record(row)
                emit((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                counts["records"] += 1
                documents.add(row["document_id"])

    sha = _atomic_write_bytes(path, writer)
    counts["documents"] = len(documents)
    manifest = {
        "complete": True,
        "file": path.name,
        "sha256": sha,
        "records": counts["records"],
        "documents": counts["documents"],
        "embedding_model": embedding_model,
        "embedding_dimensions": dimensions,
        "embedding_config": embedding_config,
        "record_version": RECORD_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_bytes(
        manifest_path(path),
        lambda emit: emit(json.dumps(manifest, indent=2).encode("utf-8")),
    )
    return manifest


# --------------------------------------------------------------------------- validation
@dataclass
class JsonlValidation:
    path: str
    exists: bool = False
    records: int = 0
    documents: int = 0
    errors: list[str] = field(default_factory=list)
    manifest_ok: bool = False
    manifest: dict[str, Any] | None = None
    dimensions: set[int] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    configs: set[str] = field(default_factory=set)

    @property
    def valid(self) -> bool:
        return self.exists and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "valid": self.valid,
            "records": self.records,
            "documents": self.documents,
            "manifest_ok": self.manifest_ok,
            "dimensions": sorted(self.dimensions),
            "embedding_models": sorted(self.models),
            "embedding_configs": sorted(self.configs),
            "errors": self.errors[:50],
            "error_count": len(self.errors),
        }


def validate_record(record: Any, expected_dimensions: int | None = None) -> list[str]:
    problems: list[str] = []
    if not isinstance(record, dict):
        return ["record is not a JSON object"]
    for key in REQUIRED_FIELDS:
        if key not in record:
            problems.append(f"missing field '{key}'")
    if problems:
        return problems
    if not isinstance(record["text"], str) or not record["text"].strip():
        problems.append("text is empty")
    embedding = record["embedding"]
    if not isinstance(embedding, list) or not embedding:
        problems.append("embedding is not a non-empty list")
    else:
        if len(embedding) != record["embedding_dimensions"]:
            problems.append(
                f"embedding has {len(embedding)} values but embedding_dimensions={record['embedding_dimensions']}"
            )
        if expected_dimensions is not None and len(embedding) != expected_dimensions:
            problems.append(f"embedding has {len(embedding)} dimensions; database expects {expected_dimensions}")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in embedding):
            problems.append("embedding contains non-numeric or non-finite values")
    meta = record["metadata"]
    if not isinstance(meta, dict):
        problems.append("metadata is not an object")
    else:
        for key in REQUIRED_METADATA:
            if key not in meta:
                problems.append(f"metadata missing '{key}'")
    return problems


def iter_jsonl(path: Path) -> Iterator[tuple[int, Any, str | None]]:
    """Yield (line_number, record_or_None, parse_error_or_None)."""
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"line {line_no}: invalid JSON ({exc.msg})"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_jsonl(path: Path, expected_dimensions: int | None = None) -> JsonlValidation:
    result = JsonlValidation(path=str(path))
    if not path.exists():
        result.errors.append("file does not exist")
        return result
    result.exists = True
    seen_chunks: set[str] = set()
    documents: set[str] = set()
    positions: set[tuple[str, int]] = set()
    try:
        for line_no, record, error in iter_jsonl(path):
            if error:
                result.errors.append(error)
                continue
            problems = validate_record(record, expected_dimensions)
            if problems:
                result.errors.extend(f"line {line_no}: {p}" for p in problems)
                continue
            result.records += 1
            if record["chunk_id"] in seen_chunks:
                result.errors.append(f"line {line_no}: duplicate chunk_id {record['chunk_id']}")
            seen_chunks.add(record["chunk_id"])
            pos = (record["document_id"], record["metadata"]["chunk_index"])
            if pos in positions:
                result.errors.append(f"line {line_no}: duplicate chunk_index for document {pos[0]}")
            positions.add(pos)
            documents.add(record["document_id"])
            result.dimensions.add(record["embedding_dimensions"])
            result.models.add(record["embedding_model"])
            if record["metadata"].get("embedding_config"):
                result.configs.add(record["metadata"]["embedding_config"])
    except UnicodeDecodeError:
        result.errors.append("file is not valid UTF-8")
    result.documents = len(documents)
    if len(result.dimensions) > 1:
        result.errors.append(f"mixed embedding dimensions in file: {sorted(result.dimensions)}")

    mpath = manifest_path(path)
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
            result.manifest = manifest
            result.manifest_ok = bool(
                manifest.get("complete")
                and manifest.get("sha256") == file_sha256(path)
                and manifest.get("records") == result.records
            )
        except (json.JSONDecodeError, OSError):
            result.manifest_ok = False
    if not result.manifest_ok:
        result.errors.append(
            "manifest missing or does not match file contents — the file may be partial or edited; "
            "regenerate it with `python -m app.cli export-jsonl`"
        )
    return result


# --------------------------------------------------------------------------- loading
@dataclass
class LoadReport:
    documents_loaded: int = 0
    chunks_loaded: int = 0
    documents_pruned: int = 0
    cache_entries: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def load_jsonl(
    conn: psycopg.Connection,
    path: Path,
    *,
    embedding_model: str,
    dimensions: int,
    text_search_config: str,
    prune: bool = False,
    allow_unverified: bool = False,
) -> LoadReport:
    """Validate the whole file first, then load it. Each document is replaced in its
    own transaction, so re-running after a failure never creates duplicates."""
    validation = validate_jsonl(path, expected_dimensions=dimensions)
    blocking = [e for e in validation.errors if allow_unverified is False or not e.startswith("manifest")]
    if not validation.exists or blocking:
        raise ValueError("embedding.jsonl failed validation: " + "; ".join(validation.errors[:5]))
    if validation.models - {embedding_model}:
        raise ValueError(f"embedding.jsonl was produced by {sorted(validation.models)}, not {embedding_model}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, record, _ in iter_jsonl(path):
        grouped[record["document_id"]].append(record)

    report = LoadReport()
    for document_id, records in grouped.items():
        records.sort(key=lambda r: r["metadata"]["chunk_index"])
        first = records[0]
        meta = first["metadata"]
        config = meta.get("embedding_config") or f"{embedding_model}|dim={dimensions}"
        doc = repo.DocumentRecord(
            document_id=document_id,
            relative_path=first["source_file"],
            filename=meta.get("filename") or Path(first["source_file"]).name,
            title=meta["title"],
            content_hash=meta.get("document_content_hash") or "",
            source_url=meta.get("document_source_url"),
            size_bytes=0,
            encoding=None,
            file_modified_at=None,
            metadata={"loaded_from_jsonl": True},
        )
        chunks = [
            repo.ChunkRecord(
                chunk_id=r["chunk_id"],
                chunk_index=r["metadata"]["chunk_index"],
                text=r["text"],
                heading_path=list(r["metadata"]["heading_path"]),
                source_urls=list(r["metadata"]["source_urls"]),
                metadata=r["metadata"].get("chunk_metadata") or {},
                content_hash=r["metadata"]["content_hash"],
                token_count=int(r["metadata"].get("token_count") or max(1, len(r["text"]) // 4)),
                char_count=int(r["metadata"].get("char_count") or len(r["text"])),
                embedding=[float(v) for v in r["embedding"]],
                embedding_model=r["embedding_model"],
                embedding_config=r["metadata"].get("embedding_config") or config,
            )
            for r in records
        ]
        with conn.transaction():
            repo.replace_document(conn, doc, chunks, embedding_config=config,
                                  text_search_config=text_search_config)
            report.cache_entries += repo.put_cached_embeddings(
                conn,
                ((c.content_hash, c.embedding) for c in chunks),
                embedding_config=config,
                embedding_model=embedding_model,
                dimensions=dimensions,
            )
        report.documents_loaded += 1
        report.chunks_loaded += len(chunks)

    if prune:
        present = set(grouped)
        for document_id, row in repo.get_all_documents(conn).items():
            if document_id not in present and row["status"] != "deleted":
                with conn.transaction():
                    repo.mark_document_deleted(conn, document_id)
                report.documents_pruned += 1
    return report


def reconcile(conn: psycopg.Connection, path: Path, dimensions: int) -> dict[str, Any]:
    """Compare the JSONL snapshot with PostgreSQL without modifying either."""
    validation = validate_jsonl(path, expected_dimensions=dimensions)
    file_chunks: dict[str, str] = {}
    if validation.exists:
        for _, record, error in iter_jsonl(path):
            if error is None and isinstance(record, dict) and "chunk_id" in record:
                file_chunks[record["chunk_id"]] = record.get("metadata", {}).get("content_hash", "")
    rows = conn.execute(
        """
        SELECT c.chunk_id, c.content_hash FROM chunks c JOIN documents d ON d.document_id = c.document_id
        WHERE d.status <> 'deleted'
        """
    ).fetchall()
    db_chunks = {r["chunk_id"]: r["content_hash"] for r in rows}
    only_file = sorted(set(file_chunks) - set(db_chunks))
    only_db = sorted(set(db_chunks) - set(file_chunks))
    mismatched = sorted(k for k in set(file_chunks) & set(db_chunks) if file_chunks[k] != db_chunks[k])
    consistent = validation.valid and not only_file and not only_db and not mismatched
    if consistent:
        advice = "JSONL and PostgreSQL are consistent."
    elif only_db and not only_file:
        advice = "PostgreSQL is newer than the JSONL file: run `python -m app.cli export-jsonl`."
    elif only_file and not only_db:
        advice = "JSONL contains chunks missing from PostgreSQL: run `python -m app.cli load-jsonl`."
    else:
        advice = ("Both differ. PostgreSQL is authoritative: run `python -m app.cli export-jsonl`, or restore "
                  "from the file with `python -m app.cli load-jsonl --prune`.")
    return {
        "consistent": consistent,
        "jsonl": validation.as_dict(),
        "database_chunks": len(db_chunks),
        "jsonl_chunks": len(file_chunks),
        "only_in_jsonl": only_file[:20],
        "only_in_database": only_db[:20],
        "hash_mismatches": mismatched[:20],
        "counts": {"only_in_jsonl": len(only_file), "only_in_database": len(only_db),
                   "hash_mismatches": len(mismatched)},
        "advice": advice,
    }
