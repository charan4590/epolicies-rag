"""Protected ingestion endpoints. Runs execute in a background thread; their
status is persisted in PostgreSQL (``ingestion_runs`` / ``ingestion_events``).
Request bodies cannot choose filesystem paths — only the configured
MARKDOWN_DIR is ever read."""

from __future__ import annotations

import logging
import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import require_admin, settings_dep
from app.config import Settings
from app.database import repository as repo
from app.database.connection import connection
from app.ingestion.embedder import document_config
from app.ingestion.markdown_loader import SourceDirectoryError
from app.ingestion.pipeline import IngestionPipeline, build_plan
from app.schemas.api_models import IngestAccepted, IngestionRun, IngestRequest, SourceStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])
_start_lock = threading.Lock()


def _public_report(report: dict) -> dict:
    """Drop absolute filesystem paths from reports returned over HTTP."""
    cleaned = {k: v for k, v in report.items() if k != "source_dir"}
    if isinstance(cleaned.get("jsonl"), dict):
        cleaned["jsonl"] = {k: v for k, v in cleaned["jsonl"].items() if k != "path"}
    return cleaned


@router.get("/source-status", response_model=SourceStatus)
def source_status(settings: Settings = Depends(settings_dep)) -> SourceStatus:
    """Compare files in MARKDOWN_DIR with the index without processing anything."""
    try:
        with connection() as conn:
            plan = build_plan(settings, conn)
            conn.rollback()
    except SourceDirectoryError:
        return SourceStatus(source_dir_configured=True, source_dir_exists=False, discovered=0, new=[], modified=[],
                            unchanged=0, deleted=[], unreadable=[],
                            error="The configured Markdown directory does not exist on the server.")
    status = plan.as_status()
    return SourceStatus(source_dir_configured=True, source_dir_exists=True, discovered=status["discovered"],
                        new=status["new"], modified=status["modified"], unchanged=status["unchanged"],
                        deleted=status["deleted"], unreadable=status["unreadable"])


@router.post("", response_model=IngestAccepted, status_code=202, dependencies=[Depends(require_admin)])
def start_ingestion(body: IngestRequest, settings: Settings = Depends(settings_dep)) -> IngestAccepted:
    if body.dry_run:
        with connection() as conn:
            plan = build_plan(settings, conn, force=body.mode == "force")
            conn.rollback()
        status = plan.as_status()
        return IngestAccepted(run_id=None, status="dry_run",
                              message=f"{len(status['new'])} new, {len(status['modified'])} modified, "
                                      f"{status['unchanged']} unchanged, {len(status['deleted'])} deleted")
    with _start_lock:
        with connection() as conn:
            if repo.ingestion_lock_held(conn):
                raise HTTPException(status_code=409, detail="An ingestion run is already in progress.")
            run_id = repo.create_run(conn, mode=body.mode, triggered_by="api",
                                     embedding_config=document_config(settings).config_id)

    def worker(run_id: uuid.UUID) -> None:
        try:
            IngestionPipeline(settings).run(mode=body.mode, retry_failed_only=body.retry_failed_only,
                                            limit=body.limit, triggered_by="api", run_id=run_id)
        except Exception as exc:  # noqa: BLE001 - status is persisted for the caller to poll
            logger.exception("api_ingestion_failed")
            with connection() as conn:
                repo.update_run(conn, run_id, report={}, status="failed", error=str(exc)[:1000])

    threading.Thread(target=worker, args=(run_id,), name=f"ingest-{run_id}", daemon=True).start()
    return IngestAccepted(run_id=run_id, status="running", message="Ingestion started. Poll /ingest/runs/{run_id}.")


@router.get("/runs", response_model=list[IngestionRun], dependencies=[Depends(require_admin)])
def list_runs(limit: int = 20) -> list[IngestionRun]:
    with connection() as conn:
        rows = repo.list_runs(conn, max(1, min(limit, 100)))
    return [IngestionRun(**{**r, "report": _public_report(r["report"] or {})}) for r in rows]


@router.get("/runs/{run_id}", response_model=IngestionRun, dependencies=[Depends(require_admin)])
def get_run(run_id: uuid.UUID) -> IngestionRun:
    with connection() as conn:
        run = repo.get_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return IngestionRun(**{**run, "report": _public_report(run["report"] or {})})
