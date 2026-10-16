# ePolicies RAG — convenience targets. Run from epolicies-rag/.
PY := backend/.venv/bin/python
CLI := cd backend && .venv/bin/python -m app.cli

.PHONY: help setup-backend setup-frontend db-init discover ingest ingest-dry ingest-force resume rebuild status \
        export-jsonl validate-jsonl load-jsonl reconcile api web test test-unit test-integration \
        frontend-check frontend-build eval eval-answers db-up db-down

help:
	@grep -E '^[a-z-]+:' Makefile | sed 's/:.*//' | sort | tr '\n' ' '; echo

setup-backend:
	python3.12 -m venv backend/.venv && backend/.venv/bin/pip install --upgrade pip && backend/.venv/bin/pip install -r backend/requirements-dev.txt

setup-frontend:
	cd frontend && npm ci

db-up:            ## optional: PostgreSQL 17 + pgvector via Docker Compose
	docker compose up -d db

db-down:
	docker compose down

db-init:
	$(CLI) init-db

discover:
	$(CLI) discover --list

ingest:
	$(CLI) ingest

ingest-dry:
	$(CLI) ingest --dry-run

ingest-force:
	$(CLI) ingest --force

resume:
	$(CLI) resume

rebuild:          ## DESTRUCTIVE: requires CONFIRM=1
	@test "$(CONFIRM)" = "1" || (echo "Refusing: run 'make rebuild CONFIRM=1' (drops chunks and cached embeddings)"; exit 1)
	$(CLI) rebuild --confirm

status:
	$(CLI) status

export-jsonl:
	$(CLI) export-jsonl

validate-jsonl:
	$(CLI) validate-jsonl

load-jsonl:
	$(CLI) load-jsonl

reconcile:
	$(CLI) reconcile

api:
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

web:
	cd frontend && npm run dev

test: test-unit test-integration

test-unit:
	cd backend && .venv/bin/python -m pytest tests/unit tests/api

test-integration:   ## needs TEST_DATABASE_URL pointing at a dedicated *test* database
	@test -n "$(TEST_DATABASE_URL)" || (echo "Set TEST_DATABASE_URL=postgresql://.../epolicies_test"; exit 1)
	cd backend && TEST_DATABASE_URL=$(TEST_DATABASE_URL) .venv/bin/python -m pytest

frontend-check:
	cd frontend && npm run check

frontend-build:
	cd frontend && npm run build

eval:
	$(CLI) evaluate --dataset eval/epolicies_eval.jsonl

eval-answers:
	$(CLI) evaluate --dataset eval/epolicies_eval.jsonl --with-answers
