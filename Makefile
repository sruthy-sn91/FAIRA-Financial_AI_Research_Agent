##############################################################################
# Financial AI Research Agent — Makefile
#
# Usage: make <target>
# On Windows, run these commands in Git Bash, WSL, or PowerShell where noted.
##############################################################################

.PHONY: help install ingest api ui dev \
        docker-build docker-up docker-up-prod docker-down docker-restart \
        docker-logs docker-logs-api docker-logs-ui docker-logs-mlflow \
        docker-shell-api docker-shell-ui docker-ingest docker-ps \
        test clean

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Financial AI Research Agent"
	@echo ""
	@echo "  Setup:"
	@echo "    make install         Install Python dependencies into active venv"
	@echo "    make ingest          Download SEC filings and build ChromaDB index"
	@echo ""
	@echo "  Local dev (no Docker):"
	@echo "    make api             FastAPI backend on :8000 (with --reload)"
	@echo "    make ui              Streamlit UI on :8501"
	@echo "    make dev             Both api and ui in parallel"
	@echo ""
	@echo "  Docker — development:"
	@echo "    make docker-build    Build the app image (api + ui share one image)"
	@echo "    make docker-up       Start mlflow + api + ui  (direct ports)"
	@echo "    make docker-down     Stop and remove containers"
	@echo "    make docker-restart  Rebuild image then restart all services"
	@echo ""
	@echo "  Docker — production:"
	@echo "    make docker-up-prod  Add nginx reverse proxy on :80 (profile: prod)"
	@echo ""
	@echo "  Docker — operations:"
	@echo "    make docker-ps          List running containers and health status"
	@echo "    make docker-logs        Tail logs for all services"
	@echo "    make docker-logs-api    Tail API logs only"
	@echo "    make docker-logs-ui     Tail UI logs only"
	@echo "    make docker-logs-mlflow Tail MLflow logs only"
	@echo "    make docker-shell-api   Open bash shell inside api container"
	@echo "    make docker-shell-ui    Open bash shell inside ui container"
	@echo "    make docker-ingest      Run SEC ingestion pipeline inside Docker"
	@echo ""
	@echo "  Other:"
	@echo "    make test            Run test suite with pytest"
	@echo "    make clean           Remove Python cache files"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

ingest:
	python -m ingestion.sec_downloader
	python -m ingestion.embedder

# ── Local development ─────────────────────────────────────────────────────────
api:
	uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/app.py --server.port 8501

dev:
	@echo "Starting API on :8000 and UI on :8501 in parallel..."
	uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload &
	streamlit run ui/app.py --server.port 8501

# ── Docker: build ─────────────────────────────────────────────────────────────
docker-build:
	docker compose build

# ── Docker: start / stop ──────────────────────────────────────────────────────
docker-up:
	docker compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  MLflow  → http://localhost:5000"
	@echo "  API     → http://localhost:8000"
	@echo "  UI      → http://localhost:8501"

docker-up-prod:
	docker compose --profile prod up -d
	@echo ""
	@echo "Services started (prod mode):"
	@echo "  Nginx   → http://localhost  (routes to api + ui + mlflow)"
	@echo "  MLflow  → http://localhost/mlflow"
	@echo "  API     → http://localhost/api"

docker-down:
	docker compose --profile prod down

docker-restart:
	docker compose --profile prod down
	docker compose build --no-cache
	docker compose up -d

# ── Docker: operations ────────────────────────────────────────────────────────
docker-ps:
	docker compose ps

docker-logs:
	docker compose logs -f

docker-logs-api:
	docker compose logs -f api

docker-logs-ui:
	docker compose logs -f ui

docker-logs-mlflow:
	docker compose logs -f mlflow

docker-shell-api:
	docker compose exec api bash

docker-shell-ui:
	docker compose exec ui bash

# Run the ingestion pipeline inside a fresh container (shares the ./data volume)
docker-ingest:
	docker compose run --rm api python -m ingestion.sec_downloader
	docker compose run --rm api python -m ingestion.embedder

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
