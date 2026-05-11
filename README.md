# FAIRA — Financial AI Research Agent

**F**inancial **AI** **R**esearch **A**gent — an autonomous research system that answers complex financial questions by reasoning over SEC filings, Federal Reserve data, and live market data in real time.

Powered by **LangGraph** · **RAG over SEC 10-K filings** · **Real-time SSE streaming** · **Docker Compose**

---

## How It Works

```
User Question
     │
     ▼
┌─────────────┐     tool calls      ┌──────────────────────────────────┐
│  LLM Agent  │ ──────────────────► │  Tools                           │
│ (LangGraph) │                     │  ├─ search_sec_filings  (RAG)    │
│             │ ◄────────────────── │  ├─ get_fred_data       (FRED)   │
│             │    tool results      │  ├─ get_stock_data      (Yahoo)  │
└─────────────┘                     │  ├─ compare_stocks      (Yahoo)  │
     │                              │  └─ run_python_calc     (e2b)    │
     │  enough evidence?            └──────────────────────────────────┘
     ▼
┌─────────────┐
│  Synthesize │  →  Structured markdown report (600–1000 words)
└─────────────┘         with inline citations and quantitative analysis
     │
     ▼ SSE stream (token by token)
┌─────────────┐
│ Streamlit UI│  →  Live typewriter rendering + agent activity feed
└─────────────┘
```

The agent loops — retrieving data, assessing completeness, and retrieving more — until it has filing citations, macro data, and market data. It then synthesises an institutional-quality research report.

---

## Features

- **Autonomous multi-step research** — LangGraph graph with conditional edges; agent decides when it has enough evidence
- **RAG over SEC 10-K filings** — ChromaDB vector store with `sentence-transformers` embeddings; search by company or across the full corpus
- **Five specialised tools** — SEC EDGAR, FRED macroeconomic data, Yahoo Finance quotes, multi-stock peer comparison, and sandboxed Python calculations via e2b
- **Real-time SSE streaming** — FastAPI streams agent activity and the final report token-by-token to the browser
- **Two LLM backends** — Ollama (local, free) or Groq cloud API (fast, no GPU required)
- **MLflow experiment tracking** — every research run logged with query, sources, and report length
- **Full Docker orchestration** — multi-stage image, health-checked startup chain, nginx reverse proxy for production

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph + LangChain |
| LLM (local) | Ollama — `llama3.1:8b` |
| LLM (cloud) | Groq — `llama-3.3-70b-versatile` |
| Vector store | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| SEC filings | `sec-edgar-downloader` |
| Macro data | FRED API (`fredapi`) |
| Market data | `yfinance` |
| Code execution | e2b sandboxed Python |
| API server | FastAPI + `sse-starlette` |
| Frontend | Streamlit |
| Experiment tracking | MLflow |
| Containerisation | Docker + Docker Compose |
| Reverse proxy | nginx |
| CI/CD | GitHub Actions |

---

## Project Structure

```
FAIRA-Financial_AI_Research_Agent/
│
├── agent/
│   ├── graph.py          # LangGraph graph: nodes, edges, and conditional routers
│   ├── nodes.py          # Node functions: agent_node, tool_node, synthesize_node
│   ├── prompts.py        # All LLM prompt templates (separated from logic)
│   ├── state.py          # ResearchState TypedDict (shared graph memory)
│   └── tools.py          # Five LangChain tools the agent can call
│
├── api/
│   └── server.py         # FastAPI: /health, /research, /research/stream (SSE)
│
├── ui/
│   └── app.py            # Streamlit frontend with live streaming and FAIRA animation
│
├── ingestion/
│   ├── sec_downloader.py # Downloads 10-K filings from SEC EDGAR
│   └── embedder.py       # Chunks filings, embeds them, loads into ChromaDB
│
├── tests/
│   └── test_tools.py     # Unit tests for all five tools (mocked, no live API calls)
│
├── nginx/
│   └── docker.conf       # nginx reverse proxy config for Docker production
│
├── scripts/
│   └── deploy.sh         # SSH deployment script (used by GitHub Actions)
│
├── .github/workflows/
│   └── deploy.yml        # CI: run tests → deploy to EC2 on push to main
│
├── Dockerfile            # Multi-stage build: builder (compile) + runtime (lean)
├── docker-compose.yml    # Orchestrates mlflow → api → ui → nginx
├── Makefile              # Dev and Docker convenience commands
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variable template
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.12 recommended |
| Docker Desktop | Required for the Docker setup |
| Ollama | For local LLM — [ollama.com](https://ollama.com) |
| FRED API key | Free at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) |
| e2b API key | Free tier at [e2b.dev](https://e2b.dev) |
| Groq API key | Optional, for cloud LLM — [console.groq.com](https://console.groq.com) |

---

## Option A — Local Development (no Docker)

### 1. Clone and install

```bash
git clone https://github.com/sruthy-sn91/FAIRA-Financial_AI_Research_Agent.git
cd FAIRA-Financial_AI_Research_Agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
FRED_API_KEY=your_key_here
E2B_API_KEY=your_key_here

# Only needed when LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

### 3. Start Ollama and pull the model

```bash
ollama pull llama3.1:8b
```

Ollama starts automatically on system boot. If it isn't running, open the Ollama desktop app or run `ollama serve` in a separate terminal.

### 4. Ingest SEC filings

Downloads 10-K filings for five regional banks (KeyCorp, Regions Financial, Fifth Third, Huntington, Citizens Financial) and builds the ChromaDB vector index.

```bash
python -m ingestion.sec_downloader
python -m ingestion.embedder
```

Takes 5–10 minutes on first run. Results are cached in `data/` — you only need to run this once.

### 5. Start the API server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the Streamlit UI (new terminal)

```bash
streamlit run ui/app.py
```

Open **http://localhost:8501**.

---

## Option B — Docker (Recommended)

### 1. Clone and configure

```bash
git clone https://github.com/sruthy-sn91/FAIRA-Financial_AI_Research_Agent.git
cd FAIRA-Financial_AI_Research_Agent
cp .env.example .env
# Edit .env with your API keys
```

### 2. Build and start all services

```bash
docker compose up --build
```

> **First build takes 15–20 minutes** — PyTorch (~530 MB) and other C-extension packages must compile. Subsequent builds use Docker's layer cache and complete in seconds.

Once running:

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI + Swagger docs | http://localhost:8000/docs |
| MLflow tracking | http://localhost:5000 |

### 3. Ingest SEC filings inside Docker

```bash
docker compose run --rm api python -m ingestion.sec_downloader
docker compose run --rm api python -m ingestion.embedder
```

### 4. Production mode — nginx on port 80

```bash
docker compose --profile prod up --build
```

Everything is served through **http://localhost**:

| Route | Destination |
|---|---|
| `/` | Streamlit UI |
| `/api/` | FastAPI backend |
| `/mlflow/` | MLflow tracking UI |

---

## Switching LLM Provider

No code changes needed — set `LLM_PROVIDER` in `.env`:

**Ollama (local, default)** — free, requires ~5 GB RAM:
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

**Groq (cloud, faster responses)** — free tier available, no GPU required:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok", "llm_provider": "..."}` |
| `POST` | `/research` | Synchronous — waits for full report before responding |
| `GET` | `/research/stream?query=...` | SSE streaming — streams activity and report tokens live |
| `GET` | `/docs` | Swagger interactive API documentation |

### Streaming example

```bash
curl -N "http://localhost:8000/research/stream?query=Analyze+the+credit+risk+of+KeyCorp"
```

Each SSE event is a JSON payload:

```jsonc
{"type": "status",      "text": "Searching SEC filings (KEY): net interest margin"}
{"type": "tool_result", "tool": "search_sec_filings", "preview": "..."}
{"type": "token",       "text": "The "}
{"type": "complete",    "report": "...", "sources": [...], "iteration_count": 3}
```

### Synchronous example

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare net interest margin across KeyCorp and Regions Financial"}'
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests mock external API calls — no live FRED, Yahoo Finance, or e2b connections required. Tests pass without any API keys.

```
tests/test_tools.py::TestFredTool::test_returns_formatted_data         PASSED
tests/test_tools.py::TestFredTool::test_missing_api_key_returns_error   PASSED
tests/test_tools.py::TestFredTool::test_invalid_series_returns_error    PASSED
tests/test_tools.py::TestYfinanceTool::test_returns_price_and_metrics   PASSED
tests/test_tools.py::TestYfinanceTool::test_handles_invalid_ticker      PASSED
tests/test_tools.py::TestYfinanceTool::test_compare_stocks_returns_table PASSED
tests/test_tools.py::TestSecFilingsTool::test_returns_formatted_passages PASSED
tests/test_tools.py::TestSecFilingsTool::test_no_results_returns_graceful_message PASSED
tests/test_tools.py::TestE2bTool::test_basic_calculation_output         PASSED
tests/test_tools.py::TestE2bTool::test_financial_ratio_calculation      PASSED
tests/test_tools.py::TestE2bTool::test_execution_error_is_captured      PASSED
tests/test_tools.py::TestChunking::test_chunks_cover_full_text          PASSED
tests/test_tools.py::TestChunking::test_overlap_creates_shared_content  PASSED
tests/test_tools.py::TestChunking::test_empty_text_yields_nothing       PASSED
```

---

## Makefile Reference

```bash
# Setup
make install          # pip install -r requirements.txt
make ingest           # download filings + build ChromaDB index

# Local development
make api              # FastAPI on :8000 (with --reload)
make ui               # Streamlit on :8501
make dev              # both in parallel

# Docker — development
make docker-build     # build the app image
make docker-up        # start mlflow + api + ui (direct ports)
make docker-down      # stop all containers
make docker-restart   # rebuild and restart

# Docker — production
make docker-up-prod   # add nginx reverse proxy on :80

# Operations
make docker-ps            # container health status
make docker-logs          # tail all service logs
make docker-logs-api      # tail API logs only
make docker-logs-ui       # tail UI logs only
make docker-shell-api     # bash shell inside api container
make docker-shell-ui      # bash shell inside ui container
make docker-ingest        # run ingestion pipeline inside Docker

# Tests
make test             # pytest tests/ -v
```

---

## CI/CD Pipeline

GitHub Actions runs automatically on every push to `main`:

1. **Test** — installs dependencies, runs `pytest tests/ -v` with dummy API keys
2. **Deploy** — SSH into EC2, pull latest code, rebuild changed containers, health check

The deploy step only runs on direct pushes to `main` (not pull requests) and only after tests pass.

### Required GitHub Secrets

Add these under `Settings → Secrets and variables → Actions`:

| Secret | Value |
|---|---|
| `EC2_HOST` | Public IP or hostname of your EC2 instance |
| `EC2_USER` | SSH username (e.g. `ubuntu`) |
| `EC2_SSH_KEY` | Contents of your `.pem` private key file |
| `GROQ_API_KEY` | Groq API key (used in production) |
| `FRED_API_KEY` | FRED API key |
| `E2B_API_KEY` | e2b API key |

---

## Example Research Questions

- *Analyze the credit risk of regional banks given the current interest rate environment*
- *Compare net interest margin trends across KeyCorp, Regions Financial, and Fifth Third*
- *What are the key risk factors disclosed by Huntington Bancshares in their latest 10-K?*
- *How has the Fed Funds rate affected regional bank profitability since 2022?*
- *Analyze loan loss provisions and credit quality at Citizens Financial Group*

---

## License

MIT
