# Financial AI Research Agent

An autonomous multi-tool financial research agent built with LangGraph, RAG over SEC filings, and real-time streaming output.

## What It Does

Submit a research question → the agent autonomously retrieves SEC filings, macroeconomic data, and price history → produces a structured, cited research report streamed in real time.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent | LangGraph + LangChain |
| LLM (local) | Ollama (Llama 3.1 8B) |
| LLM (production) | Groq (Llama 3.3 70B) |
| Vector Store | ChromaDB + sentence-transformers |
| Data Sources | SEC EDGAR, FRED, yfinance |
| Code Execution | e2b sandbox |
| API | FastAPI + SSE |
| UI | Streamlit |
| Observability | MLflow |
| Deployment | AWS EC2 + nginx + Docker |

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd Financial_AI_Research_Agent

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Install and start Ollama (local dev)
# Download from https://ollama.com
ollama pull llama3.1:8b

# 6. Run the full stack
docker compose up
```

## Project Structure

```
agent/       # LangGraph agent logic
ingestion/   # SEC filing downloader + ChromaDB embedder
api/         # FastAPI streaming backend
ui/          # Streamlit research interface
tests/       # Unit tests
data/        # Local data (git-ignored)
```
