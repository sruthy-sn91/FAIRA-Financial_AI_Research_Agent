# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Compile C-extension packages (chromadb, tokenizers, sentence-transformers).
# build-essential is only needed here; the runtime image never sees it.
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into /deps so the whole tree can be copied to the runtime stage.
# --prefix separates installed packages from the system Python, letting us
# COPY only the packages without the build toolchain.
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Final image: no compiler, no build headers — just the app and its packages.
FROM python:3.12-slim AS runtime

WORKDIR /app

# libgomp1 : OpenMP runtime required by sentence-transformers at runtime
# curl     : used by Docker healthchecks (see docker-compose.yml)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages + console scripts (uvicorn, streamlit, mlflow…)
# /deps mirrors the /usr/local layout, so COPY merges them cleanly.
COPY --from=builder /deps /usr/local

# Copy application source (what's excluded lives in .dockerignore)
COPY agent/     ./agent/
COPY api/       ./api/
COPY ui/        ./ui/
COPY ingestion/ ./ingestion/

# Create mount points; Docker volumes will overlay these at runtime
RUN mkdir -p data/filings data/chroma mlflow_runs

# Single ENV layer — overridable at runtime via docker-compose environment:
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLM_PROVIDER=ollama \
    OLLAMA_MODEL=llama3.1:8b \
    CHROMA_PATH=/app/data/chroma \
    FILINGS_PATH=/app/data/filings \
    MLFLOW_TRACKING_URI=/app/mlflow_runs \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Document which ports each process binds to (informational; compose maps them)
EXPOSE 8000 8501
