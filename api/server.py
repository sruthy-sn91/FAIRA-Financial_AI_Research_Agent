"""
FastAPI Research Agent Server
==============================
Exposes the LangGraph agent as a REST API with SSE streaming.

Endpoints:
  GET  /health                    - server health check
  POST /research                  - synchronous (waits for full report)
  GET  /research/stream           - SSE streaming (events sent as agent progresses)

Run locally:
  uvicorn api.server:app --reload --port 8000
"""

import asyncio
import json
import logging
import os
import traceback
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()
log = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial AI Research Agent",
    description="Autonomous financial research with RAG over SEC filings",
    version="1.0.0",
)

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Analyze the credit risk of regional banks given the current rate environment."
            }
        }
    }


class ResearchResponse(BaseModel):
    query: str
    final_report: str
    retrieved_sources: list[str]
    iteration_count: int


# ── SSE event helpers ─────────────────────────────────────────────────────────

def make_event(event_type: str, **kwargs) -> dict:
    """
    Build a structured SSE event payload.

    Event types:
      status   - agent is doing something (fetching data, calling a tool)
      token    - a single token from the streaming synthesis LLM
      sources  - list of citations collected during research
      complete - final report is ready (also includes full report text)
      error    - something went wrong
    """
    return {"type": event_type, **kwargs}


# ── Core streaming generator ──────────────────────────────────────────────────

async def research_event_stream(query: str) -> AsyncIterator[dict]:
    """
    Async generator that runs the LangGraph agent and yields SSE events.

    LangGraph's graph.stream() is synchronous — we run it in a thread pool
    via run_in_executor() so it doesn't block the FastAPI event loop.
    Thread→async communication uses loop.call_soon_threadsafe so queue
    wakeups are delivered on the correct event loop without data races.

    Architecture:
      Thread pool  →  queue  →  async generator  →  SSE response  →  browser
    """
    from agent.graph import stream_research

    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()  # signals that the thread is done
    loop = asyncio.get_running_loop()

    def run_agent():
        """
        Runs in a background thread. Calls LangGraph synchronously,
        puts events into the queue as each node completes.
        """
        try:
            for update in stream_research(query):
                loop.call_soon_threadsafe(queue.put_nowait, update)

        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"__error__": str(e), "__traceback__": traceback.format_exc()},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    # Start the agent in a background thread
    loop.run_in_executor(None, run_agent)

    all_sources = []

    # Consume queue events and translate them to SSE events
    while True:
        update = await queue.get()

        # Thread finished
        if update is SENTINEL:
            break

        # Thread raised an exception
        if isinstance(update, dict) and "__error__" in update:
            yield make_event("error", message=update["__error__"])
            break

        # Process each node update
        for node_name, node_data in update.items():

            # ── agent node: LLM decided to call tool(s) ──────────────
            if node_name == "agent":
                messages = node_data.get("messages", [])
                for msg in messages:
                    tool_calls = getattr(msg, "tool_calls", [])
                    for tc in tool_calls:
                        tool_name = tc["name"]
                        tool_args = tc["args"]

                        # Produce a human-readable status message
                        status_text = _tool_call_to_status(tool_name, tool_args)
                        yield make_event("status", text=status_text, tool=tool_name)

            # ── tool node: tool returned results ──────────────────────
            elif node_name == "tools":
                messages = node_data.get("messages", [])
                sources = node_data.get("retrieved_sources", [])
                all_sources.extend(sources)

                for msg in messages:
                    tool_name = getattr(msg, "name", "tool")
                    # Send a brief preview of the tool result
                    preview = str(msg.content)[:200].replace("\n", " ")
                    yield make_event(
                        "tool_result",
                        tool=tool_name,
                        preview=preview,
                        sources=list(all_sources),
                    )

            # ── synthesize node: final report ready ───────────────────
            elif node_name == "synthesize":
                report = node_data.get("final_report", "")
                iteration_count = node_data.get("iteration_count", 0)

                if report:
                    # Stream the report token by token for a live-typing effect
                    words = report.split(" ")
                    for i, word in enumerate(words):
                        # Send each word (with trailing space) as a token event
                        token = word + (" " if i < len(words) - 1 else "")
                        yield make_event("token", text=token)
                        # Small delay creates the typewriter effect in the browser
                        await asyncio.sleep(0.01)

                    # Send the completion event with the full report
                    yield make_event(
                        "complete",
                        report=report,
                        sources=list(all_sources),
                        iteration_count=iteration_count,
                    )


def _tool_call_to_status(tool_name: str, tool_args: dict) -> str:
    """Convert a tool call into a readable status message for the UI."""
    if tool_name == "search_sec_filings":
        query = tool_args.get("query", "")
        ticker = tool_args.get("ticker", "")
        scope = f" ({ticker})" if ticker else " (all companies)"
        return f"Searching SEC filings{scope}: \"{query[:60]}\""

    elif tool_name == "get_fred_data":
        series = tool_args.get("series_id", "")
        return f"Fetching FRED data: {series}"

    elif tool_name == "get_stock_data":
        ticker = tool_args.get("ticker", "")
        return f"Fetching market data for {ticker}"

    elif tool_name == "compare_stocks":
        tickers = tool_args.get("tickers", [])
        return f"Comparing stocks: {', '.join(tickers)}"

    elif tool_name == "run_python_calculation":
        first_line = str(tool_args.get("code", "")).split("\n")[0][:60]
        return f"Running calculation: {first_line}"

    return f"Calling tool: {tool_name}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple health check. Used by nginx, Docker, and the CI/CD pipeline."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    return {
        "status": "ok",
        "llm_provider": provider,
        "version": "1.0.0",
    }


@app.get("/research/stream")
async def research_stream(
    query: str = Query(..., description="The financial research question to investigate"),
):
    """
    Stream research agent output as Server-Sent Events.

    The client receives a stream of JSON events:
      {"type": "status",      "text": "Searching SEC filings..."}
      {"type": "tool_result", "tool": "search_sec_filings", "preview": "..."}
      {"type": "token",       "text": "The "}
      {"type": "token",       "text": "credit "}
      ...
      {"type": "complete",    "report": "...", "sources": [...]}

    Connect with EventSource in JavaScript or httpx in Python.
    """
    if not query.strip():
        return JSONResponse({"error": "query cannot be empty"}, status_code=400)

    log.info(f"[/research/stream] query='{query[:80]}...'")

    async def event_generator():
        async for event in research_event_stream(query):
            # SSE format: each event is a JSON string
            yield {"data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@app.post("/research", response_model=ResearchResponse)
async def research_sync(request: ResearchRequest):
    """
    Synchronous research endpoint. Waits for the full report before responding.
    Useful for programmatic API clients that don't need streaming.
    """
    if not request.query.strip():
        return JSONResponse({"error": "query cannot be empty"}, status_code=400)

    log.info(f"[/research] query='{request.query[:80]}...'")

    from agent.graph import run_research

    # run_research is synchronous — run it in a thread so we don't block
    result = await asyncio.to_thread(run_research, request.query)

    return ResearchResponse(
        query=request.query,
        final_report=result["final_report"],
        retrieved_sources=result["retrieved_sources"],
        iteration_count=result["iteration_count"],
    )
