"""
LangGraph Node Functions
========================
Each function here is a NODE in the agent graph.
A node receives the current state, does something, and returns a partial state update.

LangGraph merges the returned dict back into the state automatically.
Nodes never see the full graph — they only see state in, state out.
"""

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.prompts import (
    ASSESS_COMPLETENESS_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
)
from agent.state import ResearchState
from agent.tools import ALL_TOOLS

load_dotenv()
log = logging.getLogger(__name__)

MAX_ITERATIONS = 5  # Safety cap: agent won't loop more than this many times


# ── LLM Factory ───────────────────────────────────────────────────────────────

def get_llm(streaming: bool = False):
    """
    Return the configured LLM based on LLM_PROVIDER env var.

    LLM_PROVIDER=ollama  → local Ollama (for development)
    LLM_PROVIDER=groq    → Groq cloud API (for production/deployment)

    The rest of the agent code doesn't care which LLM is used —
    both implement the same LangChain interface.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            streaming=streaming,
            temperature=0.1,   # low temperature = more deterministic for research
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            streaming=streaming,
            temperature=0.1,
        )


# ── Node 1: Agent (LLM decides what to do next) ───────────────────────────────

def agent_node(state: ResearchState) -> dict:
    """
    The core reasoning node. The LLM sees the conversation history and
    decides: call a tool OR write the final answer.

    This node is called repeatedly in the loop. Each time, the LLM sees
    all previous tool results and decides what to do next.
    """
    log.info(f"[agent_node] iteration={state['iteration_count']}, "
             f"messages={len(state['messages'])}")

    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Build the full message list: system prompt + conversation history
    messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT)] + state["messages"]

    response = llm_with_tools.invoke(messages)

    log.info(f"[agent_node] response has {len(getattr(response, 'tool_calls', []))} tool calls")

    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1,
    }


# ── Node 2: Tool Executor ─────────────────────────────────────────────────────

def tool_node(state: ResearchState) -> dict:
    """
    Executes whatever tool the LLM called in the previous agent_node.

    How tool-calling works in LangChain:
    1. agent_node returns an AIMessage with tool_calls=[{name, args}]
    2. tool_node reads those tool_calls, runs the actual Python function
    3. Returns ToolMessage(s) with the results back into state
    4. Next agent_node invocation sees those results and reasons further
    """
    from langchain_core.messages import ToolMessage

    # Build a lookup: tool name → tool function
    tool_map = {t.name: t for t in ALL_TOOLS}

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    tool_messages = []
    new_sources = []

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc["id"]

        log.info(f"[tool_node] calling {tool_name}({tool_args})")

        if tool_name not in tool_map:
            result = f"Error: tool '{tool_name}' not found"
        else:
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Tool error: {e}"
                log.error(f"[tool_node] {tool_name} failed: {e}")

        # Track source citations for the final report
        if tool_name == "search_sec_filings":
            new_sources.append(f"SEC: {tool_args.get('query', '')[:50]}")
        elif tool_name == "get_fred_data":
            new_sources.append(f"FRED: {tool_args.get('series_id', '')}")
        elif tool_name in ("get_stock_data", "compare_stocks"):
            tickers = tool_args.get("ticker") or str(tool_args.get("tickers", ""))
            new_sources.append(f"Market: {tickers}")

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    return {
        "messages": tool_messages,
        "retrieved_sources": state["retrieved_sources"] + new_sources,
    }


# ── Node 3: Synthesize Final Report ──────────────────────────────────────────

def synthesize_node(state: ResearchState) -> dict:
    """
    Called when the agent has enough evidence. Writes the final research report.

    Uses a separate LLM call with a synthesis-focused prompt so the output
    is structured and complete rather than conversational.
    """
    log.info("[synthesize_node] writing final report")

    llm = get_llm(streaming=False)

    # Build full context: system + history + synthesis instruction
    messages = (
        [SystemMessage(content=RESEARCH_SYSTEM_PROMPT)]
        + state["messages"]
        + [HumanMessage(content=SYNTHESIS_PROMPT)]
    )

    response = llm.invoke(messages)
    report = response.content

    log.info(f"[synthesize_node] report length: {len(report)} chars")

    return {
        "final_report": report,
        "is_complete": True,
        "messages": [AIMessage(content=report)],
    }


# ── Routing Functions (Conditional Edges) ────────────────────────────────────

def should_use_tools(state: ResearchState) -> Literal["tools", "synthesize"]:
    """
    After agent_node: did the LLM call a tool, or is it done?

    If the LLM returned tool_calls → go execute them
    If no tool_calls (or max iterations hit) → go synthesize the report
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if state["iteration_count"] >= MAX_ITERATIONS:
        log.info("[router] max iterations reached → synthesize")
        return "synthesize"

    if tool_calls:
        log.info(f"[router] {len(tool_calls)} tool call(s) → tools")
        return "tools"

    log.info("[router] no tool calls → synthesize")
    return "synthesize"


def should_continue_research(state: ResearchState) -> Literal["agent", "synthesize"]:
    """
    After tool_node: does the agent need more data, or is it ready to write?

    Strategy: after each round of tool calls, check if we have enough
    sources. If we have 3+ different source types, proceed to synthesis.
    Otherwise, loop back to agent_node for another retrieval round.
    """
    sources = state["retrieved_sources"]
    iterations = state["iteration_count"]

    # Count how many different source types we have
    has_sec = any("SEC:" in s for s in sources)
    has_fred = any("FRED:" in s for s in sources)
    has_market = any("Market:" in s for s in sources)
    source_diversity = sum([has_sec, has_fred, has_market])

    log.info(f"[router] sources={len(sources)}, diversity={source_diversity}, "
             f"iterations={iterations}")

    # Ready to synthesize if: diverse sources OR approaching max iterations
    if source_diversity >= 2 and len(sources) >= 3:
        log.info("[router] sufficient evidence → synthesize")
        return "synthesize"
    elif iterations >= MAX_ITERATIONS - 1:
        log.info("[router] approaching max iterations → synthesize")
        return "synthesize"
    else:
        log.info("[router] need more data → agent")
        return "agent"
