"""
LangGraph Research Agent
========================
Wires nodes and edges into a compiled, runnable graph.

Graph topology:

    START
      │
      ▼
  [agent_node] ──── should_use_tools() ────►  [synthesize_node] ──► END
      ▲
      │          └──► [tool_node] ──── should_continue_research() ──┘
      │                                         │
      └─────────────────────────────────────────┘ (loop back)

The agent loop:
  1. agent_node: LLM sees history, decides which tool to call
  2. tool_node: executes the tool, appends result to messages
  3. Router checks: enough sources? → synthesize; else → back to agent
  4. synthesize_node: writes the final report from all gathered evidence
"""

import logging
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes import agent_node, should_continue_research, should_use_tools, synthesize_node, tool_node
from agent.state import ResearchState

log = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Construct and compile the research agent graph.

    Returns a compiled LangGraph that can be invoked with:
        graph.invoke({"query": "...", "messages": [HumanMessage(...)]})
    """
    # ── Define the graph ──────────────────────────────────────────────
    graph = StateGraph(ResearchState)

    # ── Add nodes ─────────────────────────────────────────────────────
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("synthesize", synthesize_node)

    # ── Add edges ─────────────────────────────────────────────────────

    # Entry point: always start at the agent node
    graph.add_edge(START, "agent")

    # After agent: conditional — did it call tools or is it done?
    graph.add_conditional_edges(
        "agent",
        should_use_tools,
        {
            "tools": "tools",
            "synthesize": "synthesize",
        },
    )

    # After tools: conditional — enough evidence or need more?
    graph.add_conditional_edges(
        "tools",
        should_continue_research,
        {
            "agent": "agent",       # loop: retrieve more
            "synthesize": "synthesize",  # done: write report
        },
    )

    # After synthesis: always end
    graph.add_edge("synthesize", END)

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────────────────

def run_research(query: str) -> dict:
    """
    Run the research agent synchronously. Waits for the full report.

    Returns:
        dict with keys: final_report, retrieved_sources, iteration_count
    """
    graph = build_graph()
    initial_state = _make_initial_state(query)
    final_state = graph.invoke(initial_state)
    return {
        "final_report": final_state.get("final_report", ""),
        "retrieved_sources": final_state.get("retrieved_sources", []),
        "iteration_count": final_state.get("iteration_count", 0),
    }


def stream_research(query: str):
    """
    Run the research agent in streaming mode.

    Yields:
        dicts of the form {"node_name": {partial state changes}}
        one dict per node execution
    """
    graph = build_graph()
    initial_state = _make_initial_state(query)
    for update in graph.stream(initial_state, stream_mode="updates"):
        yield update


def _make_initial_state(query: str) -> dict:
    return {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "retrieved_sources": [],
        "iteration_count": 0,
        "final_report": "",
        "is_complete": False,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Analyze the credit risk of regional banks given the current interest rate environment. "
        "Focus on KeyCorp, Regions Financial, and Fifth Third Bancorp."
    )

    print(f"\nResearch Query: {query}")
    print("=" * 60)
    print("Running agent (this may take 1-3 minutes)...\n")

    for update in stream_research(query):
        node_name = list(update.keys())[0]
        node_data = update[node_name]

        if node_name == "agent":
            msgs = node_data.get("messages", [])
            for msg in msgs:
                tool_calls = getattr(msg, "tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        print(f"[{node_name}] >> calling {tc['name']}({list(tc['args'].keys())})")

        elif node_name == "tools":
            msgs = node_data.get("messages", [])
            for msg in msgs:
                preview = str(msg.content)[:120].replace("\n", " ")
                print(f"[{node_name}] << {msg.name}: {preview}...")

        elif node_name == "synthesize":
            report = node_data.get("final_report", "")
            print(f"\n{'='*60}")
            print("FINAL RESEARCH REPORT")
            print(f"{'='*60}")
            print(report)
            sources = node_data.get("retrieved_sources", [])
