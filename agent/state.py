"""
Agent State Definition
======================
The ResearchState is the shared memory that flows through every node in the graph.

Key concept: Annotated[list, add_messages] tells LangGraph to APPEND new messages
to the list rather than replace it. Without this annotation, each node would
overwrite the message history instead of extending it.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    # The original user research question — set once, never changed
    query: str

    # Full conversation history: HumanMessage, AIMessage, ToolMessage
    # add_messages reducer: new messages are appended, not overwritten
    messages: Annotated[list, add_messages]

    # Citations collected during research (used in completeness check)
    retrieved_sources: list[str]

    # How many agent→tool→agent cycles have completed (loop guard)
    iteration_count: int

    # The final written report — empty until synthesize_node runs
    final_report: str

    # Set to True by synthesize_node to signal the graph to stop
    is_complete: bool
