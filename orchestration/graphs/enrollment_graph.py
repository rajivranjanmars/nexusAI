"""
LangGraph graph: Enrollment workflow.

A specialised subgraph for handling enrollment-related queries.  Follows the
same pattern as the student-query graph but can be extended with
enrollment-specific nodes (e.g. eligibility checks, seat availability).

Pipeline:
    fetch_data → build_context → llm_reasoning → END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from orchestration.state import WorkflowState
from orchestration.nodes.rag_retrieve_node import rag_retrieve_node
from orchestration.nodes.build_context_node import build_context_node
from orchestration.nodes.llm_reasoning_node import llm_reasoning_node
from shared.logger import get_logger

logger = get_logger(__name__)


def build_enrollment_graph() -> StateGraph:
    """Construct and compile the enrollment workflow graph.

    Returns:
        A compiled :class:`langgraph.graph.StateGraph`.
    """
    graph = StateGraph(WorkflowState)

    # ── Add nodes ────────────────────────────────────────────────────────
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("llm_reasoning", llm_reasoning_node)

    # ── Wire edges ───────────────────────────────────────────────────────
    graph.set_entry_point("rag_retrieve")
    graph.add_edge("rag_retrieve", "build_context")
    graph.add_edge("build_context", "llm_reasoning")
    graph.add_edge("llm_reasoning", END)

    logger.info("Enrollment graph compiled")
    return graph.compile()


# Pre-compiled singleton.
enrollment_app = build_enrollment_graph()
