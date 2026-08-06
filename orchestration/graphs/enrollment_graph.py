"""
LangGraph graph: Enrollment workflow.

A specialised subgraph for handling enrollment-related queries.  Follows the
same pattern as the student-query graph but can be extended with
enrollment-specific nodes (e.g. eligibility checks, seat availability).

Pipeline:
    rag_retrieve → disambiguate → (narrowing Q→END | continue→build_context) →
        build_context → llm_reasoning → END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from orchestration.state import WorkflowState
from orchestration.nodes.rag_retrieve_node import rag_retrieve_node
from orchestration.nodes.disambiguate_node import disambiguate_node
from orchestration.nodes.build_context_node import build_context_node
from orchestration.nodes.llm_reasoning_node import llm_reasoning_node
from shared.logger import get_logger

logger = get_logger(__name__)


def _route_after_disambiguate(state: WorkflowState) -> str:
    """End early when a narrowing question was produced."""
    if state.get("llm_response") and (state.get("metadata") or {}).get("prechecked_response"):
        return END
    return "build_context"


def build_enrollment_graph() -> StateGraph:
    """Construct and compile the enrollment workflow graph.

    Returns:
        A compiled :class:`langgraph.graph.StateGraph`.
    """
    graph = StateGraph(WorkflowState)

    # ── Add nodes ────────────────────────────────────────────────────────
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("disambiguate", disambiguate_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("llm_reasoning", llm_reasoning_node)

    # ── Wire edges ───────────────────────────────────────────────────────
    graph.set_entry_point("rag_retrieve")
    graph.add_edge("rag_retrieve", "disambiguate")
    graph.add_conditional_edges(
        "disambiguate",
        _route_after_disambiguate,
        {
            "build_context": "build_context",
            END: END,
        },
    )
    graph.add_edge("build_context", "llm_reasoning")
    graph.add_edge("llm_reasoning", END)

    logger.info("Enrollment graph compiled")
    return graph.compile()


# Pre-compiled singleton.
enrollment_app = build_enrollment_graph()
