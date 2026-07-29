"""
LangGraph graph: Student query workflow.

Builds a ``StateGraph`` with the following pipeline::

    detect_workflow → cache_check → (HIT→END | MISS→route) →
        fetch_data → build_context → llm_reasoning → cache_store → END

The semantic cache is checked immediately after workflow classification.
If a cached response is found (similarity ≥ threshold), the graph short-
circuits directly to END, avoiding the LLM reasoning step entirely.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

_memory = MemorySaver()

from orchestration.state import WorkflowState
from orchestration.router import route_after_detection
from orchestration.nodes.detect_workflow_node import detect_workflow_node
from orchestration.nodes.cache_node import cache_check_node, cache_store_node
from orchestration.nodes.fetch_data_node import fetch_data_node
from orchestration.nodes.build_context_node import build_context_node
from orchestration.nodes.llm_reasoning_node import llm_reasoning_node
from orchestration.nodes.precheck_smalltalk_node import precheck_smalltalk_node
from orchestration.nodes.rag_retrieve_node import rag_retrieve_node
from shared.logger import get_logger

logger = get_logger(__name__)


def _route_after_cache(state: WorkflowState) -> str:
    """Conditional edge after cache check.

    Returns ``"__end__"`` on cache hit, otherwise delegates to the
    standard workflow router.

    Args:
        state: Current workflow state.

    Returns:
        Next node name.
    """
    if state.get("cache_hit"):
        logger.info("Cache HIT — skipping pipeline, returning cached response")
        return END
    return route_after_detection(state)


def _route_after_smalltalk(state: WorkflowState) -> str:
    """End early when the smalltalk precheck already produced a reply."""
    if state.get("llm_response") and (state.get("metadata") or {}).get("prechecked_response"):
        return END
    return "detect_workflow"

def build_student_query_graph() -> StateGraph:
    """Construct and compile the student-query workflow graph.

    Returns:
        A compiled :class:`langgraph.graph.StateGraph` ready for invocation.
    """
    graph = StateGraph(WorkflowState)

    # ── Add nodes ────────────────────────────────────────────────────────
    graph.add_node("precheck_smalltalk", precheck_smalltalk_node)
    graph.add_node("detect_workflow", detect_workflow_node)
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("llm_reasoning", llm_reasoning_node)
    graph.add_node("cache_store", cache_store_node)

    # ── Wire edges ───────────────────────────────────────────────────────
    graph.set_entry_point("precheck_smalltalk")

    graph.add_conditional_edges(
        "precheck_smalltalk",
        _route_after_smalltalk,
        {
            "detect_workflow": "detect_workflow",
            END: END,
        },
    )

    # detect_workflow → cache_check (always)
    graph.add_edge("detect_workflow", "cache_check")

    # cache_check → END (on hit) or fetch_data/build_context (on miss)
    graph.add_conditional_edges(
        "cache_check",
        _route_after_cache,
        {
            "fetch_data": "fetch_data",
            "rag_retrieve": "rag_retrieve",
            "build_context": "build_context",
            END: END,
        },
    )

    graph.add_edge("fetch_data", "rag_retrieve")
    graph.add_edge("rag_retrieve", "build_context")
    graph.add_edge("build_context", "llm_reasoning")
    graph.add_edge("llm_reasoning", "cache_store")
    graph.add_edge("cache_store", END)

    logger.info("Student query graph compiled (with semantic cache)")
    return graph.compile(checkpointer=_memory)


# Pre-compiled singleton — import and invoke directly.
student_query_app = build_student_query_graph()
