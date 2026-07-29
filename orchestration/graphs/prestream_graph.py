"""
Pre-stream graph: runs the pipeline up to ``build_context`` and stops.

Used by the streaming endpoint to prepare context (classify, cache-check,
fetch data, RAG retrieve, build context) without running the LLM reasoning
step.  The proxy then streams LLM tokens directly via ``generate_response_stream``.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from orchestration.state import WorkflowState
from orchestration.router import route_after_detection
from orchestration.nodes.detect_workflow_node import detect_workflow_node
from orchestration.nodes.precheck_smalltalk_node import precheck_smalltalk_node
from orchestration.nodes.cache_node import cache_check_node
from orchestration.nodes.fetch_data_node import fetch_data_node
from orchestration.nodes.build_context_node import build_context_node
from orchestration.nodes.rag_retrieve_node import rag_retrieve_node
from shared.logger import get_logger

logger = get_logger(__name__)


def _route_after_cache(state: WorkflowState) -> str:
    """Conditional edge after cache check.

    Returns ``"__end__"`` on cache hit, otherwise delegates to the
    standard workflow router.
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

def build_prestream_graph() -> StateGraph:
    """Construct a graph that runs classification → cache → fetch → RAG → context.

    Stops BEFORE llm_reasoning so the proxy can stream the LLM call directly.
    On cache hit the graph short-circuits to END as usual.
    """
    graph = StateGraph(WorkflowState)

    graph.add_node("precheck_smalltalk", precheck_smalltalk_node)
    graph.add_node("detect_workflow", detect_workflow_node)
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("build_context", build_context_node)

    graph.set_entry_point("precheck_smalltalk")
    graph.add_conditional_edges(
        "precheck_smalltalk",
        _route_after_smalltalk,
        {
            "detect_workflow": "detect_workflow",
            END: END,
        },
    )
    graph.add_edge("detect_workflow", "cache_check")

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
    graph.add_edge("build_context", END)

    logger.info("Pre-stream graph compiled (context-only, stateless)")
    return graph.compile()


# Pre-compiled singleton
prestream_app = build_prestream_graph()
