"""
Node: Build context from Redis session cache + pgvector semantic search.

Populates ``state["context"]`` with a merged context object.
Uses ``actor_id`` with fallback to ``student_id`` for backward compatibility.
"""

from __future__ import annotations

from orchestration.conversation_memory import get_history, get_lead_progress
from orchestration.state import WorkflowState
from mcp_server.tools.context_tools import get_context
from shared.logger import get_logger

logger = get_logger(__name__)


async def build_context_node(state: WorkflowState) -> WorkflowState:
    """Assemble context from cached sessions and vector search.

    Merges the actor data already in state with session/semantic results
    from the context tools layer.  Also injects conversation history from
    Redis so the downstream LLM reasoning node has multi-turn context.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with ``context`` populated.
    """
    actor_id = state.get("actor_id") or state.get("student_id", "")
    session_id = state.get("session_id") or actor_id
    user_input = state.get("user_input", "")
    app_id = state.get("app_id", "")

    try:
        if actor_id:
            ctx = await get_context(actor_id, user_input)
        else:
            # Guests should stay grounded to current RAG results instead of
            # inheriting any actor/session context.
            ctx = {
                "student_id": "",
                "session": {},
                "semantic_results": [],
            }

        # Merge in actor_context if available (falls back to student_data)
        actor_data = state.get("actor_context") or state.get("student_data") or {}
        ctx["student_data"] = actor_data

        # Inject conversation history for the session (if any)
        if app_id and session_id:
            history = await get_history(app_id, session_id)
            ctx["conversation_history"] = history
            if not state.get("lead_progress"):
                state["lead_progress"] = await get_lead_progress(app_id, session_id)

        state["context"] = ctx
        logger.info("build_context_node: assembled context for %s", session_id or actor_id)
    except Exception as exc:
        logger.error("build_context_node failed: %s", exc)
        state["error"] = "Context build failed"
        state["context"] = {"actor_id": actor_id}

    return state
