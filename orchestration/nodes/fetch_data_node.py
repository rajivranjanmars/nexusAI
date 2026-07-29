"""
Node: Fetch actor data from PostgreSQL.

Populates ``state["actor_context"]`` by querying the database layer.
Falls back to ``state["student_id"]`` for backward compatibility.
"""

from __future__ import annotations

from orchestration.state import WorkflowState
from mcp_server.tools.student_tools import get_student
from shared.logger import get_logger

logger = get_logger(__name__)


async def fetch_data_node(state: WorkflowState) -> WorkflowState:
    """Pull actor record from PostgreSQL and add it to state.

    Uses ``actor_id`` if present, falls back to deprecated ``student_id``.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with ``actor_context`` populated.
    """
    actor_id = state.get("actor_id") or state.get("student_id", "")

    # Skip DB lookup entirely when there is no actor to look up
    if not actor_id:
        state["actor_context"] = {}
        state["student_data"] = {}
        logger.info("fetch_data_node: no actor_id, skipping DB fetch")
        return state

    try:
        data = await get_student(actor_id)
        if "error" in data:
            logger.warning("Actor lookup returned error: %s", data)
            # Don't set state["error"] — pipeline can still continue with empty data

        state["actor_context"] = data
        # Backward compatibility
        state["student_data"] = data
        logger.info("fetch_data_node: loaded data for %s", actor_id)
    except Exception as exc:
        logger.error("fetch_data_node failed: %s", exc)
        state["actor_context"] = {}
        state["student_data"] = {}

    return state
