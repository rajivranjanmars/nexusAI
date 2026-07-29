"""
Conditional edge router for LangGraph workflows.

Reads the ``detected_workflow`` field from state and uses the dynamic
WorkflowConfig to determine the name of the next node to execute.
"""

from __future__ import annotations

from orchestration.state import WorkflowState
from orchestration.workflow_config import get_workflow_config
from shared.logger import get_logger

logger = get_logger(__name__)


def route_after_detection(state: WorkflowState) -> str:
    """Conditional edge function: decide which node to run next."""
    workflow = state.get("detected_workflow", "general")
    app_id = state.get("app_id")
    
    config = get_workflow_config(app_id)
    
    # Resolve the route from dynamic config, fallback to fetch_data
    next_node = config.routes.get(workflow, "fetch_data")

    logger.info(
        "Router: workflow=%s → next_node=%s (app_id=%s)",
        workflow,
        next_node,
        app_id
    )
    return next_node
