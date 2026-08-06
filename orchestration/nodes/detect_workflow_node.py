"""
Node: Classify intent using the fast LLM model.

Populates ``state["detected_workflow"]`` via the LLM classifier and
accumulates token usage into ``state["token_usage"]``.
"""

from __future__ import annotations

import time

from orchestration.state import WorkflowState
from llm.classifier import classify_intent
from orchestration.workflow_policy import resolve_effective_workflow
from orchestration.workflow_config import get_workflow_config, sticky_workflow
from orchestration.conversation_memory import clear_lead_progress, get_lead_progress
from shared.helper_buttons import EXIT_WORKFLOW_KEYWORDS
from shared.logger import get_logger

logger = get_logger(__name__)


def _accumulate_usage(state: WorkflowState, usage: dict) -> None:
    """Merge a single LLM usage dict into the cumulative state total."""
    existing = state.get("token_usage") or {}
    state["token_usage"] = {
        "prompt_tokens": existing.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0),
        "completion_tokens": existing.get("completion_tokens", 0) + usage.get("completion_tokens", 0),
        "total_tokens": existing.get("total_tokens", 0) + usage.get("total_tokens", 0),
        "model": usage.get("model", existing.get("model")),
    }


async def detect_workflow_node(state: WorkflowState) -> WorkflowState:
    """Classify the user's intent and update state.

    Args:
        state: Current workflow state with ``user_input``.

    Returns:
        Updated state with ``detected_workflow`` and ``token_usage`` populated.
    """
    if state.get("force_workflow"):
        workflow, policy_reason = resolve_effective_workflow(
            state["force_workflow"],
            app_id=state.get("app_id"),
            actor_id=state.get("actor_id") or state.get("student_id"),
        )
        state["detected_workflow"] = workflow
        if policy_reason:
            state["metadata"] = {
                **(state.get("metadata") or {}),
                "workflow_policy": policy_reason,
            }
            logger.info(policy_reason)
        logger.info("detect_workflow_node: forced workflow '%s'", workflow)
        return state

    actor_id = state.get("actor_id") or state.get("student_id") or ""
    is_guest = not actor_id
    if is_guest:
        state["metadata"] = {
            **(state.get("metadata") or {}),
            "guest_public_workflow": True,
        }

    # Fetch lead progress early for stickiness
    app_id = state.get("app_id") or ""
    session_id = state.get("session_id") or actor_id
    if app_id and session_id and not state.get("lead_progress"):
        state["lead_progress"] = await get_lead_progress(app_id, session_id)

    # Resolve stickiness config
    config = get_workflow_config(app_id)
    sticky_name = sticky_workflow(config)

    # ponytail: the progress record is still lead-shaped because lead_capture is
    # the only elicit workflow. Generalise the store when a second one exists.
    lead_progress = state.get("lead_progress", {})
    user_input_lower = state.get("user_input", "").lower()
    exit_requested = any(kw in user_input_lower for kw in EXIT_WORKFLOW_KEYWORDS)

    # Stickiness: Lock into the sticky workflow if form is active, recently touched,
    # and the user didn't ask to exit.
    if sticky_name and lead_progress.get("status") == "in_progress":
        session_abandoned = False
        updated_at = lead_progress.get("updated_at")
        if updated_at:
            abandon_timeout = config.workflow_response_config[sticky_name].abandon_after_seconds
            if (time.time() - float(updated_at)) > abandon_timeout:
                session_abandoned = True

        if not exit_requested and not session_abandoned:
            workflow = sticky_name
            state["detected_workflow"] = workflow
            if is_guest:
                state["metadata"] = {
                    **(state.get("metadata") or {}),
                    "guest_detected_workflow": workflow,
                }
            logger.info("detect_workflow_node: forced sticky workflow '%s'", workflow)
            return state

        if exit_requested or session_abandoned:
            # Drop the stale in-progress lock so a later message can't re-stick.
            # An abandoned session only re-enters the sticky workflow via an explicit
            # callback request, same as classify_intent below would route it.
            await clear_lead_progress(app_id, session_id)
            state["lead_progress"] = {}
            reason = "exit keyword" if exit_requested else "inactivity (session abandoned)"
            logger.info("detect_workflow_node: %s cleared sticky %s lock", reason, sticky_name)

    try:
        workflow, usage = await classify_intent(
            state["user_input"],
            app_id=state.get("app_id") or "",
            actor_id=state.get("actor_id") or state.get("student_id") or "",
            actor_type=state.get("actor_type") or "",
        )
        state["detected_workflow"] = workflow
        if is_guest:
            state["metadata"] = {
                **(state.get("metadata") or {}),
                "guest_detected_workflow": workflow,
            }
        _accumulate_usage(state, usage)
        logger.info("detect_workflow_node: classified as '%s'", workflow)
    except Exception as exc:
        logger.error("detect_workflow_node failed: %s", exc)
        state["error"] = "Classification failed"
        state["detected_workflow"] = "general"

    return state
