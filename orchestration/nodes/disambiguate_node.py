"""
Disambiguation node: ask the user to narrow down ambiguous results before answering.

When RAG retrieval yields chunks scattered across multiple facets (e.g., many
programmes, many specializations), this node identifies the highest-value axis
to ask about and surfaces a single clarification question. The question is marked
as prechecked so the graph short-circuits and returns it to the user immediately.

On the following turn, the node detects the pending clarification and clears it,
allowing the normal answer pipeline to proceed with the narrowed context.
"""

from __future__ import annotations

from orchestration.conversation_memory import (
    get_clarification,
    set_clarification,
    clear_clarification,
)
from orchestration.state import WorkflowState
from orchestration.workflow_config import get_workflow_config
from shared.disambiguation import Axis, top_axis
from shared.logger import get_logger

logger = get_logger(__name__)


async def disambiguate_node(state: WorkflowState) -> WorkflowState:
    """Optionally ask a narrowing question before attempting to answer.

    Returns state unchanged at any early exit. On success:
      - Sets state["llm_response"] to a single-sentence question
      - Sets state["metadata"]["prechecked_response"] = True
      - Stores pending clarification in Redis and state["clarification"]
    """

    # 1. Extract keys, return early if missing
    app_id = state.get("app_id")
    session_key = state.get("session_id") or state.get("actor_id")
    if not app_id or not session_key:
        return state

    # 2. Validate workflow strategy
    workflow = state.get("detected_workflow") or "general"
    config = get_workflow_config(app_id)
    wf = config.workflow_response_config.get(workflow) or config.workflow_response_config.get("general")
    if wf is None or wf.strategy != "answer":
        return state

    # 3. Check budget: only one clarification before answering
    prev_pending = False
    try:
        prev = await get_clarification(app_id, session_key)
        prev_pending = bool(prev and prev.get("pending"))
    except Exception as exc:
        logger.warning("Failed to read pending clarification for %s/%s: %s", app_id, session_key, exc)
        # On read failure, treat as "no pending clarification"
        prev_pending = False

    if prev_pending:
        # Check if user's reply matches any of the offered options
        reply = (state.get("user_input") or "").strip().casefold()
        options = [str(o).casefold() for o in (prev.get("options") or [])]
        matched = any(o and o in reply for o in options)

        # Only a short, non-interrogative reply reads as a failed answer
        # attempt ("M.Tech"). A sentence or a question is the user changing
        # the subject — answer that instead of nagging about the old facet.
        looks_like_answer = len(reply.split()) <= 3 and "?" not in reply

        # If they tried to answer but missed, retry the same facet once
        if not matched and looks_like_answer and not prev.get("retried"):
            question = f"Sorry, I didn't catch which {prev.get('axis', 'option')} you meant — could you type it exactly?"
            state["llm_response"] = question
            metadata = dict(state.get("metadata") or {})
            metadata["prechecked_response"] = True
            state["metadata"] = metadata

            pending2 = {**prev, "retried": True}
            state["clarification"] = pending2

            try:
                await set_clarification(app_id, session_key, pending2)
            except Exception as exc:
                logger.warning("Failed to update pending clarification for %s/%s: %s", app_id, session_key, exc)

            logger.info("Retrying clarification for %s on %s/%s", prev.get("axis", "option"), app_id, session_key)
            return state

        # Otherwise (matched, OR already retried): clear and proceed to answer
        try:
            await clear_clarification(app_id, session_key)
        except Exception as exc:
            logger.warning("Failed to clear pending clarification for %s/%s: %s", app_id, session_key, exc)
        state["clarification"] = None
        logger.info("Clarification budget spent for %s/%s — proceeding to answer", app_id, session_key)
        return state

    # 4. Return unchanged if no RAG context
    rag = state.get("rag_context") or []
    if not rag:
        return state

    # 5. Compute top axis, return unchanged if no clear winner
    axes = [Axis(name, prefix) for name, prefix in config.disambiguation.axes]
    if not axes:
        return state

    result = top_axis(rag, axes, config.disambiguation.min_gain)
    if result is None:
        return state

    axis_name, options = result
    if len(options) < 2:
        return state

    # 6. ASK
    shown = [o.upper() for o in options[:4]]
    question = f"Just to make sure I get this right — which {axis_name} are you asking about: {', '.join(shown)}?"

    state["llm_response"] = question
    metadata = dict(state.get("metadata") or {})
    metadata["prechecked_response"] = True
    state["metadata"] = metadata

    pending = {
        "pending": True,
        "axis": axis_name,
        "options": options,
        "asked_for": state.get("user_input", ""),
    }
    state["clarification"] = pending

    try:
        await set_clarification(app_id, session_key, pending)
    except Exception as exc:
        logger.warning("Failed to persist pending clarification for %s/%s: %s", app_id, session_key, exc)
        # On write failure, still return the question (it's in state)

    logger.info(
        "Asking about %s with %d options for %s/%s",
        axis_name,
        len(options),
        app_id,
        session_key,
    )
    return state
