"""
MCP Tools — Chat completion via LangGraph orchestration.

Exposes ``chat_complete`` which runs the full student query pipeline:
intent classification → cache check → data fetch → context build →
LLM reasoning → cache store, returning a structured response.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from fastapi import HTTPException

from orchestration.graphs.student_query_graph import student_query_app
from orchestration.graphs.prestream_graph import prestream_app
from shared.helper_buttons import build_public_helper_buttons, normalize_lead_progress_payload
from shared.logger import get_logger

logger = get_logger(__name__)


def _normalize_lead_progress(lead_progress: Any, *, user_input: str = "") -> Any:
    """Normalize lead-progress payloads for API responses."""

    return normalize_lead_progress_payload(lead_progress, user_input=user_input)


def _build_helper_buttons(lead_progress: Any, *, user_input: str = "") -> list[dict[str, str]]:
    """Return public helper buttons for the current conversation turn."""

    return build_public_helper_buttons(lead_progress=lead_progress, user_input=user_input)


async def chat_complete(
    user_input: str,
    actor_id: str,
    actor_type: str = "student",
    app_id: str = "",
    session_id: str = "",
    force_workflow: Optional[str] = None,
    verified_phone: str = "",
) -> Dict[str, Any]:
    """Run the full LangGraph pipeline and return a structured chat response.

    Args:
        user_input: The raw message from the student.
        actor_id: Actor identifier for data retrieval.
        actor_type: Actor type discriminator.
        app_id: Originating application identifier.
        verified_phone: Phone number already verified via OTP at session
            start, if any — lets lead_capture skip re-asking for it.

    Returns:
        A dict with ``response``, ``workflow``, and ``token_usage``.
    """
    logger.info(
        "chat_complete called",
        extra={"actor_id": actor_id, "app_id": app_id},
    )

    # Enforce actor_id for lead_capture workflow
    if force_workflow == "lead_capture" and not actor_id:
        raise ValueError("lead_capture requires a valid actor_id")

    initial_state = {
        "actor_id": actor_id,
        "student_id": actor_id,
        "actor_type": actor_type,
        "app_id": app_id,
        "session_id": session_id or actor_id,
        "user_input": user_input,
        "force_workflow": force_workflow,
        "verified_phone": verified_phone,
    }

    thread_id = session_id or actor_id or str(uuid.uuid4())

    try:
        final_state = await student_query_app.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        response_text = final_state.get("llm_response", "")
        workflow = final_state.get("detected_workflow", "general")
        token_usage = final_state.get("token_usage", {})
        cache_hit = final_state.get("cache_hit", False)
        error = final_state.get("error")

        result: Dict[str, Any] = {
            "response": response_text,
            "workflow": workflow,
            "cache_hit": cache_hit,
            "token_usage": token_usage,
            "rag_context": final_state.get("rag_context", []),
            "answer_confidence": final_state.get("answer_confidence", 0.0),
            "retrieval_debug": final_state.get("retrieval_debug", {}),
        }

        if "lead_progress" in final_state and workflow == "lead_capture":
            raw_lead_progress = final_state["lead_progress"]
            result["lead_progress"] = _normalize_lead_progress(
                raw_lead_progress,
                user_input=user_input,
            )
            result["helper_buttons"] = _build_helper_buttons(
                raw_lead_progress,
                user_input=user_input,
            )

        if error:
            result["error"] = error
            logger.warning("chat_complete finished with error: %s", error)

        logger.info(
            "chat_complete finished",
            extra={
                "workflow": workflow,
                "cache_hit": cache_hit,
                "response_len": len(response_text),
            },
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat_complete pipeline failed: %s", exc)
        return {
            "response": (
                "I encountered a brief hiccup while processing your request. "
                "Please try again."
            ),
            "workflow": "general",
            "cache_hit": False,
            "error": "Internal pipeline error",
        }


async def chat_complete_prestream(
    user_input: str,
    actor_id: str,
    actor_type: str = "student",
    app_id: str = "",
    session_id: str = "",
    force_workflow: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the pipeline up to context building and return prepared state.

    Used by the streaming endpoint so LLM reasoning can be streamed directly.
    If a cache hit occurs, the full response is returned and no streaming
    is needed.

    The prestream graph is stateless (no checkpointer) so every invocation
    starts with a clean slate and always produces a fresh ``rag_context``.

    Returns:
        A dict with ``cache_hit``, ``workflow``, ``context``, ``rag_context``, etc.
    """
    logger.info(
        "chat_complete_prestream called",
        extra={"actor_id": actor_id, "app_id": app_id},
    )

    if force_workflow == "lead_capture" and not actor_id:
        raise ValueError("lead_capture requires a valid actor_id")

    initial_state = {
        "actor_id": actor_id,
        "student_id": actor_id,
        "actor_type": actor_type,
        "app_id": app_id,
        "session_id": session_id or actor_id,
        "user_input": user_input,
        "force_workflow": force_workflow,
    }

    try:
        # The prestream graph has no MemorySaver — it is stateless by design.
        # Every invocation runs a fresh classify → cache-check → fetch → RAG → context
        # pipeline so that rag_context is always current.
        final_state = await prestream_app.ainvoke(initial_state)

        prechecked_response = bool((final_state.get("metadata") or {}).get("prechecked_response"))

        raw_lead_progress = final_state.get("lead_progress", {})
        normalized_lead_progress = _normalize_lead_progress(
            raw_lead_progress,
            user_input=user_input,
        )
        return {
            "cache_hit": final_state.get("cache_hit", False) or prechecked_response,
            "cached_response": final_state.get("llm_response", ""),
            "workflow": final_state.get("detected_workflow", "general"),
            "context": final_state.get("context", {}),
            "rag_context": final_state.get("rag_context", []),
            "answer_confidence": final_state.get("answer_confidence", 0.0),
            "retrieval_debug": final_state.get("retrieval_debug", {}),
            "lead_progress": normalized_lead_progress,
            "helper_buttons": _build_helper_buttons(
                raw_lead_progress,
                user_input=user_input,
            ),
            "app_id": app_id,
            "actor_id": actor_id,
            "actor_type": actor_type,
            "session_id": session_id or actor_id,
            "user_input": user_input,
            "error": final_state.get("error"),
        }

    except Exception as exc:
        logger.exception("chat_complete_prestream failed: %s", exc)
        return {
            "cache_hit": False,
            "cached_response": "",
            "workflow": "general",
            "context": {},
            "rag_context": [],
            "answer_confidence": 0.0,
            "retrieval_debug": {},
            "lead_progress": {},
            "helper_buttons": [],
            "app_id": app_id,
            "actor_id": actor_id,
            "actor_type": actor_type,
            "session_id": session_id or actor_id,
            "user_input": user_input,
            "error": "Internal prestream error",
        }
