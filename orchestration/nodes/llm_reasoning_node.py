"""
Node: Call the smart LLM model for deep reasoning.

Reads context from state and produces a final ``llm_response``.
Accumulates token usage from the reasoning call into ``state["token_usage"]``.
"""

from __future__ import annotations

import json
import re

from db.app_registry import resolve_by_app_id
from orchestration.conversation_memory import append_turn, set_lead_progress
from orchestration.state import WorkflowState
from orchestration.workflow_config import get_workflow_config
from orchestration.workflow_policy import workflow_uses_public_rag
from llm.reasoner import generate_response
from shared.helper_buttons import normalize_lead_progress_payload
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


def _workflow_includes_sources(app_id: str | None, workflow: str | None, user_query: str = "") -> bool:
    """Whether the current workflow config allows appending sources."""
    config = get_workflow_config(app_id)
    workflow_name = (workflow or "").strip().lower()
    workflow_config = config.workflow_response_config.get(
        workflow_name,
        config.workflow_response_config.get("general"),
    )
    return bool(workflow_config and workflow_config.include_sources)


def _append_sources(
    response: str,
    rag_context: list[dict],
    *,
    app_id: str | None,
    workflow: str | None,
    user_query: str = "",
) -> str:
    """Append a deduplicated Sources section when RAG content was used."""
    if not response or not rag_context:
        return response

    if not _workflow_includes_sources(app_id, workflow, user_query):
        return response

    if "Sources:" in response:
        return response

    sources: list[str] = []
    seen: set[str] = set()
    source_index = 0
    for item in rag_context:
        source_url = (item or {}).get("source_url")
        if not source_url or source_url in seen:
            continue
        seen.add(source_url)
        source_index += 1
        heading = (item or {}).get("section_heading") or source_url
        score = float((item or {}).get("score", 0.0) or 0.0)
        sources.append(f"[{source_index}] {heading} - {source_url}")
        if len(sources) >= 3:
            break

    if not sources:
        return response

    sources_block = "\n".join(sources)
    return f"{response.rstrip()}\n\nSources:\n{sources_block}"


async def llm_reasoning_node(state: WorkflowState) -> WorkflowState:
    """Generate a context-aware response using the smart model.

    Args:
        state: Current workflow state (must contain ``context`` and ``user_input``).

    Returns:
        Updated state with ``llm_response`` and ``token_usage`` populated.
    """
    context = state.get("context") or {}
    context["app_id"] = state.get("app_id", "")
    context["workflow"] = state.get("detected_workflow", "general")
    context["actor_id"] = state.get("actor_id", "")
    context["rag_context"] = state.get("rag_context", [])
    context["answer_confidence"] = state.get("answer_confidence", 0.0)
    context["lead_progress"] = state.get("lead_progress", {})
    context["verified_phone"] = state.get("verified_phone", "")
    workflow = state.get("detected_workflow", "general")
    actor_id = state.get("actor_id") or state.get("student_id") or ""
    context["strict_rag"] = bool(state.get("app_id")) and workflow_uses_public_rag(workflow)
    context["guest_mode"] = not bool(actor_id)

    # Inject persona from app_config
    app_id = state.get("app_id")
    app_context = resolve_by_app_id(app_id) if app_id else None
    app_config = app_context.app_config if app_context else {}
    if app_config and "persona" in app_config:
        context["persona"] = app_config["persona"]

    user_input = state.get("user_input", "")

    try:
        response, usage = await generate_response(
            context=context,
            user_query=user_input,
            app_id=state.get("app_id") or "",
            actor_id=state.get("actor_id") or state.get("student_id") or "",
            actor_type=state.get("actor_type") or "",
        )

        logger.debug("llm_reasoning_node raw output (%d chars): %.500s", len(response), response)

        # Parse and execute lead capture tools output by the LLM
        from mcp_server.tools import lead_tools

        upsert_matches = re.findall(r"<upsert_lead>(.*?)</upsert_lead>", response, re.DOTALL)
        for upsert_raw in upsert_matches:
            try:
                payload = json.loads(upsert_raw)
                session_id = payload.pop("session_id", context["actor_id"])
                if "phone" in payload:
                    payload["mobile"] = payload.pop("phone")
                await lead_tools.upsert_lead(session_id=session_id, **payload)
            except Exception as e:
                logger.error("Failed to parse and execute upsert_lead: %s", e)

        if upsert_matches:
            # Clean the response for the user
            response = re.sub(r"<upsert_lead>.*?</upsert_lead>", "", response, flags=re.DOTALL).strip()

        progress_match = re.search(r"<lead_progress>\s*(.*?)\s*</lead_progress>", response, re.DOTALL)
        if progress_match:
            try:
                progress_payload = json.loads(progress_match.group(1))
                state["lead_progress"] = normalize_lead_progress_payload(
                    progress_payload,
                    user_input=user_input,
                )
            except Exception as e:
                logger.error("Failed to parse lead_progress: %s", e)

            # Clean the response for the user
            response = re.sub(r"<lead_progress>.*?</lead_progress>", "", response, flags=re.DOTALL).strip()

        # Safety net: if stripping XML blocks left an empty response but we
        # successfully parsed lead_progress, provide a fallback greeting so
        # the user never sees a blank message.
        if not response and state.get("lead_progress"):
            progress = state["lead_progress"]
            current_step = progress.get("current_step", "name")
            step_prompts = {
                "name": "Hi there! 👋 I'd love to help you get started with your admission journey. Could you share your full name?",
                "mobile": "Thanks. Could you share your 10-digit mobile number so our admissions team can reach you?",
                "email": "Great! Could you share your email address so we can stay in touch?",
                "program_level": "What level of program are you interested in — Bachelors, Masters, or Diploma?",
                "program_names": "Which specific program(s) are you interested in?",
                "state": "Which state are you located in?",
                "city": "And which city or district?",
                "address": "Lastly, could you share your full address for our records?",
            }
            response = step_prompts.get(
                current_step,
                "Hi there! 👋 I'd love to help you get started. Could you share your full name?",
            )
            logger.warning(
                "LLM response was empty after XML strip; using fallback for step '%s'",
                current_step,
            )

        response = _append_sources(
            response,
            state.get("rag_context") or [],
            app_id=state.get("app_id"),
            workflow=workflow,
            user_query=user_input,
        )

        state["llm_response"] = response
        _accumulate_usage(state, usage)

        # Persist the turn in conversation memory
        app_id = state.get("app_id", "")
        actor_id = state.get("actor_id") or state.get("student_id", "")
        session_id = state.get("session_id") or actor_id
        if app_id and session_id:
            try:
                if workflow == "lead_capture" and state.get("lead_progress"):
                    await set_lead_progress(app_id, session_id, state["lead_progress"])
                await append_turn(app_id, session_id, user_input, response)
            except Exception as mem_exc:
                logger.warning("Failed to persist conversation turn: %s", mem_exc)

        logger.info(
            "llm_reasoning_node: generated response (%d chars)", len(response)
        )
    except Exception as exc:
        logger.error("llm_reasoning_node failed: %s", exc)
        state["error"] = "LLM reasoning failed"
        state["llm_response"] = (
            "I encountered a brief hiccup while processing your request. "
            "Please try again in a moment."
        )

    return state
