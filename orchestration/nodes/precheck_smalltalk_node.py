"""
Node: Fast-path greetings and light small talk without hitting RAG.

This node recognizes a narrow set of conversational openers and courtesy
messages and returns a direct response early. It is intentionally small and
conservative so knowledge-seeking questions still flow through the normal
pipeline.
"""

from __future__ import annotations

import re

from orchestration.conversation_memory import append_turn, get_history
from orchestration.state import WorkflowState

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s']")

_GREETING_PATTERNS = (
    re.compile(r"^(hi|hii|hello|hey|heyy)$", re.IGNORECASE),
    re.compile(r"^good (morning|afternoon|evening)$", re.IGNORECASE),
)
_HOW_ARE_YOU_PATTERNS = (
    re.compile(r"^(how are you|how are you doing|how's it going|hows it going)\??$", re.IGNORECASE),
    re.compile(r"^(what's up|whats up|sup)\??$", re.IGNORECASE),
)
_THANKS_PATTERNS = (
    re.compile(r"^(thanks|thank you|thx|ty)$", re.IGNORECASE),
)
_BYE_PATTERNS = (
    re.compile(r"^(bye|goodbye|see you|see ya|talk to you later)$", re.IGNORECASE),
)


def _normalize(text: str) -> str:
    stripped = _PUNCT_RE.sub(" ", (text or "").strip().lower())
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _match_smalltalk_response(user_input: str) -> str | None:
    normalized = _normalize(user_input)
    if not normalized:
        return None

    if any(pattern.match(normalized) for pattern in _HOW_ARE_YOU_PATTERNS):
        return "I'm doing well, thanks. How can I help you today?"

    if any(pattern.match(normalized) for pattern in _THANKS_PATTERNS):
        return "You're welcome. If you'd like, I can assist with admissions, fees, eligibility, or programmes."

    if any(pattern.match(normalized) for pattern in _BYE_PATTERNS):
        return "Goodbye. If you need anything later, I can assist with admissions, programmes, or eligibility."

    if any(pattern.match(normalized) for pattern in _GREETING_PATTERNS):
        return "Hi! Welcome to LPU Distance Education. How can I assist you today?"

    return None


async def precheck_smalltalk_node(state: WorkflowState) -> WorkflowState:
    """Short-circuit greetings and small talk before cache/RAG."""
    if state.get("force_workflow"):
        return state

    app_id = state.get("app_id") or ""
    session_id = state.get("session_id") or state.get("actor_id") or state.get("student_id") or ""
    history = await get_history(app_id, session_id) if app_id and session_id else []
    is_first_turn = len(history) == 0

    response = _match_smalltalk_response(state.get("user_input", ""))
    if not response:
        return state

    if not is_first_turn and response == "Hi! Welcome to LPU Distance Education. How can I assist you today?":
        response = "How can I help with information related to admissions, programmes, or eligibility?"

    if app_id and session_id:
        try:
            await append_turn(app_id, session_id, state.get("user_input", ""), response)
        except Exception:
            pass

    state["llm_response"] = response
    state["metadata"] = {
        **(state.get("metadata") or {}),
        "prechecked_response": True,
        "response_source": "smalltalk_precheck",
        "is_first_turn": is_first_turn,
    }
    return state
