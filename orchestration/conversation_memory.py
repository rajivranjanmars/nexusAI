"""
Conversation memory backed by Redis.

Stores the last N user/assistant turns per session (app_id + session_key).
Configurable via environment variables (or shared config).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from db.redis_client import redis_client

# Configuration (can be overridden via env vars)
MAX_TURNS = int(os.getenv("CHAT_MEMORY_MAX_TURNS", "10"))  # Number of turns (user+assistant) to keep
MEMORY_TTL = int(os.getenv("CHAT_MEMORY_TTL_SECONDS", "3600"))  # 1 hour default


def _memory_key(app_id: str, session_key: str) -> str:
    return f"conv:{app_id}:{session_key}"


def _lead_progress_key(app_id: str, session_key: str) -> str:
    return f"lead_progress:{app_id}:{session_key}"


async def get_history(app_id: str, session_key: str) -> List[Dict[str, str]]:
    """Retrieve conversation history for a session.

    Returns a list of messages in the form:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    if not app_id or not session_key:
        return []
    key = _memory_key(app_id, session_key)
    raw = await redis_client.get(key)
    # redis_client returns None or the stored Python object (list)
    return raw or []


async def append_turn(app_id: str, session_key: str, user_msg: str, assistant_msg: str) -> None:
    """Append a user/assistant turn to the session history.

    Keeps only the most recent ``MAX_TURNS`` exchanges (i.e. ``MAX_TURNS * 2`` messages).
    """
    if not app_id or not session_key:
        return
    key = _memory_key(app_id, session_key)
    history: List[Dict[str, str]] = await get_history(app_id, session_key)
    # Append the new turn
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    # Trim to the most recent turns
    max_messages = MAX_TURNS * 2
    history = history[-max_messages:]
    await redis_client.set(key, history, ttl=MEMORY_TTL)


async def get_lead_progress(app_id: str, session_key: str) -> Dict[str, Any]:
    """Return persisted lead-capture progress for a session."""
    if not app_id or not session_key:
        return {}
    key = _lead_progress_key(app_id, session_key)
    raw = await redis_client.get(key)
    return raw if isinstance(raw, dict) else {}


async def set_lead_progress(app_id: str, session_key: str, lead_progress: Dict[str, Any]) -> None:
    """Persist lead-capture progress for a session."""
    if not app_id or not session_key or not lead_progress:
        return
    key = _lead_progress_key(app_id, session_key)
    await redis_client.set(key, lead_progress, ttl=MEMORY_TTL)


async def clear_lead_progress(app_id: str, session_key: str) -> None:
    """Delete persisted lead-capture progress for a session (e.g. on exit)."""
    if not app_id or not session_key:
        return
    key = _lead_progress_key(app_id, session_key)
    await redis_client.delete(key)


def _clarification_key(app_id: str, session_key: str) -> str:
    return f"clarification:{app_id}:{session_key}"


async def get_clarification(app_id: str, session_key: str) -> Dict[str, Any]:
    """Return persisted clarification state for a session."""
    if not app_id or not session_key:
        return {}
    key = _clarification_key(app_id, session_key)
    raw = await redis_client.get(key)
    return raw if isinstance(raw, dict) else {}


async def set_clarification(app_id: str, session_key: str, clarification: Dict[str, Any]) -> None:
    """Persist clarification state for a session."""
    if not app_id or not session_key or not clarification:
        return
    key = _clarification_key(app_id, session_key)
    await redis_client.set(key, clarification, ttl=MEMORY_TTL)


async def clear_clarification(app_id: str, session_key: str) -> None:
    """Delete persisted clarification state for a session (e.g. on exit)."""
    if not app_id or not session_key:
        return
    key = _clarification_key(app_id, session_key)
    await redis_client.delete(key)
