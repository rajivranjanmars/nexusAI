"""
Pluggable data-resolver registry for app-token response enrichment.

Each app can declare ``data_resolvers`` in its ``app_config`` to request
additional data payloads in the token response.  Resolvers are registered
by name and executed dynamically at token-issuance time.

Example YAML entry::

    app_config:
      data_resolvers:
        - resolver: student_profile
          key: student
          requires_actor: true
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, List, Optional

from db.models.student import Student
from db.postgres import get_session
from shared.logger import get_logger
from sqlalchemy import select

logger = get_logger(__name__)

# ── Resolver registry ───────────────────────────────────────────────────────

_RESOLVER_REGISTRY: Dict[str, Callable[..., Coroutine[Any, Any, Optional[Dict[str, Any]]]]] = {}


def register_resolver(name: str):
    """Decorator that registers an async resolver function by name.

    Args:
        name: Unique resolver identifier referenced in ``app_config.data_resolvers``.

    Returns:
        Decorator wrapping the resolver function.
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, Optional[Dict[str, Any]]]],
    ) -> Callable[..., Coroutine[Any, Any, Optional[Dict[str, Any]]]]:
        if name in _RESOLVER_REGISTRY:
            logger.warning("Overwriting resolver: %s", name)
        _RESOLVER_REGISTRY[name] = func
        logger.debug("Registered resolver: %s", name)
        return func

    return decorator


async def run_resolvers(
    includes: List[Dict[str, Any]],
    actor_id: str,
) -> Optional[Dict[str, Any]]:
    """Execute each resolver declared in ``data_resolvers``.

    Args:
        includes: List of resolver declarations from ``app_config.data_resolvers``.
        actor_id: The actor identifier from the token request.

    Returns:
        Merged dictionary of all resolver outputs keyed by their declared
        ``key``, or ``None`` if no resolvers produced data.
    """
    if not includes:
        return None

    result: Dict[str, Any] = {}

    async def _run_one(entry: Dict[str, Any]) -> None:
        resolver_name = entry.get("resolver", "")
        key = entry.get("key", resolver_name)
        requires_actor = entry.get("requires_actor", False)

        if requires_actor and not actor_id:
            logger.debug(
                "Skipping resolver %s — requires actor but none provided",
                resolver_name,
            )
            return

        func = _RESOLVER_REGISTRY.get(resolver_name)
        if func is None:
            logger.warning("Unknown resolver requested: %s", resolver_name)
            return

        try:
            data = await func(actor_id=actor_id, config=entry)
            if data is not None:
                result[key] = data
        except Exception:
            logger.exception("Resolver %s failed", resolver_name)

    await asyncio.gather(*[_run_one(entry) for entry in includes])

    return result if result else None


# ── Built-in resolvers ──────────────────────────────────────────────────────


@register_resolver("student_profile")
async def resolve_student_profile(
    *, actor_id: str, config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Fetch a student record and return a front-end-friendly profile dict.

    Args:
        actor_id: The student identifier (``student_id`` column).
        config: Full resolver entry from ``app_config.data_resolvers``.

    Returns:
        Dictionary with student profile fields, or ``None`` if not found.
    """
    if not actor_id:
        return None

    async with get_session() as session:
        row = await session.execute(
            select(Student).where(Student.student_id == actor_id)
        )
        student = row.scalar_one_or_none()

    if student is None:
        logger.warning("student_profile resolver: student %s not found", actor_id)
        return None

    return {
        "name": f"{student.first_name} {student.last_name}",
        "regNo": student.student_id,
        "attendance": student.attendance,
        "cgpa": student.gpa,
        "feeStatus": student.fee_status,
        "nextExam": student.next_exam.isoformat() if student.next_exam else None,
        "activeCourses": student.active_courses,
    }
