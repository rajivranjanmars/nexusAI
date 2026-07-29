"""
MCP Resource — Student profile blobs.

Exposes a student profile as an MCP Resource that can be read by MCP
clients.  The resource URI pattern is ``student://{student_id}/profile``.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from db.models.student import Student
from db.postgres import get_session
from shared.logger import get_logger
from sqlalchemy import select

logger = get_logger(__name__)


async def read_student_profile(student_id: str) -> Dict[str, Any]:
    """Return the full profile blob for a student.

    This is registered as an MCP Resource so that clients can subscribe
    to or read student data without invoking a tool.

    Args:
        student_id: The external student identifier.

    Returns:
        A JSON-serialisable dict with the student profile.
    """
    logger.info("Reading student profile resource", extra={"student_id": student_id})

    async with get_session() as session:
        result = await session.execute(
            select(Student).where(Student.student_id == student_id)
        )
        student = result.scalar_one_or_none()

    if student is None:
        return {"error": "student_not_found", "student_id": student_id}

    profile = student.to_dict()
    logger.debug("Student profile resource loaded: %s", student_id)
    return profile
