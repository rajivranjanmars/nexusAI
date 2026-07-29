"""
MCP Tools — Student operations.

Exposes ``get_student`` and ``update_student`` as MCP tools that delegate
to the database layer.  On mutations, the semantic response cache is
invalidated to prevent serving stale data.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import select

from db.models.student import Student
from db.postgres import get_session
from cache_module.semantic_cache import invalidate_cache
from shared.logger import get_logger

logger = get_logger(__name__)


async def get_student(student_id: str) -> Dict[str, Any]:
    """Retrieve a student record by their unique student ID.

    Args:
        student_id: The external student identifier.

    Returns:
        A dict representation of the student, or an error dict if not found.
    """
    logger.info("get_student called", extra={"student_id": student_id})

    async with get_session() as session:
        result = await session.execute(
            select(Student).where(Student.student_id == student_id)
        )
        student = result.scalar_one_or_none()

    if student is None:
        logger.warning("Student not found: %s", student_id)
        return {"error": "student_not_found", "student_id": student_id}

    student_dict = student.to_dict()

    if student_dict.get("enrollment_status") == "Not Active":
        academic_fields = [
            "gpa", "attendance", "total_classes_held", "classes_attended",
            "total_semester_classes", "fee_status", "next_exam", "active_courses"
        ]
        for field in academic_fields:
            student_dict[field] = "Not applicable for non-enrolled users"

    return student_dict


async def update_student(
    student_id: str,
    updates: Dict[str, Any],
    app_id: str | None = None,
) -> Dict[str, Any]:
    """Validate and apply updates to a student record.

    Writes an audit log entry alongside the update.

    Args:
        student_id: The external student identifier.
        updates: A dict of field names → new values.
        app_id: Optional originating app identifier for scoped cache invalidation.

    Returns:
        The updated student dict, or an error dict.
    """
    logger.info(
        "update_student called",
        extra={"student_id": student_id, "fields": list(updates.keys())},
    )

    # Whitelist of updatable columns
    allowed_fields = {
        "first_name",
        "last_name",
        "email",
        "program",
        "enrollment_status",
        "gpa",
        "notes",
    }
    invalid = set(updates.keys()) - allowed_fields
    if invalid:
        return {"error": "invalid_fields", "fields": list(invalid)}

    async with get_session() as session:
        result = await session.execute(
            select(Student).where(Student.student_id == student_id)
        )
        student = result.scalar_one_or_none()

        if student is None:
            logger.warning("Student not found for update: %s", student_id)
            return {"error": "student_not_found", "student_id": student_id}

        for field, value in updates.items():
            setattr(student, field, value)

        # Ensure the instance is refreshed so accessing any expired or
        # lazily-loaded attributes does not attempt IO outside the
        # session/greenlet context.
        await session.flush()
        await session.refresh(student)
        updated = student.to_dict()

    logger.info("Student updated successfully: %s", student_id)

    # Invalidate cached responses that depend on student data.
    try:
        await invalidate_cache(
            app_id=app_id,
            mutation_type="student_update",
            actor_id=student_id,
        )
    except Exception as exc:
        logger.warning("Cache invalidation after student update failed: %s", exc)

    return updated
