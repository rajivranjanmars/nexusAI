"""
MCP Tools — Workflow detection.

Exposes ``detect_workflow`` which delegates intent classification to the
``llm.classifier`` module.
"""

from __future__ import annotations

from llm.classifier import classify_intent
from shared.logger import get_logger

logger = get_logger(__name__)


async def detect_workflow(user_input: str, student_id: str) -> str:
    """Classify the user's intent and return the appropriate workflow name.

    Args:
        user_input: The raw message from the student.
        student_id: Student identifier (logged for traceability).

    Returns:
        A workflow name string: ``"grade_inquiry"``, ``"enrollment"``,
        or ``"general"``.
    """
    logger.info(
        "detect_workflow called",
        extra={"student_id": student_id},
    )

    workflow, _usage = await classify_intent(user_input)

    logger.info(
        "Workflow detected: %s for student %s",
        workflow,
        student_id,
    )
    return workflow
