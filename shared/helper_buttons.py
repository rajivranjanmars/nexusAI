"""
Helpers for deterministic helper-button generation.

Two button-set modes:
  • Q&A mode  (any non-lead_capture workflow) — topic buttons + hb1_start_application
  • Lead form mode (lead_capture workflow)     — only hb_go_back
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from shared.logger import get_logger

logger = get_logger(__name__)

# ── Persistent button constants ──────────────────────────────────────────────

REQUEST_CALLBACK: dict[str, str] = {
    "id": "request_callback",
    "text": "Admission Query",
    # "callback" keyword ensures classifier naturally routes to lead_capture
    "value": "I'd like to request a callback.",
}

GO_BACK_BUTTON: dict[str, str] = {
    "id": "go_back",
    "text": "Other Query",
    # Must contain an EXIT_WORKFLOW_KEYWORDS phrase so it actually breaks the
    # lead_capture sticky lock (see detect_workflow_node.py), not just fall
    # through to the classifier — which never runs while the lock holds.
    "value": "go back, I have other questions",
}

# Typed or button-triggered phrases that break a sticky/in-progress workflow
# lock. Kept in sync with GO_BACK_BUTTON above — both must round-trip through
# the same check in detect_workflow_node.py.
EXIT_WORKFLOW_KEYWORDS: tuple[str, ...] = ("go back",)

# ── Q&A topic button configuration ───────────────────────────────────────────

_MAX_TOTAL_HELPER_BUTTONS = 2

_PROGRAM_HINTS: dict[str, str] = {
    "bca": "BCA",
    "bba": "BBA",
    "b.com": "B.Com",
    "bcom": "B.Com",
    "ba": "BA",
    "b.sc": "B.Sc",
    "bsc": "B.Sc",
    "mca": "MCA",
    "mba": "MBA",
    "m.com": "M.Com",
    "mcom": "M.Com",
    "ma": "MA",
    "m.sc": "M.Sc",
    "msc": "M.Sc",
    "dit": "DIT",
    "dbm": "DBM",
}

_TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "fees": ("fee", "fees", "cost", "costs", "tuition"),
    "eligibility": ("eligibility", "eligible", "qualification", "criteria"),
    "duration": ("duration", "semester", "semesters", "year", "years"),
    "specialisations": ("specialisation", "specialisations", "specialization", "specializations"),
    "placements": ("placement", "placements", "career", "jobs"),
    "admission_process": ("admission", "apply", "application", "enroll", "enrollment"),
    "syllabus": ("curriculum", "syllabus", "subjects"),
}

_PROGRAM_FOLLOW_UPS: dict[str, list[dict[str, str]]] = {
    "default": [
        {
            "id": "course_eligibility",
            "text": "Eligibility criteria",
            "value": "What is the eligibility criteria for this course?",
            "topics": {"eligibility"},
        },
        {
            "id": "course_duration",
            "text": "Course duration",
            "value": "What is the duration of this course?",
            "topics": {"duration"},
        },
        {
            "id": "course_admission_process",
            "text": "Admission process",
            "value": "What is the admission process for this course?",
            "topics": {"admission_process"},
        },
    ],
    "MBA": [
        {
            "id": "mba_specialisations",
            "text": "MBA specialisations",
            "value": "Tell me about the specialisations available in the MBA program.",
            "topics": {"specialisations"},
        },
        {
            "id": "mba_eligibility",
            "text": "MBA eligibility",
            "value": "What is the eligibility criteria for the MBA program?",
            "topics": {"eligibility"},
        },
        {
            "id": "mba_duration",
            "text": "MBA duration",
            "value": "What is the duration of the MBA program?",
            "topics": {"duration"},
        },
        {
            "id": "mba_placements",
            "text": "MBA placements",
            "value": "Tell me about placement support for MBA students.",
            "topics": {"placements"},
        },
    ],
    "MCA": [
        {
            "id": "mca_specialisations",
            "text": "MCA curriculum",
            "value": "Tell me about the curriculum or core subjects in the MCA program.",
            "topics": {"syllabus"},
        },
        {
            "id": "mca_eligibility",
            "text": "MCA eligibility",
            "value": "What is the eligibility criteria for the MCA program?",
            "topics": {"eligibility"},
        },
        {
            "id": "mca_duration",
            "text": "MCA duration",
            "value": "What is the duration of the MCA program?",
            "topics": {"duration"},
        },
    ],
    "BCA": [
        {
            "id": "bca_subjects",
            "text": "BCA subjects",
            "value": "Tell me about the main subjects covered in the BCA program.",
            "topics": {"syllabus"},
        },
        {
            "id": "bca_eligibility",
            "text": "BCA eligibility",
            "value": "What is the eligibility criteria for the BCA program?",
            "topics": {"eligibility"},
        },
        {
            "id": "bca_duration",
            "text": "BCA duration",
            "value": "What is the duration of the BCA program?",
            "topics": {"duration"},
        },
    ],
}


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_lead_progress_payload(
    lead_progress: Any,
    *,
    user_input: str = "",
) -> Any:
    """Return a lead-progress payload without duplicated public helper buttons."""

    if not isinstance(lead_progress, dict):
        return lead_progress

    normalized = dict(lead_progress)
    normalized.pop("helper_buttons", None)
    return normalized


# ── Lead form step-specific buttons ───────────────────────────────────────────

_PROGRAM_LEVEL_BUTTONS: list[dict[str, str]] = [
    {"id": "program_level_bachelors", "text": "🎓 Bachelors", "value": "Bachelors"},
    {"id": "program_level_masters",   "text": "🎓 Masters",   "value": "Masters"},
    {"id": "program_level_diploma",   "text": "📜 Diploma",   "value": "Diploma"},
]

_PROGRAM_NAME_BUTTONS: dict[str, list[dict[str, str]]] = {
    "bachelors": [
        {"id": "prog_bca",   "text": "BCA",   "value": "BCA"},
        {"id": "prog_bba",   "text": "BBA",   "value": "BBA"},
        {"id": "prog_bcom",  "text": "B.Com", "value": "B.Com"},
        {"id": "prog_ba",    "text": "BA",    "value": "BA"},
        {"id": "prog_bsc",   "text": "B.Sc",  "value": "B.Sc"},
        {"id": "prog_blis",  "text": "BLIS",  "value": "BLIS"},
        {"id": "prog_other", "text": "Other", "value": "Other"},
    ],
    "masters": [
        {"id": "prog_mca",   "text": "MCA",   "value": "MCA"},
        {"id": "prog_mba",   "text": "MBA",   "value": "MBA"},
        {"id": "prog_mcom",  "text": "M.Com", "value": "M.Com"},
        {"id": "prog_msc",   "text": "M.Sc",  "value": "M.Sc"},
        {"id": "prog_ma",    "text": "MA",    "value": "MA"},
        {"id": "prog_mlis",  "text": "MLIS",  "value": "MLIS"},
        {"id": "prog_other", "text": "Other", "value": "Other"},
    ],
    "diploma": [
        {"id": "prog_dba",   "text": "DBA",   "value": "DBA"},
        {"id": "prog_dca",   "text": "DCA",   "value": "DCA"},
        {"id": "prog_dlis",  "text": "DLIS",  "value": "DLIS"},
        {"id": "prog_other", "text": "Other", "value": "Other"},
    ],
}

_STATE_BUTTONS: list[dict[str, str]] = [
    {"id": "state_punjab",       "text": "Punjab",          "value": "Punjab"},
    {"id": "state_haryana",      "text": "Haryana",         "value": "Haryana"},
    {"id": "state_up",           "text": "Uttar Pradesh",   "value": "Uttar Pradesh"},
    {"id": "state_delhi",        "text": "Delhi",           "value": "Delhi"},
    {"id": "state_rajasthan",    "text": "Rajasthan",       "value": "Rajasthan"},
    {"id": "state_maharashtra",  "text": "Maharashtra",     "value": "Maharashtra"},
    {"id": "state_mp",           "text": "Madhya Pradesh",  "value": "Madhya Pradesh"},
    {"id": "state_bihar",        "text": "Bihar",           "value": "Bihar"},
    {"id": "state_karnataka",    "text": "Karnataka",       "value": "Karnataka"},
    {"id": "state_other",        "text": "Others",           "value": "Other"},
]


def _resolve_program_level(lead_progress: dict) -> str:
    """Extract the chosen program level from lead_progress fields."""
    # Check fields_collected to see if program_level is already captured.
    # The actual value may appear in the lead_progress payload directly,
    # or we can infer from conversation context stored during upsert.
    level = str(lead_progress.get("program_level", "")).strip().lower()
    if level:
        return level

    # Fallback: scan collected field names for a level hint
    collected = lead_progress.get("fields_collected", [])
    if "program_level" not in collected:
        return ""

    # If the field is collected but we don't have the value in lead_progress,
    # return empty — the buttons won't show (safe fallback to free text).
    return ""


def build_lead_form_buttons(
    lead_progress: Any = None,
) -> list[dict[str, str]]:
    """Return step-specific option buttons during lead form collection.

    Args:
        lead_progress: The current lead_progress dict from the LLM stream.

    Returns:
        A list of button dicts. Always includes Go Back as the last button.
        Steps with predefined choices also include the option buttons.
    """
    buttons: list[dict[str, str]] = []

    if not isinstance(lead_progress, dict):
        return [GO_BACK_BUTTON]

    current_step = str(lead_progress.get("current_step", "")).strip().lower()
    status = str(lead_progress.get("status", "")).strip().lower()

    if status == "completed":
        return []

    if current_step == "program_level":
        buttons.extend(_PROGRAM_LEVEL_BUTTONS)

    elif current_step == "program_names":
        level = _resolve_program_level(lead_progress)
        level_buttons = _PROGRAM_NAME_BUTTONS.get(level, [])
        if level_buttons:
            buttons.extend(level_buttons)

    elif current_step == "state":
        buttons.extend(_STATE_BUTTONS)

    # Go Back is always last
    buttons.append(GO_BACK_BUTTON)
    return buttons


def build_public_helper_buttons(
    *,
    lead_progress: Any,
    user_input: str = "",
) -> list[dict[str, str]]:
    """Build user-facing helper buttons for Q&A mode.

    Total buttons (contextual topic buttons + the persistent
    ``request_callback``) are capped at ``_MAX_TOTAL_HELPER_BUTTONS``.
    """

    add_callback = True
    if isinstance(lead_progress, dict):
        if lead_progress.get("status") == "completed" or lead_progress.get("percent_complete", 0) >= 100:
            add_callback = False

    max_topic_buttons = _MAX_TOTAL_HELPER_BUTTONS - (1 if add_callback else 0)

    topic_buttons: list[dict[str, str]] = []

    if isinstance(lead_progress, dict) and max_topic_buttons > 0:
        detected_program = _detect_program(user_input)
        detected_topics = _detect_topics(user_input)

        if detected_program or detected_topics:
            candidates = _PROGRAM_FOLLOW_UPS.get(
                detected_program or "",
                _PROGRAM_FOLLOW_UPS["default"],
            )

            seen_ids: set[str] = set()
            for candidate in candidates:
                candidate_topics = set(candidate.get("topics", set()))
                if candidate_topics & detected_topics:
                    continue
                button_id = str(candidate["id"])
                if button_id in seen_ids:
                    continue
                topic_buttons.append(
                    {
                        "id": button_id,
                        "text": str(candidate["text"]),
                        "value": str(candidate["value"]),
                    }
                )
                seen_ids.add(button_id)
                if len(topic_buttons) >= max_topic_buttons:
                    break

    # request_callback always appears last, unless the lead form is already completed
    if add_callback:
        topic_buttons.append(REQUEST_CALLBACK)

    return topic_buttons


def build_slot_buttons(
    axis: str,
    options: Sequence[str],
    *,
    max_options: int = 4,
) -> list[dict[str, str]]:
    """Return dynamically-derived option buttons for narrowing questions.

    Args:
        axis: The axis/category name (e.g., "program"). Empty/blank returns [].
        options: List of option values (already ranked by relevance).
            Blank entries (after strip) are skipped. Case-insensitive
            duplicates are collapsed, keeping the first occurrence.
        max_options: Maximum number of option buttons to return (excluding Go Back).
            Defaults to 4.

    Returns:
        A list of button dicts with "id", "text", and "value" keys.
        Always includes GO_BACK_BUTTON as the last entry.
        Returns [] if axis is empty/blank or options is empty.
    """
    # Empty axis or options -> no buttons
    axis_str = str(axis).strip() if axis else ""
    if not axis_str:
        return []
    if not options:
        return []

    buttons: list[dict[str, str]] = []
    seen: set[str] = set()  # Track case-insensitive duplicates

    for option in options:
        if len(buttons) >= max_options:
            break

        option_str = str(option).strip()
        if not option_str:  # Skip blank options
            continue

        # Case-insensitive deduplication
        lower_option = option_str.lower()
        if lower_option in seen:
            continue
        seen.add(lower_option)

        # Build slug: lowercase, replace non-alphanumeric with "_", strip leading/trailing "_"
        slug = re.sub(r"[^a-z0-9]+", "_", lower_option).strip("_")

        button: dict[str, str] = {
            "id": f"slot_{axis_str.lower()}_{slug}",
            "text": option_str.upper(),
            "value": option_str,
        }
        buttons.append(button)

    # Always append Go Back as the last entry
    buttons.append(GO_BACK_BUTTON)
    return buttons


# ── Private helpers ───────────────────────────────────────────────────────────

def _detect_program(user_input: str) -> str | None:
    """Detect a specific program name from the latest user message."""

    lowered = f" {str(user_input or '').lower()} "
    for token, program_name in _PROGRAM_HINTS.items():
        pattern = rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
        if re.search(pattern, lowered):
            return program_name
    return None


def _detect_topics(user_input: str) -> set[str]:
    """Detect the admissions topics already mentioned in the latest user message."""

    lowered = str(user_input or "").lower()
    topics: set[str] = set()
    for topic, patterns in _TOPIC_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            topics.add(topic)
    return topics


def _demo() -> None:
    # In-progress lead + detected program -> at most 2 buttons total (1 topic + callback)
    buttons = build_public_helper_buttons(lead_progress={"status": "in_progress"}, user_input="tell me about MBA fees")
    assert len(buttons) <= 2, buttons
    assert buttons[-1]["id"] == "request_callback", buttons

    # Completed lead form -> no callback, up to 2 topic buttons allowed
    buttons = build_public_helper_buttons(lead_progress={"status": "completed"}, user_input="tell me about MBA fees")
    assert len(buttons) <= 2, buttons
    assert all(b["id"] != "request_callback" for b in buttons), buttons

    # No program/topic detected -> just the callback
    buttons = build_public_helper_buttons(lead_progress={"status": "in_progress"}, user_input="hello")
    assert buttons == [REQUEST_CALLBACK], buttons

    # build_slot_buttons: basic case with three programs
    buttons = build_slot_buttons("program", ["mba", "bba", "bca"])
    assert len(buttons) == 4, buttons  # 3 options + GO_BACK
    assert buttons[0]["id"] == "slot_program_mba", buttons
    assert buttons[0]["text"] == "MBA", buttons
    assert buttons[0]["value"] == "mba", buttons
    assert buttons[-1] == GO_BACK_BUTTON, buttons


if __name__ == "__main__":
    _demo()
    print("ok")
