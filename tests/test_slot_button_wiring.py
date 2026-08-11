"""Tests for clarification slot button wiring in chat pipeline."""

import pytest

from backend_proxy.main import _resolve_helper_buttons, _normalise_chat_downstream
from shared.helper_buttons import GO_BACK_BUTTON


def test_resolve_helper_buttons_with_pending_clarification():
    """With a pending clarification, return slot buttons for the axis."""
    clarification = {
        "pending": True,
        "axis": "program",
        "options": ["mba", "bba", "bca"],
        "asked_for": "program selection",
    }
    buttons = _resolve_helper_buttons(
        "general",
        raw_lead_progress=None,
        user_input="",
        clarification=clarification,
    )

    assert len(buttons) == 4  # 3 options + GO_BACK
    assert buttons[0]["id"] == "slot_program_mba"
    assert buttons[0]["value"] == "mba"
    assert buttons[1]["id"] == "slot_program_bba"
    assert buttons[1]["value"] == "bba"
    assert buttons[2]["id"] == "slot_program_bca"
    assert buttons[2]["value"] == "bca"
    assert buttons[3] == GO_BACK_BUTTON


def _baseline(workflow="general", lead_progress=None):
    """What _resolve_helper_buttons returns when no clarification is in play."""
    return _resolve_helper_buttons(
        workflow,
        raw_lead_progress=lead_progress,
        user_input="",
        clarification=None,
    )


@pytest.mark.parametrize(
    "clarification",
    [
        None,
        {"pending": True},                                          # no axis/options
        {"pending": False, "axis": "program", "options": ["mba"]},   # not pending
        {"pending": True, "axis": "program", "options": []},         # no options
        {"pending": True, "axis": "", "options": ["mba"]},           # blank axis
        "not-a-dict",
    ],
)
def test_resolve_helper_buttons_falls_through(clarification):
    """Anything that is not a well-formed pending clarification is ignored.

    Asserting equality with the no-clarification baseline is what catches a
    slot button leaking into the normal path; an isinstance check would not.
    """
    buttons = _resolve_helper_buttons(
        "general",
        raw_lead_progress=None,
        user_input="",
        clarification=clarification,
    )

    assert buttons == _baseline()
    assert not any(b.get("id", "").startswith("slot_") for b in buttons)


def test_resolve_helper_buttons_lead_capture_with_no_clarification():
    """workflow=lead_capture with no clarification returns lead form buttons."""
    buttons = _baseline("lead_capture", {"step": "first_name"})

    assert GO_BACK_BUTTON in buttons
    assert not any(b.get("id", "").startswith("slot_") for b in buttons)


def test_normalise_chat_downstream_passes_clarification():
    """_normalise_chat_downstream passes through clarification from downstream dict."""
    downstream = {
        "response": "test response",
        "workflow": "general",
        "cache_hit": False,
        "clarification": {
            "pending": True,
            "axis": "program",
            "options": ["mba", "bba"],
        },
    }

    result = _normalise_chat_downstream(downstream)

    assert result["clarification"] == downstream["clarification"]
    assert result["clarification"]["pending"] is True
    assert result["clarification"]["axis"] == "program"


def test_normalise_chat_downstream_no_clarification():
    """_normalise_chat_downstream yields None when clarification is absent."""
    downstream = {
        "response": "test response",
        "workflow": "general",
        "cache_hit": False,
    }

    result = _normalise_chat_downstream(downstream)

    assert result["clarification"] is None


def test_normalise_chat_downstream_fallback_dict_has_clarification():
    """_normalise_chat_downstream fallback dict includes clarification=None."""
    # Test with a non-dict, non-string value to trigger fallback
    downstream = 12345

    result = _normalise_chat_downstream(downstream)

    assert "clarification" in result
    assert result["clarification"] is None
