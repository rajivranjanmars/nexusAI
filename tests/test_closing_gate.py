"""Tests for the closing-offer gate.

The gate exists because frequency is a cross-turn property: prose can only say
"always offer" (robotic) or "never offer" (curt). These tests pin the cases
where the answer changes between consecutive turns.
"""

import pytest

from llm.prompt_builder import (
    _closing_instruction,
    _last_assistant_turn,
    closing_allowed,
)
from orchestration.workflow_config import WorkflowResponseConfig


def _config(strategy="answer"):
    return WorkflowResponseConfig(
        default_style="conversational",
        include_sources=True,
        max_rag_chars=8000,
        history_limit=6,
        strategy=strategy,
    )


HIGH = 0.9
LOW = 0.1


def test_last_assistant_turn_picks_most_recent():
    history = "User: hi\nYou: hello there\nUser: fees?\nYou: MBA is 5L"
    assert _last_assistant_turn(history) == "MBA is 5L"


def test_last_assistant_turn_empty_when_no_prior_reply():
    assert _last_assistant_turn("User: hi") == ""
    assert _last_assistant_turn("") == ""
    assert _last_assistant_turn("(no prior conversation)") == ""


def test_allowed_on_a_clean_answer_turn():
    history = "User: hi\nYou: The MBA fee is 5L.\nUser: and the duration?"
    assert closing_allowed(history, _config(), HIGH) is True


def test_allowed_on_the_very_first_turn():
    assert closing_allowed("(no prior conversation)", _config(), HIGH) is True


@pytest.mark.parametrize(
    "previous_reply",
    [
        "The fee is 5L. Let me know if you need anything else!",
        "Duration is 2 years. Feel free to ask more.",
        "Happy to help with admissions.",
        "I'm here to help.",
        "Do you have further questions?",
    ],
)
def test_suppressed_when_previous_turn_already_offered(previous_reply):
    """Back-to-back offers are exactly what reads as a machine."""
    history = f"User: fees?\nYou: {previous_reply}\nUser: duration?"
    assert closing_allowed(history, _config(), HIGH) is False


def test_suppressed_when_previous_turn_ended_in_a_question():
    history = "User: fees?\nYou: Which programme are you asking about?\nUser: MBA"
    assert closing_allowed(history, _config(), HIGH) is False


def test_suppressed_for_elicit_workflows():
    """lead_capture is filling a form; an open offer derails it."""
    history = "User: hi\nYou: What is your name?"
    assert closing_allowed(history, _config(strategy="elicit"), HIGH) is False


def test_suppressed_when_this_turn_ends_in_a_clarifying_question():
    """Low confidence means we ask rather than answer — no offer on top."""
    history = "User: hi\nYou: The MBA fee is 5L."
    assert closing_allowed(history, _config(), LOW) is False


def test_alternates_across_consecutive_turns():
    """The property the gate exists for: offer, then don't, then offer again."""
    cfg = _config()

    turn1 = "(no prior conversation)"
    assert closing_allowed(turn1, cfg, HIGH) is True

    # Model took the offer, so the next turn must not repeat it.
    turn2 = "User: fees?\nYou: MBA is 5L. Anything else you'd like to know?"
    assert closing_allowed(turn2, cfg, HIGH) is False

    # Model stayed plain, so an offer is available again.
    turn3 = "User: duration?\nYou: Two years, full time."
    assert closing_allowed(turn3, cfg, HIGH) is True


def test_instruction_text_differs_by_verdict():
    allowed = _closing_instruction(True)
    denied = _closing_instruction(False)

    assert "CLOSING:" in allowed and "CLOSING:" in denied
    assert "never a stock sign-off" in allowed
    assert "Do not offer further help" in denied
    assert allowed != denied


def test_marker_match_is_case_insensitive():
    history = "User: fees?\nYou: MBA is 5L. LET ME KNOW if you need more."
    assert closing_allowed(history, _config(), HIGH) is False
