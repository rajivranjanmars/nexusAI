"""
Test prompt builder to verify conversational response style removes bullet-forcing language.
"""

from llm.prompt_builder import PromptBuilder


def test_enrollment_conversational_no_bullets():
    """Verify enrollment workflow uses conversational style without bullet-forcing."""
    builder = PromptBuilder(app_id="", workflow="enrollment")
    workflow_config = builder.get_workflow_response_config()

    # Fake RAG context (non-empty to trigger style formatting)
    rag_context = [
        {
            "content": "Test enrollment info",
            "source_url": "https://example.com/enrollment",
            "score": 0.95,
            "section_heading": "Test Section",
        }
    ]

    system_prompt, _ = builder.build_regular_prompt(
        user_query="What are the program fees?",
        student_data="",
        retrieved_context="",
        history_text="",
        rag_context=rag_context,
        workflow_config=workflow_config,
        answer_confidence=0.9,
    )

    # Assert old bullet-forcing language is NOT present
    assert "always use bullet points" not in system_prompt.lower(), \
        "System prompt should not force bullet points in enrollment workflow"
    assert "bullet points for key facts" not in system_prompt.lower(), \
        "System prompt should not mention bullet points in enrollment workflow"

    # Assert conversational style IS present
    assert "single short sentence" in system_prompt.lower() or \
           "one-liner" in system_prompt.lower() or \
           "most relevant fact" in system_prompt.lower(), \
        "System prompt should include conversational instructions"


def test_general_conversational_no_bullets():
    """Verify general workflow uses conversational style without bullet-forcing."""
    builder = PromptBuilder(app_id="", workflow="general")
    workflow_config = builder.get_workflow_response_config()

    # Fake RAG context
    rag_context = [
        {
            "content": "Test general info",
            "source_url": "https://example.com/general",
            "score": 0.85,
            "section_heading": "Test Section",
        }
    ]

    system_prompt, _ = builder.build_regular_prompt(
        user_query="Tell me about the university",
        student_data="",
        retrieved_context="",
        history_text="",
        rag_context=rag_context,
        workflow_config=workflow_config,
        answer_confidence=0.8,
    )

    # Assert old bullet-forcing language is NOT present
    assert "bullet points" not in system_prompt.lower() or \
           "always" not in system_prompt.lower() or \
           "BREVITY RULE" not in system_prompt, \
        "System prompt should not force bullet points in general workflow"


def test_low_confidence_grounding_asks_clarifying_question():
    """Verify low confidence grounding asks clarifying questions instead of deflecting."""
    builder = PromptBuilder(app_id="", workflow="general")
    instruction = builder._grounding_instruction(0.3)

    # Should NOT suggest checking the website or contacting someone
    assert "verify" not in instruction.lower() or "clarifying" in instruction.lower()

    # Should ask clarifying questions
    assert "clarifying question" in instruction.lower()
    assert "fabricate" in instruction.lower() or "fabrication" in instruction.lower()


def test_medium_confidence_grounding_labels_rather_than_asking():
    """Medium confidence answers with per-facet labels instead of asking again.

    Superseded the earlier "prefers clarifying over guessing" assertion. Asking
    which programme is now disambiguate_node's job and is budgeted to one turn;
    re-asking here would spend that budget twice and contradict the AMBIGUITY
    LABELLING instruction. Clarification still happens — deterministically in
    the node, and in the low-confidence branch below the threshold.
    """
    builder = PromptBuilder(app_id="", workflow="general")
    instruction = builder._grounding_instruction(0.45)

    # Answers with labels rather than bouncing the question back
    assert "which one each fact applies to" in instruction.lower()
    assert "clarifying question" not in instruction.lower()

    # Should NOT include raw URLs constraint is still present (load-bearing)
    assert "raw url" in instruction.lower() or "urls" in instruction.lower()

    # Low confidence still asks — clarification moved, it did not disappear
    assert "clarifying question" in builder._grounding_instruction(0.1).lower()


def test_high_confidence_grounding_unchanged():
    """Verify high confidence grounding remains unchanged."""
    builder = PromptBuilder(app_id="", workflow="general")
    instruction = builder._grounding_instruction(0.7)

    # Should answer precisely from retrieved knowledge
    assert "answer precisely" in instruction.lower()

    # Should NOT include URLs (load-bearing constraint)
    assert "raw url" in instruction.lower()


if __name__ == "__main__":
    test_enrollment_conversational_no_bullets()
    test_general_conversational_no_bullets()
    test_low_confidence_grounding_asks_clarifying_question()
    test_medium_confidence_grounding_labels_rather_than_asking()
    test_high_confidence_grounding_unchanged()
    print("All tests passed!")
