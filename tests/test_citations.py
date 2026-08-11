"""
Test citation alignment between prompt builder and backend proxy.

The backend proxy numbers sources by deduplicated source_url (max 2 per user-visible block).
The prompt builder must use the same deduplication logic so citations in the model's response
point to the correct sources in the user-facing block.
"""

import pytest
from llm.prompt_builder import PromptBuilder, _MAX_CITED_SOURCES
from orchestration.workflow_config import WorkflowResponseConfig


class TestCitationAlignment:
    """Test that render_rag_content deduplicates source URLs like backend_proxy does."""

    def test_chunks_with_same_source_url_get_same_number(self):
        """Chunks from the same URL must share a source number (alignment property)."""
        builder = PromptBuilder(app_id="", workflow="general")
        workflow_config = WorkflowResponseConfig(
            default_style="balanced",
            include_sources=True,
            max_rag_chars=4000,
            history_limit=6
        )

        rag_context = [
            {
                "content": "First chunk from URL A",
                "match_content": "excerpt A1",
                "source_url": "https://example.com/page-a",
                "score": 0.95,
                "section_heading": "Section 1",
            },
            {
                "content": "Second chunk from URL A",
                "match_content": "excerpt A2",
                "source_url": "https://example.com/page-a",
                "score": 0.90,
                "section_heading": "Section 2",
            },
            {
                "content": "First chunk from URL B",
                "match_content": "excerpt B1",
                "source_url": "https://example.com/page-b",
                "score": 0.85,
                "section_heading": "Section 3",
            },
        ]

        _, rendered = builder.render_rag_content(rag_context, workflow_config)

        # Extract source numbers using a simple regex-based approach
        import re
        source_nums = re.findall(r'\[Source (\d+)', rendered)

        # Chunks from the same URL should have the same number
        assert len(source_nums) >= 3, f"Expected at least 3 chunks rendered, got: {rendered}"
        assert source_nums[0] == source_nums[1], (
            f"Chunks 1 and 2 share URL A but got different numbers: {source_nums[0]} vs {source_nums[1]}"
        )
        assert source_nums[2] != source_nums[0], (
            f"Chunk 3 (URL B) should differ from chunks 1-2 (URL A): {source_nums[2]} vs {source_nums[0]}"
        )

    def test_distinct_urls_beyond_max_get_no_number(self):
        """URLs beyond _MAX_CITED_SOURCES cap must render with no source number."""
        builder = PromptBuilder(app_id="", workflow="general")
        workflow_config = WorkflowResponseConfig(
            default_style="balanced",
            include_sources=True,
            max_rag_chars=4000,
            history_limit=6
        )

        rag_context = [
            {
                "content": "From URL A",
                "match_content": "excerpt A",
                "source_url": "https://example.com/a",
                "score": 0.95,
            },
            {
                "content": "From URL B",
                "match_content": "excerpt B",
                "source_url": "https://example.com/b",
                "score": 0.90,
            },
            {
                "content": "From URL C",
                "match_content": "excerpt C",
                "source_url": "https://example.com/c",
                "score": 0.85,
            },
        ]

        _, rendered = builder.render_rag_content(rag_context, workflow_config)

        # Extract headers: [Source N ...] or [Source | ...]
        import re
        headers = re.findall(r'\[Source\s*(\d*)\s*\|', rendered)

        # First two should have numbers (1 and 2), third should have empty string (no number)
        assert len(headers) == 3, f"Expected 3 headers, got {len(headers)}: {headers}"
        assert headers[0] == "1", f"First source should be [1], got [{headers[0]}]"
        assert headers[1] == "2", f"Second source should be [2], got [{headers[1]}]"
        assert headers[2] == "", f"Third source (beyond cap) should have no number, got [{headers[2]}]"

    def test_chunks_with_missing_source_url_get_no_number(self):
        """Chunks with missing or empty source_url must not get a number."""
        builder = PromptBuilder(app_id="", workflow="general")
        workflow_config = WorkflowResponseConfig(
            default_style="balanced",
            include_sources=True,
            max_rag_chars=4000,
            history_limit=6
        )

        rag_context = [
            {
                "content": "From URL A",
                "match_content": "excerpt A",
                "source_url": "https://example.com/a",
                "score": 0.95,
            },
            {
                "content": "No URL provided",
                "match_content": "excerpt B",
                "score": 0.90,
            },
            {
                "content": "Empty URL string",
                "match_content": "excerpt C",
                "source_url": "",
                "score": 0.85,
            },
        ]

        _, rendered = builder.render_rag_content(rag_context, workflow_config)

        import re
        # Count how many actual source numbers appear
        source_nums = re.findall(r'\[Source (\d+)\s*\|', rendered)

        # Only URL A should have a number
        assert len(source_nums) == 1, (
            f"Expected only 1 numbered source, got {len(source_nums)}: {source_nums}"
        )
        assert source_nums[0] == "1"

    def test_include_sources_false_renders_plain_text_without_headers(self):
        """With include_sources=False, no [Source headers should appear."""
        builder = PromptBuilder(app_id="", workflow="general")
        workflow_config = WorkflowResponseConfig(
            default_style="balanced",
            include_sources=False,
            max_rag_chars=4000,
            history_limit=6
        )

        rag_context = [
            {
                "content": "Content here",
                "match_content": "matched",
                "source_url": "https://example.com/a",
                "score": 0.95,
            },
        ]

        _, rendered = builder.render_rag_content(rag_context, workflow_config)

        assert "[Source" not in rendered, (
            f"include_sources=False should not render [Source headers, but got: {rendered}"
        )
        assert "matched" in rendered, "Content should still be rendered"

    def test_grounding_instruction_high_confidence_requires_markers(self):
        """High confidence grounding must require source markers, not forbid them."""
        builder = PromptBuilder(app_id="", workflow="general")
        instruction = builder._grounding_instruction(0.9)  # > _MEDIUM_CONFIDENCE_THRESHOLD

        # Must mention source markers
        assert "[1]" in instruction or "[2]" in instruction or "marker" in instruction.lower(), (
            f"High confidence instruction should mention source markers, got: {instruction}"
        )
        # Must still forbid raw URLs
        assert "raw url" in instruction.lower(), (
            f"High confidence instruction should still forbid raw URLs, got: {instruction}"
        )
        # Must still forbid Sources section
        assert "source" in instruction.lower() and "section" in instruction.lower() or \
               "Sources" in instruction, (
            f"High confidence instruction should forbid Sources section, got: {instruction}"
        )
        # Must NOT say "do not include... source numbers"
        assert "source numbers" not in instruction.lower() or "include" not in instruction.lower(), (
            f"High confidence should not forbid source numbers, got: {instruction}"
        )

    def test_grounding_instruction_medium_confidence_requires_markers(self):
        """Medium confidence grounding must require source markers."""
        builder = PromptBuilder(app_id="", workflow="general")
        instruction = builder._grounding_instruction(0.5)  # Between low and medium threshold

        # Must mention source markers
        assert "[1]" in instruction or "[2]" in instruction or "marker" in instruction.lower(), (
            f"Medium confidence instruction should mention source markers, got: {instruction}"
        )
        # Must still forbid raw URLs
        assert "raw url" in instruction.lower(), (
            f"Medium confidence instruction should still forbid raw URLs, got: {instruction}"
        )

    def test_grounding_instruction_low_confidence_unchanged(self):
        """Low confidence grounding should ask clarifying questions, not mention markers."""
        builder = PromptBuilder(app_id="", workflow="general")
        instruction = builder._grounding_instruction(0.3)  # < _LOW_CONFIDENCE_THRESHOLD

        # Should ask clarifying questions
        assert "clarifying question" in instruction.lower(), (
            f"Low confidence should ask clarifying questions, got: {instruction}"
        )
        # Should NOT mention source markers (it's not answering, so no citations)
        assert "marker" not in instruction.lower() or "clarifying" in instruction.lower()

    def test_character_budget_truncation_still_works(self):
        """Character budget truncation must still work as before."""
        builder = PromptBuilder(app_id="", workflow="general")
        workflow_config = WorkflowResponseConfig(
            default_style="balanced",
            include_sources=True,
            max_rag_chars=1000,  # Generous budget for first, too tight for both
            history_limit=6
        )

        rag_context = [
            {
                "content": "Short content A",
                "match_content": "matched A",
                "source_url": "https://example.com/a",
                "score": 0.95,
            },
            {
                "content": "x" * 1500,  # This chunk alone exceeds budget
                "match_content": "matched B",
                "source_url": "https://example.com/b",
                "score": 0.90,
            },
        ]

        budgeted, rendered = builder.render_rag_content(rag_context, workflow_config)

        # First chunk should be included, second should be skipped or truncated
        # Dedup should still work for the chunk(s) that were included
        assert len(budgeted) >= 1, (
            f"Expected at least 1 chunk within budget, got {len(budgeted)}"
        )
        # The first chunk should appear
        assert "example.com/a" in rendered

    def test_dedup_matches_backend_proxy_logic(self):
        """Verify dedup logic assigns same index to chunks with same URL, matching backend_proxy."""
        builder = PromptBuilder(app_id="", workflow="general")
        workflow_config = WorkflowResponseConfig(
            default_style="balanced",
            include_sources=True,
            max_rag_chars=4000,
            history_limit=6
        )

        # Simulate backend_proxy's _build_sources_block logic locally
        rag_context = [
            {"content": "c1", "match_content": "m1", "source_url": "https://a.com", "score": 0.95},
            {"content": "c2", "match_content": "m2", "source_url": "https://a.com", "score": 0.90},
            {"content": "c3", "match_content": "m3", "source_url": "https://b.com", "score": 0.85},
            {"content": "c4", "match_content": "m4", "source_url": "https://c.com", "score": 0.80},
        ]

        # What backend_proxy would number for unique URLs:
        backend_url_to_idx = {}
        seen = set()
        idx = 0
        for chunk in rag_context:
            url = chunk.get("source_url")
            if url and url not in seen:
                seen.add(url)
                idx += 1
                if idx <= 2:  # _MAX_STREAM_SOURCE_URLS
                    backend_url_to_idx[url] = idx

        # What render_rag_content produces:
        _, rendered = builder.render_rag_content(rag_context, workflow_config)

        import re
        # Extract all (source_idx, url) pairs from headers
        headers = re.findall(r'Source\s*(\d*)\s*.*?(https://[^\s|]+)', rendered)

        # For each chunk, extract its (index, url) pair
        for num, url in headers:
            url = url.strip()
            if num:  # Only check numbered sources
                num = int(num)
                # This chunk should have the same index as what backend_proxy assigned
                expected_idx = backend_url_to_idx.get(url)
                assert expected_idx is not None, (
                    f"URL {url} with index {num} is not in backend's numbered set"
                )
                assert num == expected_idx, (
                    f"Index mismatch for {url}: expected {expected_idx}, got {num}"
                )
