"""
Modular prompt builder for app-specific prompt hierarchy.

This module handles the construction of system and user prompts based on:
1. App-level system prompts and persona
2. Workflow-specific response configurations
3. Context (RAG, conversation history, etc.)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from shared.logger import get_logger
from shared.prompt_loader import render_prompt
from shared.disambiguation import Axis, rank_axes
from orchestration.workflow_config import get_workflow_config, WorkflowResponseConfig
from db.app_registry import resolve_by_app_id

logger = get_logger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 0.35
_MEDIUM_CONFIDENCE_THRESHOLD = 0.60
# Must match backend_proxy.main._MAX_STREAM_SOURCE_URLS — the user only ever
# sees this many numbered sources, so the model must not cite beyond it.
_MAX_CITED_SOURCES = 2

# Phrases that mean the previous turn already offered further help. Offering
# again immediately is what makes a bot read as a machine.
_CLOSING_OFFER_MARKERS = (
    "anything else",
    "let me know",
    "feel free",
    "happy to help",
    "here to help",
    "further questions",
    "more questions",
)


def _last_assistant_turn(history_text: str) -> str:
    """Return the most recent assistant line from the rendered history.

    ``reasoner`` renders history as alternating ``User: ...`` / ``You: ...``
    lines. Empty string when there is no prior assistant turn.
    """
    for line in reversed((history_text or "").splitlines()):
        if line.startswith("You: "):
            return line[len("You: "):].strip()
    return ""


def closing_allowed(
    history_text: str,
    workflow_config: WorkflowResponseConfig,
    answer_confidence: float,
) -> bool:
    """Whether this turn may end by inviting the user to ask something else.

    Frequency is a cross-turn property, so prose cannot control it: an
    instruction to offer help fires every turn and reads robotic, while an
    instruction never to offer reads curt. The decision belongs in code.
    """
    if workflow_config.strategy != "answer":
        # An elicit workflow is filling a form; an open offer derails it.
        return False
    if answer_confidence < _LOW_CONFIDENCE_THRESHOLD:
        # This turn ends in a clarifying question — don't stack an offer on it.
        return False

    last = _last_assistant_turn(history_text).casefold()
    if last.endswith("?"):
        return False
    return not any(marker in last for marker in _CLOSING_OFFER_MARKERS)


def _closing_instruction(allowed: bool) -> str:
    """Return the closing-line instruction for this turn."""
    if allowed:
        return (
            "CLOSING: you may end with one short, natural line inviting the user to "
            "ask about anything else. Phrase it in your own words for this specific "
            "conversation — never a stock sign-off. Omit it entirely if the answer "
            "already stands on its own.\n\n"
        )
    return (
        "CLOSING: end on the answer itself. Do not offer further help and do not "
        "list other topics you can cover.\n\n"
    )


class PromptBuilder:
    """Builds prompts using app and workflow configuration hierarchy."""
    
    def __init__(self, app_id: str = "", workflow: str = "general"):
        self.app_id = app_id
        self.workflow = workflow
        self.app_context = resolve_by_app_id(app_id) if app_id else None
        
    def get_app_system_prompt(self) -> Tuple[str, Dict[str, Any]]:
        """Get app-specific system prompt and persona configuration."""
        if not self.app_context or not self.app_context.app_config:
            return "", {}
        
        system_config = self.app_context.app_config.get("system_prompt", {})
        base_prompt = system_config.get("base", "")
        persona = system_config.get("persona", {})
        
        return base_prompt, persona
    
    def get_workflow_response_config(self) -> WorkflowResponseConfig:
        """Get response configuration for the current workflow."""
        config = get_workflow_config(self.app_id)
        
        # Get workflow-specific config, fallback to general
        workflow_config = config.workflow_response_config.get(
            self.workflow, 
            config.workflow_response_config.get("general")
        )
        
        # If no config found, create default
        if not workflow_config:
            from orchestration.workflow_config import WorkflowResponseConfig
            workflow_config = WorkflowResponseConfig(
                default_style="balanced",
                include_sources=True,
                max_rag_chars=4000,
                history_limit=6
            )
        
        return workflow_config
    
    def build_system_prompt(self, base_system: str) -> str:
        """Build system prompt using app and workflow configuration hierarchy."""
        app_system, persona = self.get_app_system_prompt()
        workflow_response_config = self.get_workflow_response_config()
        
        # Start with base system prompt
        system_prompt = base_system
        
        # Add app-specific system prompt (highest priority)
        if app_system:
            system_prompt = f"{app_system}\n\n{system_prompt}" if system_prompt else app_system
        
        # Add workflow-specific system suffix from app config
        app_system_suffix = ""
        if self.app_context and self.app_context.app_config:
            workflow_config_section = self.app_context.app_config.get("workflow_config", {}).get(self.workflow, {})
            app_system_suffix = workflow_config_section.get("system_suffix", "")

        effective_system_suffix = app_system_suffix or workflow_response_config.system_suffix or ""
        if effective_system_suffix:
            system_prompt += f"\n\n{effective_system_suffix}"
        
        # Add persona configuration
        if persona:
            persona_rules = "\n".join(f"- {rule}" for rule in persona.get("rules", []))
            persona_section = (
                f"\n\nAPP-SPECIFIC PERSONA & TONE:\n"
                f"You must adopt the following persona:\n"
                f"Name: {persona.get('name', 'Assistant')}\n"
                f"Tone: {persona.get('tone', 'Helpful and professional')}\n"
            )
            if persona_rules:
                persona_section += f"\nAdditional Rules:\n{persona_rules}\n"
            if persona.get("example_response"):
                persona_section += f"\nExample Response Style:\n{persona.get('example_response')}\n"
            
            system_prompt += persona_section
        
        return system_prompt
    
    def render_rag_content(
        self,
        rag_context: list[dict],
        workflow_config: WorkflowResponseConfig,
    ) -> Tuple[list[dict], str]:
        """Render hydrated website content within the configured character budget."""

        budgeted_chunks: list[dict] = []
        rendered_chunks: list[str] = []
        chars_used = 0

        # Build a map of source_url to deduplicated source index, matching backend_proxy's logic.
        source_url_to_idx: dict[str, int] = {}
        source_index = 0
        for chunk in rag_context:
            source_url = chunk.get("source_url", "")
            if source_url and source_url not in source_url_to_idx:
                if source_index < _MAX_CITED_SOURCES:
                    source_index += 1
                    source_url_to_idx[source_url] = source_index

        for chunk in rag_context:
            matched_text = str(chunk.get("match_content", "") or "").strip()
            hydrated_text = str(chunk.get("content", "") or "").strip()
            if not matched_text and not hydrated_text:
                continue

            heading = chunk.get("section_heading", "")
            source_url = chunk.get("source_url", "")
            confidence = float(chunk.get("score", 0.0) or 0.0)

            # Use deduplicated source index, or None if beyond cap or missing URL.
            source_idx = source_url_to_idx.get(source_url)

            header = "[Source"
            if source_idx:
                header += f" {source_idx}"
            if heading:
                header += f" | {heading}"
            if source_url:
                header += f" | {source_url}"
            header += f" | confidence: {confidence:.0%}]"

            body_parts = []
            if matched_text:
                body_parts.append(f"Matched excerpt:\n{matched_text}")
            if hydrated_text and hydrated_text != matched_text:
                body_parts.append(f"Additional context:\n{hydrated_text}")

            block_body = "\n\n".join(body_parts).strip()
            if not block_body:
                continue

            block = f"{header}\n{block_body}\n"
            plain_text = block_body
            if chars_used + len(block) > workflow_config.max_rag_chars:
                remaining = workflow_config.max_rag_chars - chars_used
                if remaining <= 200:
                    break
                visible_text = block_body[: max(0, remaining - len(header) - 4)]
                plain_text = visible_text
                block = f"{header}\n{visible_text}...\n"

            budgeted_chunks.append(chunk)
            chars_used += len(block)

            if workflow_config.include_sources:
                rendered_chunks.append(block.strip())
            else:
                rendered_chunks.append(plain_text.strip())

        return budgeted_chunks, "\n\n".join(rendered_chunks).strip()

    def _grounding_instruction(self, answer_confidence: float) -> str:
        """Return confidence-aware grounding instructions for the model."""

        if answer_confidence < _LOW_CONFIDENCE_THRESHOLD:
            return (
                "The retrieved knowledge does not contain a confident answer. "
                "Do not fabricate. Instead of suggesting the user verify elsewhere, ask one short, specific clarifying question "
                "to narrow down what they need (e.g., which programme, which specific detail, which year). "
                "Keep the question brief and constructive to move the conversation forward."
            )
        if answer_confidence < _MEDIUM_CONFIDENCE_THRESHOLD:
            return (
                "Answer from the retrieved knowledge where it clearly addresses the question. "
                #"Include the source marker (e.g., [1], [2]) when citing information; never invent a marker. "
                # Asking which programme is disambiguate_node's job, and it is
                # budgeted to one turn. Telling the model to ask here would spend
                # that budget a second time and contradict AMBIGUITY LABELLING.
                "If the evidence covers more than one programme or topic, say which one each fact "
                "applies to rather than asking the user to choose. "
                "Do not include raw URLs or a Sources section — the system will attach them automatically."
            )
        return (
            "Answer precisely from the retrieved knowledge below. "
            #"Include the source marker (e.g., [1], [2]) when citing information; never invent a marker. "
            "Do not include raw URLs or a Sources section — the system will attach them automatically. "
            "Do not add unsupported facts."
        )

    def build_strict_rag_prompt(
        self,
        user_query: str,
        rag_context: list[dict],
        workflow_config: WorkflowResponseConfig,
        answer_confidence: float,
        history_text: str = "",
    ) -> Tuple[str, str]:
        """Build prompts for strict RAG mode."""
        config = get_workflow_config(self.app_id)
        response_style = config.response_styles.get(workflow_config.default_style)
        
        rendered_chunks = []
        for idx, chunk in enumerate(rag_context, start=1):
            if workflow_config.include_sources:
                rendered_chunks.append(
                    f"[Chunk {idx}]\n"
                    f"Source: {chunk.get('source_url', '(unknown)')}\n"
                    f"Similarity: {chunk.get('score', 0.0):.3f}\n"
                    f"Content:\n{chunk.get('content', '')}"
                )
            else:
                rendered_chunks.append(
                    f"[Chunk {idx}]\n"
                    f"Content:\n{chunk.get('content', '')}"
                )
        
        system_prompt = (
            "You are an institutional assistant speaking directly to the user. "
            "Answer ONLY from the website content provided below. "
            "Do not use prior knowledge. Do not infer missing facts. "
            "Do not mention retrieved context, chunks, sources, database results, or provided text. "
            "Speak naturally using first-person assistant language like 'I' and 'I can help'. "
            "If the website content does not contain the answer, say briefly that you could not find that information. "
        )
        system_prompt += f"\n{self._grounding_instruction(answer_confidence)}\n"
        
        # Add style-specific formatting instructions
        if response_style and response_style.format_instructions:
            system_prompt += f"\n{response_style.format_instructions}\n"
        else:
            system_prompt += (
                "Format responses with:\n"
                "- Clear section headers\n"
                "- Bullet points for key information\n"
            )
        
        # Source URLs are appended by the proxy — do not ask the LLM to emit them inline.
        system_prompt += "- Do not include raw URLs or a Sources section in your reply.\n"

        system_prompt += "\n" + _closing_instruction(
            closing_allowed(history_text, workflow_config, answer_confidence)
        )

        user_prompt = (
            f"Question:\n{user_query}\n\n"
            f"Website content:\n\n{chr(10).join(rendered_chunks)}\n\n"
            "Write a concise, human answer grounded only in the website content above. "
            "Do not mention missing sources, retrieved information, or provided context."
        )
        
        return system_prompt, user_prompt
    
    def build_regular_prompt(
        self,
        user_query: str,
        student_data: str,
        retrieved_context: str,
        history_text: str,
        rag_context: list[dict],
        workflow_config: WorkflowResponseConfig,
        answer_confidence: float,
        actor_id: str = "",
        lead_progress: Optional[Dict[str, Any]] = None,
        verified_phone: str = "",
    ) -> Tuple[str, str]:
        """Build prompts for regular (non-strict RAG) mode."""
        workflow_prompt_ids = {
            "enrollment": "workflow_enrollment",
            "lead_capture": "workflow_lead_capture",
        }
        prompt_id = workflow_prompt_ids.get(self.workflow, "reasoning_synthesis")

        try:
            system_prompt, user_prompt = render_prompt(
                prompt_id,
                user_input=user_query,
                student_data=student_data,
                context=retrieved_context,
                actor_id=actor_id,
                lead_progress=json.dumps(lead_progress or {}, default=str),
                verified_phone=verified_phone or "",
            )
        except Exception:
            system_prompt, user_prompt = render_prompt(
                "reasoning_synthesis",
                user_input=user_query,
                student_data=student_data,
                context=retrieved_context,
                actor_id=actor_id,
                lead_progress=json.dumps(lead_progress or {}, default=str),
                verified_phone=verified_phone or "",
            )
        
        if rag_context:
            config = get_workflow_config(self.app_id)
            response_style = config.response_styles.get(workflow_config.default_style)
            
            base_instructions = (
                "\n\nYou have been given institutional knowledge context. Use ONLY this information\n"
                "to answer the question. Do NOT invent facts. If the information is insufficient,\n"
                "briefly say that you do not have that specific detail right now.\n\n"
                "CONVERSATION RULE: Never reference 'chunks', 'context', or 'database'.\n"
                "Speak naturally as if you inherently know this. Use first-person assistant language.\n\n"
            )
            base_instructions += f"{self._grounding_instruction(answer_confidence)}\n\n"
            
            # Add style-specific formatting instructions
            if response_style and response_style.format_instructions:
                base_instructions += f"{response_style.format_instructions}\n\n"
            else:
                # Conversational strategy is a workflow-contract concern, not a prompt-assembly concern.
                base_instructions += "BREVITY RULE: Answer concisely with the most relevant fact. Avoid unnecessary detail.\n\n"

            # Add ambiguity labeling instruction if facets remain unresolved
            try:
                cfg = get_workflow_config(self.app_id)
                dis = cfg.disambiguation
                axes = [Axis(n, p) for n, p in dis.axes]
                ranked = rank_axes(rag_context, axes)
                unresolved = [name for name, gain in ranked if gain > dis.min_gain]

                if unresolved:
                    axes_str = ", ".join(unresolved)
                    base_instructions += (
                        f"AMBIGUITY LABELLING: the retrieved information still varies by {axes_str}. "
                        f"For any fact whose value depends on {axes_str}, state which one it applies to "
                        f"inline, e.g. 'Fee (online): ...'. State facts that do NOT vary plainly, with no label. "
                        f"Do not ask the user which one they meant — you have already asked once this conversation.\n\n"
                    )
            except Exception:
                logger.warning("Failed to compute unresolved facets for ambiguity labeling")

            # Source URLs are appended cleanly by the proxy after the LLM reply.
            # Never ask the LLM to emit raw URLs or a Sources section inline.
            base_instructions += "Do not include raw URLs or a 'Sources:' section in your reply. The system will attach sources automatically.\n"
            
            base_instructions += "Do not return JSON or any response schema. Return only the final answer.\n"
            system_prompt += base_instructions

        # Applies with or without RAG — an elicit turn needs the ban just as
        # much as an answered one needs the permission.
        system_prompt += "\n\n" + _closing_instruction(
            closing_allowed(history_text, workflow_config, answer_confidence)
        )

        return system_prompt, user_prompt
