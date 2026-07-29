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
from orchestration.workflow_config import get_workflow_config, WorkflowResponseConfig
from db.app_registry import resolve_by_app_id

logger = get_logger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 0.35
_MEDIUM_CONFIDENCE_THRESHOLD = 0.60
_LEAD_CAPTURE_FACT_TOKENS = frozenset({
    "fee",
    "fees",
    "cost",
    "costs",
    "tuition",
    "eligibility",
    "duration",
    "admission",
    "semester",
    "semesters",
    "annual",
    "lumpsum",
    "curriculum",
    "syllabus",
    "placement",
    "placements",
    "scholarship",
    "scholarships",
})


def is_lead_capture_fact_query(user_query: str) -> bool:
    """Return whether a lead-capture query is asking for concrete institutional facts."""

    tokens = {
        token.strip(".,?!:;()[]{}\"'").lower()
        for token in (user_query or "").split()
        if token.strip()
    }
    return bool(tokens & _LEAD_CAPTURE_FACT_TOKENS)


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

        for idx, chunk in enumerate(rag_context, start=1):
            matched_text = str(chunk.get("match_content", "") or "").strip()
            hydrated_text = str(chunk.get("content", "") or "").strip()
            if not matched_text and not hydrated_text:
                continue

            heading = chunk.get("section_heading", "")
            source_url = chunk.get("source_url", "")
            confidence = float(chunk.get("score", 0.0) or 0.0)
            header = f"[Source {idx}"
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
                "Acknowledge uncertainty clearly, avoid fabrication, and suggest where the user can verify the answer."
            )
        if answer_confidence < _MEDIUM_CONFIDENCE_THRESHOLD:
            return (
                "Answer from the retrieved knowledge, but note uncertainty where the evidence is incomplete. "
                "Do not include raw URLs or a Sources section — the system will attach them automatically."
            )
        return (
            "Answer precisely from the retrieved knowledge below. "
            "Do not include raw URLs, source numbers, or a Sources section — the system will attach them automatically. "
            "Do not add unsupported facts."
        )

    def _is_lead_capture_fact_query(self, user_query: str) -> bool:
        """Return whether a lead-capture query is asking for concrete institutional facts."""
        return is_lead_capture_fact_query(user_query)

    def build_strict_rag_prompt(
        self,
        user_query: str,
        rag_context: list[dict],
        workflow_config: WorkflowResponseConfig,
        answer_confidence: float,
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
                # Default formatting based on workflow
                if self.workflow == "lead_capture":
                    base_instructions += (
                        "LEAD CAPTURE FORMAT: Always use bullet points for factual details (fees, eligibility, dates). "
                        "Keep the overall response concise and warm.\n\n"
                    )
                elif self.workflow == "general":
                    base_instructions += (
                        "TRIAGE RULE: Keep replies short and natural. Clarify intent when needed, "
                        "and move admissions- or university-related conversations toward concrete next steps.\n\n"
                    )
                elif self.workflow == "enrollment":
                    base_instructions += (
                        "ENROLLMENT FORMAT: Always use bullet points for key facts (fees, eligibility, dates, duration). "
                        "Answer the question first, then suggest the most relevant next step if helpful.\n\n"
                    )
                else:
                    base_instructions += "BREVITY RULE: Always answer in short, crisp bullet points. No preamble.\n\n"

            if self.workflow == "lead_capture" and self._is_lead_capture_fact_query(user_query):
                base_instructions += (
                    "LEAD CAPTURE FACT RULE: The user is asking for concrete admissions information. "
                    "Answer ONLY from the retrieved website content. If exact figures or details are present, "
                    "state them directly and exactly. If the exact detail is not present, say briefly that you "
                    "could not find that exact information right now. Do NOT speculate. Do NOT say fees may change. "
                    "Do NOT mention scholarships, offers, brochures, or admissions-team follow-up unless those "
                    "details are explicitly present in the retrieved website content.\n\n"
                )
            
            # Source URLs are appended cleanly by the proxy after the LLM reply.
            # Never ask the LLM to emit raw URLs or a Sources section inline.
            base_instructions += "Do not include raw URLs or a 'Sources:' section in your reply. The system will attach sources automatically.\n"
            
            base_instructions += "Do not return JSON or any response schema. Return only the final answer.\n"
            system_prompt += base_instructions
        
        return system_prompt, user_prompt
