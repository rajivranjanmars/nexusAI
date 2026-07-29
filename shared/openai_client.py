"""
Shared OpenAI client utility.

Provides singleton OpenAI clients for LLM and embedding operations to eliminate
duplicate initialization code across multiple modules.
"""

from __future__ import annotations

import openai

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

_llm_client: openai.AsyncOpenAI | None = None
_embedding_client: openai.AsyncOpenAI | None = None


def get_llm_client() -> openai.AsyncOpenAI:
    """Return a singleton async OpenAI-compatible client for LLM operations."""
    global _llm_client
    if _llm_client is None:
        _llm_client = openai.AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=30,
            max_retries=0,  # We handle retries ourselves for finer control
        )
        logger.info(
            "LLM client initialized",
            extra={"base_url": settings.llm_base_url},
        )
    return _llm_client


def get_embedding_client() -> openai.AsyncOpenAI:
    """Return a singleton async OpenAI-compatible client for embedding operations."""
    global _embedding_client
    if _embedding_client is None:
        if not settings.embedding_api_key:
            raise RuntimeError(
                "Embedding API key is required for remote embedding provider"
            )
        _embedding_client = openai.AsyncOpenAI(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
        )
        logger.info(
            "Embedding client initialized",
            extra={"base_url": settings.embedding_base_url},
        )
    return _embedding_client


def reset_clients() -> None:
    """Reset the client singletons (primarily for testing)."""
    global _llm_client, _embedding_client
    _llm_client = None
    _embedding_client = None
