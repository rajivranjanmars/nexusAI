"""
Shared data models for the precision RAG retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from shared.logger import get_logger

logger = get_logger(__name__)


class ChunkType(str, Enum):
    """Supported chunk storage shapes."""

    PARENT = "parent"
    CHILD = "child"
    STANDALONE = "standalone"


@dataclass
class ChunkRecord:
    """A single stored chunk as returned from ChromaDB."""

    chunk_id: str
    content: str
    app_id: str
    source_url: str
    source_type: str
    source_ref: str
    chunk_type: ChunkType
    parent_chunk_id: Optional[str]
    chunk_index: int
    section_heading: Optional[str]
    entity_tags: list[str] = field(default_factory=list)
    content_hash: str = ""
    crawl_job_id: Optional[str] = None


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with scoring metadata."""

    record: ChunkRecord
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    hydrated_content: str = ""


@dataclass
class RetrievalResult:
    """Full output of the retrieval pipeline for one query."""

    query: str
    expanded_queries: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    answer_confidence: float = 0.0
    retrieval_debug: dict[str, Any] = field(default_factory=dict)
