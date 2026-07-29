"""
Shared ChromaDB client utility.

Provides a singleton ChromaDB client and collection accessor to eliminate
duplicate initialization code across multiple scripts and modules.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

_chroma_client: chromadb.ClientAPI | None = None
KNOWLEDGE_CHUNKS_COLLECTION = "knowledge_chunks"
KNOWLEDGE_PARENTS_COLLECTION = "knowledge_chunks_parent"


def get_chroma_client() -> chromadb.ClientAPI:
    """Return a singleton persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_db_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection(name: str = KNOWLEDGE_CHUNKS_COLLECTION) -> chromadb.Collection:
    """Get or create a ChromaDB collection with cosine similarity."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def get_parent_collection() -> chromadb.Collection:
    """Return the parent-chunk collection used by hierarchical retrieval."""

    return get_collection(KNOWLEDGE_PARENTS_COLLECTION)


def get_parent_chunks_by_ids(parent_chunk_ids: list[str]) -> dict[str, str]:
    """Fetch parent chunk documents keyed by their chunk IDs."""

    if not parent_chunk_ids:
        return {}

    collection = get_parent_collection()
    try:
        result = collection.get(
            ids=parent_chunk_ids,
            include=["documents"],
        )
    except Exception as exc:
        logger.warning("Parent chunk fetch failed for %d ids: %s", len(parent_chunk_ids), exc)
        return {}

    ids = result.get("ids") or []
    documents = result.get("documents") or []
    return {chunk_id: document for chunk_id, document in zip(ids, documents) if chunk_id and document}
