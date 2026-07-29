"""
pgvector-backed vector store for semantic search.

Connects to PostgreSQL + pgvector using ``PGVECTOR_URL``, generates embeddings
via an OpenAI-compatible endpoint (``EMBEDDING_BASE_URL``), and exposes
``embed_and_store`` / ``semantic_search`` async helpers.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

import openai
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

# �"?�"? Embedding client �"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?

_embedding_client: Optional[openai.AsyncOpenAI] = None
_local_embedding_model: Optional[object] = None


def _get_embedding_client() -> openai.AsyncOpenAI:
    """Return (or lazily create) the async OpenAI-compatible embedding client."""
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
    return _embedding_client


def _get_local_embedding_model() -> object:
    """Return the local sentence-transformers model instance."""
    global _local_embedding_model
    if _local_embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.error(
                "Local embedding provider requested but sentence-transformers is not installed"
            )
            raise RuntimeError(
                "sentence-transformers must be installed for local embeddings"
            ) from exc

        _local_embedding_model = SentenceTransformer(settings.embedding_local_model)
    return _local_embedding_model


async def _embed_local(text: str) -> List[float]:
    model = await asyncio.to_thread(_get_local_embedding_model)
    vector = await asyncio.to_thread(
        model.encode,
        text,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vector.tolist()


async def _embed_remote(text: str) -> List[float]:
    client = _get_embedding_client()
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


async def _embed(text: str) -> List[float]:
    """Generate an embedding vector for *text*.

    Supports remote, local, and auto fallback modes.
    """
    provider = settings.embedding_provider.lower()
    if provider == "local":
        return await _embed_local(text)
    if provider == "auto":
        try:
            return await _embed_remote(text)
        except Exception as exc:
            logger.warning(
                "Remote embedding failed; falling back to local provider: %s",
                exc,
            )
            return await _embed_local(text)
    return await _embed_remote(text)


# �"?�"? Database helpers �"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?

def _get_connection() -> psycopg2.extensions.connection:
    """Create a synchronous psycopg2 connection with pgvector registered."""
    conn = psycopg2.connect(settings.pgvector_url)
    return conn


def _ensure_table(conn: psycopg2.extensions.connection, dim: Optional[int] = None) -> None:
    """Create the ``embeddings`` table if it does not exist.

    Args:
        conn: psycopg2 connection.
        dim: Optional vector dimension to use when creating the table. If not
            provided, defaults to 1536 for compatibility with remote providers.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        # Prefer explicit dim argument; fall back to 1536.
        dim = int(dim) if dim is not None else 1536
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                metadata    JSONB DEFAULT '{{}}',
                embedding   vector({dim})
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embeddings_vector
            ON embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )
        conn.commit()


# �"?�"? Public API �"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?


async def embed_and_store(
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Embed *text* and upsert into the ``embeddings`` table.

    Args:
        doc_id: Unique identifier for the document (used as PK).
        text: The raw text to embed.
        metadata: Optional JSON-serialisable metadata dict.

    Returns:
        The ``doc_id`` that was stored.
    """
    vector = await _embed(text)

    conn = _get_connection()
    try:
        # If table does not exist, create it with the embedding dimension
        # matching the vector we just generated.
        _ensure_table(conn, dim=len(vector))
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings (id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET content   = EXCLUDED.content,
                              metadata  = EXCLUDED.metadata,
                              embedding = EXCLUDED.embedding;
                """,
                (
                    doc_id or str(uuid.uuid4()),
                    text,
                    json.dumps(metadata or {}),
                    vector,
                ),
            )
            conn.commit()
    finally:
        conn.close()

    logger.info("Stored embedding for doc_id=%s", doc_id)
    return doc_id


async def semantic_search(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Find the *top_k* most semantically similar documents.

    Args:
        query: Natural-language query string.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys ``id``, ``content``, ``metadata``, ``score``.
    """
    query_vector = await _embed(query)

    conn = _get_connection()
    try:
        _ensure_table(conn)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM   embeddings
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s;
                """,
                (query_vector, query_vector, top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    results = [
        {
            "id": row["id"],
            "content": row["content"],
            "metadata": row["metadata"],
            "score": float(row["score"]),
        }
        for row in rows
    ]
    logger.debug("Semantic search returned %d results for query", len(results))
    return results