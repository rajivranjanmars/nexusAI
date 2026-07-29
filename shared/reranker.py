"""
Cross-encoder reranker for the precision RAG pipeline.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time

from shared.logger import get_logger
from shared.retrieval_models import RetrievedChunk

logger = get_logger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker_model = None
_load_lock = threading.Lock()


def _load_model():
    """Load and memoize the reranker model once per process."""

    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    with _load_lock:
        if _reranker_model is not None:
            return _reranker_model
        try:
            from sentence_transformers import CrossEncoder

            started_at = time.monotonic()
            logger.info("Loading reranker model %s", _MODEL_NAME)
            _reranker_model = CrossEncoder(_MODEL_NAME, max_length=512)
            logger.info(
                "Loaded reranker model %s in %.2fs",
                _MODEL_NAME,
                time.monotonic() - started_at,
            )
        except Exception as exc:
            logger.error("Reranker load failed: %s", exc)
            _reranker_model = None

    return _reranker_model


def warmup() -> None:
    """Preload the reranker at startup to reduce first-query latency."""

    _load_model()


def _apply_fallback_scores(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Fallback to pre-reranker scores when the reranker is unavailable."""

    for chunk in chunks:
        chunk.rerank_score = chunk.rrf_score
        # Preserve chunk.final_score (set by _apply_lexical_boosts)
    return sorted(chunks, key=lambda item: float(item.final_score or 0.0), reverse=True)[:top_k]


def _predict_scores(model, query: str, chunks: list[RetrievedChunk]) -> list[float]:
    """Run synchronous cross-encoder inference."""

    pairs = [(query, chunk.record.content[:512]) for chunk in chunks]
    scores = model.predict(pairs, show_progress_bar=False)
    return [float(score) for score in scores]


async def rerank(query: str, chunks: list[RetrievedChunk], top_k: int = 6) -> list[RetrievedChunk]:
    """Rerank retrieved chunks using a cross-encoder with sigmoid calibration."""

    if not chunks:
        return []

    model = _load_model()
    if model is None:
        return _apply_fallback_scores(chunks, top_k)

    try:
        scores = await asyncio.to_thread(_predict_scores, model, query, chunks)
    except Exception as exc:
        logger.warning("Cross-encoder inference failed, falling back to RRF: %s", exc)
        return _apply_fallback_scores(chunks, top_k)

    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = score
        if score >= 0:
            exp_value = math.exp(-score)
            chunk.final_score = 1.0 / (1.0 + exp_value)
        else:
            exp_value = math.exp(score)
            chunk.final_score = exp_value / (1.0 + exp_value)

    return sorted(chunks, key=lambda item: item.final_score, reverse=True)[:top_k]
