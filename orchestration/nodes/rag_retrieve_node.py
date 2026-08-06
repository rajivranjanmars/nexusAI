"""
Precision RAG retrieval node.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Optional

from cache_module.vector_store import _embed
from orchestration.conversation_memory import get_history
from orchestration.state import WorkflowState
from orchestration.workflow_config import RetrievalConfig, get_workflow_config
from shared.bm25_index import get_or_build_index
from shared.chroma_client import (
    KNOWLEDGE_CHUNKS_COLLECTION,
    get_collection,
    get_parent_chunks_by_ids,
)
from shared.logger import get_logger
from shared.query_expander import expand_query, condense_query_with_history
from shared.reranker import rerank
from shared.retrieval_models import ChunkRecord, ChunkType, RetrievalResult, RetrievedChunk

logger = get_logger(__name__)

# Fallback constants — used only when RetrievalConfig cannot be loaded.
_FALLBACK_DENSE_TOP_K = 30
_FALLBACK_SPARSE_TOP_K = 30
_FALLBACK_RRF_K = 60
_FALLBACK_RERANK_INPUT_SIZE = 25
_FALLBACK_RERANK_OUTPUT_SIZE = 6
_FALLBACK_MIN_FINAL_SCORE = 0.35
_FALLBACK_MAX_CHUNKS_PER_SOURCE = 2

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PROGRAM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "mba": ("mba", "master of business administration"),
    "management": ("management",),
    "mca": ("mca", "master of computer applications"),
    "mcom": ("mcom", "m.com", "master of commerce"),
    "ma": ("ma", "m.a", "master of arts"),
    "msc": ("msc", "m.sc", "master of science"),
    "bca": ("bca", "bachelor of computer applications"),
    "bba": ("bba", "bachelor of business administration"),
    "bcom": ("bcom", "b.com", "bachelor of commerce"),
    "ba": ("ba", "b.a", "bachelor of arts"),
    "bsc": ("bsc", "b.sc", "bachelor of science"),
}
_FEE_TOKENS = frozenset({
    "fee",
    "fees",
    "annual",
    "annually",
    "semester",
    "semesters",
    "exam",
    "lumpsum",
    "amount",
    "payment",
    "tuition",
    "cost",
})


def _get_retrieval_config(app_id: str | None) -> RetrievalConfig:
    """Resolve the retrieval config for an app, with graceful fallback."""

    try:
        return get_workflow_config(app_id).retrieval_config
    except Exception:
        logger.warning("Failed to load RetrievalConfig for app %s, using fallback", app_id)
        from orchestration.workflow_config import _parse_retrieval_config
        return _parse_retrieval_config({})


def _coerce_chunk_type(value: object) -> ChunkType:
    """Normalize raw metadata values into supported chunk types."""

    try:
        return ChunkType(str(value or ChunkType.STANDALONE.value).lower())
    except ValueError:
        return ChunkType.STANDALONE


def _coerce_entity_tags(value: object) -> list[str]:
    """Normalize persisted entity tags from JSON or list storage."""

    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return []


def _tokenize(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens for deterministic query heuristics."""

    return set(_TOKEN_RE.findall((text or "").lower()))


def _extract_program_terms(query: str) -> set[str]:
    """Expand program aliases in the query into canonical lexical hints."""

    lowered_query = (query or "").lower()
    matched_terms: set[str] = set()
    for canonical, aliases in _PROGRAM_SYNONYMS.items():
        if any(alias in lowered_query for alias in aliases):
            matched_terms.update(aliases)
            if canonical == "mba":
                matched_terms.update(_PROGRAM_SYNONYMS["management"])
    return matched_terms


def _score_lexical_relevance(
    chunk: RetrievedChunk,
    query: str,
    rc: RetrievalConfig,
) -> tuple[float, dict[str, Any]]:
    """Assign deterministic boosts/penalties for high-signal lexical matches."""

    boosts = rc.lexical_boosts
    query_tokens = _tokenize(query)
    program_terms = _extract_program_terms(query)
    is_fee_query = bool(query_tokens & _FEE_TOKENS)

    source_url = (chunk.record.source_url or "").lower()
    section_heading = (chunk.record.section_heading or "").lower()
    content = (chunk.hydrated_content or chunk.record.content or "").lower()
    source_ref = (chunk.record.source_ref or "").lower()
    entity_tags = {str(item).lower() for item in chunk.record.entity_tags}
    searchable = " ".join([
        source_url,
        source_ref,
        section_heading,
        content[:1200],
        " ".join(sorted(entity_tags)),
    ])

    boost = 0.0
    reasons: list[str] = []

    if chunk.record.source_type == "structured":
        boost += boosts.get("structured_data", 0.75)
        reasons.append("structured_data_boost")

    if program_terms:
        strong_searchable = " ".join([
            source_url,
            source_ref,
            section_heading,
            " ".join(sorted(entity_tags)),
        ])
        if any(term in strong_searchable for term in program_terms):
            boost += boosts.get("strong_program_match", 0.20)
            reasons.append("strong_program_match")
        elif any(term in searchable for term in program_terms):
            boost += boosts.get("weak_program_match", 0.08)
            reasons.append("weak_program_match")
        else:
            boost += boosts.get("program_miss", -0.15)
            reasons.append("program_miss")

    if is_fee_query:
        matched_fee_terms = [term for term in _FEE_TOKENS if term in searchable]
        if matched_fee_terms:
            boost += boosts.get("fee_match", 0.08)
            reasons.append("fee_match")
        else:
            boost += boosts.get("fee_miss", -0.05)
            reasons.append("fee_miss")

    noisy_path_hits = [
        token for token in rc.noisy_path_tokens
        if token in source_url or token in source_ref
    ]
    if noisy_path_hits and (program_terms or is_fee_query):
        boost += boosts.get("noisy_path", -0.12)
        reasons.append("noisy_path")

    if "/programmes/" in source_url or "/programmes/" in source_ref:
        boost += boosts.get("programme_path", 0.04)
        reasons.append("programme_path")

    if section_heading and any(term in section_heading for term in program_terms | _FEE_TOKENS):
        boost += boosts.get("heading_match", 0.04)
        reasons.append("heading_match")

    # Path allowlist boost
    if rc.path_allowlist and any(pattern in source_url for pattern in rc.path_allowlist):
        boost += rc.path_allowlist_boost
        reasons.append("path_allowlist")

    # Path blocklist penalty
    if rc.path_blocklist and any(pattern in source_url for pattern in rc.path_blocklist):
        boost += rc.path_blocklist_penalty
        reasons.append("path_blocklist")

    return boost, {
        "is_fee_query": is_fee_query,
        "program_terms": sorted(program_terms),
        "reasons": reasons,
    }


def _apply_lexical_boosts(
    chunks: list[RetrievedChunk],
    query: str,
    rc: RetrievalConfig,
) -> list[RetrievedChunk]:
    """Reorder retrieval candidates with deterministic lexical boosts."""

    w = rc.scoring_weights
    w_dense = w.get("dense", 0.45)
    w_sparse = w.get("sparse", 0.35)
    w_rrf = w.get("rrf", 0.20)

    rescored: list[RetrievedChunk] = []
    for chunk in chunks:
        boost, _debug = _score_lexical_relevance(chunk, query, rc)
        chunk.final_score = max(
            chunk.rrf_score,
            chunk.dense_score * w_dense + chunk.sparse_score * w_sparse + chunk.rrf_score * w_rrf + boost,
        )
        rescored.append(chunk)
    return sorted(rescored, key=lambda item: item.final_score, reverse=True)


def _apply_program_guardrails(chunks: list[RetrievedChunk], query: str) -> list[RetrievedChunk]:
    """Drop generic chunks when the query clearly targets a specific program."""

    program_terms = _extract_program_terms(query)
    if not program_terms:
        return chunks

    guarded: list[RetrievedChunk] = []
    fallback: list[RetrievedChunk] = []
    for chunk in chunks:
        source_url = (chunk.record.source_url or "").lower()
        source_ref = (chunk.record.source_ref or "").lower()
        section_heading = (chunk.record.section_heading or "").lower()
        content = (chunk.hydrated_content or chunk.record.content or "").lower()
        entity_tags = " ".join(str(item).lower() for item in chunk.record.entity_tags)
        searchable = " ".join([source_url, source_ref, section_heading, content[:1600], entity_tags])

        if any(term in searchable for term in program_terms):
            guarded.append(chunk)
            continue

        fallback.append(chunk)

    if guarded:
        return guarded
    return fallback


def _meta_to_record(chunk_id: str, document: str, metadata: dict[str, Any], app_id: str) -> ChunkRecord:
    """Convert raw Chroma result fields into a shared chunk record."""

    return ChunkRecord(
        chunk_id=chunk_id,
        content=document,
        app_id=str(metadata.get("app_id", app_id)),
        source_url=str(metadata.get("source_url", "")),
        source_type=str(metadata.get("source_type", "crawl")),
        source_ref=str(metadata.get("source_ref", "")),
        chunk_type=_coerce_chunk_type(metadata.get("chunk_type")),
        parent_chunk_id=metadata.get("parent_chunk_id") or None,
        chunk_index=int(metadata.get("chunk_index", 0)),
        section_heading=metadata.get("section_heading") or None,
        entity_tags=_coerce_entity_tags(metadata.get("entity_tags")),
        content_hash=str(metadata.get("content_hash", "")),
        crawl_job_id=metadata.get("crawl_job_id") or None,
    )


async def _run_dense_query(
    query: str,
    app_id: str,
    rc: RetrievalConfig,
) -> dict[str, tuple[ChunkRecord, float]]:
    """Run one dense retrieval query and return best-score hits keyed by chunk ID."""

    collection = await asyncio.to_thread(get_collection, KNOWLEDGE_CHUNKS_COLLECTION)
    query_embedding = await _embed(query)
    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[query_embedding],
        n_results=min(rc.dense_top_k, 100),
        where={"app_id": app_id},
        include=["documents", "metadatas", "distances"],
    )

    dense_hits: dict[str, tuple[ChunkRecord, float]] = {}
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        meta = metadata or {}
        if meta.get("app_id") not in {None, "", app_id}:
            continue
        similarity = max(0.0, 1.0 - float(distance))
        current = dense_hits.get(chunk_id)
        if current is not None and current[1] >= similarity:
            continue
        dense_hits[chunk_id] = (_meta_to_record(chunk_id, document, meta, app_id), similarity)

    return dense_hits


def _build_rrf_candidates(
    dense_results: dict[str, tuple[ChunkRecord, float]],
    sparse_results: dict[str, tuple[ChunkRecord, float]],
    rc: RetrievalConfig,
) -> list[RetrievedChunk]:
    """Fuse dense and sparse candidates with reciprocal rank fusion."""

    rrf_k = rc.rrf_k
    all_chunk_ids = set(dense_results) | set(sparse_results)
    dense_ranked = sorted(dense_results.items(), key=lambda item: item[1][1], reverse=True)
    sparse_ranked = sorted(sparse_results.items(), key=lambda item: item[1][1], reverse=True)

    dense_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(dense_ranked)}
    sparse_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(sparse_ranked)}

    scored_ids: list[tuple[str, float]] = []
    for chunk_id in all_chunk_ids:
        score = 0.0
        if chunk_id in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[chunk_id])
        if chunk_id in sparse_ranks:
            score += 1.0 / (rrf_k + sparse_ranks[chunk_id])
        scored_ids.append((chunk_id, score))

    candidates: list[RetrievedChunk] = []
    for chunk_id, rrf_score in sorted(scored_ids, key=lambda item: item[1], reverse=True)[:rc.rerank_input_size]:
        dense_record, dense_score = dense_results.get(chunk_id, (None, 0.0))
        if dense_record is None:
            dense_record, _ = sparse_results.get(chunk_id, (None, 0.0))
        if dense_record is None:
            continue
        _, sparse_score = sparse_results.get(chunk_id, (None, 0.0))
        candidates.append(
            RetrievedChunk(
                record=dense_record,
                dense_score=dense_score,
                sparse_score=float(sparse_score or 0.0),
                rrf_score=rrf_score,
            )
        )

    return candidates


def _dedupe_sources(
    chunks: list[RetrievedChunk],
    rc: RetrievalConfig,
) -> list[RetrievedChunk]:
    """Keep at most ``max_chunks_per_source`` chunks per source URL."""

    max_per_source = rc.max_chunks_per_source
    counts: dict[str, int] = {}
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        source_key = chunk.record.source_url or chunk.record.source_ref or chunk.record.chunk_id
        count = counts.get(source_key, 0)
        if count >= max_per_source:
            continue
        counts[source_key] = count + 1
        deduped.append(chunk)
    return deduped


async def run_retrieval(query: str, app_id: str) -> RetrievalResult:
    """Run the full precision retrieval pipeline for one user query."""

    rc = _get_retrieval_config(app_id)
    started_at = time.monotonic()
    retrieval_debug: dict[str, Any] = {}

    expanded_queries = await expand_query(query, app_id)
    retrieval_debug["expanded_queries"] = expanded_queries
    retrieval_debug["t_expansion"] = round(time.monotonic() - started_at, 3)

    dense_task_group = [_run_dense_query(expanded_query, app_id, rc) for expanded_query in expanded_queries]
    sparse_task = get_or_build_index(app_id)
    dense_outputs, bm25_index = await asyncio.gather(
        asyncio.gather(*dense_task_group, return_exceptions=True),
        sparse_task,
    )

    dense_results: dict[str, tuple[ChunkRecord, float]] = {}
    for expanded_query, output in zip(expanded_queries, dense_outputs):
        if isinstance(output, Exception):
            logger.warning("Dense retrieval failed for query '%s': %s", expanded_query[:80], output)
            continue
        for chunk_id, payload in output.items():
            current = dense_results.get(chunk_id)
            if current is None or payload[1] > current[1]:
                dense_results[chunk_id] = payload
    retrieval_debug["dense_candidate_count"] = len(dense_results)
    retrieval_debug["t_dense"] = round(time.monotonic() - started_at, 3)

    sparse_results: dict[str, tuple[ChunkRecord, float]] = {}
    if bm25_index is not None:
        try:
            for record, score in bm25_index.search(query, top_k=rc.sparse_top_k):
                if record.app_id != app_id:
                    continue
                sparse_results[record.chunk_id] = (record, score)
        except Exception as exc:
            logger.warning("Sparse retrieval failed for app %s: %s", app_id, exc)
    retrieval_debug["sparse_candidate_count"] = len(sparse_results)
    retrieval_debug["t_sparse"] = round(time.monotonic() - started_at, 3)

    candidates = _build_rrf_candidates(dense_results, sparse_results, rc)
    candidates = _apply_lexical_boosts(candidates, query, rc)[:rc.rerank_input_size]
    retrieval_debug["rrf_candidate_count"] = len(candidates)
    retrieval_debug["t_rrf"] = round(time.monotonic() - started_at, 3)

    reranked = await rerank(query, candidates, top_k=rc.rerank_output_size)
    for chunk in reranked:
        lexical_boost, lexical_debug = _score_lexical_relevance(chunk, query, rc)
        chunk.final_score = max(0.0, min(1.0, float(chunk.final_score or 0.0) + lexical_boost))
        chunk.record.entity_tags = sorted(set(chunk.record.entity_tags))
        setattr(chunk, "lexical_debug", lexical_debug)
    retrieval_debug["t_rerank"] = round(time.monotonic() - started_at, 3)
    reranked = _apply_program_guardrails(reranked, query)
    retrieval_debug["guarded_chunk_count"] = len(reranked)

    parent_ids = sorted(
        {
            chunk.record.parent_chunk_id
            for chunk in reranked
            if chunk.record.chunk_type == ChunkType.CHILD and chunk.record.parent_chunk_id
        }
    )
    parent_documents = await asyncio.to_thread(get_parent_chunks_by_ids, parent_ids)
    for chunk in reranked:
        if chunk.record.chunk_type == ChunkType.CHILD and chunk.record.parent_chunk_id:
            chunk.hydrated_content = (
                parent_documents.get(chunk.record.parent_chunk_id) or chunk.record.content
            )
        else:
            chunk.hydrated_content = chunk.record.content

    final_chunks = _dedupe_sources(reranked, rc)
    retrieval_debug["final_chunk_count"] = len(final_chunks)
    retrieval_debug["t_total"] = round(time.monotonic() - started_at, 3)

    return RetrievalResult(
        query=query,
        expanded_queries=expanded_queries,
        chunks=final_chunks,
        answer_confidence=max((chunk.final_score for chunk in final_chunks), default=0.0),
        retrieval_debug=retrieval_debug,
    )


def _to_rag_context(
    chunk: RetrievedChunk,
    rc: RetrievalConfig,
) -> Optional[dict[str, Any]]:
    """Project a retrieved chunk into the workflow state's rag_context shape."""

    if chunk.final_score < rc.min_final_score:
        return None

    return {
        "content": chunk.hydrated_content or chunk.record.content,
        "match_content": chunk.record.content,
        "source_url": chunk.record.source_url,
        "source_type": chunk.record.source_type,
        "section_heading": chunk.record.section_heading or "",
        "score": round(chunk.final_score, 3),
        "raw_score": round(chunk.rerank_score, 4),
        "dense_score": round(chunk.dense_score, 4),
        "sparse_score": round(chunk.sparse_score, 4),
        "rrf_score": round(chunk.rrf_score, 4),
        "chunk_type": chunk.record.chunk_type.value,
        "parent_chunk_id": chunk.record.parent_chunk_id or "",
        "entity_tags": chunk.record.entity_tags,
        "lexical_debug": getattr(chunk, "lexical_debug", {}),
        "source_ref": chunk.record.source_ref,
    }


async def rag_retrieve_node(state: WorkflowState) -> WorkflowState:
    """Retrieve precision-ranked RAG chunks and attach them to workflow state."""

    app_id = state.get("app_id")
    user_input = state.get("user_input")

    if not app_id or not user_input:
        state["rag_context"] = []
        state["answer_confidence"] = 0.0
        state["retrieval_debug"] = {}
        return state

    try:
        rc = _get_retrieval_config(app_id)

        actor_id = state.get("actor_id") or state.get("student_id") or ""
        session_id = state.get("session_id") or actor_id
        retrieval_query = user_input

        if app_id and session_id:
            try:
                history = await get_history(app_id, session_id)
                if history:
                    retrieval_query = await condense_query_with_history(user_input, history, app_id)
                    if retrieval_query != user_input:
                        logger.info(
                            "Query condensation for app %s: '%s' -> '%s'",
                            app_id,
                            user_input[:80],
                            retrieval_query[:80],
                        )
            except Exception as exc:
                logger.warning("Query condensation failed for app %s, using original query: %s", app_id, exc)

        retrieval_result = await run_retrieval(retrieval_query, app_id)
        rag_context = [
            item
            for item in (_to_rag_context(chunk, rc) for chunk in retrieval_result.chunks)
            if item is not None
        ]

        state["rag_context"] = rag_context
        state["answer_confidence"] = retrieval_result.answer_confidence
        state["retrieval_debug"] = retrieval_result.retrieval_debug

        logger.info(
            "Precision RAG retrieved %d chunks for app %s (confidence=%.3f, t_total=%ss)",
            len(rag_context),
            app_id,
            retrieval_result.answer_confidence,
            retrieval_result.retrieval_debug.get("t_total", "n/a"),
        )
    except Exception as exc:
        logger.error("Precision RAG retrieval failed for app %s: %s", app_id, exc)
        state["rag_context"] = []
        state["answer_confidence"] = 0.0
        state["retrieval_debug"] = {"error": str(exc)}

    return state
