"""
BM25 index helpers for per-app sparse retrieval.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from shared.chroma_client import KNOWLEDGE_CHUNKS_COLLECTION, get_collection
from shared.config import settings
from shared.logger import get_logger
from shared.retrieval_models import ChunkRecord, ChunkType

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_bm25_cache: dict[str, "BM25AppIndex"] = {}


def _tokenize(text: str) -> list[str]:
    """Tokenize text for lightweight sparse retrieval."""

    return _TOKEN_RE.findall((text or "").lower())


def _coerce_chunk_type(value: str | None) -> ChunkType:
    """Convert raw metadata into a safe chunk type."""

    try:
        return ChunkType((value or ChunkType.STANDALONE.value).lower())
    except ValueError:
        return ChunkType.STANDALONE


def _coerce_entity_tags(value: object) -> list[str]:
    """Normalize persisted entity tags from list or JSON string."""

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


def _record_to_dict(record: ChunkRecord) -> dict:
    """Serialize a chunk record for on-disk caching."""

    payload = asdict(record)
    payload["chunk_type"] = record.chunk_type.value
    return payload


def _record_from_dict(payload: dict) -> ChunkRecord:
    """Deserialize a chunk record from cache storage."""

    return ChunkRecord(
        chunk_id=str(payload.get("chunk_id", "")),
        content=str(payload.get("content", "")),
        app_id=str(payload.get("app_id", "")),
        source_url=str(payload.get("source_url", "")),
        source_type=str(payload.get("source_type", "crawl")),
        source_ref=str(payload.get("source_ref", "")),
        chunk_type=_coerce_chunk_type(str(payload.get("chunk_type", ChunkType.STANDALONE.value))),
        parent_chunk_id=payload.get("parent_chunk_id") or None,
        chunk_index=int(payload.get("chunk_index", 0)),
        section_heading=payload.get("section_heading") or None,
        entity_tags=[str(item) for item in payload.get("entity_tags", []) if str(item).strip()],
        content_hash=str(payload.get("content_hash", "")),
        crawl_job_id=payload.get("crawl_job_id") or None,
    )


class BM25AppIndex:
    """In-memory BM25 index for one application."""

    def __init__(self, app_id: str, chunk_records: list[ChunkRecord]) -> None:
        self.app_id = app_id
        self.chunk_records = chunk_records
        tokenized_corpus = [_tokenize(record.content) for record in chunk_records]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 30) -> list[tuple[ChunkRecord, float]]:
        """Return normalized BM25 matches for a query."""

        tokens = _tokenize(query)
        if not tokens or not self.chunk_records:
            return []

        scores = list(self.bm25.get_scores(tokens))
        if not scores:
            return []

        max_score = max(scores)
        if max_score <= 0:
            return []

        indexed_scores = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        matches: list[tuple[ChunkRecord, float]] = []
        for index, score in indexed_scores[:top_k]:
            normalized = float(score / max_score) if max_score else 0.0
            if normalized <= 0.01:
                continue
            matches.append((self.chunk_records[index], normalized))
        return matches


def _index_path(app_id: str) -> Path:
    """Return the on-disk path for a persisted BM25 cache."""

    return Path(settings.chroma_db_dir) / f"bm25_{app_id}.json"


def save_index(index: BM25AppIndex) -> None:
    """Persist chunk records used to rebuild the BM25 index quickly."""

    path = _index_path(index.app_id)
    payload = [_record_to_dict(record) for record in index.chunk_records]
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:
        logger.warning("BM25 save failed for app %s: %s", index.app_id, exc)


def _load_index(app_id: str) -> Optional[BM25AppIndex]:
    """Load a previously persisted BM25 cache from disk."""

    path = _index_path(app_id)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return None
        records = [_record_from_dict(item) for item in payload if isinstance(item, dict)]
        if not records:
            return None
        return BM25AppIndex(app_id, records)
    except Exception as exc:
        logger.warning("BM25 load failed for app %s: %s", app_id, exc)
        return None


def _build_record(chunk_id: str, document: str, metadata: dict, app_id: str) -> ChunkRecord:
    """Convert raw Chroma metadata into a chunk record."""

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


async def get_or_build_index(app_id: str) -> Optional[BM25AppIndex]:
    """Return a cached BM25 index for the given app."""

    if app_id in _bm25_cache:
        return _bm25_cache[app_id]

    disk_index = _load_index(app_id)
    if disk_index is not None:
        _bm25_cache[app_id] = disk_index
        return disk_index

    try:
        collection = await asyncio.to_thread(get_collection, KNOWLEDGE_CHUNKS_COLLECTION)
        result = await asyncio.to_thread(
            collection.get,
            where={"app_id": app_id},
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not ids:
            return None

        records = [
            _build_record(chunk_id, document, metadata or {}, app_id)
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        ]
        index = BM25AppIndex(app_id, records)
        _bm25_cache[app_id] = index
        save_index(index)
        return index
    except Exception as exc:
        logger.error("BM25 index build failed for app %s: %s", app_id, exc)
        return None


def invalidate_index(app_id: str) -> None:
    """Drop in-memory and persisted BM25 state for one app."""

    _bm25_cache.pop(app_id, None)
    try:
        _index_path(app_id).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("BM25 cache cleanup failed for app %s: %s", app_id, exc)
