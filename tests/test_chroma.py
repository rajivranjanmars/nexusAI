"""Tests for Chroma metadata filtering used by the semantic cache."""

from __future__ import annotations

import chromadb


def test_response_cache_metadata_filter_is_tenant_scoped() -> None:
    """Return only entries matching workflow, cache scope, app, and actor."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("response_cache_test")
    collection.add(
        ids=["matching", "other-app"],
        documents=["matching response", "other response"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {
                "workflow": "enrollment",
                "cache_scope": "shared",
                "app_id": "app-a",
                "actor_id": "__none__",
            },
            {
                "workflow": "enrollment",
                "cache_scope": "shared",
                "app_id": "app-b",
                "actor_id": "__none__",
            },
        ],
    )

    result = collection.get(
        where={
            "$and": [
                {"workflow": "enrollment"},
                {"cache_scope": "shared"},
                {"app_id": "app-a"},
                {"actor_id": "__none__"},
            ]
        }
    )

    assert result["ids"] == ["matching"]
