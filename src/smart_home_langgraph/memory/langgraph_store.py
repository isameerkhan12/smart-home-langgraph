# ---------------------------------------------------------------------------
# memory/langgraph_store.py
# Purpose: Build the native LangGraph Postgres store for long-term memory.
# ---------------------------------------------------------------------------
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore

from smart_home_langgraph.config.settings import get_settings


@dataclass
class ManagedStore:
    """Opened store plus the context manager that owns its resources."""

    store: BaseStore
    manager: AbstractContextManager[BaseStore]


def create_postgres_store() -> AbstractContextManager[BaseStore] | None:
    """Return a configured PostgresStore context manager when DB settings exist."""
    settings = get_settings()
    # If either value is missing, disable LTM store and run without Postgres.
    if not settings.postgres_uri or not settings.gemini_api_key:
        return None

    # Embedding model used for semantic search vectors.
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=settings.gemini_embedding_dims,
    )

    # from_conn_string returns a context manager (not an opened store yet).
    # We open/close it in open_postgres_store()/close_postgres_store().
    return PostgresStore.from_conn_string(
        # Standard PostgreSQL connection string.
        settings.postgres_uri,
        # Vector index configuration used by pgvector-backed semantic search.

        index={
            # Embedding dimension must match the selected embedding model.
            "dims": settings.gemini_embedding_dims,
            # Callable used to convert memory text into embedding vectors.
            "embed": embeddings,
            # Value fields to embed from each stored memory document.
            # Our memory records will include a "content" field.
            "fields": ["content"],
            # Similarity metric for nearest-neighbor search.
            "distance_type": "cosine",
            # Approximate nearest-neighbor index config in pgvector.
            "ann_index_config": {
                # HNSW gives strong recall/latency tradeoff for semantic search.
                "kind": "hnsw",
                # Full-precision vector storage type.
                "vector_type": "vector",
                # HNSW graph connectivity parameter.
                "m": 16,
                # HNSW build-time search width parameter.
                "ef_construction": 64,
            },
        },
    )


def open_postgres_store() -> ManagedStore | None:
    """Open the configured PostgresStore and run setup once for this session."""
    manager = create_postgres_store()
    if manager is None:
        return None

    # Enter context manager to acquire an active store connection.
    store = manager.__enter__()
    # Idempotent setup: creates required tables/extensions/indexes if missing.
    store.setup()
    # Keep both objects so caller can use store now and close manager later.
    return ManagedStore(store=store, manager=manager)


def close_postgres_store(managed_store: ManagedStore | None) -> None:
    """Close an opened PostgresStore if one was created."""
    if managed_store is None:
        return
    # Exit context manager to release DB resources cleanly.
    managed_store.manager.__exit__(None, None, None)
