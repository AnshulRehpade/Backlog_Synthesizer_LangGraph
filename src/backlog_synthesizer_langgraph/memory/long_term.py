"""Long-Term Memory implementation using vector store for semantic search.

Provides embedding generation, storage, and semantic search over backlog items.
Uses EmbeddingTool and VectorSearchTool protocol interfaces for portability.
Implements a 30-day retention policy for stored entries.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from backlog_synthesizer_langgraph.tools.interfaces import (
    EmbeddingTool,
    SearchResult,
    VectorSearchTool,
)

# Default retention period in days for stored entries.
_DEFAULT_RETENTION_DAYS = 30


class LongTermMemory:
    """Vector-backed long-term memory for semantic search over backlog items.

    Stores item embeddings via VectorSearchTool and generates embeddings via
    EmbeddingTool. Items are tagged with a stored_at timestamp to support
    time-based retention purging.

    Args:
        embedding_tool: Implementation of EmbeddingTool for generating embeddings.
        vector_search_tool: Implementation of VectorSearchTool for storage and search.
        retention_days: Number of days to retain stored entries. Defaults to 30.
    """

    def __init__(
        self,
        embedding_tool: EmbeddingTool,
        vector_search_tool: VectorSearchTool,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        self._embedding_tool = embedding_tool
        self._vector_search_tool = vector_search_tool
        self._retention_days = retention_days

    @property
    def retention_days(self) -> int:
        """Number of days entries are retained before becoming eligible for purging."""
        return self._retention_days

    def store_item(
        self,
        item_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Generate an embedding for content and store it in the vector store.

        The item is stored with a `stored_at` ISO 8601 timestamp in its metadata
        to support retention-based purging.
        """
        embedding = self._embedding_tool.generate_embedding(content)

        stored_metadata: dict[str, Any] = {
            **(metadata or {}),
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "content": content,
        }

        self._vector_search_tool.store(item_id, embedding, stored_metadata)

    def search_similar(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for items semantically similar to the query text."""
        query_embedding = self._embedding_tool.generate_embedding(query)
        return self._vector_search_tool.query_similar(query_embedding, top_k)

    def purge_expired(self, reference_time: datetime | None = None) -> list[str]:
        """Remove entries that have exceeded the retention period.

        Returns list of item IDs that were identified as expired.
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        cutoff = reference_time - timedelta(days=self._retention_days)
        cutoff_iso = cutoff.isoformat()

        try:
            if hasattr(self._vector_search_tool, "query_similar_filtered"):
                dummy_embedding = self._embedding_tool.generate_embedding("purge query")
                results = self._vector_search_tool.query_similar_filtered(
                    dummy_embedding,
                    top_k=10000,
                    where={"stored_at": {"$lt": cutoff_iso}},
                )
            else:
                dummy_embedding = self._embedding_tool.generate_embedding("purge query")
                results = self._vector_search_tool.query_similar(dummy_embedding, 10000)
                results = [
                    r for r in results
                    if r.metadata.get("stored_at", "") < cutoff_iso
                    and r.metadata.get("stored_at") is not None
                ]
        except Exception:
            return []

        expired_ids = [r.item_id for r in results]
        return expired_ids
