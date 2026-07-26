"""Concrete VectorSearchTool implementation using ChromaDB.

Satisfies Requirement 9.2 (vector search through Tool interface)
and Requirement 9.6 (translate implementation-specific exceptions to ToolError).
"""

from backlog_synthesizer_langgraph.tools.errors import PermanentToolError, TransientToolError
from backlog_synthesizer_langgraph.tools.interfaces import SearchResult


class ChromaVectorSearchTool:
    """VectorSearchTool implementation backed by ChromaDB.

    Provides vector similarity search and storage using a Chroma collection.
    All Chroma-specific exceptions are translated to ToolError subtypes.

    Args:
        collection_name: Name of the Chroma collection to use.
        persist_directory: Optional directory for persisting the Chroma database.
            If None, uses an in-memory ephemeral client.
    """

    def __init__(
        self,
        collection_name: str = "backlog_items",
        persist_directory: str | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._client = self._create_client()
        self._collection = self._get_or_create_collection()

    def _create_client(self):  # noqa: ANN202
        """Create a ChromaDB client."""
        try:
            import chromadb

            if self._persist_directory:
                settings = chromadb.Settings(
                    anonymized_telemetry=False,
                    persist_directory=self._persist_directory,
                    is_persistent=True,
                )
                return chromadb.Client(settings)
            else:
                return chromadb.EphemeralClient()
        except ImportError as e:
            raise PermanentToolError(
                "chromadb library is not installed. "
                "Install it with: pip install chromadb",
                original_error=e,
            ) from e
        except Exception as e:
            raise PermanentToolError(
                f"Failed to create ChromaDB client: {e}",
                original_error=e,
            ) from e

    def _get_or_create_collection(self):  # noqa: ANN202
        """Get or create the Chroma collection."""
        try:
            return self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise PermanentToolError(
                f"Failed to get or create collection '{self._collection_name}': {e}",
                original_error=e,
            ) from e

    def store(self, item_id: str, embedding: list[float], metadata: dict) -> None:
        """Store an embedding with associated metadata in the Chroma collection."""
        if not item_id:
            raise PermanentToolError("item_id must be a non-empty string")

        if not embedding:
            raise PermanentToolError("embedding must be a non-empty list of floats")

        sanitized_metadata = self._sanitize_metadata(metadata)

        try:
            self._collection.upsert(
                ids=[item_id],
                embeddings=[embedding],
                metadatas=[sanitized_metadata] if sanitized_metadata else None,
            )
        except ValueError as e:
            raise PermanentToolError(
                f"Invalid data for storage: {e}",
                original_error=e,
            ) from e
        except Exception as e:
            if self._is_transient_error(e):
                raise TransientToolError(
                    f"Transient error storing item '{item_id}': {e}",
                    original_error=e,
                ) from e
            raise PermanentToolError(
                f"Failed to store item '{item_id}': {e}",
                original_error=e,
            ) from e

    def query_similar(
        self, embedding: list[float], top_k: int
    ) -> list[SearchResult]:
        """Query the vector store for similar items."""
        return self.query_similar_filtered(embedding, top_k, where=None)

    def query_similar_filtered(
        self,
        embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Query the vector store with optional metadata filtering."""
        if not embedding:
            raise PermanentToolError("embedding must be a non-empty list of floats")

        if top_k < 1:
            raise PermanentToolError("top_k must be at least 1")

        try:
            query_kwargs: dict = {
                "query_embeddings": [embedding],
                "n_results": top_k,
                "include": ["metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where

            results = self._collection.query(**query_kwargs)
        except Exception as e:
            if self._is_transient_error(e):
                raise TransientToolError(
                    f"Transient error querying vector store: {e}",
                    original_error=e,
                ) from e
            raise PermanentToolError(
                f"Failed to query vector store: {e}",
                original_error=e,
            ) from e

        return self._parse_query_results(results)

    def _parse_query_results(self, results: dict) -> list[SearchResult]:
        """Parse Chroma query results into SearchResult objects."""
        search_results: list[SearchResult] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []

        for i, item_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0.0
            score = 1.0 - distance

            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

            search_results.append(
                SearchResult(
                    item_id=item_id,
                    score=score,
                    metadata=metadata,
                )
            )

        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata to only include ChromaDB-compatible value types."""
        if not metadata:
            return {}

        sanitized: dict = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[str(key)] = value
            elif value is None:
                continue
            else:
                sanitized[str(key)] = str(value)

        return sanitized

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        """Determine if an error is transient (retryable)."""
        transient_indicators = (
            "timeout",
            "connection",
            "unavailable",
            "temporary",
            "resource exhausted",
            "too many requests",
        )
        error_msg = str(error).lower()
        return any(indicator in error_msg for indicator in transient_indicators)
