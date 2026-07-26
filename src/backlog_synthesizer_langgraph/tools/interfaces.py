"""Protocol classes defining tool interfaces for the Backlog Synthesizer system.

These protocols allow tool implementations to be swapped without modifying
agent logic, satisfying Requirement 9 (Modular Tool Abstractions).
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SearchResult:
    """Result returned by vector similarity search."""

    item_id: str
    score: float
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class DocumentParsingTool(Protocol):
    """Interface for document parsing operations.

    Provides PDF-to-text conversion.
    Implementations must translate internal exceptions into ToolError types.

    Note: Text chunking is NOT part of this interface. Chunking is a pure
    in-memory computation (no external I/O) and is handled internally by
    ParserAgent using tiktoken. The Protocol pattern is reserved for
    external dependencies that may need swapping (PDF libraries, APIs, etc.).
    """

    def pdf_to_text(self, content: bytes) -> str:
        """Convert PDF binary content to plain text.

        Args:
            content: Raw PDF file bytes.

        Returns:
            Extracted text content preserving paragraph boundaries.

        Raises:
            ToolError: If parsing fails.
        """
        ...


@runtime_checkable
class EmbeddingTool(Protocol):
    """Interface for generating text embeddings.

    Provides embedding vector generation for semantic search.
    Implementations must translate internal exceptions into ToolError types.
    """

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ToolError: If embedding generation fails.
        """
        ...


@runtime_checkable
class VectorSearchTool(Protocol):
    """Interface for vector similarity search and storage.

    Provides methods to store embeddings and query for similar items.
    Implementations must translate internal exceptions into ToolError types.
    """

    def query_similar(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        """Query the vector store for similar items.

        Args:
            embedding: The query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects ordered by similarity score (descending).

        Raises:
            ToolError: If the query fails.
        """
        ...

    def query_similar_filtered(
        self,
        embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Query the vector store with optional metadata filtering.

        Args:
            embedding: The query embedding vector.
            top_k: Maximum number of results to return.
            where: Optional metadata filter dict.

        Returns:
            List of SearchResult objects ordered by similarity score (descending).

        Raises:
            ToolError: If the query fails.
        """
        ...

    def store(self, item_id: str, embedding: list[float], metadata: dict) -> None:
        """Store an embedding with associated metadata.

        Args:
            item_id: Unique identifier for the item.
            embedding: The embedding vector to store.
            metadata: Associated metadata dictionary.

        Raises:
            ToolError: If storage fails.
        """
        ...


@runtime_checkable
class LLMGenerationTool(Protocol):
    """Interface for LLM text generation.

    Provides text generation capabilities for agents.
    Implementations must translate internal exceptions into ToolError types.
    """

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using an LLM.

        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: Optional system prompt for context/instruction.

        Returns:
            The generated text response.

        Raises:
            ToolError: If generation fails.
        """
        ...
