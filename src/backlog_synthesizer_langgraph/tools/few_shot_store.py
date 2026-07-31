"""Few-shot example store for improving Story Writer output quality.

Stores high-quality user story examples in a vector store and retrieves
semantically similar examples to inject into prompts. This enables
dynamic few-shot prompting — the LLM sees relevant examples before
generating each story, significantly improving output consistency.

Usage:
    store = FewShotStore(embedding_tool, vector_search_tool)
    store.add_example(transcript="...", story_json="...", description="auth feature")
    examples = store.get_similar_story_examples("SSO support needed", top_k=2)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from backlog_synthesizer_langgraph.tools.interfaces import (
    EmbeddingTool,
    VectorSearchTool,
)

logger = logging.getLogger(__name__)


@dataclass
class FewShotExample:
    """A single few-shot example for story generation."""

    transcript: str
    story_json: str
    description: str
    tags: list[str] = field(default_factory=list)


class FewShotStore:
    """Vector-backed store for few-shot story generation examples.

    Stores example transcript→story pairs and retrieves the most relevant
    ones for a given input text using semantic similarity.

    The StoryWriterAgent already checks for this interface:
        examples = self._few_shot_store.get_similar_story_examples(text, top_k=2)

    Args:
        embedding_tool: Tool for generating text embeddings.
        vector_search_tool: Tool for storing and searching embeddings.
        collection_prefix: Prefix for item IDs in the vector store.
    """

    def __init__(
        self,
        embedding_tool: EmbeddingTool,
        vector_search_tool: VectorSearchTool,
        collection_prefix: str = "few_shot_",
    ) -> None:
        self._embedding_tool = embedding_tool
        self._vector_search_tool = vector_search_tool
        self._prefix = collection_prefix
        self._examples: dict[str, FewShotExample] = {}

    def add_example(
        self,
        example_id: str,
        transcript: str,
        story_json: str,
        description: str,
        tags: list[str] | None = None,
    ) -> None:
        """Add a few-shot example to the store.

        Args:
            example_id: Unique identifier for this example.
            transcript: The source transcript text.
            story_json: The expected JSON output for this transcript.
            description: Short description of what this example demonstrates.
            tags: Optional tags for categorization.
        """
        example = FewShotExample(
            transcript=transcript,
            story_json=story_json,
            description=description,
            tags=tags or [],
        )

        # Store in memory for retrieval
        self._examples[example_id] = example

        # Store embedding in vector store for similarity search
        item_id = f"{self._prefix}{example_id}"
        embedding = self._embedding_tool.generate_embedding(transcript)
        metadata = {
            "example_id": example_id,
            "description": description,
            "tags": ",".join(tags) if tags else "",
        }
        self._vector_search_tool.store(item_id, embedding, metadata)

    def get_similar_story_examples(
        self, query_text: str, top_k: int = 2
    ) -> list[FewShotExample]:
        """Retrieve the most similar few-shot examples for a given text.

        This is the method that StoryWriterAgent calls internally.

        Args:
            query_text: The input text to find similar examples for.
            top_k: Maximum number of examples to return.

        Returns:
            List of FewShotExample objects, ordered by relevance.
        """
        try:
            query_embedding = self._embedding_tool.generate_embedding(query_text)
            results = self._vector_search_tool.query_similar(query_embedding, top_k)
        except Exception as e:
            logger.warning("Few-shot retrieval failed: %s", e)
            return []

        examples = []
        for result in results:
            # Extract example_id from the result
            example_id = result.metadata.get("example_id", "")
            if example_id in self._examples:
                examples.append(self._examples[example_id])
            elif result.item_id.startswith(self._prefix):
                # Reconstruct from item_id
                eid = result.item_id[len(self._prefix):]
                if eid in self._examples:
                    examples.append(self._examples[eid])

        return examples[:top_k]

    def load_from_json(self, data: list[dict[str, Any]]) -> int:
        """Bulk load examples from a list of dicts.

        Expected format:
            [{"id": "ex1", "transcript": "...", "story_json": "...",
              "description": "...", "tags": ["auth"]}]

        Args:
            data: List of example dicts.

        Returns:
            Number of examples loaded.
        """
        count = 0
        for item in data:
            try:
                self.add_example(
                    example_id=item["id"],
                    transcript=item["transcript"],
                    story_json=item.get("story_json", "{}"),
                    description=item.get("description", ""),
                    tags=item.get("tags", []),
                )
                count += 1
            except Exception as e:
                logger.warning("Failed to load few-shot example %s: %s", item.get("id"), e)
        return count
