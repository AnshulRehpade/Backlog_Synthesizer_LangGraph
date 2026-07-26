"""Gap Detection Agent for the Backlog Synthesizer system.

Compares extracted items against existing backlog tickets in Long-Term Memory
to classify each item as NEW, DUPLICATE, or CONFLICT.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5

Uses EmbeddingTool + VectorSearchTool (via LongTermMemory) for semantic
similarity comparison. Classification thresholds are configurable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from backlog_synthesizer_langgraph.memory.long_term import LongTermMemory
from backlog_synthesizer_langgraph.models.extraction import ExtractedItem
from backlog_synthesizer_langgraph.models.gap_detection import (
    ConflictFlag,
    DuplicateFlag,
    GapReport,
    GapReportEntry,
)
from backlog_synthesizer_langgraph.tools.interfaces import SearchResult

logger = logging.getLogger(__name__)


class GapDetectionAgent:
    """Agent that classifies extracted items against existing backlog.

    For each extracted item, performs a semantic similarity search against
    Long-Term Memory (existing backlog tickets). Based on the similarity
    score and configured thresholds, classifies each item as:
    - NEW: No similar existing ticket (score < conflict_threshold)
    - CONFLICT: Similar but contradictory (conflict_threshold <= score < duplicate_threshold)
    - DUPLICATE: Very similar to existing ticket (score >= duplicate_threshold)

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """

    def __init__(
        self,
        long_term_memory: LongTermMemory,
        duplicate_threshold: float = 0.80,
        conflict_threshold: float = 0.50,
    ) -> None:
        """Initialize GapDetectionAgent with dependencies.

        Args:
            long_term_memory: LongTermMemory for semantic search against existing tickets.
            duplicate_threshold: Similarity score at or above which items are DUPLICATE.
            conflict_threshold: Similarity score at or above which items may be CONFLICT.
        """
        self._memory = long_term_memory
        self._duplicate_threshold = duplicate_threshold
        self._conflict_threshold = conflict_threshold

    async def analyze_gaps(
        self,
        items: list[ExtractedItem],
    ) -> GapReport:
        """Analyze extracted items against existing backlog for gaps.

        For each item, searches Long-Term Memory for similar tickets and
        classifies the item based on similarity scores.

        Args:
            items: List of ExtractedItem objects from the parser.

        Returns:
            A GapReport with classified entries and summary counts.
        """
        if not items:
            return GapReport(
                entries=[],
                total_new=0,
                total_duplicates=0,
                total_conflicts=0,
                total_unprocessed=0,
            )

        # Process all items concurrently
        coros = [self._classify_item(item) for item in items]
        entries = await asyncio.gather(*coros)

        # Compute summary counts
        total_new = sum(1 for e in entries if e.classification == "new")
        total_duplicates = sum(1 for e in entries if e.classification == "duplicate")
        total_conflicts = sum(1 for e in entries if e.classification == "conflict")
        total_unprocessed = sum(1 for e in entries if e.classification == "unprocessed")

        return GapReport(
            entries=list(entries),
            total_new=total_new,
            total_duplicates=total_duplicates,
            total_conflicts=total_conflicts,
            total_unprocessed=total_unprocessed,
        )

    async def _classify_item(self, item: ExtractedItem) -> GapReportEntry:
        """Classify a single extracted item against the existing backlog.

        Args:
            item: The extracted item to classify.

        Returns:
            A GapReportEntry with classification and similarity details.
        """
        try:
            # Search for similar items in long-term memory (sync call → run in thread)
            results: list[SearchResult] = await asyncio.to_thread(
                self._memory.search_similar,
                item.text,
                5,  # top_k
            )
        except Exception as e:
            logger.warning("Search failed for item '%s': %s", item.text[:50], e)
            return GapReportEntry(
                item=item,
                classification="unprocessed",
                gap_type="UNPROCESSED",
                confidence=0.0,
                error_reason=f"Similarity search failed: {e}",
            )

        if not results:
            # No similar items found → NEW
            return GapReportEntry(
                item=item,
                classification="new",
                gap_type="NEW",
                confidence=0.95,
                similarity_score=0.0,
            )

        # Use the best (highest score) match
        best_match = results[0]
        score = best_match.score

        if score >= self._duplicate_threshold:
            # DUPLICATE — very similar to existing ticket
            return GapReportEntry(
                item=item,
                classification="duplicate",
                gap_type="DUPLICATE",
                confidence=score,
                similarity_score=score,
                similar_ticket_id=best_match.item_id,
                duplicate_info=DuplicateFlag(
                    extracted_item=item,
                    matching_ticket_id=best_match.item_id,
                    similarity_score=score,
                ),
            )
        elif score >= self._conflict_threshold:
            # CONFLICT — similar but potentially contradictory
            return GapReportEntry(
                item=item,
                classification="conflict",
                gap_type="CONFLICT",
                confidence=score,
                similarity_score=score,
                similar_ticket_id=best_match.item_id,
                conflict_info=ConflictFlag(
                    item_a=item,
                    item_b_ticket_id=best_match.item_id,
                    similarity_score=score,
                    contradiction_description=(
                        f"Similar to existing ticket {best_match.item_id} "
                        f"(score: {score:.2f}) but may represent a conflicting requirement"
                    ),
                ),
            )
        else:
            # NEW — below conflict threshold
            return GapReportEntry(
                item=item,
                classification="new",
                gap_type="NEW",
                confidence=0.9,
                similarity_score=score,
            )
