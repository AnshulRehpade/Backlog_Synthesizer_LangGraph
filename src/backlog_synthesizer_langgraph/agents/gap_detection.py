"""Gap Detection Agent for the Backlog Synthesizer system.

Compares extracted items against existing backlog tickets in Long-Term Memory
to classify each item as NEW, DUPLICATE, or CONFLICT.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5

Uses EmbeddingTool + VectorSearchTool (via LongTermMemory) for semantic
similarity comparison. Classification thresholds are configurable.

Embedding similarity measures topical closeness, not agreement: a paraphrase can
score below the duplicate threshold and a contradiction can score above it. When
an LLMGenerationTool is supplied, any item whose best match scores at or above
the conflict threshold is verified by the LLM, which decides duplicate, conflict
or unrelated. Without an LLM tool (or if the call fails), the threshold result stands.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from backlog_synthesizer_langgraph.memory.long_term import LongTermMemory
from backlog_synthesizer_langgraph.models.extraction import ExtractedItem
from backlog_synthesizer_langgraph.models.gap_detection import (
    RESOLUTIONS,
    ConflictFlag,
    DuplicateFlag,
    GapReport,
    GapReportEntry,
)
from backlog_synthesizer_langgraph.tools.interfaces import LLMGenerationTool, SearchResult

logger = logging.getLogger(__name__)

_VERIFY_SYSTEM_PROMPT = (
    "You compare a requirement from a meeting against an existing backlog ticket. "
    "Answer with JSON only: {\"verdict\": \"duplicate\" | \"conflict\" | \"unrelated\", "
    "\"reason\": \"<one sentence>\"}.\n"
    "- duplicate: the requirement asks for the same thing the ticket already covers, "
    "even if worded differently.\n"
    "- conflict: same area, but the requirement contradicts, replaces, or changes the "
    "ticket's scope, constraints, or approach (e.g. different limits, sync vs async, "
    "one technology instead of another).\n"
    "- unrelated: a different piece of work that merely shares vocabulary.\n"
    "For a conflict, also answer two questions as booleans:\n"
    "- \"is_product_work\": is the requirement something to build or change in the "
    "product? false for meeting logistics, agenda or planning notes (sizing, "
    "prioritizing, scheduling) and tentative ideas.\n"
    "- \"replaces_ticket\": does building the requirement mean removing, undoing, or "
    "redoing differently what the ticket describes (its limits, technology, "
    "architecture, or behavior)? false when the ticket's work stays in place and the "
    "requirement adds to it, covers one part of it, or is the next phase after it, "
    "even if the ticket alone would not be enough. The ticket's status says whether "
    "its work already exists."
)

_VERDICTS = {"duplicate", "conflict", "unrelated"}
# ConflictFlag.similarity_score is validated to this range.
_CONFLICT_SCORE_MIN, _CONFLICT_SCORE_MAX = 0.50, 0.85


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
        llm_tool: LLMGenerationTool | None = None,
        ticket_ids: list[str] | None = None,
    ) -> None:
        """Initialize GapDetectionAgent with dependencies.

        Args:
            long_term_memory: LongTermMemory for semantic search against existing tickets.
            duplicate_threshold: Similarity score at or above which items are DUPLICATE.
            conflict_threshold: Similarity score at or above which items may be CONFLICT.
            llm_tool: Optional LLM used to verify matches at or above conflict_threshold.
            ticket_ids: If given, only these backlog tickets can be matched. Memory also
                holds items from earlier runs and other backlogs, which are not tickets
                in the backlog being analyzed.
        """
        self._memory = long_term_memory
        self._duplicate_threshold = duplicate_threshold
        self._conflict_threshold = conflict_threshold
        self._llm_tool = llm_tool
        self._ticket_ids = ticket_ids

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
        if self._ticket_ids is not None and not self._ticket_ids:
            # Empty backlog: nothing to match against.
            return self._new_entry(item, 0.0)

        where = {"ticket_id": {"$in": self._ticket_ids}} if self._ticket_ids else None
        try:
            # Search for similar items in long-term memory (sync call → run in thread)
            results: list[SearchResult] = await asyncio.to_thread(
                self._memory.search_similar,
                item.text,
                5,  # top_k
                where,
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

        if score < self._conflict_threshold:
            # NEW — below conflict threshold
            return self._new_entry(item, score)

        # Embedding similarity can't tell agreement from contradiction, so let the
        # LLM make the call when one is available.
        verdict, reason, suggestion = await self._verify_with_llm(item, best_match)
        if verdict == "unrelated":
            return self._new_entry(item, score)
        if verdict == "duplicate":
            return self._duplicate_entry(item, best_match, score)
        if verdict == "conflict":
            return self._conflict_entry(item, best_match, score, reason, suggestion)

        # No verdict (no LLM tool, or the call failed) — fall back to thresholds.
        if score >= self._duplicate_threshold:
            return self._duplicate_entry(item, best_match, score)
        return self._conflict_entry(
            item,
            best_match,
            score,
            f"Similar to existing ticket {best_match.item_id} "
            f"(score: {score:.2f}) but may represent a conflicting requirement",
        )

    @staticmethod
    def _new_entry(item: ExtractedItem, score: float) -> GapReportEntry:
        return GapReportEntry(
            item=item,
            classification="new",
            gap_type="NEW",
            confidence=0.9,
            similarity_score=score,
        )

    @staticmethod
    def _duplicate_entry(
        item: ExtractedItem, match: SearchResult, score: float
    ) -> GapReportEntry:
        return GapReportEntry(
            item=item,
            classification="duplicate",
            gap_type="DUPLICATE",
            confidence=score,
            similarity_score=score,
            similar_ticket_id=match.item_id,
            duplicate_info=DuplicateFlag(
                extracted_item=item,
                matching_ticket_id=match.item_id,
                similarity_score=score,
            ),
        )

    @staticmethod
    def _conflict_entry(
        item: ExtractedItem,
        match: SearchResult,
        score: float,
        description: str,
        suggestion: str | None = None,
    ) -> GapReportEntry:
        return GapReportEntry(
            item=item,
            classification="conflict",
            gap_type="CONFLICT",
            confidence=score,
            similarity_score=score,
            similar_ticket_id=match.item_id,
            conflict_info=ConflictFlag(
                item_a=item,
                item_b_ticket_id=match.item_id,
                similarity_score=min(max(score, _CONFLICT_SCORE_MIN), _CONFLICT_SCORE_MAX),
                contradiction_description=description,
                suggested_resolution=suggestion or "meeting_wins",
            ),
        )

    async def _verify_with_llm(
        self, item: ExtractedItem, match: SearchResult
    ) -> tuple[str | None, str, str | None]:
        """Ask the LLM whether an item duplicates, conflicts with, or is unrelated to its match.

        Returns:
            (verdict, reason, suggestion). verdict is None when there is no LLM
            tool or the call or its output is unusable, so the caller falls back
            to thresholds. suggestion is a RESOLUTIONS key for conflicts, else None.
        """
        if self._llm_tool is None:
            return None, "", None

        ticket_text = match.metadata.get("content") or match.metadata.get("title") or ""
        status = match.metadata.get("status")
        status_text = f" (status: {status})" if status else ""
        prompt = (
            f"Existing ticket {match.item_id}{status_text}: {ticket_text}\n"
            f"New requirement: {item.text}\n"
        )
        try:
            raw = await asyncio.to_thread(
                self._llm_tool.generate, prompt, _VERIFY_SYSTEM_PROMPT
            )
        except Exception as e:
            logger.warning("LLM verification failed for '%s': %s", item.text[:50], e)
            return None, "", None

        return self._parse_verdict(raw)

    @staticmethod
    def _parse_verdict(raw: Any) -> tuple[str | None, str, str | None]:
        if not isinstance(raw, str):
            return None, "", None
        found = re.search(r"\{.*\}", raw, re.DOTALL)
        if not found:
            return None, "", None
        try:
            data = json.loads(found.group(0))
        except json.JSONDecodeError:
            return None, "", None
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict not in _VERDICTS:
            return None, "", None
        suggestion = _suggest_resolution(data) if verdict == "conflict" else None
        return verdict, str(data.get("reason", "")).strip(), suggestion


def _suggest_resolution(data: dict) -> str | None:
    """Derive a RESOLUTIONS key from the judge's two yes/no answers.

    Asking narrow factual questions and mapping them in code is steadier than
    asking the model to pick a resolution label directly.
    """
    if data.get("is_product_work") is False:
        return "keep_ticket"
    if data.get("replaces_ticket") is False:
        return "keep_both"
    if data.get("replaces_ticket") is True:
        return "meeting_wins"
    suggestion = str(data.get("suggestion", "")).strip().lower()
    return suggestion if suggestion in RESOLUTIONS else None
