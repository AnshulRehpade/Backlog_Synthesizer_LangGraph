"""Conditional edge functions for the Backlog Synthesizer LangGraph pipeline.

These functions implement the three-tier decision pattern:
    Level 1: Conditional edges (deterministic, zero cost)
    Level 2: LLM fallback (for unexpected situations)
    Level 3: Human in the loop (for critical decisions — future)

Routing logic lives here, never inside nodes.
"""

from __future__ import annotations

import logging

from langgraph.graph import END

from graph.state import PipelineState

logger = logging.getLogger(__name__)


def route_after_parser(state: PipelineState) -> str:
    """Determine the next step after the Parser node completes.

    Three-tier decision pattern:

    Level 1 — Conditional edge (deterministic, zero LLM cost):
        - If status == "permanent_failure" → "error_node"
        - If parser_output is None → "error_node"
        - If 0 items extracted → "llm_fallback_node"
        - If < 3 items extracted → "llm_fallback_node"
          (too few items — may indicate vague transcript)
        - If all items have confidence < 0.3 → "llm_fallback_node"
          (parser uncertain about everything)
        - Otherwise → "gap_detection" (success)

    Args:
        state: The current PipelineState after parser_node.

    Returns:
        One of: "error_node", "llm_fallback_node", or "gap_detection".
    """
    # Check for permanent failure
    status = state.get("status", "in_progress")
    if status == "permanent_failure":
        logger.info("Routing after parser: permanent_failure → error_node")
        return "error_node"

    # Check parser_output exists
    parser_output = state.get("parser_output")
    if parser_output is None:
        logger.info("Routing after parser: parser_output is None → error_node")
        return "error_node"

    # Get extracted items
    items = parser_output.items if hasattr(parser_output, "items") else []
    item_count = len(items)

    # Check for 0 items
    if item_count == 0:
        retry_count = state.get("retry_count", 0)
        if retry_count >= 1:
            logger.info(
                "Routing after parser: 0 items extracted, retry_count=%d → error_node",
                retry_count,
            )
            return "error_node"
        logger.info("Routing after parser: 0 items extracted → llm_fallback_node")
        return "llm_fallback_node"

    # Check for < 3 items (too few — may indicate vague transcript)
    if item_count < 3:
        retry_count = state.get("retry_count", 0)
        if retry_count >= 1:
            logger.info(
                "Routing after parser: only %d items (< 3), retry_count=%d → error_node",
                item_count,
                retry_count,
            )
            return "error_node"
        logger.info(
            "Routing after parser: only %d items (< 3) → llm_fallback_node",
            item_count,
        )
        return "llm_fallback_node"

    # Check if all items have confidence < 0.3
    all_low_confidence = all(
        item.confidence < 0.3 for item in items
    )
    if all_low_confidence:
        retry_count = state.get("retry_count", 0)
        if retry_count >= 1:
            logger.info(
                "Routing after parser: all %d items have confidence < 0.3, "
                "retry_count=%d → error_node",
                item_count,
                retry_count,
            )
            return "error_node"
        logger.info(
            "Routing after parser: all %d items have confidence < 0.3 → llm_fallback_node",
            item_count,
        )
        return "llm_fallback_node"

    # Success path
    logger.info(
        "Routing after parser: success (%d items extracted) → gap_detection",
        item_count,
    )
    return "gap_detection"


def route_after_gap_detection(state: PipelineState) -> str:
    """Determine the next step after the Gap Detection node completes.

    Three-tier decision pattern:

    Level 1 — Conditional edge (deterministic, zero LLM cost):
        - If status == "permanent_failure" → "error_node"
        - If gap_output is None → "error_node"
        - If all items are DUPLICATE (new_count == 0 and conflict_count == 0)
          → "llm_fallback_node" (nothing new found — unexpected)
        - If new_count + conflict_count == 0
          → "llm_fallback_node" (nothing to process downstream)
        - If duplicate_count / total_items > 0.8
          → "llm_fallback_node" (more than 80% duplicates — flag for review)
        - Otherwise → "story_writer" (success)

    Args:
        state: The current PipelineState after gap_detection_node.

    Returns:
        One of: "error_node", "llm_fallback_node", or "story_writer".
    """
    # Check for permanent failure
    status = state.get("status", "in_progress")
    if status == "permanent_failure":
        logger.info("Routing after gap_detection: permanent_failure → error_node")
        return "error_node"

    # Check gap_output exists
    gap_output = state.get("gap_output")
    if gap_output is None:
        logger.info("Routing after gap_detection: gap_output is None → error_node")
        return "error_node"

    # Compute totals
    new_count = gap_output.total_new
    conflict_count = gap_output.total_conflicts
    duplicate_count = gap_output.total_duplicates
    total_items = len(gap_output.entries)

    # Check: all items are duplicates (new + conflict == 0)
    if new_count == 0 and conflict_count == 0:
        retry_count = state.get("retry_count", 0)
        if retry_count >= 1:
            logger.info(
                "Routing after gap_detection: all duplicates, retry_count=%d → error_node",
                retry_count,
            )
            return "error_node"
        logger.info(
            "Routing after gap_detection: 0 new + 0 conflicts (all duplicates) → llm_fallback_node"
        )
        return "llm_fallback_node"

    # Check: >80% duplicates
    if total_items > 0 and (duplicate_count / total_items) > 0.8:
        retry_count = state.get("retry_count", 0)
        if retry_count >= 1:
            logger.info(
                "Routing after gap_detection: >80%% duplicates (%d/%d), "
                "retry_count=%d → error_node",
                duplicate_count,
                total_items,
                retry_count,
            )
            return "error_node"
        logger.info(
            "Routing after gap_detection: >80%% duplicates (%d/%d) → llm_fallback_node",
            duplicate_count,
            total_items,
        )
        return "llm_fallback_node"

    # Success path
    logger.info(
        "Routing after gap_detection: success (%d new, %d conflicts, %d duplicates) → story_writer",
        new_count,
        conflict_count,
        duplicate_count,
    )
    return "story_writer"


def route_after_story_writer(state: PipelineState) -> str:
    """Determine the next step after the Story Writer node completes.

    Three-tier decision pattern:

    Level 1 — Conditional edge (deterministic, zero LLM cost):
        - If status == "permanent_failure" → "error_node"
        - If story_output is None → "error_node"
        - If 0 epics generated → "llm_fallback_node"
        - If >50% stories tagged needs_refinement → "llm_fallback_node"
        - Otherwise → END (success)

    Level 2 — LLM fallback node handles unexpected situations
        (routed to via "llm_fallback_node" return value)

    Level 3 — Human in the loop for critical decisions (future)

    Args:
        state: The current PipelineState after story_writer_node.

    Returns:
        One of: "error_node", "llm_fallback_node", or END.
    """
    # Check for permanent failure
    status = state.get("status", "in_progress")
    if status == "permanent_failure":
        logger.info("Routing after story_writer: permanent_failure → error_node")
        return "error_node"

    # Check story_output exists
    story_output = state.get("story_output")
    if story_output is None:
        logger.info("Routing after story_writer: story_output is None → error_node")
        return "error_node"

    # Check for 0 epics
    retry_count = state.get("retry_count", 0)
    if not story_output.epics:
        if retry_count >= 1:
            logger.info(
                "Routing after story_writer: 0 epics generated, but retry_count=%d → error_node",
                retry_count,
            )
            return "error_node"
        logger.info("Routing after story_writer: 0 epics generated → llm_fallback_node")
        return "llm_fallback_node"

    # Check if >50% stories need refinement
    total_stories = 0
    needs_refinement_count = 0
    for epic in story_output.epics:
        for story in epic.stories:
            total_stories += 1
            if story.needs_refinement:
                needs_refinement_count += 1

    if total_stories > 0 and (needs_refinement_count / total_stories) > 0.5:
        if retry_count >= 1:
            logger.info(
                "Routing after story_writer: %d/%d need refinement (>50%%), "
                "retry_count=%d → error_node",
                needs_refinement_count,
                total_stories,
                retry_count,
            )
            return "error_node"
        logger.info(
            "Routing after story_writer: %d/%d stories need refinement (>50%%) → llm_fallback_node",
            needs_refinement_count,
            total_stories,
        )
        return "llm_fallback_node"

    # Success path
    logger.info(
        "Routing after story_writer: success (%d stories, %d epics) → END",
        total_stories,
        len(story_output.epics),
    )
    return END


def route_after_llm_fallback(state: PipelineState) -> str:
    """Determine the next step after the LLM Fallback node completes.

    Reads the last react_note to find what action was taken by the fallback
    node, then routes accordingly:

    Retry actions (graph routes back to the target node):
        - "retry_parser" → "parser"
        - "lower_threshold" → "gap_detection"
        - "retry_story_writer" → "story_writer"

    Proceed actions (continue pipeline from where it left off):
        - "proceed_with_low_confidence" → "gap_detection"
        - "proceed_with_conflicts_only" → "story_writer"

    Human review:
        - "request_review" → "human_review"

    Terminal actions (pipeline ends here):
        - "return_partial", "return_nothing_new", "halt",
          "request_better_transcript", or anything else → END

    Args:
        state: The current PipelineState after llm_fallback_node.

    Returns:
        One of: "parser", "gap_detection", "story_writer", "human_review", or END.
    """
    # Read the last react_note to find the action
    react_notes = state.get("react_notes", [])
    if not react_notes:
        logger.info("Routing after llm_fallback: no react_notes → END")
        return END

    last_note = react_notes[-1] if react_notes else {}
    action = last_note.get("action", "") if isinstance(last_note, dict) else ""

    # Retry actions → route back to target node
    if action == "retry_parser":
        logger.info("Routing after llm_fallback: retry_parser → parser")
        return "parser"
    if action == "lower_threshold":
        logger.info("Routing after llm_fallback: lower_threshold → gap_detection")
        return "gap_detection"
    if action == "retry_story_writer":
        logger.info("Routing after llm_fallback: retry_story_writer → story_writer")
        return "story_writer"

    # Proceed actions → continue from appropriate step
    if action == "proceed_with_low_confidence":
        logger.info("Routing after llm_fallback: proceed_with_low_confidence → gap_detection")
        return "gap_detection"
    if action == "proceed_with_conflicts_only":
        logger.info("Routing after llm_fallback: proceed_with_conflicts_only → story_writer")
        return "story_writer"

    # Human review
    if action == "request_review":
        logger.info("Routing after llm_fallback: request_review → human_review")
        return "human_review"

    # Terminal actions — pipeline ends
    logger.info("Routing after llm_fallback: action='%s' → END", action)
    return END


def route_after_human_review(state: PipelineState) -> str:
    """Determine the next step after human review completes.

    Reads the status and current_step to decide where to route:
        - If status == "permanent_failure" → END (human rejected)
        - If current_step indicates gap_detection context → "gap_detection"
        - If current_step indicates story_writer context → "story_writer"
        - Otherwise → END

    Args:
        state: The current PipelineState after human_review_node.

    Returns:
        One of: "gap_detection", "story_writer", or END.
    """
    status = state.get("status", "in_progress")
    if status == "permanent_failure":
        logger.info("Routing after human_review: permanent_failure → END")
        return END

    # Read the last react_note to find where to continue
    react_notes = state.get("react_notes", [])
    if react_notes:
        last_note = react_notes[-1] if react_notes else {}
        action = last_note.get("action", "") if isinstance(last_note, dict) else ""

        if action in ("approve_gap_detection", "modify_threshold"):
            logger.info("Routing after human_review: → gap_detection")
            return "gap_detection"
        if action == "approve_story_writer":
            logger.info("Routing after human_review: → story_writer")
            return "story_writer"

    logger.info("Routing after human_review: → END")
    return END
