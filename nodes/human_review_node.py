"""Human Review node — Level 3 of the three-tier decision pattern.

Uses LangGraph's interrupt() to pause graph execution and wait for
human input. When the pipeline encounters a situation requiring human
judgment (e.g., ambiguous duplicates, conflicting data), the LLM fallback
routes here.

The graph pauses at interrupt(), persists state via the checkpointer,
and resumes when the human provides a decision via Command(resume=...).

Never raises exceptions — always returns state gracefully.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from langgraph.types import interrupt

from graph.state import PipelineState
from backlog_synthesizer_langgraph.memory.audit_log import AuditLog
from backlog_synthesizer_langgraph.models.memory import AuditEntry
from backlog_synthesizer_langgraph.observability import get_tracer

logger = logging.getLogger(__name__)

# Module-level audit log instance
_audit_log = AuditLog()


def get_audit_log() -> AuditLog:
    """Return the module-level audit log for external access (e.g., testing)."""
    return _audit_log


async def human_review_node(state: PipelineState) -> PipelineState:
    """LangGraph node that pauses execution for human review.

    Implements Level 3 of the three-tier decision pattern. Uses
    LangGraph's interrupt() to pause the graph and present a review
    summary to the human operator.

    The human can:
    - "approve" → continue to the appropriate next node
    - "reject" → set status to permanent_failure
    - "modify_threshold" → route back to gap_detection with adjusted threshold

    Args:
        state: The current PipelineState.

    Returns:
        Updated PipelineState reflecting the human decision.
    """
    tracer = get_tracer()
    start_time = time.time()

    # Start OTel span
    span = None
    if tracer is not None:
        span = tracer.start_span("human_review")
        span.set_attribute("session_id", state["session_id"])

    try:
        return await _execute_human_review(state, start_time, span)
    except Exception as e:
        # Safety net — never crash the pipeline
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("Unexpected error in human_review_node: %s", e)

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.end()

        return {
            "current_step": "human_review",
            "status": "partial_failure",
            "needs_human_review": False,
            "errors": [
                {
                    "node": "human_review",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }


async def _execute_human_review(
    state: PipelineState, start_time: float, span: object | None
) -> PipelineState:
    """Core logic for human review node.

    Args:
        state: Current pipeline state.
        start_time: When the node started.
        span: Optional OTel span.

    Returns:
        Updated state dict reflecting human decision.
    """
    # Build review summary for the human
    review_summary = _build_review_summary(state)

    # Determine which step triggered the review
    triggering_step = _get_triggering_step(state)

    # Pause execution and wait for human input
    # LangGraph persists state here via the checkpointer.
    # Execution resumes when the caller provides Command(resume=<decision>).
    human_decision = interrupt({
        "message": "Human review required",
        "current_step": triggering_step,
        "summary": review_summary,
        "options": ["approve", "reject", "modify_threshold"],
    })

    duration_ms = int((time.time() - start_time) * 1000)

    # Handle human decision
    if isinstance(human_decision, dict):
        decision = human_decision.get("decision", "reject")
    elif isinstance(human_decision, str):
        decision = human_decision
    else:
        decision = "reject"

    logger.info("Human review decision: %s (triggering_step: %s)", decision, triggering_step)

    # Record span attributes
    if span is not None:
        span.set_attribute("decision", decision)
        span.set_attribute("triggering_step", triggering_step)
        span.set_attribute("duration_ms", duration_ms)
        span.end()

    # Log to audit log
    _log_to_audit(
        session_id=state["session_id"],
        decision=decision,
        triggering_step=triggering_step,
        duration_ms=duration_ms,
    )

    # Route based on decision
    if decision == "approve":
        # Determine where to continue based on what triggered the review
        if triggering_step == "gap_detection":
            action = "approve_gap_detection"
        else:
            action = "approve_story_writer"

        return {
            "current_step": "human_review",
            "status": "in_progress",
            "needs_human_review": False,
            "errors": [],
            "react_notes": [
                {
                    "node": "human_review",
                    "action": action,
                    "note": f"Human approved: continuing from {triggering_step}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    elif decision == "modify_threshold":
        return {
            "current_step": "human_review",
            "status": "in_progress",
            "needs_human_review": False,
            "errors": [],
            "react_notes": [
                {
                    "node": "human_review",
                    "action": "modify_threshold",
                    "note": "Human requested threshold modification — re-running gap detection",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    else:
        # "reject" or unknown → permanent failure
        return {
            "current_step": "human_review",
            "status": "permanent_failure",
            "needs_human_review": False,
            "errors": [],
            "react_notes": [
                {
                    "node": "human_review",
                    "action": "reject",
                    "note": f"Human rejected: pipeline terminated (decision: {decision})",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }


def _build_review_summary(state: PipelineState) -> str:
    """Build a concise summary for human review.

    Args:
        state: Current pipeline state.

    Returns:
        Multi-line summary string.
    """
    lines = []

    # What triggered the review
    react_notes = state.get("react_notes", [])
    if react_notes:
        last_note = react_notes[-1] if react_notes else {}
        if isinstance(last_note, dict):
            lines.append(f"Trigger: {last_note.get('note', 'Unknown')}")

    # Parser output summary
    parser_output = state.get("parser_output")
    if parser_output is not None:
        items = parser_output.items if hasattr(parser_output, "items") else []
        lines.append(f"Extracted items: {len(items)}")

    # Gap output summary
    gap_output = state.get("gap_output")
    if gap_output is not None:
        lines.append(
            f"Gap analysis: {gap_output.total_new} new, "
            f"{gap_output.total_duplicates} duplicates, "
            f"{gap_output.total_conflicts} conflicts"
        )

    # Story output summary
    story_output = state.get("story_output")
    if story_output is not None:
        epic_count = len(story_output.epics)
        story_count = sum(len(e.stories) for e in story_output.epics)
        lines.append(f"Stories: {story_count} in {epic_count} epics")

    # Errors
    errors = state.get("errors", [])
    if errors:
        lines.append(f"Errors: {len(errors)}")

    return "\n".join(lines) if lines else "No additional context available."


def _get_triggering_step(state: PipelineState) -> str:
    """Determine which pipeline step triggered the human review.

    Looks at react_notes to find the step that led to request_review.

    Args:
        state: Current pipeline state.

    Returns:
        Name of the triggering step (e.g., "gap_detection", "story_writer").
    """
    react_notes = state.get("react_notes", [])
    # Walk backwards to find the llm_fallback note that triggered review
    for note in reversed(react_notes):
        if isinstance(note, dict) and note.get("node") == "llm_fallback":
            action = note.get("action", "")
            if action == "request_review":
                # The current_step recorded in the note tells us what triggered it
                # but since llm_fallback overwrites current_step, we look at
                # what step the gap_output or parser_output suggests
                break

    # Infer from state: if gap_output exists, it was gap_detection
    if state.get("gap_output") is not None:
        return "gap_detection"
    if state.get("parser_output") is not None:
        return "parser"
    return "unknown"


def _log_to_audit(
    session_id: str,
    decision: str,
    triggering_step: str,
    duration_ms: int,
) -> None:
    """Record the human review decision to the audit log.

    Args:
        session_id: Pipeline session ID.
        decision: The human's decision.
        triggering_step: Which step triggered the review.
        duration_ms: Time spent in human review (including wait time).
    """
    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        agent_name="human_review",
        input_summary=f"triggering_step={triggering_step}",
        output_summary=f"decision={decision}",
        duration_ms=duration_ms,
    )

    try:
        _audit_log.record(session_id, entry)
    except Exception as e:
        logger.warning("Failed to record audit log entry: %s", e)
