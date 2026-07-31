"""Gap Detection node — wraps GapDetectionAgent as a LangGraph node function.

This node:
1. Reads parser_output.items and inputs.backlog_tickets from state
2. Loads backlog tickets into Long-Term Memory
3. Calls purge_expired() to enforce 30-day retention
4. Calls GapDetectionAgent.analyze_gaps() unchanged
5. Stores extracted items in Long-Term Memory AFTER gap detection
6. Writes gap_output back to state
7. Records OTel span and audit log entry

Never raises exceptions — always returns state gracefully.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from graph.state import PipelineState
from backlog_synthesizer_langgraph.agents.gap_detection import GapDetectionAgent
from backlog_synthesizer_langgraph.config import PipelineConfig
from backlog_synthesizer_langgraph.memory.audit_log import AuditLog
from backlog_synthesizer_langgraph.memory.long_term import LongTermMemory
from backlog_synthesizer_langgraph.models.memory import AuditEntry
from backlog_synthesizer_langgraph.observability import get_tracer
from backlog_synthesizer_langgraph.tools.errors import (
    PermanentToolError,
    TransientToolError,
)

logger = logging.getLogger(__name__)

# Module-level audit log instance
_audit_log = AuditLog()


def get_audit_log() -> AuditLog:
    """Return the module-level audit log for external access (e.g., testing)."""
    return _audit_log


async def gap_detection_node(state: PipelineState) -> PipelineState:
    """LangGraph node that classifies extracted items against existing backlog.

    Reads parser_output from state, loads backlog tickets into Long-Term Memory,
    invokes GapDetectionAgent.analyze_gaps(), then stores new items in memory.

    Error handling:
        - TransientToolError → appends error, sets status to "partial_failure"
        - PermanentToolError → appends error, sets status to "permanent_failure"
        - Any other exception → appends error, sets status to "partial_failure"
        - Never raises out of the node — always returns state.

    Args:
        state: The current PipelineState.

    Returns:
        Updated PipelineState with gap_output populated (or errors appended).
    """
    tracer = get_tracer()
    start_time = time.time()

    # Start OTel span
    span = None
    if tracer is not None:
        span = tracer.start_span("agent.gap_detection")
        span.set_attribute("session_id", state["session_id"])

    try:
        return await _execute_gap_detection(state, start_time, span)
    except Exception as e:
        # Catch-all safety net — guarantees we never raise out of the node.
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("Unexpected error in gap_detection_node: %s", e)

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.end()

        return {
            "current_step": "gap_detection",
            "status": "partial_failure",
            "errors": [
                {
                    "node": "gap_detection",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }


async def _execute_gap_detection(
    state: PipelineState, start_time: float, span: object | None
) -> PipelineState:
    """Core logic for the gap detection node, separated for clarity.

    Args:
        state: The current pipeline state.
        start_time: Timestamp when the node started.
        span: Optional OTel span for tracing.

    Returns:
        Updated state dict with gap_output or error information.
    """
    # 1. Read parser_output from state
    parser_output = state.get("parser_output")
    if parser_output is None:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "gap_detection",
            "error_type": "MissingInput",
            "message": "parser_output is None — cannot detect gaps without extracted items",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "gap_detection",
            "status": "permanent_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    extracted_items = parser_output.items

    # 1b. Confidence-based filtering — remove noise before gap detection
    confidence_threshold = 0.3
    filtered_items = [item for item in extracted_items if item.confidence >= confidence_threshold]
    filtered_out = len(extracted_items) - len(filtered_items)
    if filtered_out > 0:
        logger.info(
            "Confidence filter: removed %d/%d items below %.2f threshold",
            filtered_out, len(extracted_items), confidence_threshold,
        )
    extracted_items = filtered_items
    items_count = len(extracted_items)

    if span is not None:
        span.set_attribute("items_processed", items_count)
        if filtered_out > 0:
            span.set_attribute("items_filtered_low_confidence", filtered_out)

    # 2. Initialize tools via PipelineConfig
    try:
        config = PipelineConfig.from_env()
        embedding_tool = config.create_embedding_tool()
        vector_search_tool = config.create_vector_search_tool()
    except PermanentToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "gap_detection",
            "error_type": "PermanentToolError",
            "message": f"Failed to create tools: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "gap_detection",
            "status": "permanent_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    # 3. Create Long-Term Memory instance
    long_term_memory = LongTermMemory(
        embedding_tool=embedding_tool,
        vector_search_tool=vector_search_tool,
    )

    # 4. Purge expired entries (30-day retention)
    try:
        expired_ids = long_term_memory.purge_expired()
        if expired_ids:
            logger.info("Purged %d expired entries from long-term memory", len(expired_ids))
    except Exception as e:
        # Purge failure is non-fatal — log and continue
        logger.warning("Failed to purge expired entries: %s", e)

    # 5. Load existing backlog tickets into Long-Term Memory
    inputs = state.get("inputs")
    backlog_tickets = inputs.backlog_tickets if inputs else []

    try:
        for ticket in backlog_tickets:
            content = f"{ticket.title} {ticket.description}"
            metadata = {
                "ticket_id": ticket.id,
                "title": ticket.title,
                "status": ticket.status,
                "tags": ",".join(ticket.tags) if ticket.tags else "",
            }
            long_term_memory.store_item(
                item_id=ticket.id,
                content=content,
                metadata=metadata,
            )
    except TransientToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "gap_detection",
            "error_type": "TransientToolError",
            "message": f"Transient error loading backlog tickets: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.warning(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "gap_detection",
            "status": "partial_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }
    except PermanentToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "gap_detection",
            "error_type": "PermanentToolError",
            "message": f"Permanent error loading backlog tickets: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "gap_detection",
            "status": "permanent_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    # 6. Create GapDetectionAgent and analyze gaps
    agent = GapDetectionAgent(
        long_term_memory=long_term_memory,
        duplicate_threshold=config.gap_detection_duplicate_threshold,
        conflict_threshold=config.gap_detection_conflict_threshold,
    )

    try:
        result = await agent.analyze_gaps(extracted_items)
    except TransientToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "gap_detection",
            "error_type": "TransientToolError",
            "message": f"Transient error during gap analysis: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.warning(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "gap_detection",
            "status": "partial_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }
    except PermanentToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "gap_detection",
            "error_type": "PermanentToolError",
            "message": f"Permanent error during gap analysis: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "gap_detection",
            "status": "permanent_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    # 7. Store extracted items in Long-Term Memory AFTER gap detection
    #    (never before — prevents self-matching bug)
    try:
        for i, item in enumerate(extracted_items):
            item_id = f"{state['session_id']}_item_{i}"
            metadata = {
                "item_type": item.item_type,
                "confidence": item.confidence,
                "tags": ",".join(item.tags) if item.tags else "",
                "session_id": state["session_id"],
            }
            long_term_memory.store_item(
                item_id=item_id,
                content=item.text,
                metadata=metadata,
            )
    except Exception as e:
        # Non-fatal — items not stored for future searches but pipeline continues
        logger.warning("Failed to store extracted items in long-term memory: %s", e)

    # 8. Compute metrics
    duration_ms = int((time.time() - start_time) * 1000)

    # 9. Record OTel span attributes
    if span is not None:
        span.set_attribute("items_processed", items_count)
        span.set_attribute("duplicate_count", result.total_duplicates)
        span.set_attribute("conflict_count", result.total_conflicts)
        span.set_attribute("new_count", result.total_new)
        span.set_attribute("unprocessed_count", result.total_unprocessed)
        span.set_attribute("duration_ms", duration_ms)
        span.set_attribute("backlog_tickets_loaded", len(backlog_tickets))
        span.end()

    # 10. Log to audit log
    _log_to_audit(
        session_id=state["session_id"],
        items_processed=items_count,
        duplicate_count=result.total_duplicates,
        conflict_count=result.total_conflicts,
        new_count=result.total_new,
        unprocessed_count=result.total_unprocessed,
        duration_ms=duration_ms,
    )

    # 11. Build react notes
    react_notes = []
    react_notes.append({
        "node": "gap_detection",
        "note": (
            f"Analyzed {items_count} items: "
            f"{result.total_new} new, "
            f"{result.total_duplicates} duplicates, "
            f"{result.total_conflicts} conflicts, "
            f"{result.total_unprocessed} unprocessed."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # 12. Determine status
    status = state.get("status", "in_progress")

    logger.info(
        "Gap detection completed: %d items → %d new, %d dup, %d conflict, %d unproc, %dms",
        items_count,
        result.total_new,
        result.total_duplicates,
        result.total_conflicts,
        result.total_unprocessed,
        duration_ms,
    )

    return {
        "gap_output": result,
        "current_step": "gap_detection",
        "status": status,
        "errors": [],
        "react_notes": react_notes,
        "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
    }


def _log_to_audit(
    session_id: str,
    items_processed: int,
    duplicate_count: int,
    conflict_count: int,
    new_count: int,
    unprocessed_count: int,
    duration_ms: int,
) -> None:
    """Record the gap detection invocation to the audit log.

    Args:
        session_id: Pipeline session ID.
        items_processed: Total items analyzed.
        duplicate_count: Items classified as duplicates.
        conflict_count: Items classified as conflicts.
        new_count: Items classified as new.
        unprocessed_count: Items that could not be processed.
        duration_ms: Processing duration.
    """
    input_summary = f"items_processed={items_processed}"
    output_summary = (
        f"new={new_count}, duplicates={duplicate_count}, "
        f"conflicts={conflict_count}, unprocessed={unprocessed_count}"
    )

    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        agent_name="gap_detection",
        input_summary=input_summary,
        output_summary=output_summary,
        duration_ms=duration_ms,
    )

    try:
        _audit_log.record(session_id, entry)
    except Exception as e:
        logger.warning("Failed to record audit log entry: %s", e)
