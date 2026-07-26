"""Error node — handles permanent failures and partial failure recovery.

This node is reached when routing determines the pipeline cannot continue
normally (permanent failure, missing output, etc.). It:
1. Logs the error to audit log
2. Records OTel span with ERROR status
3. Updates status to "permanent_failure"
4. Returns state with whatever partial results exist

Never raises — always returns state gracefully.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from graph.state import PipelineState
from backlog_synthesizer_langgraph.observability import get_tracer

logger = logging.getLogger(__name__)


async def error_node(state: PipelineState) -> PipelineState:
    """LangGraph node that handles pipeline failures gracefully.

    Reached via conditional edges when:
    - A node sets status to "permanent_failure"
    - story_output is None after story_writer completes
    - Any unrecoverable error occurs

    This node:
    - Logs the accumulated errors
    - Records an OTel span with error status
    - Sets final status to "permanent_failure"
    - Preserves any partial results that exist in state

    Args:
        state: The current PipelineState (with errors already appended).

    Returns:
        Updated PipelineState with status set to "permanent_failure".
    """
    tracer = get_tracer()
    start_time = time.time()

    # Start OTel span
    span = None
    if tracer is not None:
        span = tracer.start_span("error_node")
        span.set_attribute("session_id", state["session_id"])
        span.set_attribute("error", True)

    # Gather error context
    existing_errors = state.get("errors", [])
    previous_step = state.get("current_step", "unknown")

    # Log accumulated errors
    error_count = len(existing_errors)
    logger.error(
        "Pipeline entered error_node from step '%s' with %d accumulated errors. "
        "Session: %s",
        previous_step,
        error_count,
        state["session_id"],
    )
    for err in existing_errors:
        if isinstance(err, dict):
            logger.error(
                "  [%s] %s: %s",
                err.get("node", "unknown"),
                err.get("error_type", "Unknown"),
                err.get("message", "No message"),
            )

    # Determine what partial results exist
    has_parser_output = state.get("parser_output") is not None
    has_gap_output = state.get("gap_output") is not None
    has_story_output = state.get("story_output") is not None

    partial_results_summary = (
        f"Partial results: parser={'yes' if has_parser_output else 'no'}, "
        f"gap={'yes' if has_gap_output else 'no'}, "
        f"story={'yes' if has_story_output else 'no'}"
    )
    logger.info(partial_results_summary)

    # Record span details
    duration_ms = int((time.time() - start_time) * 1000)
    if span is not None:
        span.set_attribute("previous_step", previous_step)
        span.set_attribute("accumulated_error_count", error_count)
        span.set_attribute("has_parser_output", has_parser_output)
        span.set_attribute("has_gap_output", has_gap_output)
        span.set_attribute("has_story_output", has_story_output)
        span.set_attribute("duration_ms", duration_ms)

        # Set span status to ERROR
        try:
            from opentelemetry.trace import StatusCode
            span.set_status(StatusCode.ERROR, "Pipeline failed — entered error node")
        except ImportError:
            pass

        span.end()

    # Build react note documenting the failure
    react_note = {
        "node": "error_node",
        "note": (
            f"Pipeline terminated with permanent failure. "
            f"Previous step: {previous_step}. "
            f"Errors: {error_count}. "
            f"{partial_results_summary}"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "current_step": "error_node",
        "status": "permanent_failure",
        "errors": [],  # Don't append new errors — existing ones are preserved via reducer
        "react_notes": [react_note],
        "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
    }
