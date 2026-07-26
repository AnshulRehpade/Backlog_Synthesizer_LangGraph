"""Parser node — wraps ParserAgent as a LangGraph node function.

This node:
1. Reads raw documents from state["inputs"].documents
2. Initializes ParserAgent using PipelineConfig
3. Calls ParserAgent.parse_documents() unchanged
4. Writes parser_output back to state
5. Records OTel span and audit log entry

Never raises exceptions — always returns state gracefully.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from graph.state import PipelineState
from backlog_synthesizer_langgraph.agents.parser import ParserAgent
from backlog_synthesizer_langgraph.config import PipelineConfig
from backlog_synthesizer_langgraph.memory.audit_log import AuditLog
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


async def parser_node(state: PipelineState) -> PipelineState:
    """LangGraph node that extracts structured items from raw documents.

    Reads documents from state["inputs"], invokes ParserAgent.parse_documents(),
    and writes the ExtractionResult back to state as parser_output.

    Error handling:
        - TransientToolError → appends error, sets status to "partial_failure"
        - PermanentToolError → appends error, sets status to "permanent_failure"
        - Any other exception → appends error, sets status to "partial_failure"
        - Never raises out of the node — always returns state.

    Args:
        state: The current PipelineState.

    Returns:
        Updated PipelineState with parser_output populated (or errors appended).
    """
    tracer = get_tracer()
    start_time = time.time()

    # Start OTel span
    span = None
    if tracer is not None:
        span = tracer.start_span("agent.parser")
        span.set_attribute("session_id", state["session_id"])

    try:
        return await _execute_parser(state, start_time, span)
    except Exception as e:
        # Catch-all safety net — guarantees we never raise out of the node.
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("Unexpected error in parser_node: %s", e)

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.end()

        return {
            "current_step": "parser",
            "status": "partial_failure",
            "errors": [
                {
                    "node": "parser",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }


async def _execute_parser(
    state: PipelineState, start_time: float, span: object | None
) -> PipelineState:
    """Core logic for the parser node, separated for clarity.

    Args:
        state: The current pipeline state.
        start_time: Timestamp when the node started.
        span: Optional OTel span for tracing.

    Returns:
        Updated state dict with parser_output or error information.
    """
    # 1. Read inputs from state
    inputs = state.get("inputs")
    if inputs is None:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "parser",
            "error_type": "MissingInput",
            "message": "inputs is None — cannot parse without session inputs",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "parser",
            "status": "permanent_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    documents = inputs.documents
    document_count = len(documents)

    if span is not None:
        span.set_attribute("document_count", document_count)

    # 2. Check for empty document list
    if document_count == 0:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "parser",
            "error_type": "MissingInput",
            "message": "No documents provided — nothing to parse",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.warning(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "parser",
            "status": "partial_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    # 3. Initialize ParserAgent using PipelineConfig
    try:
        config = PipelineConfig.from_env()
        llm_tool = config.create_llm_tool()
        document_parser = config.create_document_parser()
    except PermanentToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "parser",
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
            "current_step": "parser",
            "status": "permanent_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    agent = ParserAgent(llm_tool=llm_tool, document_parser=document_parser)

    # 4. Call ParserAgent.parse_documents()
    try:
        result = await agent.parse_documents(documents)
    except TransientToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "parser",
            "error_type": "TransientToolError",
            "message": f"Transient error during document parsing: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.warning(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "parser",
            "status": "partial_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }
    except PermanentToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "parser",
            "error_type": "PermanentToolError",
            "message": f"Permanent error during document parsing: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "parser",
            "status": "permanent_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    # 5. Compute metrics
    duration_ms = int((time.time() - start_time) * 1000)
    items_extracted = len(result.items)
    chunks_created = result.metadata.get("chunks_created", 0)
    input_tokens = result.metadata.get("input_tokens", 0)
    output_tokens = result.metadata.get("output_tokens", 0)

    # Count low-confidence items (warnings)
    low_confidence_items = [item for item in result.items if item.confidence < 0.3]
    low_confidence_count = len(low_confidence_items)

    # 6. Record OTel span attributes
    if span is not None:
        span.set_attribute("document_count", document_count)
        span.set_attribute("chunks_created", chunks_created)
        span.set_attribute("items_extracted", items_extracted)
        span.set_attribute("duration_ms", duration_ms)
        span.set_attribute("input_tokens", input_tokens)
        span.set_attribute("output_tokens", output_tokens)
        if low_confidence_count > 0:
            span.set_attribute("low_confidence_items", low_confidence_count)
        span.end()

    # 7. Log to audit log
    _log_to_audit(
        session_id=state["session_id"],
        document_count=document_count,
        items_extracted=items_extracted,
        low_confidence_count=low_confidence_count,
        duration_ms=duration_ms,
    )

    # 8. Build errors from document-level failures
    errors_to_append = []
    for doc_error in result.errors:
        errors_to_append.append({
            "node": "parser",
            "error_type": "DocumentError",
            "message": f"{doc_error.filename}: {doc_error.reason}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # 9. Build react notes
    react_notes = []
    note_parts = [
        f"Parsed {document_count} documents → {items_extracted} items extracted "
        f"({chunks_created} chunks)."
    ]
    if low_confidence_count > 0:
        note_parts.append(
            f" Warning: {low_confidence_count} items have confidence < 0.3."
        )
    if result.errors:
        note_parts.append(
            f" {len(result.errors)} document(s) had parsing errors."
        )

    react_notes.append({
        "node": "parser",
        "note": "".join(note_parts),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # 10. Determine status
    status = state.get("status", "in_progress")

    logger.info(
        "Parser completed: %d documents, %d items extracted, %d chunks, %dms",
        document_count,
        items_extracted,
        chunks_created,
        duration_ms,
    )

    return {
        "parser_output": result,
        "current_step": "parser",
        "status": status,
        "errors": errors_to_append,
        "react_notes": react_notes,
        "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
    }


def _log_to_audit(
    session_id: str,
    document_count: int,
    items_extracted: int,
    low_confidence_count: int,
    duration_ms: int,
) -> None:
    """Record the parser invocation to the audit log.

    Args:
        session_id: Pipeline session ID.
        document_count: Number of documents processed.
        items_extracted: Number of items extracted.
        low_confidence_count: Items with confidence < 0.3.
        duration_ms: Processing duration.
    """
    input_summary = f"documents={document_count}"
    output_summary = (
        f"items_extracted={items_extracted}, "
        f"low_confidence={low_confidence_count}"
    )

    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        agent_name="parser",
        input_summary=input_summary,
        output_summary=output_summary,
        duration_ms=duration_ms,
    )

    try:
        _audit_log.record(session_id, entry)
    except Exception as e:
        logger.warning("Failed to record audit log entry: %s", e)
