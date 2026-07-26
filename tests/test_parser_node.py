"""Tests for the Parser node and route_after_parser routing logic.

All LLM and document parsing calls are mocked — no real API calls.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure graph/ and nodes/ are importable
sys.path.insert(0, ".")

from graph.routing import route_after_parser
from graph.state import PipelineState
from nodes.parser_node import get_audit_log, parser_node

from backlog_synthesizer_langgraph.models.extraction import (
    DocumentError,
    ExtractionResult,
    ExtractedItem,
)
from backlog_synthesizer_langgraph.models.inputs import (
    DocumentType,
    InputDocument,
    SessionInputs,
)


# ─── Test Fixtures ────────────────────────────────────────────────


def _make_document(
    filename: str = "meeting_notes.txt",
    doc_type: DocumentType = DocumentType.TRANSCRIPT_TXT,
    content: str = "We need a password reset feature for all users. The current auth system is too slow.",
) -> InputDocument:
    """Create a test InputDocument."""
    content_bytes = content.encode("utf-8")
    return InputDocument(
        filename=filename,
        document_type=doc_type,
        content=content_bytes,
        size_bytes=len(content_bytes),
    )


def _make_session_inputs(
    documents: list[InputDocument] | None = None,
) -> SessionInputs:
    """Create minimal test SessionInputs."""
    if documents is None:
        documents = [_make_document()]
    return SessionInputs(
        session_id="test-session-parser",
        documents=documents,
        backlog_tickets=[],
    )


def _make_extracted_item(
    text: str = "Users need a password reset feature",
    confidence: float = 0.85,
    item_type: str = "feature_request",
    tags: list[str] | None = None,
) -> ExtractedItem:
    """Create a test ExtractedItem."""
    return ExtractedItem(
        item_type=item_type,
        text=text,
        source_chunk_index=0,
        confidence=confidence,
        tags=tags or ["auth", "security"],
    )


def _make_extraction_result(
    items: list[ExtractedItem] | None = None,
    errors: list[DocumentError] | None = None,
) -> ExtractionResult:
    """Create a test ExtractionResult."""
    if items is None:
        items = [
            _make_extracted_item("Password reset feature needed", 0.9),
            _make_extracted_item("Auth system is too slow", 0.8, "pain_point"),
            _make_extracted_item("Need OAuth2 support", 0.75),
        ]
    return ExtractionResult(
        items=items,
        errors=errors or [],
        metadata={
            "document_count": 1,
            "chunks_created": 2,
            "input_tokens": 50,
            "output_tokens": 150,
        },
    )


def _make_base_state(
    documents: list[InputDocument] | None = None,
    parser_output: ExtractionResult | None = None,
    retry_count: int = 0,
    status: str = "in_progress",
) -> PipelineState:
    """Create a base PipelineState for testing."""
    inputs = _make_session_inputs(documents)
    return {
        "inputs": inputs,
        "session_id": "test-session-parser",
        "parser_output": parser_output,
        "gap_output": None,
        "story_output": None,
        "current_step": "parser",
        "status": status,
        "errors": [],
        "react_notes": [],
        "trace_id": "test-trace-parser",
        "total_duration_ms": 0,
        "retry_count": retry_count,
    }


def _mock_llm_extraction_response() -> str:
    """Create a mock LLM JSON response for item extraction."""
    return json.dumps([
        {
            "item_type": "feature_request",
            "text": "Password reset feature needed for all users",
            "confidence": 0.9,
            "stakeholder": "Product Manager",
            "section_heading": "Auth Discussion",
            "tags": ["auth", "security"],
        },
        {
            "item_type": "pain_point",
            "text": "Current auth system is too slow",
            "confidence": 0.8,
            "stakeholder": None,
            "section_heading": "Auth Discussion",
            "tags": ["auth", "performance"],
        },
        {
            "item_type": "feature_request",
            "text": "Need OAuth2 support for third-party integrations",
            "confidence": 0.75,
            "stakeholder": "Engineering Lead",
            "section_heading": "Integration Requirements",
            "tags": ["auth", "integration"],
        },
    ])


# ─── Node Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parser_node_reads_documents_from_state():
    """Test that the node reads documents from state['inputs'].documents."""
    documents = [_make_document()]
    state = _make_base_state(documents=documents)

    mock_llm = MagicMock()
    mock_llm.generate.return_value = _mock_llm_extraction_response()
    mock_parser = MagicMock()

    with patch("nodes.parser_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config.create_document_parser.return_value = mock_parser
        mock_config_cls.from_env.return_value = mock_config

        result = await parser_node(state)

    # LLM should have been called (documents were read and chunked)
    assert mock_llm.generate.called
    assert result["current_step"] == "parser"


@pytest.mark.asyncio
async def test_parser_node_writes_parser_output_to_state():
    """Test that the node writes parser_output to state."""
    state = _make_base_state()

    mock_llm = MagicMock()
    mock_llm.generate.return_value = _mock_llm_extraction_response()
    mock_parser = MagicMock()

    with patch("nodes.parser_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config.create_document_parser.return_value = mock_parser
        mock_config_cls.from_env.return_value = mock_config

        result = await parser_node(state)

    assert result["parser_output"] is not None
    assert isinstance(result["parser_output"], ExtractionResult)
    assert len(result["parser_output"].items) > 0


@pytest.mark.asyncio
async def test_parser_node_updates_current_step():
    """Test that the node sets current_step to 'parser'."""
    state = _make_base_state()

    mock_llm = MagicMock()
    mock_llm.generate.return_value = _mock_llm_extraction_response()
    mock_parser = MagicMock()

    with patch("nodes.parser_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config.create_document_parser.return_value = mock_parser
        mock_config_cls.from_env.return_value = mock_config

        result = await parser_node(state)

    assert result["current_step"] == "parser"


@pytest.mark.asyncio
async def test_parser_node_otel_span_recorded():
    """Test that OTel span is created with correct attributes."""
    state = _make_base_state()

    mock_llm = MagicMock()
    mock_llm.generate.return_value = _mock_llm_extraction_response()
    mock_parser_tool = MagicMock()

    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span

    with patch("nodes.parser_node.PipelineConfig") as mock_config_cls, \
         patch("nodes.parser_node.get_tracer", return_value=mock_tracer):
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config.create_document_parser.return_value = mock_parser_tool
        mock_config_cls.from_env.return_value = mock_config

        await parser_node(state)

    # Verify span was started with correct name
    mock_tracer.start_span.assert_called_once_with("agent.parser")

    # Verify key span attributes were set
    set_attr_calls = {
        call.args[0]: call.args[1]
        for call in mock_span.set_attribute.call_args_list
    }
    assert set_attr_calls["session_id"] == "test-session-parser"
    assert set_attr_calls["document_count"] == 1
    assert "items_extracted" in set_attr_calls
    assert "chunks_created" in set_attr_calls
    assert "duration_ms" in set_attr_calls
    assert "input_tokens" in set_attr_calls
    assert "output_tokens" in set_attr_calls

    # Verify span was ended
    mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_parser_node_audit_log_entry_created():
    """Test that an audit log entry is recorded."""
    state = _make_base_state()

    mock_llm = MagicMock()
    mock_llm.generate.return_value = _mock_llm_extraction_response()
    mock_parser_tool = MagicMock()

    with patch("nodes.parser_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config.create_document_parser.return_value = mock_parser_tool
        mock_config_cls.from_env.return_value = mock_config

        await parser_node(state)

    audit_log = get_audit_log()
    entries = audit_log.get_entries("test-session-parser")
    assert len(entries) > 0

    latest_entry = entries[-1]
    assert latest_entry.agent_name == "parser"
    assert "documents=1" in latest_entry.input_summary
    assert "items_extracted=" in latest_entry.output_summary


@pytest.mark.asyncio
async def test_parser_node_transient_error_handled():
    """Test that TransientToolError sets status to partial_failure."""
    from backlog_synthesizer_langgraph.tools.errors import TransientToolError

    state = _make_base_state()

    with patch("nodes.parser_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = MagicMock()
        mock_config.create_document_parser.return_value = MagicMock()
        mock_config_cls.from_env.return_value = mock_config

        # Patch the agent's parse_documents to raise TransientToolError
        with patch("nodes.parser_node.ParserAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.parse_documents = AsyncMock(
                side_effect=TransientToolError("Rate limit exceeded")
            )
            mock_agent_cls.return_value = mock_agent

            result = await parser_node(state)

    assert result["current_step"] == "parser"
    assert result["status"] == "partial_failure"
    assert len(result["errors"]) > 0
    assert "TransientToolError" in result["errors"][0]["error_type"]


@pytest.mark.asyncio
async def test_parser_node_permanent_error_handled():
    """Test that PermanentToolError sets status to permanent_failure."""
    from backlog_synthesizer_langgraph.tools.errors import PermanentToolError

    state = _make_base_state()

    with patch("nodes.parser_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.side_effect = PermanentToolError("No API key")
        mock_config_cls.from_env.return_value = mock_config

        result = await parser_node(state)

    assert result["current_step"] == "parser"
    assert result["status"] == "permanent_failure"
    assert len(result["errors"]) > 0
    assert "PermanentToolError" in result["errors"][0]["error_type"]


# ─── Routing Tests ────────────────────────────────────────────────


def test_routing_returns_gap_detection_on_success():
    """Test routing returns 'gap_detection' on successful extraction."""
    state = _make_base_state(parser_output=_make_extraction_result())

    result = route_after_parser(state)
    assert result == "gap_detection"


def test_routing_returns_error_node_on_permanent_failure():
    """Test routing returns 'error_node' when status is permanent_failure."""
    state = _make_base_state(status="permanent_failure")

    result = route_after_parser(state)
    assert result == "error_node"


def test_routing_returns_error_node_when_parser_output_none():
    """Test routing returns 'error_node' when parser_output is None."""
    state = _make_base_state(parser_output=None)
    state["status"] = "in_progress"

    result = route_after_parser(state)
    assert result == "error_node"


def test_routing_returns_llm_fallback_on_zero_items():
    """Test routing returns 'llm_fallback_node' when 0 items extracted."""
    extraction = _make_extraction_result(items=[])
    state = _make_base_state(parser_output=extraction)

    result = route_after_parser(state)
    assert result == "llm_fallback_node"


def test_routing_returns_llm_fallback_on_fewer_than_3_items():
    """Test routing returns 'llm_fallback_node' when < 3 items extracted."""
    items = [
        _make_extracted_item("Only one item", 0.9),
        _make_extracted_item("Only two items", 0.8),
    ]
    extraction = _make_extraction_result(items=items)
    state = _make_base_state(parser_output=extraction)

    result = route_after_parser(state)
    assert result == "llm_fallback_node"


def test_routing_returns_llm_fallback_when_all_low_confidence():
    """Test routing returns 'llm_fallback_node' when all items confidence < 0.3."""
    items = [
        _make_extracted_item("Item 1", confidence=0.1),
        _make_extracted_item("Item 2", confidence=0.2),
        _make_extracted_item("Item 3", confidence=0.25),
        _make_extracted_item("Item 4", confidence=0.15),
    ]
    extraction = _make_extraction_result(items=items)
    state = _make_base_state(parser_output=extraction)

    result = route_after_parser(state)
    assert result == "llm_fallback_node"


def test_routing_returns_gap_detection_when_some_high_confidence():
    """Test routing returns 'gap_detection' when at least one item has confidence >= 0.3."""
    items = [
        _make_extracted_item("Low confidence", confidence=0.1),
        _make_extracted_item("Low confidence", confidence=0.2),
        _make_extracted_item("High confidence", confidence=0.5),  # This one is fine
        _make_extracted_item("Low confidence", confidence=0.15),
    ]
    extraction = _make_extraction_result(items=items)
    state = _make_base_state(parser_output=extraction)

    result = route_after_parser(state)
    assert result == "gap_detection"


def test_routing_returns_error_node_with_retry_count_zero_items():
    """Test routing returns 'error_node' when retry_count >= 1 and 0 items."""
    extraction = _make_extraction_result(items=[])
    state = _make_base_state(parser_output=extraction, retry_count=1)

    result = route_after_parser(state)
    assert result == "error_node"


def test_routing_returns_error_node_with_retry_count_few_items():
    """Test routing returns 'error_node' when retry_count >= 1 and < 3 items."""
    items = [_make_extracted_item("Only one", 0.9)]
    extraction = _make_extraction_result(items=items)
    state = _make_base_state(parser_output=extraction, retry_count=1)

    result = route_after_parser(state)
    assert result == "error_node"


def test_routing_returns_error_node_with_retry_count_low_confidence():
    """Test routing returns 'error_node' when retry_count >= 1 and all low confidence."""
    items = [
        _make_extracted_item("Item 1", confidence=0.1),
        _make_extracted_item("Item 2", confidence=0.2),
        _make_extracted_item("Item 3", confidence=0.25),
    ]
    extraction = _make_extraction_result(items=items)
    state = _make_base_state(parser_output=extraction, retry_count=1)

    result = route_after_parser(state)
    assert result == "error_node"
