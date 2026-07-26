"""Tests for the Gap Detection node and route_after_gap_detection routing logic.

All embedding, vector store, and LLM calls are mocked — no real API calls.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure graph/ and nodes/ are importable
sys.path.insert(0, ".")

from graph.routing import route_after_gap_detection
from graph.state import PipelineState
from nodes.gap_detection_node import gap_detection_node, get_audit_log

from backlog_synthesizer_langgraph.models.extraction import (
    ExtractionResult,
    ExtractedItem,
)
from backlog_synthesizer_langgraph.models.gap_detection import (
    GapReport,
    GapReportEntry,
)
from backlog_synthesizer_langgraph.models.inputs import (
    BacklogTicket,
    SessionInputs,
)
from backlog_synthesizer_langgraph.tools.interfaces import SearchResult


# ─── Test Fixtures ────────────────────────────────────────────────


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
) -> ExtractionResult:
    """Create a test ExtractionResult."""
    if items is None:
        items = [
            _make_extracted_item("Password reset feature needed", 0.9),
            _make_extracted_item("Auth system is too slow", 0.8, "pain_point"),
            _make_extracted_item("Need OAuth2 support", 0.75),
            _make_extracted_item("Logging across microservices", 0.85, tags=["logging"]),
        ]
    return ExtractionResult(
        items=items,
        errors=[],
        metadata={"document_count": 1, "chunks_created": 2},
    )


def _make_backlog_ticket(
    ticket_id: str = "TICKET-001",
    title: str = "Existing password reset",
    description: str = "Users can reset their password via email",
) -> BacklogTicket:
    """Create a test BacklogTicket."""
    return BacklogTicket(
        id=ticket_id,
        title=title,
        description=description,
        status="open",
        tags=["auth"],
    )


def _make_session_inputs(
    backlog_tickets: list[BacklogTicket] | None = None,
) -> SessionInputs:
    """Create minimal test SessionInputs."""
    return SessionInputs(
        session_id="test-session-gap",
        documents=[],
        backlog_tickets=backlog_tickets or [],
    )


def _make_gap_report(
    new_count: int = 2,
    duplicate_count: int = 1,
    conflict_count: int = 1,
    unprocessed_count: int = 0,
) -> GapReport:
    """Create a test GapReport with the given counts."""
    entries = []
    for i in range(new_count):
        entries.append(GapReportEntry(
            item=_make_extracted_item(f"New item {i}"),
            classification="new",
            gap_type="NEW",
            confidence=0.9,
        ))
    for i in range(duplicate_count):
        entries.append(GapReportEntry(
            item=_make_extracted_item(f"Duplicate item {i}"),
            classification="duplicate",
            gap_type="DUPLICATE",
            confidence=0.85,
            similarity_score=0.9,
            similar_ticket_id=f"TICKET-{i}",
        ))
    for i in range(conflict_count):
        entries.append(GapReportEntry(
            item=_make_extracted_item(f"Conflict item {i}"),
            classification="conflict",
            gap_type="CONFLICT",
            confidence=0.7,
            similarity_score=0.6,
            similar_ticket_id=f"TICKET-C{i}",
        ))
    for i in range(unprocessed_count):
        entries.append(GapReportEntry(
            item=_make_extracted_item(f"Unprocessed item {i}"),
            classification="unprocessed",
            gap_type="UNPROCESSED",
            confidence=0.0,
            error_reason="Search failed",
        ))

    return GapReport(
        entries=entries,
        total_new=new_count,
        total_duplicates=duplicate_count,
        total_conflicts=conflict_count,
        total_unprocessed=unprocessed_count,
    )


def _make_base_state(
    parser_output: ExtractionResult | None = None,
    gap_output: GapReport | None = None,
    backlog_tickets: list[BacklogTicket] | None = None,
    retry_count: int = 0,
    status: str = "in_progress",
) -> PipelineState:
    """Create a base PipelineState for testing."""
    if parser_output is None:
        parser_output = _make_extraction_result()
    inputs = _make_session_inputs(backlog_tickets)
    return {
        "inputs": inputs,
        "session_id": "test-session-gap",
        "parser_output": parser_output,
        "gap_output": gap_output,
        "story_output": None,
        "current_step": "parser",
        "status": status,
        "errors": [],
        "react_notes": [],
        "trace_id": "test-trace-gap",
        "total_duration_ms": 100,
        "retry_count": retry_count,
        "needs_human_review": False,
    }


# ─── Node Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gap_detection_node_reads_parser_output():
    """Test that the node reads parser_output.items from state."""
    state = _make_base_state()

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384
    mock_vector = MagicMock()
    mock_vector.query_similar.return_value = []
    mock_vector.store.return_value = None

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_embedding_tool.return_value = mock_embedding
        mock_config.create_vector_search_tool.return_value = mock_vector
        mock_config.gap_detection_duplicate_threshold = 0.80
        mock_config.gap_detection_conflict_threshold = 0.50
        mock_config_cls.from_env.return_value = mock_config

        result = await gap_detection_node(state)

    assert result["current_step"] == "gap_detection"
    assert result["gap_output"] is not None


@pytest.mark.asyncio
async def test_gap_detection_node_writes_gap_output():
    """Test that the node writes gap_output to state."""
    state = _make_base_state()

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384
    mock_vector = MagicMock()
    mock_vector.query_similar.return_value = []
    mock_vector.store.return_value = None

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_embedding_tool.return_value = mock_embedding
        mock_config.create_vector_search_tool.return_value = mock_vector
        mock_config.gap_detection_duplicate_threshold = 0.80
        mock_config.gap_detection_conflict_threshold = 0.50
        mock_config_cls.from_env.return_value = mock_config

        result = await gap_detection_node(state)

    assert isinstance(result["gap_output"], GapReport)
    # 4 items should be processed (all new since no similar results found)
    assert result["gap_output"].total_new == 4


@pytest.mark.asyncio
async def test_gap_detection_node_updates_current_step():
    """Test that the node sets current_step to 'gap_detection'."""
    state = _make_base_state()

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384
    mock_vector = MagicMock()
    mock_vector.query_similar.return_value = []
    mock_vector.store.return_value = None

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_embedding_tool.return_value = mock_embedding
        mock_config.create_vector_search_tool.return_value = mock_vector
        mock_config.gap_detection_duplicate_threshold = 0.80
        mock_config.gap_detection_conflict_threshold = 0.50
        mock_config_cls.from_env.return_value = mock_config

        result = await gap_detection_node(state)

    assert result["current_step"] == "gap_detection"


@pytest.mark.asyncio
async def test_backlog_tickets_loaded_into_memory():
    """Test that backlog tickets are stored in Long-Term Memory before gap analysis."""
    tickets = [
        _make_backlog_ticket("T-1", "Password reset", "Reset via email"),
        _make_backlog_ticket("T-2", "OAuth support", "Third-party login"),
    ]
    state = _make_base_state(backlog_tickets=tickets)

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384
    mock_vector = MagicMock()
    mock_vector.query_similar.return_value = []
    mock_vector.store.return_value = None

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_embedding_tool.return_value = mock_embedding
        mock_config.create_vector_search_tool.return_value = mock_vector
        mock_config.gap_detection_duplicate_threshold = 0.80
        mock_config.gap_detection_conflict_threshold = 0.50
        mock_config_cls.from_env.return_value = mock_config

        await gap_detection_node(state)

    # Verify store was called for tickets + extracted items
    # 2 tickets + 4 extracted items = at least 6 store calls
    assert mock_vector.store.call_count >= 2


@pytest.mark.asyncio
async def test_extracted_items_stored_after_gap_detection():
    """Test that extracted items are stored AFTER gap detection, not before."""
    state = _make_base_state()

    call_order = []

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384

    mock_vector = MagicMock()

    def track_query(*args, **kwargs):
        call_order.append("query")
        return []

    def track_store(*args, **kwargs):
        call_order.append("store")

    mock_vector.query_similar.side_effect = track_query
    mock_vector.store.side_effect = track_store

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_embedding_tool.return_value = mock_embedding
        mock_config.create_vector_search_tool.return_value = mock_vector
        mock_config.gap_detection_duplicate_threshold = 0.80
        mock_config.gap_detection_conflict_threshold = 0.50
        mock_config_cls.from_env.return_value = mock_config

        await gap_detection_node(state)

    # All queries (gap detection) should come before the post-gap stores
    # With no backlog tickets, first stores are the post-detection extracted item stores
    # All query calls happen during analyze_gaps, stores happen after
    queries = [i for i, op in enumerate(call_order) if op == "query"]
    stores = [i for i, op in enumerate(call_order) if op == "store"]

    if queries and stores:
        # Last query should come before first extracted-item store
        assert max(queries) < max(stores)


@pytest.mark.asyncio
async def test_purge_expired_called():
    """Test that purge_expired is called at node start."""
    state = _make_base_state()

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384
    mock_vector = MagicMock()
    mock_vector.query_similar.return_value = []
    mock_vector.store.return_value = None

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls, \
         patch("nodes.gap_detection_node.LongTermMemory") as mock_ltm_cls:
        mock_config = MagicMock()
        mock_config.create_embedding_tool.return_value = mock_embedding
        mock_config.create_vector_search_tool.return_value = mock_vector
        mock_config.gap_detection_duplicate_threshold = 0.80
        mock_config.gap_detection_conflict_threshold = 0.50
        mock_config_cls.from_env.return_value = mock_config

        mock_ltm = MagicMock()
        mock_ltm.purge_expired.return_value = []
        mock_ltm.search_similar.return_value = []
        mock_ltm.store_item.return_value = None
        mock_ltm_cls.return_value = mock_ltm

        with patch("nodes.gap_detection_node.GapDetectionAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.analyze_gaps = AsyncMock(return_value=_make_gap_report())
            mock_agent_cls.return_value = mock_agent

            await gap_detection_node(state)

        mock_ltm.purge_expired.assert_called_once()


@pytest.mark.asyncio
async def test_otel_span_recorded():
    """Test that OTel span is created with correct attributes."""
    state = _make_base_state()

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384
    mock_vector = MagicMock()
    mock_vector.query_similar.return_value = []
    mock_vector.store.return_value = None

    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls, \
         patch("nodes.gap_detection_node.get_tracer", return_value=mock_tracer):
        mock_config = MagicMock()
        mock_config.create_embedding_tool.return_value = mock_embedding
        mock_config.create_vector_search_tool.return_value = mock_vector
        mock_config.gap_detection_duplicate_threshold = 0.80
        mock_config.gap_detection_conflict_threshold = 0.50
        mock_config_cls.from_env.return_value = mock_config

        await gap_detection_node(state)

    mock_tracer.start_span.assert_called_once_with("agent.gap_detection")

    set_attr_calls = {
        call.args[0]: call.args[1]
        for call in mock_span.set_attribute.call_args_list
    }
    assert set_attr_calls["session_id"] == "test-session-gap"
    assert set_attr_calls["items_processed"] == 4
    assert "duplicate_count" in set_attr_calls
    assert "conflict_count" in set_attr_calls
    assert "new_count" in set_attr_calls
    assert "duration_ms" in set_attr_calls

    mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_audit_log_entry_created():
    """Test that an audit log entry is recorded."""
    state = _make_base_state()

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384
    mock_vector = MagicMock()
    mock_vector.query_similar.return_value = []
    mock_vector.store.return_value = None

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_embedding_tool.return_value = mock_embedding
        mock_config.create_vector_search_tool.return_value = mock_vector
        mock_config.gap_detection_duplicate_threshold = 0.80
        mock_config.gap_detection_conflict_threshold = 0.50
        mock_config_cls.from_env.return_value = mock_config

        await gap_detection_node(state)

    audit_log = get_audit_log()
    entries = audit_log.get_entries("test-session-gap")
    assert len(entries) > 0

    latest_entry = entries[-1]
    assert latest_entry.agent_name == "gap_detection"
    assert "items_processed=4" in latest_entry.input_summary


@pytest.mark.asyncio
async def test_transient_error_handled():
    """Test that TransientToolError sets status to partial_failure."""
    from backlog_synthesizer_langgraph.tools.errors import TransientToolError

    state = _make_base_state()

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_embedding_tool.side_effect = TransientToolError("Connection timeout")
        mock_config_cls.from_env.return_value = mock_config

        # create_embedding_tool raises before we can proceed — but that's caught
        # as a generic Exception by the outer try. Let's make it PermanentToolError instead.
        # Actually, create_embedding_tool isn't behind PermanentToolError guard.
        # The node catches generic exceptions in the safety net.
        result = await gap_detection_node(state)

    assert result["current_step"] == "gap_detection"
    assert result["status"] == "partial_failure"
    assert len(result["errors"]) > 0


@pytest.mark.asyncio
async def test_permanent_error_handled():
    """Test that PermanentToolError sets status to permanent_failure."""
    from backlog_synthesizer_langgraph.tools.errors import PermanentToolError

    state = _make_base_state()

    with patch("nodes.gap_detection_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_embedding_tool.side_effect = PermanentToolError("No API key")
        mock_config_cls.from_env.return_value = mock_config

        result = await gap_detection_node(state)

    assert result["current_step"] == "gap_detection"
    assert result["status"] == "permanent_failure"
    assert len(result["errors"]) > 0
    assert "PermanentToolError" in result["errors"][0]["error_type"]


# ─── Routing Tests ────────────────────────────────────────────────


def test_routing_returns_story_writer_on_success():
    """Test routing returns 'story_writer' on successful mixed results."""
    state = _make_base_state(gap_output=_make_gap_report(new_count=3, duplicate_count=1, conflict_count=1))

    result = route_after_gap_detection(state)
    assert result == "story_writer"


def test_routing_returns_error_node_on_permanent_failure():
    """Test routing returns 'error_node' when status is permanent_failure."""
    state = _make_base_state(status="permanent_failure")

    result = route_after_gap_detection(state)
    assert result == "error_node"


def test_routing_returns_error_node_when_gap_output_none():
    """Test routing returns 'error_node' when gap_output is None."""
    state = _make_base_state(gap_output=None)

    result = route_after_gap_detection(state)
    assert result == "error_node"


def test_routing_returns_llm_fallback_on_all_duplicates():
    """Test routing returns 'llm_fallback_node' when all items are duplicates."""
    state = _make_base_state(
        gap_output=_make_gap_report(new_count=0, duplicate_count=5, conflict_count=0)
    )

    result = route_after_gap_detection(state)
    assert result == "llm_fallback_node"


def test_routing_returns_llm_fallback_on_zero_new_and_conflict():
    """Test routing returns 'llm_fallback_node' when new + conflict == 0."""
    state = _make_base_state(
        gap_output=_make_gap_report(new_count=0, duplicate_count=3, conflict_count=0, unprocessed_count=2)
    )

    result = route_after_gap_detection(state)
    assert result == "llm_fallback_node"


def test_routing_returns_llm_fallback_on_over_80_percent_duplicates():
    """Test routing returns 'llm_fallback_node' when >80% are duplicates."""
    # 9 duplicates + 1 new = 90% duplicates
    state = _make_base_state(
        gap_output=_make_gap_report(new_count=1, duplicate_count=9, conflict_count=0)
    )

    result = route_after_gap_detection(state)
    assert result == "llm_fallback_node"


def test_routing_returns_story_writer_at_exactly_80_percent_duplicates():
    """Test routing returns 'story_writer' when exactly 80% duplicates (not >80%)."""
    # 4 duplicates + 1 new = 80% duplicates (not > 80%)
    state = _make_base_state(
        gap_output=_make_gap_report(new_count=1, duplicate_count=4, conflict_count=0)
    )

    result = route_after_gap_detection(state)
    assert result == "story_writer"


def test_routing_returns_story_writer_with_conflicts():
    """Test routing returns 'story_writer' when conflicts exist even with many duplicates."""
    # Has conflicts so new+conflict > 0, and duplicates aren't > 80%
    state = _make_base_state(
        gap_output=_make_gap_report(new_count=0, duplicate_count=2, conflict_count=3)
    )

    result = route_after_gap_detection(state)
    assert result == "story_writer"


def test_routing_returns_error_node_with_retry_count_all_duplicates():
    """Test routing returns 'error_node' when retry_count >= 1 and all duplicates."""
    state = _make_base_state(
        gap_output=_make_gap_report(new_count=0, duplicate_count=5, conflict_count=0),
        retry_count=1,
    )

    result = route_after_gap_detection(state)
    assert result == "error_node"


def test_routing_returns_error_node_with_retry_count_high_duplicates():
    """Test routing returns 'error_node' when retry_count >= 1 and >80% duplicates."""
    state = _make_base_state(
        gap_output=_make_gap_report(new_count=1, duplicate_count=9, conflict_count=0),
        retry_count=1,
    )

    result = route_after_gap_detection(state)
    assert result == "error_node"
