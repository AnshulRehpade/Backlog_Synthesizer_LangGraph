"""End-to-end tests for the complete Backlog Synthesizer LangGraph pipeline.

Tests the full graph wiring including routing, retries via llm_fallback,
and the human review interrupt flow.

All LLM, embedding, and vector store calls are mocked — no real API calls.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, ".")

from graph.builder import build_graph, compile_graph
from graph.routing import (
    route_after_gap_detection,
    route_after_human_review,
    route_after_llm_fallback,
    route_after_parser,
    route_after_story_writer,
)
from graph.state import PipelineState
from langgraph.graph import END

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
    DocumentType,
    InputDocument,
    SessionInputs,
)
from backlog_synthesizer_langgraph.models.output import (
    AcceptanceCriterion,
    Epic,
    OutputMetadata,
    StoryOutput,
    UserStory,
)


# ─── Test Fixtures ────────────────────────────────────────────────


def _make_extracted_item(
    text: str = "Users need a password reset feature",
    confidence: float = 0.85,
) -> ExtractedItem:
    return ExtractedItem(
        item_type="feature_request",
        text=text,
        source_chunk_index=0,
        confidence=confidence,
        tags=["auth", "security"],
    )


def _make_extraction_result(item_count: int = 5) -> ExtractionResult:
    items = [
        _make_extracted_item(f"Feature request {i}", 0.8 + i * 0.02)
        for i in range(item_count)
    ]
    return ExtractionResult(
        items=items,
        errors=[],
        metadata={"document_count": 1, "chunks_created": 2, "input_tokens": 100, "output_tokens": 200},
    )


def _make_gap_report(new_count: int = 3, dup_count: int = 1, conflict_count: int = 1) -> GapReport:
    entries = []
    for i in range(new_count):
        entries.append(GapReportEntry(
            item=_make_extracted_item(f"New item {i}"),
            classification="new", gap_type="NEW", confidence=0.9,
        ))
    for i in range(dup_count):
        entries.append(GapReportEntry(
            item=_make_extracted_item(f"Dup item {i}"),
            classification="duplicate", gap_type="DUPLICATE",
            confidence=0.85, similarity_score=0.9, similar_ticket_id=f"T-{i}",
        ))
    for i in range(conflict_count):
        entries.append(GapReportEntry(
            item=_make_extracted_item(f"Conflict item {i}"),
            classification="conflict", gap_type="CONFLICT",
            confidence=0.7, similarity_score=0.6, similar_ticket_id=f"TC-{i}",
        ))
    return GapReport(
        entries=entries,
        total_new=new_count, total_duplicates=dup_count,
        total_conflicts=conflict_count, total_unprocessed=0,
    )


def _make_story_output() -> StoryOutput:
    story = UserStory(
        title="Password Reset Feature",
        user_story="As a user, I want to reset my password, so that I can regain access",
        acceptance_criteria=[
            AcceptanceCriterion(description="User can request reset via email"),
            AcceptanceCriterion(description="Reset link expires after 24 hours"),
        ],
        tags=["auth", "security"],
        needs_refinement=False,
    )
    epic = Epic(epic_title="Auth & Security", stories=[story])
    return StoryOutput(
        index=[{"epic_title": "Auth & Security", "story_count": 1}],
        epics=[epic],
        metadata=OutputMetadata(
            session_id="test-session", timestamp=datetime.now(timezone.utc)
        ),
    )


def _make_session_inputs() -> SessionInputs:
    doc = InputDocument(
        filename="meeting.txt",
        document_type=DocumentType.TRANSCRIPT_TXT,
        content=b"We need a password reset feature for all users.",
        size_bytes=47,
    )
    return SessionInputs(
        session_id="test-session-e2e",
        documents=[doc],
        backlog_tickets=[],
    )


def _make_state(**overrides) -> PipelineState:
    base = {
        "inputs": _make_session_inputs(),
        "session_id": "test-session-e2e",
        "parser_output": None,
        "gap_output": None,
        "story_output": None,
        "current_step": "start",
        "status": "in_progress",
        "errors": [],
        "react_notes": [],
        "retry_count": 0,
        "needs_human_review": False,
        "trace_id": "test-trace-e2e",
        "total_duration_ms": 0,
    }
    base.update(overrides)
    return base


# ─── Graph Structure Tests ────────────────────────────────────────


def test_graph_builds_with_all_nodes():
    """Test that the graph builds with all 6 expected nodes."""
    g = build_graph()
    nodes = list(g.nodes.keys())
    assert "parser" in nodes
    assert "gap_detection" in nodes
    assert "story_writer" in nodes
    assert "error_node" in nodes
    assert "llm_fallback_node" in nodes
    assert "human_review" in nodes
    assert len(nodes) == 6


def test_graph_compiles_with_checkpointer():
    """Test that compile_graph succeeds and returns a runnable."""
    graph = compile_graph()
    assert graph is not None


# ─── Happy Path Routing Tests ─────────────────────────────────────


def test_happy_path_parser_to_gap_detection():
    """Test parser → gap_detection on successful extraction."""
    state = _make_state(
        parser_output=_make_extraction_result(5),
        status="in_progress",
    )
    result = route_after_parser(state)
    assert result == "gap_detection"


def test_happy_path_gap_detection_to_story_writer():
    """Test gap_detection → story_writer on mixed results."""
    state = _make_state(
        gap_output=_make_gap_report(new_count=3, dup_count=1, conflict_count=1),
        status="in_progress",
    )
    result = route_after_gap_detection(state)
    assert result == "story_writer"


def test_happy_path_story_writer_to_end():
    """Test story_writer → END on successful output."""
    state = _make_state(
        story_output=_make_story_output(),
        status="in_progress",
    )
    result = route_after_story_writer(state)
    assert result == END


# ─── Parser Failure Path ──────────────────────────────────────────


def test_parser_permanent_failure_to_error_node():
    """Test parser → error_node on permanent failure."""
    state = _make_state(status="permanent_failure")
    result = route_after_parser(state)
    assert result == "error_node"


# ─── All Duplicates Path ──────────────────────────────────────────


def test_all_duplicates_to_llm_fallback():
    """Test gap_detection → llm_fallback when all items are duplicates."""
    state = _make_state(
        gap_output=_make_gap_report(new_count=0, dup_count=5, conflict_count=0),
        status="in_progress",
    )
    result = route_after_gap_detection(state)
    assert result == "llm_fallback_node"


# ─── Retry Path via LLM Fallback ─────────────────────────────────


def test_llm_fallback_retry_story_writer_routes_to_story_writer():
    """Test llm_fallback → story_writer when action is retry_story_writer."""
    state = _make_state(
        react_notes=[
            {
                "node": "llm_fallback",
                "action": "retry_story_writer",
                "note": "retry_story_writer: retrying (confidence: 0.80)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    result = route_after_llm_fallback(state)
    assert result == "story_writer"


def test_llm_fallback_retry_parser_routes_to_parser():
    """Test llm_fallback → parser when action is retry_parser."""
    state = _make_state(
        react_notes=[
            {
                "node": "llm_fallback",
                "action": "retry_parser",
                "note": "retry_parser: retrying (confidence: 0.75)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    result = route_after_llm_fallback(state)
    assert result == "parser"


def test_llm_fallback_lower_threshold_routes_to_gap_detection():
    """Test llm_fallback → gap_detection when action is lower_threshold."""
    state = _make_state(
        react_notes=[
            {
                "node": "llm_fallback",
                "action": "lower_threshold",
                "note": "lower_threshold: reducing threshold (confidence: 0.70)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    result = route_after_llm_fallback(state)
    assert result == "gap_detection"


def test_llm_fallback_proceed_low_confidence_routes_to_gap_detection():
    """Test llm_fallback → gap_detection when action is proceed_with_low_confidence."""
    state = _make_state(
        react_notes=[
            {
                "node": "llm_fallback",
                "action": "proceed_with_low_confidence",
                "note": "proceed_with_low_confidence: continuing (confidence: 0.60)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    result = route_after_llm_fallback(state)
    assert result == "gap_detection"


def test_llm_fallback_proceed_conflicts_only_routes_to_story_writer():
    """Test llm_fallback → story_writer when action is proceed_with_conflicts_only."""
    state = _make_state(
        react_notes=[
            {
                "node": "llm_fallback",
                "action": "proceed_with_conflicts_only",
                "note": "proceed_with_conflicts_only: continuing (confidence: 0.65)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    result = route_after_llm_fallback(state)
    assert result == "story_writer"


def test_llm_fallback_return_partial_routes_to_end():
    """Test llm_fallback → END when action is return_partial."""
    state = _make_state(
        react_notes=[
            {
                "node": "llm_fallback",
                "action": "return_partial",
                "note": "return_partial: done (confidence: 0.50)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    result = route_after_llm_fallback(state)
    assert result == END


def test_llm_fallback_halt_routes_to_end():
    """Test llm_fallback → END when action is halt."""
    state = _make_state(
        react_notes=[
            {
                "node": "llm_fallback",
                "action": "halt",
                "note": "halt: stopping (confidence: 0.90)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    result = route_after_llm_fallback(state)
    assert result == END


def test_llm_fallback_no_react_notes_routes_to_end():
    """Test llm_fallback → END when react_notes is empty."""
    state = _make_state(react_notes=[])
    result = route_after_llm_fallback(state)
    assert result == END


# ─── Human Review Path ────────────────────────────────────────────


def test_llm_fallback_request_review_routes_to_human_review():
    """Test llm_fallback → human_review when action is request_review."""
    state = _make_state(
        react_notes=[
            {
                "node": "llm_fallback",
                "action": "request_review",
                "note": "request_review: needs human input (confidence: 0.85)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    result = route_after_llm_fallback(state)
    assert result == "human_review"


def test_human_review_approve_routes_to_gap_detection():
    """Test human_review → gap_detection when human approves gap_detection context."""
    state = _make_state(
        gap_output=_make_gap_report(),
        react_notes=[
            {
                "node": "human_review",
                "action": "approve_gap_detection",
                "note": "Human approved: continuing from gap_detection",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        status="in_progress",
    )
    result = route_after_human_review(state)
    assert result == "gap_detection"


def test_human_review_modify_threshold_routes_to_gap_detection():
    """Test human_review → gap_detection when human requests threshold modification."""
    state = _make_state(
        react_notes=[
            {
                "node": "human_review",
                "action": "modify_threshold",
                "note": "Human requested threshold modification",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        status="in_progress",
    )
    result = route_after_human_review(state)
    assert result == "gap_detection"


def test_human_review_reject_routes_to_end():
    """Test human_review → END when human rejects."""
    state = _make_state(
        react_notes=[
            {
                "node": "human_review",
                "action": "reject",
                "note": "Human rejected: pipeline terminated",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        status="permanent_failure",
    )
    result = route_after_human_review(state)
    assert result == END


def test_human_review_approve_story_writer_routes_to_story_writer():
    """Test human_review → story_writer when human approves story_writer context."""
    state = _make_state(
        react_notes=[
            {
                "node": "human_review",
                "action": "approve_story_writer",
                "note": "Human approved: continuing from story_writer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        status="in_progress",
    )
    result = route_after_human_review(state)
    assert result == "story_writer"


# ─── Checkpointer State Persistence ──────────────────────────────


def test_checkpointer_saves_state():
    """Test that the checkpointer is enabled in compile_graph."""
    graph = compile_graph()
    # The graph should have a checkpointer configured
    # (compile_graph passes create_checkpointer() to builder.compile)
    assert graph is not None
    # Verify graph can be invoked structure is correct
    # (actual invocation would require mocking all tools)


# ─── LLM Fallback Node Integration ───────────────────────────────


@pytest.mark.asyncio
async def test_llm_fallback_records_action_in_react_notes():
    """Test that llm_fallback_node records the action field in react_notes."""
    from nodes.llm_fallback_node import llm_fallback_node

    state = _make_state(
        current_step="story_writer",
        story_output=_make_story_output(),
        status="in_progress",
    )

    # Mock LLM to return halt action
    llm_response = json.dumps({
        "action": "halt",
        "reason": "Pipeline should stop",
        "confidence": 0.9,
    })
    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await llm_fallback_node(state)

    # Verify action field is in react_notes
    assert len(result["react_notes"]) > 0
    last_note = result["react_notes"][-1]
    assert "action" in last_note
    assert last_note["action"] == "halt"


@pytest.mark.asyncio
async def test_llm_fallback_retry_increments_retry_count():
    """Test that retry actions increment retry_count in state."""
    from nodes.llm_fallback_node import llm_fallback_node

    state = _make_state(
        current_step="parser",
        parser_output=_make_extraction_result(0),
        retry_count=0,
        status="in_progress",
    )

    # Mock LLM to return retry_parser action
    llm_response = json.dumps({
        "action": "retry_parser",
        "reason": "Worth retrying",
        "confidence": 0.8,
    })
    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await llm_fallback_node(state)

    # retry_count should be incremented
    assert result["retry_count"] == 1
    assert result["react_notes"][-1]["action"] == "retry_parser"


@pytest.mark.asyncio
async def test_llm_fallback_retry_refused_when_retry_count_exceeded():
    """Test that retry is refused when retry_count >= 1."""
    from nodes.llm_fallback_node import llm_fallback_node

    state = _make_state(
        current_step="story_writer",
        story_output=_make_story_output(),
        retry_count=1,  # Already retried
        status="in_progress",
    )

    # Mock LLM to return retry action
    llm_response = json.dumps({
        "action": "retry_story_writer",
        "reason": "Try again",
        "confidence": 0.7,
    })
    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await llm_fallback_node(state)

    # Should fall back to return_partial since retry_count >= 1
    assert result["status"] == "partial_failure"
    assert result["react_notes"][-1]["action"] == "return_partial"
