"""Tests for the LLM Fallback node and updated routing logic.

All LLM calls are mocked — no real API calls.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure graph/ and nodes/ are importable
sys.path.insert(0, ".")

from graph.routing import route_after_story_writer
from graph.state import PipelineState
from nodes.llm_fallback_node import (
    _build_observation,
    _parse_llm_response,
    get_audit_log,
    llm_fallback_node,
)

from backlog_synthesizer_langgraph.models.extraction import ExtractedItem
from backlog_synthesizer_langgraph.models.gap_detection import GapReport, GapReportEntry
from backlog_synthesizer_langgraph.models.inputs import SessionInputs
from backlog_synthesizer_langgraph.models.output import (
    AcceptanceCriterion,
    Epic,
    OutputMetadata,
    StoryOutput,
    UserStory,
)


# ─── Test Fixtures ────────────────────────────────────────────────


def _make_session_inputs() -> SessionInputs:
    """Create minimal test SessionInputs."""
    return SessionInputs(
        session_id="test-session-fallback",
        documents=[],
        backlog_tickets=[],
    )


def _make_story(needs_refinement: bool = False) -> UserStory:
    """Create a test UserStory."""
    return UserStory(
        title="Test Story",
        user_story="As a user, I want something, so that I benefit",
        acceptance_criteria=[
            AcceptanceCriterion(description="Criterion A"),
            AcceptanceCriterion(description="Criterion B"),
        ],
        tags=["test-tag"],
        needs_refinement=needs_refinement,
    )


def _make_story_output(
    epics: list[Epic] | None = None,
    session_id: str = "test-session-fallback",
) -> StoryOutput:
    """Create a test StoryOutput."""
    if epics is None:
        epics = [Epic(epic_title="Test Epic", stories=[_make_story()])]
    index = [{"epic_title": e.epic_title, "story_count": len(e.stories)} for e in epics]
    return StoryOutput(
        index=index,
        epics=epics,
        metadata=OutputMetadata(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
        ),
    )


def _make_gap_report() -> GapReport:
    """Create a minimal test GapReport."""
    item = ExtractedItem(
        item_type="feature_request",
        text="Test item",
        source_chunk_index=0,
        confidence=0.9,
        tags=["test"],
    )
    entries = [
        GapReportEntry(item=item, classification="new", gap_type="NEW", confidence=0.95),
    ]
    return GapReport(
        entries=entries,
        total_new=1,
        total_duplicates=0,
        total_conflicts=0,
        total_unprocessed=0,
    )


def _make_base_state(
    story_output: StoryOutput | None = None,
    retry_count: int = 0,
    current_step: str = "story_writer",
    status: str = "in_progress",
) -> PipelineState:
    """Create a base PipelineState for testing."""
    return {
        "inputs": _make_session_inputs(),
        "session_id": "test-session-fallback",
        "parser_output": None,
        "gap_output": _make_gap_report(),
        "story_output": story_output,
        "current_step": current_step,
        "status": status,
        "errors": [],
        "react_notes": [],
        "trace_id": "test-trace-fallback",
        "total_duration_ms": 500,
        "retry_count": retry_count,
    }


def _make_llm_decision_response(
    action: str = "return_partial",
    reason: str = "Quality is acceptable for partial delivery",
    confidence: float = 0.75,
) -> str:
    """Create a mock LLM JSON response for fallback decisions."""
    return json.dumps({
        "action": action,
        "reason": reason,
        "confidence": confidence,
    })


# ─── LLM Fallback Node Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_story_writer_action_retries_correctly():
    """Test that retry_story_writer action records retry and increments retry_count."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),  # 0 epics triggers fallback
        retry_count=0,
    )

    llm_response = _make_llm_decision_response(
        action="retry_story_writer",
        reason="Zero epics likely a transient issue, retry once",
        confidence=0.8,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await llm_fallback_node(state)

    # retry_count should be incremented
    assert result["retry_count"] == 1
    # Action should be recorded in react_notes for routing
    assert result["react_notes"][-1]["action"] == "retry_story_writer"


@pytest.mark.asyncio
async def test_retry_count_prevents_infinite_loop():
    """Test that retry_count >= 1 prevents retry and returns partial instead."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=1,  # Already retried once
    )

    llm_response = _make_llm_decision_response(
        action="retry_story_writer",
        reason="Try again",
        confidence=0.7,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await llm_fallback_node(state)

    # Should NOT retry — should return partial instead
    assert result["status"] == "partial_failure"
    assert result["current_step"] == "llm_fallback"
    # react_notes should mention retry limit
    assert any("Retry limit" in note.get("note", "") for note in result["react_notes"])


@pytest.mark.asyncio
async def test_return_partial_sets_correct_status():
    """Test that return_partial action sets status to partial_failure."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=0,
    )

    llm_response = _make_llm_decision_response(
        action="return_partial",
        reason="Partial results are acceptable",
        confidence=0.85,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await llm_fallback_node(state)

    assert result["status"] == "partial_failure"
    assert result["current_step"] == "llm_fallback"
    assert len(result["react_notes"]) > 0
    assert "return_partial" in result["react_notes"][0]["note"]


@pytest.mark.asyncio
async def test_halt_sets_permanent_failure_status():
    """Test that halt action sets status to permanent_failure."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=0,
    )

    llm_response = _make_llm_decision_response(
        action="halt",
        reason="No recoverable path forward",
        confidence=0.9,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await llm_fallback_node(state)

    assert result["status"] == "permanent_failure"
    assert result["current_step"] == "llm_fallback"
    assert "halt" in result["react_notes"][0]["note"]


@pytest.mark.asyncio
async def test_llm_call_failure_defaults_to_return_partial():
    """Test that if the LLM call itself fails, we default to return_partial."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=0,
    )

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.side_effect = Exception("API key invalid")
        mock_config_cls.from_env.return_value = mock_config

        result = await llm_fallback_node(state)

    assert result["status"] == "partial_failure"
    assert result["current_step"] == "llm_fallback"
    # Should mention the failure in react_notes
    assert any("LLM call failed" in note.get("note", "") for note in result["react_notes"])


@pytest.mark.asyncio
async def test_invalid_llm_action_defaults_to_return_partial():
    """Test that an invalid action from LLM defaults to return_partial."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=0,
    )

    # LLM returns an action that doesn't exist
    llm_response = json.dumps({
        "action": "nonexistent_action",
        "reason": "I made this up",
        "confidence": 0.5,
    })

    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await llm_fallback_node(state)

    assert result["status"] == "partial_failure"
    assert result["current_step"] == "llm_fallback"


@pytest.mark.asyncio
async def test_otel_span_recorded_correctly():
    """Test that OTel span is created with correct attributes."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=0,
    )

    llm_response = _make_llm_decision_response(
        action="return_partial",
        reason="Partial OK",
        confidence=0.7,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls, \
         patch("nodes.llm_fallback_node.get_tracer", return_value=mock_tracer):
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        await llm_fallback_node(state)

    # Verify span was started with correct name
    mock_tracer.start_span.assert_called_once_with("llm_fallback")

    # Verify span attributes were set
    set_attr_calls = {
        call.args[0]: call.args[1]
        for call in mock_span.set_attribute.call_args_list
    }
    assert set_attr_calls["session_id"] == "test-session-fallback"
    assert set_attr_calls["action_chosen"] == "return_partial"
    assert set_attr_calls["confidence"] == 0.7
    assert set_attr_calls["current_step"] == "story_writer"
    assert set_attr_calls["retry_count"] == 0
    assert set_attr_calls["fallback_triggered_by"] == "story_writer"

    # Verify span was ended
    mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_audit_log_entry_created():
    """Test that an audit log entry is recorded."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=0,
    )

    llm_response = _make_llm_decision_response(
        action="return_partial",
        reason="Partial results sufficient",
        confidence=0.8,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        await llm_fallback_node(state)

    # Check audit log has an entry for this session
    audit_log = get_audit_log()
    entries = audit_log.get_entries("test-session-fallback")
    assert len(entries) > 0

    latest_entry = entries[-1]
    assert latest_entry.agent_name == "llm_fallback"
    assert "return_partial" in latest_entry.output_summary
    assert "confidence=0.80" in latest_entry.output_summary


@pytest.mark.asyncio
async def test_confidence_score_logged():
    """Test that the confidence score is correctly captured in audit log."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=0,
    )

    llm_response = _make_llm_decision_response(
        action="halt",
        reason="Unrecoverable",
        confidence=0.95,
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_response

    with patch("nodes.llm_fallback_node.PipelineConfig") as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        await llm_fallback_node(state)

    audit_log = get_audit_log()
    entries = audit_log.get_entries("test-session-fallback")
    # Find the entry for this specific test (halt action)
    halt_entries = [e for e in entries if "halt" in e.output_summary]
    assert len(halt_entries) > 0
    assert "confidence=0.95" in halt_entries[-1].output_summary


# ─── LLM Response Parsing Tests ──────────────────────────────────


def test_parse_llm_response_valid_json():
    """Test parsing a valid JSON response."""
    response = '{"action": "retry_story_writer", "reason": "worth trying", "confidence": 0.8}'
    actions = {"retry_story_writer": "desc", "return_partial": "desc", "halt": "desc"}

    action, reason, confidence = _parse_llm_response(response, actions)

    assert action == "retry_story_writer"
    assert reason == "worth trying"
    assert confidence == 0.8


def test_parse_llm_response_with_markdown_fences():
    """Test parsing response wrapped in markdown code fences."""
    response = '```json\n{"action": "halt", "reason": "no path", "confidence": 0.9}\n```'
    actions = {"retry_story_writer": "desc", "return_partial": "desc", "halt": "desc"}

    action, reason, confidence = _parse_llm_response(response, actions)

    assert action == "halt"
    assert confidence == 0.9


def test_parse_llm_response_invalid_json():
    """Test parsing an invalid JSON response defaults to return_partial."""
    response = "This is not JSON at all"
    actions = {"retry_story_writer": "desc", "return_partial": "desc", "halt": "desc"}

    action, reason, confidence = _parse_llm_response(response, actions)

    assert action == "return_partial"
    assert confidence == 0.0


def test_parse_llm_response_invalid_action():
    """Test parsing a response with an invalid action name."""
    response = '{"action": "destroy_everything", "reason": "chaos", "confidence": 0.99}'
    actions = {"retry_story_writer": "desc", "return_partial": "desc", "halt": "desc"}

    action, reason, confidence = _parse_llm_response(response, actions)

    assert action == "return_partial"
    assert confidence == 0.0


# ─── Routing Tests with retry_count ──────────────────────────────


def test_routing_retry_count_zero_routes_to_llm_fallback():
    """Test that retry_count=0 with quality issues routes to llm_fallback_node."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=0,
    )

    result = route_after_story_writer(state)
    assert result == "llm_fallback_node"


def test_routing_retry_count_1_routes_to_error_node_zero_epics():
    """Test that retry_count>=1 with 0 epics routes to error_node."""
    state = _make_base_state(
        story_output=_make_story_output(epics=[]),
        retry_count=1,
    )

    result = route_after_story_writer(state)
    assert result == "error_node"


def test_routing_retry_count_1_routes_to_error_node_refinement():
    """Test that retry_count>=1 with >50% refinement routes to error_node."""
    stories = [
        _make_story(needs_refinement=True),
        _make_story(needs_refinement=True),
        _make_story(needs_refinement=True),
        _make_story(needs_refinement=False),
    ]
    epics = [Epic(epic_title="Mixed Epic", stories=stories)]
    state = _make_base_state(
        story_output=_make_story_output(epics=epics),
        retry_count=1,
    )

    result = route_after_story_writer(state)
    assert result == "error_node"


def test_routing_success_unaffected_by_retry_count():
    """Test that successful output still returns END regardless of retry_count."""
    state = _make_base_state(
        story_output=_make_story_output(),  # Default: 1 epic with 1 good story
        retry_count=1,
    )

    result = route_after_story_writer(state)
    assert result == "__end__"
