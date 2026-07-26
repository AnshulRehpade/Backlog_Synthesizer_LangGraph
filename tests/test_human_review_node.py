"""Tests for the Human Review node (Level 3 of three-tier decision pattern).

Tests mock interrupt() to simulate human decisions without
actually pausing execution.

All external calls are mocked — no real API calls.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from graph.state import PipelineState
from nodes.human_review_node import (
    _build_review_summary,
    _get_triggering_step,
    get_audit_log,
    human_review_node,
)

from backlog_synthesizer_langgraph.models.extraction import (
    ExtractionResult,
    ExtractedItem,
)
from backlog_synthesizer_langgraph.models.gap_detection import (
    GapReport,
    GapReportEntry,
)
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
    return SessionInputs(session_id="test-session-hr", documents=[], backlog_tickets=[])


def _make_extracted_item() -> ExtractedItem:
    return ExtractedItem(
        item_type="feature_request",
        text="SSO support needed",
        source_chunk_index=0,
        confidence=0.9,
        tags=["auth", "sso"],
    )


def _make_extraction_result() -> ExtractionResult:
    return ExtractionResult(
        items=[_make_extracted_item() for _ in range(5)],
        errors=[],
        metadata={"document_count": 1, "chunks_created": 2},
    )


def _make_gap_report() -> GapReport:
    entries = [
        GapReportEntry(
            item=_make_extracted_item(),
            classification="new",
            gap_type="NEW",
            confidence=0.9,
        )
        for _ in range(3)
    ]
    return GapReport(
        entries=entries,
        total_new=3,
        total_duplicates=2,
        total_conflicts=1,
        total_unprocessed=0,
    )


def _make_story_output() -> StoryOutput:
    story = UserStory(
        title="SSO Feature",
        user_story="As a user, I want SSO, so that I can log in with my company credentials",
        acceptance_criteria=[
            AcceptanceCriterion(description="SAML 2.0 supported"),
            AcceptanceCriterion(description="OpenID Connect supported"),
        ],
        tags=["auth", "sso"],
        needs_refinement=False,
    )
    return StoryOutput(
        index=[{"epic_title": "Auth", "story_count": 1}],
        epics=[Epic(epic_title="Auth", stories=[story])],
        metadata=OutputMetadata(
            session_id="test-session-hr",
            timestamp=datetime.now(timezone.utc),
        ),
    )


def _make_state(**overrides) -> PipelineState:
    base = {
        "inputs": _make_session_inputs(),
        "session_id": "test-session-hr",
        "parser_output": _make_extraction_result(),
        "gap_output": _make_gap_report(),
        "story_output": None,
        "current_step": "llm_fallback",
        "status": "partial_failure",
        "errors": [],
        "react_notes": [
            {
                "node": "llm_fallback",
                "action": "request_review",
                "note": "request_review: too many duplicates (confidence: 0.85)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "retry_count": 0,
        "needs_human_review": True,
        "trace_id": "test-trace-hr",
        "total_duration_ms": 200,
    }
    base.update(overrides)
    return base


# ─── Node Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interrupt_called_with_correct_summary():
    """Test that interrupt() is called with the expected review payload."""
    state = _make_state()
    interrupt_payload = None

    def mock_interrupt(payload):
        nonlocal interrupt_payload
        interrupt_payload = payload
        return "approve"

    with patch("nodes.human_review_node.interrupt", side_effect=mock_interrupt):
        await human_review_node(state)

    assert interrupt_payload is not None
    assert interrupt_payload["message"] == "Human review required"
    assert interrupt_payload["current_step"] == "gap_detection"
    assert "options" in interrupt_payload
    assert "approve" in interrupt_payload["options"]
    assert "reject" in interrupt_payload["options"]
    assert "modify_threshold" in interrupt_payload["options"]
    assert "summary" in interrupt_payload


@pytest.mark.asyncio
async def test_approve_decision_sets_in_progress():
    """Test that 'approve' decision sets status to in_progress."""
    state = _make_state()

    with patch("nodes.human_review_node.interrupt", return_value="approve"):
        result = await human_review_node(state)

    assert result["status"] == "in_progress"
    assert result["needs_human_review"] is False
    assert result["current_step"] == "human_review"


@pytest.mark.asyncio
async def test_approve_decision_gap_detection_context():
    """Test that approve with gap_output present routes via approve_gap_detection action."""
    state = _make_state(gap_output=_make_gap_report())

    with patch("nodes.human_review_node.interrupt", return_value="approve"):
        result = await human_review_node(state)

    last_note = result["react_notes"][-1]
    assert last_note["action"] == "approve_gap_detection"
    assert "gap_detection" in last_note["note"]


@pytest.mark.asyncio
async def test_approve_decision_story_writer_context():
    """Test that approve with only parser_output routes via approve_story_writer action."""
    state = _make_state(gap_output=None, parser_output=None)

    with patch("nodes.human_review_node.interrupt", return_value="approve"):
        result = await human_review_node(state)

    last_note = result["react_notes"][-1]
    assert last_note["action"] == "approve_story_writer"


@pytest.mark.asyncio
async def test_reject_decision_sets_permanent_failure():
    """Test that 'reject' decision sets status to permanent_failure."""
    state = _make_state()

    with patch("nodes.human_review_node.interrupt", return_value="reject"):
        result = await human_review_node(state)

    assert result["status"] == "permanent_failure"
    assert result["needs_human_review"] is False
    last_note = result["react_notes"][-1]
    assert last_note["action"] == "reject"
    assert "terminated" in last_note["note"]


@pytest.mark.asyncio
async def test_modify_threshold_routes_to_gap_detection():
    """Test that 'modify_threshold' decision sets action for gap_detection routing."""
    state = _make_state()

    with patch("nodes.human_review_node.interrupt", return_value="modify_threshold"):
        result = await human_review_node(state)

    assert result["status"] == "in_progress"
    assert result["needs_human_review"] is False
    last_note = result["react_notes"][-1]
    assert last_note["action"] == "modify_threshold"
    assert "gap detection" in last_note["note"]


@pytest.mark.asyncio
async def test_dict_decision_format():
    """Test that decision can be passed as a dict with 'decision' key."""
    state = _make_state()

    with patch("nodes.human_review_node.interrupt", return_value={"decision": "approve"}):
        result = await human_review_node(state)

    assert result["status"] == "in_progress"
    assert result["needs_human_review"] is False


@pytest.mark.asyncio
async def test_unknown_decision_defaults_to_reject():
    """Test that an unknown decision value defaults to reject behavior."""
    state = _make_state()

    with patch("nodes.human_review_node.interrupt", return_value="something_else"):
        result = await human_review_node(state)

    assert result["status"] == "permanent_failure"
    last_note = result["react_notes"][-1]
    assert last_note["action"] == "reject"


@pytest.mark.asyncio
async def test_audit_log_entry_created():
    """Test that an audit log entry is recorded after human decision."""
    state = _make_state()

    with patch("nodes.human_review_node.interrupt", return_value="approve"):
        await human_review_node(state)

    audit_log = get_audit_log()
    entries = audit_log.get_entries("test-session-hr")
    assert len(entries) > 0

    latest = entries[-1]
    assert latest.agent_name == "human_review"
    assert "decision=approve" in latest.output_summary
    assert "triggering_step=gap_detection" in latest.input_summary


@pytest.mark.asyncio
async def test_otel_span_recorded():
    """Test that OTel span is created with correct attributes."""
    state = _make_state()

    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_span.return_value = mock_span

    with patch("nodes.human_review_node.interrupt", return_value="reject"), \
         patch("nodes.human_review_node.get_tracer", return_value=mock_tracer):
        await human_review_node(state)

    mock_tracer.start_span.assert_called_once_with("human_review")

    set_attr_calls = {
        call.args[0]: call.args[1]
        for call in mock_span.set_attribute.call_args_list
    }
    assert set_attr_calls["session_id"] == "test-session-hr"
    assert set_attr_calls["decision"] == "reject"
    assert set_attr_calls["triggering_step"] == "gap_detection"
    assert "duration_ms" in set_attr_calls
    mock_span.end.assert_called_once()


@pytest.mark.asyncio
async def test_exception_in_node_returns_gracefully():
    """Test that unexpected exceptions are caught and state is returned."""
    state = _make_state()

    with patch("nodes.human_review_node.interrupt", side_effect=RuntimeError("boom")):
        result = await human_review_node(state)

    assert result["status"] == "partial_failure"
    assert result["current_step"] == "human_review"
    assert len(result["errors"]) > 0
    assert "boom" in result["errors"][0]["message"]


@pytest.mark.asyncio
async def test_total_duration_ms_accumulated():
    """Test that total_duration_ms is accumulated from prior state."""
    state = _make_state(total_duration_ms=500)

    with patch("nodes.human_review_node.interrupt", return_value="approve"):
        result = await human_review_node(state)

    assert result["total_duration_ms"] >= 500


# ─── Helper Function Tests ────────────────────────────────────────


def test_build_review_summary_includes_gap_info():
    """Test that review summary includes gap detection info when available."""
    state = _make_state()
    summary = _build_review_summary(state)

    assert "3 new" in summary
    assert "2 duplicates" in summary
    assert "1 conflicts" in summary


def test_build_review_summary_includes_parser_info():
    """Test that review summary includes extracted items count."""
    state = _make_state()
    summary = _build_review_summary(state)

    assert "Extracted items: 5" in summary


def test_build_review_summary_includes_trigger():
    """Test that review summary includes trigger note from react_notes."""
    state = _make_state()
    summary = _build_review_summary(state)

    assert "request_review" in summary


def test_build_review_summary_includes_story_info():
    """Test that review summary includes story output when available."""
    state = _make_state(story_output=_make_story_output())
    summary = _build_review_summary(state)

    assert "Stories: 1 in 1 epics" in summary


def test_build_review_summary_empty_state():
    """Test that review summary handles empty state gracefully."""
    state = _make_state(
        parser_output=None, gap_output=None, story_output=None,
        react_notes=[], errors=[],
    )
    summary = _build_review_summary(state)

    assert summary == "No additional context available."


def test_get_triggering_step_gap_detection():
    """Test that triggering step is inferred as gap_detection when gap_output exists."""
    state = _make_state(gap_output=_make_gap_report())
    step = _get_triggering_step(state)
    assert step == "gap_detection"


def test_get_triggering_step_parser():
    """Test that triggering step is inferred as parser when only parser_output exists."""
    state = _make_state(gap_output=None, parser_output=_make_extraction_result())
    step = _get_triggering_step(state)
    assert step == "parser"


def test_get_triggering_step_unknown():
    """Test that triggering step returns 'unknown' when no outputs exist."""
    state = _make_state(gap_output=None, parser_output=None)
    step = _get_triggering_step(state)
    assert step == "unknown"
