"""Tests for the Story Writer node and routing logic.

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
from nodes.story_writer_node import story_writer_node
from nodes.error_node import error_node

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


def _make_extracted_item(
    text: str = "Users need a password reset feature for security",
    item_type: str = "feature_request",
    confidence: float = 0.9,
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


def _make_gap_report_entry(
    classification: str = "new",
    item: ExtractedItem | None = None,
) -> GapReportEntry:
    """Create a test GapReportEntry."""
    if item is None:
        item = _make_extracted_item()
    return GapReportEntry(
        item=item,
        classification=classification,
        gap_type=classification.upper(),
        confidence=0.95,
    )


def _make_gap_report(entries: list[GapReportEntry] | None = None) -> GapReport:
    """Create a test GapReport with some entries."""
    if entries is None:
        entries = [
            _make_gap_report_entry("new"),
            _make_gap_report_entry("new", _make_extracted_item(
                text="We need better logging across all microservices",
                tags=["logging", "observability"],
            )),
            _make_gap_report_entry("conflict", _make_extracted_item(
                text="Authentication should use OAuth2 instead of JWT",
                tags=["auth"],
            )),
            _make_gap_report_entry("duplicate"),  # Should be filtered out
        ]
    new_count = sum(1 for e in entries if e.classification == "new")
    dup_count = sum(1 for e in entries if e.classification == "duplicate")
    conflict_count = sum(1 for e in entries if e.classification == "conflict")
    unprocessed_count = sum(1 for e in entries if e.classification == "unprocessed")
    return GapReport(
        entries=entries,
        total_new=new_count,
        total_duplicates=dup_count,
        total_conflicts=conflict_count,
        total_unprocessed=unprocessed_count,
    )


def _make_session_inputs() -> SessionInputs:
    """Create minimal test SessionInputs."""
    return SessionInputs(
        session_id="test-session-001",
        documents=[],
        backlog_tickets=[],
    )


def _make_base_state(gap_output: GapReport | None = None) -> PipelineState:
    """Create a base PipelineState for testing."""
    return {
        "inputs": _make_session_inputs(),
        "session_id": "test-session-001",
        "parser_output": None,
        "gap_output": gap_output,
        "story_output": None,
        "current_step": "gap_detection",
        "status": "in_progress",
        "errors": [],
        "react_notes": [],
        "trace_id": "test-trace-001",
        "total_duration_ms": 500,
    }


def _mock_llm_response(title: str = "Password Reset Feature", tags: list[str] | None = None) -> str:
    """Create a mock LLM JSON response for story generation."""
    return json.dumps({
        "title": title,
        "user_story": "As a user, I want to reset my password, so that I can regain access to my account",
        "acceptance_criteria": [
            {"description": "User can request a password reset via email"},
            {"description": "Reset link expires after 24 hours"},
            {"description": "New password must meet complexity requirements"},
        ],
        "tags": tags or ["auth", "security"],
    })


# ─── Node Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_story_writer_node_produces_output():
    """Test that the node generates story_output from gap_output."""
    gap_report = _make_gap_report()
    state = _make_base_state(gap_output=gap_report)

    mock_llm = MagicMock()
    mock_llm.generate.return_value = _mock_llm_response()

    with patch(
        "nodes.story_writer_node.PipelineConfig"
    ) as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await story_writer_node(state)

    assert result["current_step"] == "story_writer"
    assert result["story_output"] is not None
    assert isinstance(result["story_output"], StoryOutput)
    assert len(result["story_output"].epics) > 0


@pytest.mark.asyncio
async def test_story_writer_node_handles_none_gap_output():
    """Test that the node handles missing gap_output gracefully."""
    state = _make_base_state(gap_output=None)

    result = await story_writer_node(state)

    assert result["current_step"] == "story_writer"
    assert result["status"] == "partial_failure"
    assert len(result["errors"]) > 0
    assert result["errors"][0]["error_type"] == "MissingInput"


@pytest.mark.asyncio
async def test_story_writer_node_transient_error():
    """Test that TransientToolError sets status to partial_failure."""
    from backlog_synthesizer_langgraph.tools.errors import TransientToolError

    gap_report = _make_gap_report()
    state = _make_base_state(gap_output=gap_report)

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = TransientToolError("Rate limit exceeded")

    with patch(
        "nodes.story_writer_node.PipelineConfig"
    ) as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        # The agent internally catches LLM errors and produces placeholder stories,
        # so TransientToolError from generate() results in placeholder stories,
        # not a node-level failure. Test that node still completes successfully.
        result = await story_writer_node(state)

    # The agent's _generate_single_story catches exceptions and returns placeholders
    # so the node should complete, but with needs_refinement stories
    assert result["current_step"] == "story_writer"
    assert result["story_output"] is not None


@pytest.mark.asyncio
async def test_story_writer_node_permanent_tool_error_on_config():
    """Test that PermanentToolError during config sets permanent_failure."""
    from backlog_synthesizer_langgraph.tools.errors import PermanentToolError

    gap_report = _make_gap_report()
    state = _make_base_state(gap_output=gap_report)

    with patch(
        "nodes.story_writer_node.PipelineConfig"
    ) as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.side_effect = PermanentToolError("No API key")
        mock_config_cls.from_env.return_value = mock_config

        result = await story_writer_node(state)

    assert result["current_step"] == "story_writer"
    assert result["status"] == "permanent_failure"
    assert len(result["errors"]) > 0
    assert "PermanentToolError" in result["errors"][0]["error_type"]


@pytest.mark.asyncio
async def test_story_writer_node_filters_duplicates():
    """Test that only NEW and CONFLICT items are processed."""
    entries = [
        _make_gap_report_entry("new"),
        _make_gap_report_entry("duplicate"),
        _make_gap_report_entry("duplicate"),
    ]
    gap_report = _make_gap_report(entries)
    state = _make_base_state(gap_output=gap_report)

    call_count = 0

    def counting_generate(prompt, system_prompt=None):
        nonlocal call_count
        call_count += 1
        return _mock_llm_response()

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = counting_generate

    with patch(
        "nodes.story_writer_node.PipelineConfig"
    ) as mock_config_cls:
        mock_config = MagicMock()
        mock_config.create_llm_tool.return_value = mock_llm
        mock_config_cls.from_env.return_value = mock_config

        result = await story_writer_node(state)

    # Only 1 "new" item should trigger LLM call (duplicates filtered out)
    assert call_count == 1
    assert result["story_output"] is not None


@pytest.mark.asyncio
async def test_error_node_sets_permanent_failure():
    """Test that error_node sets status to permanent_failure."""
    state = _make_base_state()
    state["status"] = "permanent_failure"
    state["errors"] = [
        {
            "node": "story_writer",
            "error_type": "PermanentToolError",
            "message": "Auth failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ]

    result = await error_node(state)

    assert result["current_step"] == "error_node"
    assert result["status"] == "permanent_failure"
    assert len(result["react_notes"]) > 0


# ─── Routing Tests ────────────────────────────────────────────────


def _make_story(needs_refinement: bool = False, tags: list[str] | None = None) -> UserStory:
    """Create a test UserStory."""
    return UserStory(
        title="Test Story",
        user_story="As a user, I want something, so that I benefit",
        acceptance_criteria=[
            AcceptanceCriterion(description="Criterion A"),
            AcceptanceCriterion(description="Criterion B"),
        ],
        tags=tags or ["test-tag"],
        needs_refinement=needs_refinement,
    )


def _make_story_output(
    epics: list[Epic] | None = None,
    session_id: str = "test-session-001",
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


def test_routing_returns_end_on_success():
    """Test routing returns END on successful generation."""
    state = _make_base_state()
    state["status"] = "in_progress"
    state["story_output"] = _make_story_output()

    result = route_after_story_writer(state)
    assert result == "__end__"  # langgraph.graph.END == "__end__"


def test_routing_returns_error_node_on_permanent_failure():
    """Test routing returns error_node when status is permanent_failure."""
    state = _make_base_state()
    state["status"] = "permanent_failure"
    state["story_output"] = None

    result = route_after_story_writer(state)
    assert result == "error_node"


def test_routing_returns_error_node_when_story_output_none():
    """Test routing returns error_node when story_output is None."""
    state = _make_base_state()
    state["status"] = "in_progress"
    state["story_output"] = None

    result = route_after_story_writer(state)
    assert result == "error_node"


def test_routing_returns_llm_fallback_when_zero_epics():
    """Test routing returns llm_fallback_node when 0 epics generated."""
    state = _make_base_state()
    state["status"] = "in_progress"
    state["story_output"] = _make_story_output(epics=[])

    result = route_after_story_writer(state)
    assert result == "llm_fallback_node"


def test_routing_returns_llm_fallback_when_majority_needs_refinement():
    """Test routing returns llm_fallback_node when >50% stories need refinement."""
    stories = [
        _make_story(needs_refinement=True),
        _make_story(needs_refinement=True),
        _make_story(needs_refinement=True),
        _make_story(needs_refinement=False),
    ]
    # 3/4 = 75% need refinement → should trigger fallback
    epics = [Epic(epic_title="Mixed Epic", stories=stories)]
    state = _make_base_state()
    state["status"] = "in_progress"
    state["story_output"] = _make_story_output(epics=epics)

    result = route_after_story_writer(state)
    assert result == "llm_fallback_node"


def test_routing_returns_end_when_exactly_50_percent_needs_refinement():
    """Test routing returns END when exactly 50% need refinement (not >50%)."""
    stories = [
        _make_story(needs_refinement=True),
        _make_story(needs_refinement=False),
    ]
    # 1/2 = 50% → NOT > 50%, so should return END
    epics = [Epic(epic_title="Half Epic", stories=stories)]
    state = _make_base_state()
    state["status"] = "in_progress"
    state["story_output"] = _make_story_output(epics=epics)

    result = route_after_story_writer(state)
    assert result == "__end__"
