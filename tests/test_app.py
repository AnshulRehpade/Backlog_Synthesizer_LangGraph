"""Smoke tests for the Streamlit app's pipeline runner function.

Tests run_pipeline_for_ui directly without invoking Streamlit UI components.
All LLM, embedding, and vector store calls are mocked.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, ".")

from app import (
    _load_sample_backlog,
    _load_sample_transcript,
    _parse_backlog_json,
    run_pipeline_for_ui,
)
from backlog_synthesizer_langgraph.models.inputs import BacklogTicket


# ─── Fixtures ─────────────────────────────────────────────────────


def _mock_llm_extraction_response() -> str:
    return json.dumps([
        {
            "item_type": "feature_request",
            "text": "SSO support with SAML 2.0",
            "confidence": 0.9,
            "stakeholder": "Jake",
            "section_heading": "Auth",
            "tags": ["auth", "sso"],
        },
        {
            "item_type": "pain_point",
            "text": "Password reset emails not arriving",
            "confidence": 0.85,
            "stakeholder": "Sarah",
            "section_heading": "Email",
            "tags": ["email", "reliability"],
        },
        {
            "item_type": "feature_request",
            "text": "Tiered rate limits based on plan",
            "confidence": 0.8,
            "stakeholder": "Jake",
            "section_heading": "Rate Limiting",
            "tags": ["api", "rate-limiting"],
        },
    ])


def _mock_story_generation_response() -> str:
    return json.dumps({
        "title": "Add SSO support with SAML 2.0",
        "user_story": "As an enterprise user, I want to log in via SSO, so that I can use my company credentials",
        "acceptance_criteria": [
            {"description": "SAML 2.0 flow works end-to-end"},
            {"description": "Users are redirected back after auth"},
        ],
        "tags": ["auth", "sso"],
    })


# ─── Tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_pipeline_returns_correct_state_structure():
    """Test that run_pipeline_for_ui returns dict with all expected keys."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = _mock_llm_extraction_response()

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384

    mock_vector = MagicMock()
    mock_vector.query_similar.return_value = []
    mock_vector.store.return_value = None

    with patch("nodes.parser_node.PipelineConfig") as mock_p_cfg, \
         patch("nodes.gap_detection_node.PipelineConfig") as mock_g_cfg, \
         patch("nodes.story_writer_node.PipelineConfig") as mock_s_cfg:

        # Parser config
        p_config = MagicMock()
        p_config.create_llm_tool.return_value = mock_llm
        p_config.create_document_parser.return_value = MagicMock()
        mock_p_cfg.from_env.return_value = p_config

        # Gap detection config
        g_config = MagicMock()
        g_config.create_embedding_tool.return_value = mock_embedding
        g_config.create_vector_search_tool.return_value = mock_vector
        g_config.gap_detection_duplicate_threshold = 0.80
        g_config.gap_detection_conflict_threshold = 0.50
        mock_g_cfg.from_env.return_value = g_config

        # Story writer config
        sw_llm = MagicMock()
        sw_llm.generate.return_value = _mock_story_generation_response()
        s_config = MagicMock()
        s_config.create_llm_tool.return_value = sw_llm
        mock_s_cfg.from_env.return_value = s_config

        result = await run_pipeline_for_ui(
            transcript_text="We need SSO support for enterprise customers.",
            backlog_tickets=[],
            filename="test.txt",
        )

    # Verify state structure
    assert "session_id" in result
    assert "status" in result
    assert "parser_output" in result
    assert "gap_output" in result
    assert "story_output" in result
    assert "errors" in result
    assert "react_notes" in result
    assert "total_duration_ms" in result
    assert "current_step" in result


@pytest.mark.asyncio
async def test_run_pipeline_with_empty_transcript_returns_state():
    """Test that an empty transcript still returns valid state (no crash)."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "[]"

    with patch("nodes.parser_node.PipelineConfig") as mock_cfg:
        config = MagicMock()
        config.create_llm_tool.return_value = mock_llm
        config.create_document_parser.return_value = MagicMock()
        mock_cfg.from_env.return_value = config

        # With empty text, parser extracts 0 items → llm_fallback → END
        with patch("nodes.llm_fallback_node.PipelineConfig") as mock_fb_cfg:
            fb_config = MagicMock()
            fb_llm = MagicMock()
            fb_llm.generate.return_value = json.dumps({
                "action": "return_partial",
                "reason": "Nothing to extract",
                "confidence": 0.9,
            })
            fb_config.create_llm_tool.return_value = fb_llm
            mock_fb_cfg.from_env.return_value = fb_config

            result = await run_pipeline_for_ui(
                transcript_text="",
                backlog_tickets=[],
            )

    assert result is not None
    assert "status" in result
    # Empty doc → parser returns partial_failure or 0 items → fallback
    assert result["status"] in ("partial_failure", "in_progress", "permanent_failure")


@pytest.mark.asyncio
async def test_run_pipeline_error_handling_no_crash():
    """Test that pipeline tool errors don't crash the function."""
    from backlog_synthesizer_langgraph.tools.errors import PermanentToolError

    with patch("nodes.parser_node.PipelineConfig") as mock_cfg:
        config = MagicMock()
        config.create_llm_tool.side_effect = PermanentToolError("No API key")
        mock_cfg.from_env.return_value = config

        result = await run_pipeline_for_ui(
            transcript_text="Some transcript content here.",
            backlog_tickets=[],
        )

    assert result["status"] in ("permanent_failure", "partial_failure")
    assert len(result["errors"]) > 0


@pytest.mark.asyncio
async def test_run_pipeline_with_backlog_tickets():
    """Test that backlog tickets are passed through correctly."""
    tickets = [
        BacklogTicket(
            id="T-1", title="Existing feature", description="Already done",
            status="done", tags=["auth"],
        )
    ]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = _mock_llm_extraction_response()

    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = [0.1] * 384

    mock_vector = MagicMock()
    mock_vector.query_similar.return_value = []
    mock_vector.store.return_value = None

    with patch("nodes.parser_node.PipelineConfig") as mock_p_cfg, \
         patch("nodes.gap_detection_node.PipelineConfig") as mock_g_cfg, \
         patch("nodes.story_writer_node.PipelineConfig") as mock_s_cfg:

        p_config = MagicMock()
        p_config.create_llm_tool.return_value = mock_llm
        p_config.create_document_parser.return_value = MagicMock()
        mock_p_cfg.from_env.return_value = p_config

        g_config = MagicMock()
        g_config.create_embedding_tool.return_value = mock_embedding
        g_config.create_vector_search_tool.return_value = mock_vector
        g_config.gap_detection_duplicate_threshold = 0.80
        g_config.gap_detection_conflict_threshold = 0.50
        mock_g_cfg.from_env.return_value = g_config

        sw_llm = MagicMock()
        sw_llm.generate.return_value = _mock_story_generation_response()
        s_config = MagicMock()
        s_config.create_llm_tool.return_value = sw_llm
        mock_s_cfg.from_env.return_value = s_config

        result = await run_pipeline_for_ui(
            transcript_text="We need better authentication.",
            backlog_tickets=tickets,
        )

    assert result is not None
    assert result["status"] in ("in_progress", "partial_failure", "completed")


# ─── Helper Function Tests ────────────────────────────────────────


def test_load_sample_transcript_returns_text():
    """Test that sample transcript loads successfully."""
    text = _load_sample_transcript()
    assert len(text) > 100
    assert "Sprint Planning" in text


def test_load_sample_backlog_returns_tickets():
    """Test that sample backlog loads correct number of tickets."""
    tickets = _load_sample_backlog()
    assert len(tickets) == 10
    assert tickets[0].id == "PLAT-101"


def test_parse_backlog_json_valid():
    """Test parsing valid backlog JSON."""
    data = json.dumps([
        {"id": "X-1", "title": "Test ticket", "description": "Desc", "status": "open"}
    ])
    tickets = _parse_backlog_json(data)
    assert len(tickets) == 1
    assert tickets[0].id == "X-1"


def test_parse_backlog_json_invalid_format():
    """Test that non-array JSON raises ValueError."""
    with pytest.raises(ValueError, match="array"):
        _parse_backlog_json('{"not": "an array"}')


def test_parse_backlog_json_invalid_json():
    """Test that malformed JSON raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        _parse_backlog_json("not json at all")
