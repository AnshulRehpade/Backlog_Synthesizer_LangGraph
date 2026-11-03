"""Tests for conflict review: judge suggestions, the review pause, and how stories obey it."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from backlog_synthesizer_langgraph.agents.gap_detection import GapDetectionAgent
from backlog_synthesizer_langgraph.agents.story_writer import StoryWriterAgent
from backlog_synthesizer_langgraph.models.gap_detection import GapReport
from backlog_synthesizer_langgraph.models.inputs import BacklogTicket, SessionInputs
from backlog_synthesizer_langgraph.tools.interfaces import SearchResult
from graph.builder import compile_graph
from graph.review import pending_review, resume_review, thread_config
from nodes.conflict_review_node import conflict_review_node
from tests.test_full_pipeline import (
    _make_extracted_item,
    _make_extraction_result,
    _make_session_inputs,
    _make_state,
    _make_story_output,
)

_TICKETS = [
    BacklogTicket(id="T-1", title="CSV export", description="Sync, up to 5,000 rows", status="in_progress"),
    BacklogTicket(id="T-2", title="Login", description="Password login", status="done"),
    BacklogTicket(id="T-3", title="Dashboard", description="Light theme only", status="open"),
]


def _conflict(text: str, ticket_id: str, suggestion: str):
    return GapDetectionAgent._conflict_entry(
        _make_extracted_item(text), SearchResult(ticket_id, 0.7), 0.7, f"{text} vs {ticket_id}", suggestion
    )


def _gap_report() -> GapReport:
    entries = [
        GapDetectionAgent._new_entry(_make_extracted_item("Brand new thing"), 0.1),
        _conflict("Async CSV export up to 1M rows", "T-1", "meeting_wins"),
        _conflict("Use SSO instead of passwords", "T-2", "keep_both"),
        _conflict("Maybe dark mode someday", "T-3", "keep_ticket"),
    ]
    return GapReport(entries=entries, total_new=1, total_duplicates=0, total_conflicts=3, total_unprocessed=0)


def _inputs() -> SessionInputs:
    base = _make_session_inputs()
    return base.model_copy(update={"backlog_tickets": _TICKETS})


# ─── Judge suggestions ────────────────────────────────────────────


class TestJudgeSuggestion:
    @pytest.mark.parametrize("answers, expected", [
        ({"is_product_work": False, "replaces_ticket": True}, "keep_ticket"),  # not work wins
        ({"is_product_work": True, "replaces_ticket": False}, "keep_both"),
        ({"is_product_work": True, "replaces_ticket": True}, "meeting_wins"),
        ({}, None),
    ])
    def test_suggestion_derived_from_yes_no_answers(self, answers, expected):
        import json
        reply = json.dumps({"verdict": "conflict", "reason": "r", **answers})
        assert GapDetectionAgent._parse_verdict(reply)[2] == expected

    def test_judge_prompt_includes_ticket_status(self):
        memory = MagicMock()
        memory.search_similar.return_value = [SearchResult("T-1", 0.7, {"content": "ticket", "status": "done"})]
        llm = MagicMock()
        llm.generate.return_value = '{"verdict": "unrelated", "reason": "r"}'
        import asyncio
        asyncio.run(GapDetectionAgent(memory, llm_tool=llm).analyze_gaps([_make_extracted_item()]))
        assert "T-1 (status: done)" in llm.generate.call_args.args[0]

    def test_conflict_suggestion_is_parsed(self):
        reply = '{"verdict": "conflict", "reason": "r", "suggestion": "keep_both"}'
        assert GapDetectionAgent._parse_verdict(reply) == ("conflict", "r", "keep_both")

    def test_unknown_suggestion_is_dropped(self):
        reply = '{"verdict": "conflict", "reason": "r", "suggestion": "flip a coin"}'
        assert GapDetectionAgent._parse_verdict(reply)[2] is None

    def test_suggestion_only_kept_for_conflicts(self):
        reply = '{"verdict": "duplicate", "reason": "r", "suggestion": "keep_both"}'
        assert GapDetectionAgent._parse_verdict(reply)[2] is None

    async def test_entry_carries_suggestion_with_meeting_wins_default(self):
        memory = MagicMock()
        memory.search_similar.return_value = [SearchResult("T-1", 0.7, {"content": "ticket"})]
        llm = MagicMock()
        llm.generate.return_value = '{"verdict": "conflict", "reason": "r", "suggestion": "keep_ticket"}'
        report = await GapDetectionAgent(memory, llm_tool=llm).analyze_gaps([_make_extracted_item()])
        assert report.entries[0].conflict_info.suggested_resolution == "keep_ticket"

        llm.generate.return_value = '{"verdict": "conflict", "reason": "r"}'
        report = await GapDetectionAgent(memory, llm_tool=llm).analyze_gaps([_make_extracted_item()])
        assert report.entries[0].conflict_info.suggested_resolution == "meeting_wins"


# ─── The node on its own ──────────────────────────────────────────


async def test_no_conflicts_passes_through():
    state = _make_state(gap_output=GapReport(
        entries=[GapDetectionAgent._new_entry(_make_extracted_item(), 0.1)],
        total_new=1, total_duplicates=0, total_conflicts=0, total_unprocessed=0,
    ))
    assert await conflict_review_node(state) == {}


async def test_review_off_passes_through(monkeypatch):
    monkeypatch.setenv("CONFLICT_REVIEW", "false")
    assert await conflict_review_node(_make_state(gap_output=_gap_report())) == {}


# ─── Through the real graph ───────────────────────────────────────


class _Fakes:
    def __init__(self):
        self.story_writer_inputs: list[GapReport] = []

    async def parser(self, state):
        return {"current_step": "parser", "parser_output": _make_extraction_result(4)}

    async def gap_detection(self, state):
        return {"current_step": "gap_detection", "gap_output": _gap_report()}

    async def story_writer(self, state):
        self.story_writer_inputs.append(state["gap_output"])
        return {"current_step": "story_writer", "status": "completed", "story_output": _make_story_output()}


@pytest.fixture
def paused():
    fakes = _Fakes()
    with patch("graph.builder.parser_node", fakes.parser), \
         patch("graph.builder.gap_detection_node", fakes.gap_detection), \
         patch("graph.builder.story_writer_node", fakes.story_writer):
        graph = compile_graph()
        session_id = str(uuid.uuid4())
        yield graph, session_id, fakes


async def _start(graph, session_id):
    state = _make_state(session_id=session_id, inputs=_inputs())
    return await graph.ainvoke(state, config=thread_config(session_id))


def _resolutions(report: GapReport) -> dict[int, str]:
    return {i: e.conflict_info.resolution for i, e in enumerate(report.entries) if e.conflict_info}


async def test_one_pause_lists_every_conflict_with_ticket_and_suggestion(paused):
    graph, session_id, fakes = paused
    review = pending_review(await _start(graph, session_id))

    assert review["kind"] == "conflict_review"
    assert review["message"] == "3 conflicts need a decision"
    assert [c["index"] for c in review["conflicts"]] == [1, 2, 3]
    first = review["conflicts"][0]
    assert (first["ticket_status"], first["suggestion"]) == ("in_progress", "meeting_wins")
    assert fakes.story_writer_inputs == []


async def test_accept_all_takes_every_suggestion(paused):
    graph, session_id, fakes = paused
    await _start(graph, session_id)
    state = await resume_review(graph, session_id, "accept_all")

    assert pending_review(state) is None
    assert _resolutions(fakes.story_writer_inputs[0]) == {1: "meeting_wins", 2: "keep_both", 3: "keep_ticket"}


async def test_overrides_apply_and_the_rest_take_suggestions(paused):
    graph, session_id, fakes = paused
    await _start(graph, session_id)
    await resume_review(graph, session_id, {"resolutions": {"1": "keep_ticket", "2": "nonsense"}})

    assert _resolutions(fakes.story_writer_inputs[0]) == {1: "keep_ticket", 2: "keep_both", 3: "keep_ticket"}


async def test_story_writer_retry_does_not_ask_again(paused):
    """Resolved conflicts stay resolved, so a later pass through the gate doesn't pause."""
    graph, session_id, fakes = paused
    await _start(graph, session_id)
    state = await resume_review(graph, session_id, "accept_all")
    assert await conflict_review_node(state) == {}


# ─── Story writer obeys the decisions ─────────────────────────────


def _resolved(resolution: str):
    entry = _conflict("Async CSV export up to 1M rows", "T-1", "meeting_wins")
    return entry.model_copy(update={
        "conflict_info": entry.conflict_info.model_copy(update={"resolution": resolution}),
    })


def _story_writer():
    llm = MagicMock()
    llm.generate.return_value = (
        '{"title": "Async CSV export", "user_story": "As a user, I want exports, so that I can work",'
        ' "acceptance_criteria": ["Exports 1M rows", "Runs in background"], "tags": ["export"]}'
    )
    return StoryWriterAgent(llm), llm


async def test_keep_ticket_writes_no_story():
    agent, llm = _story_writer()
    assert await agent.generate_stories([_resolved("keep_ticket")]) == []
    llm.generate.assert_not_called()


@pytest.mark.parametrize("resolution, phrase", [
    ("meeting_wins", "Frame it as an update, migration, or replacement"),
    (None, "Frame it as an update, migration, or replacement"),
    ("keep_both", "as an addition alongside the existing ticket"),
])
async def test_prompt_framing_follows_the_decision(resolution, phrase):
    agent, llm = _story_writer()
    await agent.generate_stories([_resolved(resolution)])
    assert phrase in llm.generate.call_args.args[0]


# ─── The UI panel ─────────────────────────────────────────────────


@pytest.fixture
def paused_app():
    fakes = _Fakes()
    with patch("graph.builder.parser_node", fakes.parser), \
         patch("graph.builder.gap_detection_node", fakes.gap_detection), \
         patch("graph.builder.story_writer_node", fakes.story_writer):
        app = AppTest.from_file("app.py", default_timeout=30)
        app.run()
        next(b for b in app.button if "Generate Stories" in b.label).click().run()
        yield app, fakes


def test_ui_shows_one_panel_with_suggestions_preselected(paused_app):
    app, _ = paused_app
    assert not app.exception
    assert any("3 conflicts need a decision" in i.value for i in app.info)
    assert [app.radio(key=f"cr_{i}").value for i in (1, 2, 3)] == ["meeting_wins", "keep_both", "keep_ticket"]


def test_ui_accept_all(paused_app):
    app, fakes = paused_app
    app.button(key="cr_accept_all").click().run()
    assert _resolutions(fakes.story_writer_inputs[0]) == {1: "meeting_wins", 2: "keep_both", 3: "keep_ticket"}
    assert app.session_state["result"]["status"] == "completed"


def test_ui_apply_my_choices(paused_app):
    app, fakes = paused_app
    app.radio(key="cr_1").set_value("keep_ticket")
    app.button(key="cr_apply").click().run()
    assert _resolutions(fakes.story_writer_inputs[0])[1] == "keep_ticket"
