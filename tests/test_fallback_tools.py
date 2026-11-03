"""Tests for the fallback agent's tools (FallbackToolbox)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backlog_synthesizer_langgraph.agents.fallback_tools import MAX_COMPARES, FallbackToolbox
from backlog_synthesizer_langgraph.models.inputs import BacklogTicket, SessionInputs
from tests.test_full_pipeline import _make_extraction_result, _make_gap_report

_ACTIONS = {"return_partial": "Return what we have", "request_review": "Ask a person"}


def _state(with_gap: bool = True, with_tickets: bool = True) -> dict:
    tickets = [
        BacklogTicket(id="T-0", title="CSV export", description="Sync export up to 5,000 rows",
                      status="open", tags=["export"]),
        BacklogTicket(id="TC-0", title="SSO login", description="Login with Okta", status="done"),
    ] if with_tickets else []
    return {
        "inputs": SessionInputs(session_id="s", documents=[], backlog_tickets=tickets),
        "parser_output": _make_extraction_result(3),
        # entries: [0] new, [1] duplicate of T-0, [2] conflict with TC-0
        "gap_output": _make_gap_report(1, 1, 1) if with_gap else None,
    }


def _toolbox(llm_reply: str | None = None, **state_kwargs) -> FallbackToolbox:
    llm = None
    if llm_reply is not None:
        llm = MagicMock()
        llm.generate.return_value = llm_reply
    return FallbackToolbox(_state(**state_kwargs), _ACTIONS, llm_tool=llm)


def _names(toolbox: FallbackToolbox) -> set[str]:
    return {d["name"] for d in toolbox.definitions()}


class TestDefinitions:
    def test_all_tools_offered_with_gap_output_tickets_and_llm(self):
        assert _names(_toolbox('{"verdict": "duplicate"}')) == {
            "list_items", "list_matches", "get_ticket", "compare", "reclassify", "finish",
        }

    def test_parser_step_offers_only_list_items_and_tickets(self):
        assert _names(_toolbox(with_gap=False)) == {"list_items", "get_ticket", "finish"}

    def test_compare_needs_an_llm(self):
        assert "compare" not in _names(_toolbox())

    def test_finish_action_enum_matches_step_menu(self):
        finish = next(d for d in _toolbox().definitions() if d["name"] == "finish")
        assert finish["input_schema"]["properties"]["action"]["enum"] == list(_ACTIONS)


class TestReadOnlyTools:
    async def test_list_matches_shows_label_score_and_ticket(self):
        result = await _toolbox().run("list_matches", {})
        assert "[1] duplicate, score 0.90, ticket T-0" in result.content

    async def test_list_items_shows_confidence(self):
        result = await _toolbox().run("list_items", {})
        assert "[0] feature_request, confidence 0.80" in result.content

    async def test_get_ticket(self):
        result = await _toolbox().run("get_ticket", {"ticket_id": "T-0"})
        assert "Sync export up to 5,000 rows" in result.content and not result.is_error

    async def test_get_unknown_ticket_is_error(self):
        result = await _toolbox().run("get_ticket", {"ticket_id": "NOPE"})
        assert result.is_error


class TestCompare:
    async def test_returns_verdict_and_reason(self):
        tb = _toolbox('{"verdict": "conflict", "reason": "Async 1M vs sync 5k"}')
        result = await tb.run("compare", {"index": 1, "ticket_id": "T-0"})
        assert result.content == "Verdict: conflict. Async 1M vs sync 5k"
        prompt = tb.llm_tool.generate.call_args.args[0]
        assert "CSV export Sync export up to 5,000 rows" in prompt

    async def test_limit_stops_further_llm_calls(self):
        tb = _toolbox('{"verdict": "duplicate", "reason": "same"}')
        for _ in range(MAX_COMPARES):
            await tb.run("compare", {"index": 0, "ticket_id": "T-0"})
        result = await tb.run("compare", {"index": 0, "ticket_id": "T-0"})
        assert result.is_error
        assert tb.llm_tool.generate.call_count == MAX_COMPARES

    async def test_unusable_judge_reply_is_error(self):
        result = await _toolbox("no idea").run("compare", {"index": 0, "ticket_id": "T-0"})
        assert result.is_error


class TestReclassify:
    async def test_changes_label_and_totals(self):
        tb = _toolbox()
        result = await tb.run(
            "reclassify",
            {"index": 1, "classification": "conflict", "ticket_id": "T-0", "reason": "Different limits"},
        )
        assert "now conflict (was duplicate)" in result.content
        report = tb.updated_gap_output()
        assert (report.total_new, report.total_duplicates, report.total_conflicts) == (1, 0, 2)
        assert report.entries[1].conflict_info.contradiction_description == "Different limits"

    async def test_does_not_touch_the_original_state(self):
        tb = _toolbox()
        await tb.run("reclassify", {"index": 1, "classification": "new", "reason": "Unrelated"})
        assert tb.state["gap_output"].total_duplicates == 1

    async def test_list_matches_reflects_reclassification(self):
        tb = _toolbox()
        await tb.run("reclassify", {"index": 1, "classification": "new", "reason": "Unrelated"})
        result = await tb.run("list_matches", {})
        assert "[1] new (reclassified)" in result.content

    async def test_no_changes_means_no_updated_report(self):
        assert _toolbox().updated_gap_output() is None

    @pytest.mark.parametrize("args", [
        {"index": 9, "classification": "new", "reason": "x"},             # out of range
        {"index": 1, "classification": "conflict", "reason": "x"},       # missing ticket
        {"index": 1, "classification": "maybe", "reason": "x"},          # bad label
        {"index": 1, "classification": "new", "reason": "  "},           # no reason
        {"index": 1, "classification": "duplicate", "ticket_id": "NOPE", "reason": "x"},
    ])
    async def test_bad_input_is_an_error_and_changes_nothing(self, args):
        tb = _toolbox()
        result = await tb.run("reclassify", args)
        assert result.is_error
        assert tb.reclassifications == {}


class TestFinish:
    async def test_records_decision(self):
        tb = _toolbox()
        await tb.run("finish", {"action": "request_review", "reason": "Unsure", "confidence": 0.6})
        assert (tb.decision.action, tb.decision.confidence) == ("request_review", 0.6)

    async def test_rejects_action_outside_menu(self):
        tb = _toolbox()
        result = await tb.run("finish", {"action": "delete_backlog", "reason": "x", "confidence": 1})
        assert result.is_error and tb.decision is None

    async def test_confidence_is_clamped(self):
        tb = _toolbox()
        await tb.run("finish", {"action": "return_partial", "reason": "x", "confidence": 7})
        assert tb.decision.confidence == 1.0


async def test_unknown_tool_is_error():
    assert (await _toolbox().run("drop_tables", {})).is_error


async def test_tool_not_offered_in_this_state_is_error():
    # No LLM → compare isn't offered, so calling it anyway must fail.
    assert (await _toolbox().run("compare", {"index": 0, "ticket_id": "T-0"})).is_error
