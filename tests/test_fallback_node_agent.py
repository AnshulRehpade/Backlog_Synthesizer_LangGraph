"""Tests for the LLM fallback node in agent mode (FALLBACK_MODE=agent).

The agent loop itself is tested in test_fallback_agent.py; these cover how the
node uses it: when it runs, when the menu takes over, and what lands in state.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backlog_synthesizer_langgraph.agents.fallback_agent import AgentOutcome, AgentStep
from backlog_synthesizer_langgraph.agents.fallback_tools import Decision
from graph.builder import compile_graph
from graph.review import pending_review, resume_review, thread_config
from nodes.human_review_node import _build_review_summary
from nodes.llm_fallback_node import get_audit_log, llm_fallback_node
from tests.test_fallback_agent import _call, _claude, _reply
from tests.test_full_pipeline import _make_gap_report, _make_session_inputs, _make_state
from tests.test_human_review_resume import _FakeNodes

_MENU_DECISION = ("return_partial", "menu reason", 0.5)


@pytest.fixture
def agent_mode(monkeypatch):
    monkeypatch.setenv("FALLBACK_MODE", "agent")


def _state(**overrides):
    base = {
        "session_id": f"s-{uuid.uuid4()}",
        "current_step": "gap_detection",
        "gap_output": _make_gap_report(1, 9, 0),
        "retry_count": 0,
    }
    return _make_state(**{**base, **overrides})


def _outcome(action: str | None = "proceed_with_conflicts_only", corrected=None, stopped="finished"):
    return AgentOutcome(
        decision=Decision(action, "agent reason", 0.8) if action else None,
        updated_gap_output=corrected,
        steps=[
            AgentStep("list_matches", {}, "[0] new ...", False),
            AgentStep("reclassify", {"index": 1, "classification": "new", "reason": "r"},
                      "Item [1] is now new (was duplicate).", False),
        ],
        turns=3,
        tokens_used=1234,
        stopped_because=stopped,
    )


async def _run_node(state, outcome=None, agent_error=None, llm_tool=None):
    """Run the node with the agent and the menu call both mocked."""
    config = MagicMock(fallback_mode="agent")
    config.create_llm_tool.return_value = llm_tool or MagicMock(spec=["generate", "create_message"])
    agent = AsyncMock(return_value=outcome, side_effect=agent_error)
    menu = AsyncMock(return_value=_MENU_DECISION)
    with patch("nodes.llm_fallback_node.PipelineConfig") as config_cls, \
         patch("nodes.llm_fallback_node.run_fallback_agent", agent), \
         patch("nodes.llm_fallback_node._ask_llm_for_decision", menu):
        config_cls.from_env.return_value = config
        result = await llm_fallback_node(state)
    return result, agent, menu


async def test_agent_decision_is_used_and_corrections_applied(agent_mode):
    corrected = _make_gap_report(2, 8, 0)
    result, agent, menu = await _run_node(_state(), _outcome(corrected=corrected))

    menu.assert_not_called()
    assert result["gap_output"] is corrected
    assert result["react_notes"][-1]["action"] == "proceed_with_conflicts_only"
    assert [n["action"] for n in result["react_notes"][:-1]] == ["agent_step", "agent_step"]
    assert "reclassify(index=1, classification='new')" in result["react_notes"][1]["note"]


async def test_agent_without_corrections_leaves_gap_output_alone(agent_mode):
    result, _, _ = await _run_node(_state(), _outcome(corrected=None))
    assert "gap_output" not in result


async def test_agent_gives_up_menu_decides_and_corrections_are_dropped(agent_mode):
    outcome = _outcome(action=None, corrected=_make_gap_report(2, 8, 0), stopped="turn_limit")
    result, _, menu = await _run_node(_state(), outcome)

    menu.assert_called_once()
    assert "gap_output" not in result
    assert result["react_notes"][-1]["action"] == "return_partial"
    assert result["react_notes"][-2]["action"] == "agent_gave_up"
    assert "turn_limit" in result["react_notes"][-2]["note"]


async def test_agent_crash_falls_back_to_menu(agent_mode):
    result, _, menu = await _run_node(_state(), agent_error=RuntimeError("boom"))
    menu.assert_called_once()
    assert result["react_notes"][-1]["action"] == "return_partial"


async def test_llm_without_tool_calling_uses_menu(agent_mode):
    _, agent, menu = await _run_node(_state(), _outcome(), llm_tool=MagicMock(spec=["generate"]))
    agent.assert_not_called()
    menu.assert_called_once()


async def test_menu_mode_never_runs_the_agent():
    # conftest pins FALLBACK_MODE=menu; the real config is read here.
    agent = AsyncMock()
    menu = AsyncMock(return_value=_MENU_DECISION)
    with patch("nodes.llm_fallback_node.run_fallback_agent", agent), \
         patch("nodes.llm_fallback_node._ask_llm_for_decision", menu):
        await llm_fallback_node(_state())
    agent.assert_not_called()
    menu.assert_called_once()


async def test_retry_limit_still_applies_to_agent_choice(agent_mode):
    result, _, _ = await _run_node(_state(retry_count=1), _outcome(action="lower_threshold"))
    assert result["react_notes"][-1]["action"] == "return_partial"
    assert result["status"] == "partial_failure"


async def test_audit_records_who_decided(agent_mode):
    state = _state()
    await _run_node(state, _outcome())
    entry = get_audit_log().get_entries(state["session_id"])[-1]
    assert "decided_by=agent" in entry.output_summary


def test_review_summary_shows_agent_evidence():
    notes = [
        {"node": "llm_fallback_agent", "action": "agent_step", "tool": "list_matches", "note": "list_matches() → ..."},
        {"node": "llm_fallback_agent", "action": "agent_step", "tool": "compare",
         "note": "compare(index=0, ticket_id='T-12') → Verdict: conflict. sync vs async"},
        {"node": "llm_fallback", "action": "request_review", "note": "request_review: unsure (confidence: 0.60)"},
    ]
    summary = _build_review_summary(_state(react_notes=notes))
    assert "Agent: compare(index=0, ticket_id='T-12') → Verdict: conflict" in summary
    assert "list_matches" not in summary  # only verdicts and corrections are evidence


async def test_end_to_end_agent_corrects_asks_human_then_stories_use_corrections(agent_mode):
    """Real graph: fallback agent reclassifies, hands to a human, approve writes stories."""
    fakes = _FakeNodes()
    seen_by_story_writer = []

    async def story_writer(state):
        seen_by_story_writer.append(state["gap_output"])
        return await fakes.story_writer(state)

    claude = _claude(
        _reply(_call("list_matches", "a")),
        _reply(_call("reclassify", "b", index=1, classification="new", reason="Different feature")),
        _reply(_call("finish", "c", action="request_review", reason="Several look wrong", confidence=0.6)),
    )
    config = MagicMock(fallback_mode="agent")
    config.create_llm_tool.return_value = claude

    with patch("graph.builder.parser_node", fakes.parser), \
         patch("graph.builder.gap_detection_node", fakes.gap_detection), \
         patch("graph.builder.story_writer_node", story_writer), \
         patch("nodes.llm_fallback_node.PipelineConfig") as config_cls:
        config_cls.from_env.return_value = config
        graph = compile_graph()
        session_id = str(uuid.uuid4())
        state = await graph.ainvoke(
            _make_state(session_id=session_id, inputs=_make_session_inputs()),
            config=thread_config(session_id),
        )

        review = pending_review(state)
        assert review is not None
        assert "Agent: reclassify(index=1, classification='new')" in review["summary"]
        assert "2 new, 8 duplicates" in review["summary"]

        final = await resume_review(graph, session_id, "approve")

    assert final["status"] == "completed"
    assert seen_by_story_writer[0].total_duplicates == 8
