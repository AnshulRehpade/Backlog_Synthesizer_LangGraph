"""Tests for the fallback agent loop, driven by a scripted fake Claude (no API calls)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from backlog_synthesizer_langgraph.agents import fallback_agent
from backlog_synthesizer_langgraph.agents.fallback_agent import (
    MAX_TURNS,
    run_fallback_agent,
    supports_tool_calling,
)
from backlog_synthesizer_langgraph.agents.fallback_tools import FallbackToolbox
from backlog_synthesizer_langgraph.tools.errors import TransientToolError
from tests.test_fallback_tools import _ACTIONS, _state


def _call(name: str, call_id: str = "c1", **tool_input):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=tool_input)


def _text(text: str):
    return SimpleNamespace(type="text", text=text)


def _reply(*blocks, stop_reason: str = "tool_use", tokens: int = 100):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=tokens, output_tokens=0),
    )


_FINISH = _call("finish", "f1", action="return_partial", reason="Done", confidence=0.7)


def _claude(*replies):
    """A fake LLM tool whose create_message returns the given replies in order."""
    llm = MagicMock()
    llm.create_message.side_effect = list(replies)
    return llm


async def _run(llm):
    toolbox = FallbackToolbox(_state(), _ACTIONS)
    return await run_fallback_agent(toolbox, llm, "gap_report: new=1, duplicates=1", "gap_detection")


def _sent_messages(llm, call_index: int) -> list[dict]:
    return llm.create_message.call_args_list[call_index].args[0]


async def test_investigates_corrects_and_finishes():
    llm = _claude(
        _reply(_call("list_matches")),
        _reply(_call("reclassify", "c2", index=1, classification="new", reason="Unrelated")),
        _reply(_FINISH),
    )
    outcome = await _run(llm)

    assert outcome.stopped_because == "finished"
    assert outcome.decision.action == "return_partial"
    assert [s.tool for s in outcome.steps] == ["list_matches", "reclassify", "finish"]
    assert outcome.updated_gap_output.total_duplicates == 0
    assert (outcome.turns, outcome.tokens_used) == (3, 300)


async def test_tool_results_go_back_with_matching_ids_in_one_message():
    llm = _claude(
        _reply(_call("list_matches", "a"), _call("get_ticket", "b", ticket_id="T-0")),
        _reply(_FINISH),
    )
    await _run(llm)

    results = _sent_messages(llm, 1)[-1]
    assert results["role"] == "user"
    assert [r["tool_use_id"] for r in results["content"]] == ["a", "b"]


async def test_first_message_carries_the_observation_and_tools():
    llm = _claude(_reply(_FINISH))
    await _run(llm)

    args = llm.create_message.call_args.args
    assert "duplicates=1" in args[0][0]["content"]
    assert args[1] == fallback_agent.SYSTEM_PROMPT
    assert "finish" in {tool["name"] for tool in args[2]}


async def test_failed_tool_is_reported_and_agent_can_recover():
    llm = _claude(
        _reply(_call("get_ticket", "bad", ticket_id="NOPE")),
        _reply(_FINISH),
    )
    outcome = await _run(llm)

    error = _sent_messages(llm, 1)[-1]["content"][0]
    assert error["is_error"] is True
    assert outcome.stopped_because == "finished"


async def test_invalid_finish_does_not_end_the_loop():
    llm = _claude(
        _reply(_call("finish", "x", action="delete_everything", reason="x", confidence=1)),
        _reply(_FINISH),
    )
    outcome = await _run(llm)
    assert outcome.decision.action == "return_partial"
    assert outcome.turns == 2


async def test_text_reply_gets_one_nudge_then_gives_up():
    llm = _claude(
        _reply(_text("I think it's fine."), stop_reason="end_turn"),
        _reply(_text("Really, it's fine."), stop_reason="end_turn"),
    )
    outcome = await _run(llm)

    assert outcome.stopped_because == "no_finish"
    assert outcome.decision is None
    assert "finish" in _sent_messages(llm, 1)[-1]["content"]


async def test_nudge_can_succeed():
    llm = _claude(_reply(_text("Looks fine."), stop_reason="end_turn"), _reply(_FINISH))
    assert (await _run(llm)).stopped_because == "finished"


async def test_turn_limit():
    llm = _claude(*[_reply(_call("list_matches", f"c{i}")) for i in range(MAX_TURNS + 2)])
    outcome = await _run(llm)

    assert outcome.stopped_because == "turn_limit"
    assert llm.create_message.call_count == MAX_TURNS
    assert outcome.decision is None


async def test_token_budget():
    llm = _claude(*[_reply(_call("list_matches", f"c{i}"), tokens=20_000) for i in range(3)])
    outcome = await _run(llm)

    assert outcome.stopped_because == "budget"
    assert llm.create_message.call_count == 2


async def test_api_error_gives_no_decision():
    llm = MagicMock()
    llm.create_message.side_effect = TransientToolError("overloaded")
    outcome = await _run(llm)
    assert (outcome.stopped_because, outcome.decision) == ("api_error", None)


async def test_refusal_stops_without_running_tools():
    llm = _claude(_reply(_FINISH, stop_reason="refusal"))
    outcome = await _run(llm)
    assert outcome.stopped_because == "refusal"
    assert outcome.steps == []


def test_supports_tool_calling():
    assert supports_tool_calling(MagicMock(spec=["generate", "create_message"]))
    assert not supports_tool_calling(MagicMock(spec=["generate"]))
