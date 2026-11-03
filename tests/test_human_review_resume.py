"""End-to-end tests for pausing at human review and resuming with a decision.

Runs the real compiled graph (routing, fallback, human_review, checkpointer)
with the three agent nodes replaced by fakes, so a run can be steered into
review and resumed without any API calls.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from graph.builder import compile_graph
from graph.review import pending_review, resume_review, thread_config
from tests.test_full_pipeline import (
    _make_extraction_result,
    _make_gap_report,
    _make_session_inputs,
    _make_state,
    _make_story_output,
)


class _FakeNodes:
    """Fake agent nodes that record the duplicate threshold each gap run saw."""

    def __init__(self) -> None:
        self.gap_thresholds: list[float | None] = []
        self.story_writer_calls = 0

    async def parser(self, state):
        return {"current_step": "parser", "parser_output": _make_extraction_result(5)}

    async def gap_detection(self, state):
        self.gap_thresholds.append(state.get("duplicate_threshold_override"))
        # 9 of 10 duplicates trips the ">80% duplicates" rule → llm_fallback.
        return {"current_step": "gap_detection", "gap_output": _make_gap_report(1, 9, 0)}

    async def story_writer(self, state):
        self.story_writer_calls += 1
        return {
            "current_step": "story_writer",
            "status": "completed",
            "story_output": _make_story_output(),
        }


@pytest.fixture
def paused_run():
    """Start a run that pauses at human review; yields (graph, session_id, state, fakes)."""
    fakes = _FakeNodes()
    with patch("graph.builder.parser_node", fakes.parser), \
         patch("graph.builder.gap_detection_node", fakes.gap_detection), \
         patch("graph.builder.story_writer_node", fakes.story_writer), \
         patch(
             "nodes.llm_fallback_node._ask_llm_for_decision",
             AsyncMock(return_value=("request_review", "Mostly duplicates", 0.9)),
         ):
        graph = compile_graph()
        session_id = str(uuid.uuid4())
        yield graph, session_id, fakes


async def _start(graph, session_id):
    state = _make_state(session_id=session_id, inputs=_make_session_inputs())
    return await graph.ainvoke(state, config=thread_config(session_id))


async def test_run_pauses_with_review_request(paused_run):
    graph, session_id, fakes = paused_run
    state = await _start(graph, session_id)

    review = pending_review(state)
    assert review is not None
    assert review["options"] == ["approve", "reject", "modify_threshold"]
    assert "9 duplicates" in review["summary"]
    assert fakes.story_writer_calls == 0


async def test_approve_continues_to_story_writer(paused_run):
    graph, session_id, fakes = paused_run
    await _start(graph, session_id)

    state = await resume_review(graph, session_id, "approve")

    assert pending_review(state) is None
    assert fakes.story_writer_calls == 1
    assert fakes.gap_thresholds == [None]  # gap detection was not re-run
    assert state["status"] == "completed"


async def test_reject_ends_the_run(paused_run):
    graph, session_id, fakes = paused_run
    await _start(graph, session_id)

    state = await resume_review(graph, session_id, "reject")

    assert pending_review(state) is None
    assert state["status"] == "permanent_failure"
    assert fakes.story_writer_calls == 0


async def test_modify_threshold_reruns_gap_detection_with_new_value(paused_run):
    graph, session_id, fakes = paused_run
    await _start(graph, session_id)

    await resume_review(
        graph, session_id, {"decision": "modify_threshold", "duplicate_threshold": 0.9}
    )

    assert fakes.gap_thresholds[:2] == [None, 0.9]


def test_pending_review_none_for_finished_state():
    assert pending_review({"status": "completed"}) is None
