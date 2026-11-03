"""Tests for GapDetectionAgent classification, including LLM verification of matches."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, ".")

from backlog_synthesizer_langgraph.agents.gap_detection import GapDetectionAgent
from backlog_synthesizer_langgraph.models.extraction import ExtractedItem
from backlog_synthesizer_langgraph.tools.errors import TransientToolError
from backlog_synthesizer_langgraph.tools.interfaces import SearchResult


def _item(text: str = "Use OAuth2 instead of JWT") -> ExtractedItem:
    return ExtractedItem(
        item_type="decision", text=text, source_chunk_index=0, confidence=0.9
    )


def _agent(score: float, llm_reply=None, llm_error: Exception | None = None):
    """Agent whose memory returns a single match at `score` and whose LLM replies as given."""
    memory = MagicMock()
    memory.search_similar.return_value = [
        SearchResult(
            item_id="T-1",
            score=score,
            metadata={"content": "JWT session management Use JWT tokens"},
        )
    ]
    llm = MagicMock()
    if llm_error is not None:
        llm.generate.side_effect = llm_error
    else:
        llm.generate.return_value = llm_reply
    return GapDetectionAgent(memory, llm_tool=llm if llm_reply or llm_error else None), llm


def _verdict(verdict: str, reason: str = "because") -> str:
    return f'{{"verdict": "{verdict}", "reason": "{reason}"}}'


async def _classify(agent: GapDetectionAgent):
    report = await agent.analyze_gaps([_item()])
    return report.entries[0]


class TestThresholdFallback:
    """Without an LLM tool, classification is by similarity thresholds alone."""

    @pytest.mark.parametrize(
        "score, expected",
        [(0.30, "new"), (0.60, "conflict"), (0.80, "duplicate"), (0.95, "duplicate")],
    )
    async def test_thresholds(self, score, expected):
        agent, _ = _agent(score)
        assert (await _classify(agent)).classification == expected


class TestLLMVerification:
    async def test_paraphrase_below_duplicate_threshold_becomes_duplicate(self):
        agent, llm = _agent(0.749, _verdict("duplicate"))
        entry = await _classify(agent)
        assert entry.classification == "duplicate"
        assert entry.duplicate_info.matching_ticket_id == "T-1"
        llm.generate.assert_called_once()

    async def test_contradiction_above_duplicate_threshold_becomes_conflict(self):
        agent, _ = _agent(0.810, _verdict("conflict", "Async 1M rows contradicts sync 5000 rows"))
        entry = await _classify(agent)
        assert entry.classification == "conflict"
        assert "contradicts" in entry.conflict_info.contradiction_description

    async def test_conflict_score_clamped_to_model_range(self):
        agent, _ = _agent(0.97, _verdict("conflict"))
        entry = await _classify(agent)
        assert entry.conflict_info.similarity_score == 0.85

    async def test_unrelated_becomes_new(self):
        agent, _ = _agent(0.62, _verdict("unrelated"))
        assert (await _classify(agent)).classification == "new"

    async def test_prompt_includes_ticket_text_and_item(self):
        agent, llm = _agent(0.70, _verdict("duplicate"))
        await _classify(agent)
        prompt = llm.generate.call_args.args[0]
        assert "JWT session management" in prompt
        assert "Use OAuth2 instead of JWT" in prompt

    async def test_items_below_conflict_threshold_skip_the_llm(self):
        agent, llm = _agent(0.30, _verdict("duplicate"))
        assert (await _classify(agent)).classification == "new"
        llm.generate.assert_not_called()

    async def test_reply_wrapped_in_prose_is_parsed(self):
        agent, _ = _agent(0.70, f"Sure! {_verdict('duplicate')} Hope that helps.")
        assert (await _classify(agent)).classification == "duplicate"


class TestLLMFailureFallsBackToThresholds:
    @pytest.mark.parametrize("reply", ["not json at all", '{"verdict": "maybe"}', "{bad json}"])
    async def test_unusable_reply(self, reply):
        agent, _ = _agent(0.60, reply)
        assert (await _classify(agent)).classification == "conflict"

    async def test_non_string_reply(self):
        agent, _ = _agent(0.85, MagicMock())
        assert (await _classify(agent)).classification == "duplicate"

    async def test_llm_raises(self):
        agent, _ = _agent(0.60, llm_error=TransientToolError("rate limited"))
        assert (await _classify(agent)).classification == "conflict"


class TestBacklogScoping:
    """Matches are restricted to tickets in the current backlog."""

    async def test_search_filtered_to_given_ticket_ids(self):
        agent, _ = _agent(0.30)
        agent._ticket_ids = ["T-1", "T-2"]
        await _classify(agent)
        where = agent._memory.search_similar.call_args.args[2]
        assert where == {"ticket_id": {"$in": ["T-1", "T-2"]}}

    async def test_empty_backlog_skips_search(self):
        agent, _ = _agent(0.95)
        agent._ticket_ids = []
        assert (await _classify(agent)).classification == "new"
        agent._memory.search_similar.assert_not_called()

    async def test_no_ticket_ids_searches_unfiltered(self):
        agent, _ = _agent(0.30)
        await _classify(agent)
        assert agent._memory.search_similar.call_args.args[2] is None
