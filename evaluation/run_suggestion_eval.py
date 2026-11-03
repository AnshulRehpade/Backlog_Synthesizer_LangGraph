"""Score the judge's suggested resolutions against labeled real conflicts.

Usage:
    python evaluation/run_suggestion_eval.py

Each case in evaluation/conflict_suggestions.json is a meeting item, the ticket
it was flagged against, and the acceptable resolutions. The real judge (gap
detection's LLM check, temperature 0) is asked about each pair.

Scoring is by outcome: if the judge changes the verdict instead of suggesting a
resolution, "duplicate" counts as keep_ticket (no story) and "unrelated" as
keep_both (a story is written alongside the ticket).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from backlog_synthesizer_langgraph.agents.gap_detection import GapDetectionAgent
from backlog_synthesizer_langgraph.config import PipelineConfig
from backlog_synthesizer_langgraph.models.extraction import ExtractedItem
from backlog_synthesizer_langgraph.tools.interfaces import SearchResult

DATASET = Path("evaluation/conflict_suggestions.json")
_OUTCOME_OF_VERDICT = {"duplicate": "keep_ticket", "new": "keep_both"}


async def _judge(agent_llm, case: dict) -> tuple[str, str]:
    """(outcome, reason) for one case."""
    t = case["ticket"]
    memory = MagicMock()
    # Same metadata gap_detection_node stores for a backlog ticket.
    memory.search_similar.return_value = [SearchResult(t["id"], 0.7, {
        "content": f"{t['title']} {t['description']}", "title": t["title"], "status": t["status"],
    })]
    item = ExtractedItem(item_type="feature_request", text=case["item"], source_chunk_index=0, confidence=0.9)
    entry = (await GapDetectionAgent(memory, llm_tool=agent_llm).analyze_gaps([item])).entries[0]
    if entry.classification == "conflict":
        return entry.conflict_info.suggested_resolution, entry.conflict_info.contradiction_description
    return _OUTCOME_OF_VERDICT.get(entry.classification, entry.classification), f"verdict: {entry.classification}"


async def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    llm = PipelineConfig.from_env().create_llm_tool(temperature=0.0)
    correct = 0
    for case in cases:
        outcome, reason = await _judge(llm, case)
        ok = outcome in case["acceptable"]
        correct += ok
        print(f"{'✅' if ok else '❌'} {case['id']}: got {outcome:<12} want {'/'.join(case['acceptable'])}")
        print(f"     {reason[:150]}")
    print(f"\n{correct}/{len(cases)} acceptable")


if __name__ == "__main__":
    asyncio.run(main())
