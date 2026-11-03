"""Group items from one meeting that describe the same piece of work.

The parser extracts a problem and its fix as separate items ("CSV export times
out" and "move CSV export to an async queue"), and people repeat requests.
Gap detection compares each item with the backlog, never with the other items,
so without this step each would become its own story.

One LLM call sees every story-eligible item with the ticket it matched and
returns groups; each group becomes one story. Any unusable answer falls back to
one group per item, which is the behavior without this step.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from backlog_synthesizer_langgraph.models.gap_detection import GapReport
from backlog_synthesizer_langgraph.tools.interfaces import LLMGenerationTool

logger = logging.getLogger(__name__)

# Items that become stories; duplicates and unprocessed items don't.
_STORY_CLASSIFICATIONS = ("new", "related", "conflict")

_SYSTEM_PROMPT = (
    "You turn requirements from one meeting into backlog stories. Group the items "
    "that are the same piece of work, so each group becomes one story: a problem "
    "and the change that fixes it, the same request said twice, or a request and a "
    "detail of how to build it. Keep separate pieces of work apart even when they "
    "touch the same existing ticket (for example new limits and a dashboard showing "
    "them are two stories). Answer with JSON only: {\"groups\": [[<index>, ...], ...]}, "
    "using every index exactly once; an item with nothing to merge is a group of one."
)


async def assign_groups(report: GapReport, llm_tool: LLMGenerationTool | None) -> GapReport:
    """Return the report with each story-eligible entry's `group` set.

    Entries in the same group describe the same piece of work. Duplicates and
    unprocessed entries keep group None.
    """
    eligible = [i for i, e in enumerate(report.entries) if e.classification in _STORY_CLASSIFICATIONS]
    groups = [[i] for i in eligible]
    if llm_tool is not None and len(eligible) > 1:
        asked = await _ask_for_groups(report, eligible, llm_tool)
        if asked is not None:
            groups = asked

    entries = list(report.entries)
    for number, members in enumerate(groups):
        for i in members:
            entries[i] = entries[i].model_copy(update={"group": number})
    merged = sum(1 for g in groups if len(g) > 1)
    if merged:
        logger.info("Grouped %d items into %d pieces of work", len(eligible), len(groups))
    return report.model_copy(update={"entries": entries})


async def _ask_for_groups(
    report: GapReport, eligible: list[int], llm_tool: LLMGenerationTool
) -> list[list[int]] | None:
    lines = []
    for i in eligible:
        e = report.entries[i]
        ticket = f"{e.classification} vs {e.similar_ticket_id}" if e.similar_ticket_id else e.classification
        lines.append(f"[{i}] ({ticket}) {e.item.text}")
    prompt = "Items:\n" + "\n".join(lines)
    try:
        raw = await asyncio.to_thread(llm_tool.generate, prompt, _SYSTEM_PROMPT)
    except Exception as e:
        logger.warning("Work grouping failed, writing one story per item: %s", e)
        return None
    groups = _parse_groups(raw, eligible)
    if groups is None:
        logger.warning("Work grouping answer unusable, writing one story per item")
    return groups


def _parse_groups(raw: Any, eligible: list[int]) -> list[list[int]] | None:
    """Groups from the reply, or None unless they cover every eligible index exactly once."""
    if not isinstance(raw, str):
        return None
    found = re.search(r"\{.*\}", raw, re.DOTALL)
    if not found:
        return None
    try:
        groups = json.loads(found.group(0)).get("groups")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(groups, list) or not all(isinstance(g, list) and g for g in groups):
        return None
    flat = [i for g in groups for i in g]
    if sorted(flat) != sorted(eligible):
        return None
    return [sorted(g) for g in groups]
