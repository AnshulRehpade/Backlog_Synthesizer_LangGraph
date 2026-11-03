"""Conflict Review node — a person settles every conflict before stories are written.

A conflict means the meeting contradicts an existing ticket (e.g. "async, 1M rows"
vs. a ticket that says "sync, 5,000 rows"). Which one wins is a product decision,
so the graph pauses here once per run with all unresolved conflicts, each with
the judge's suggested resolution. The reviewer can accept every suggestion at
once or override individual ones.

The node sits in front of story_writer on every path. With no unresolved
conflicts, or with CONFLICT_REVIEW=false, it passes straight through.

Resume with Command(resume=...) using one of:
    "accept_all"                              — take every suggestion
    {"resolutions": {"<entry index>": "keep_ticket", ...}}
                                              — overrides; others take the suggestion
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langgraph.types import interrupt

from graph.state import PipelineState
from backlog_synthesizer_langgraph.config import PipelineConfig
from backlog_synthesizer_langgraph.models.gap_detection import RESOLUTIONS, GapReport

logger = logging.getLogger(__name__)


async def conflict_review_node(state: PipelineState) -> PipelineState:
    """Pause for a person to settle conflicts, then record their decisions."""
    gap_output = state.get("gap_output")
    pending = _pending_conflicts(gap_output)
    if not pending or not PipelineConfig.from_env().conflict_review:
        return {}

    tickets = {t.id: t for t in state["inputs"].backlog_tickets} if state.get("inputs") else {}
    request = {
        "kind": "conflict_review",
        "message": f"{len(pending)} conflict{'s' if len(pending) != 1 else ''} need a decision",
        # Grouped by ticket so every decision about one ticket is made together.
        "conflicts": sorted(
            (_describe(i, gap_output.entries[i], tickets) for i in pending),
            key=lambda c: (c["ticket_id"], c["index"]),
        ),
        "options": dict(RESOLUTIONS),
    }

    # interrupt() pauses the graph here; on resume the node re-runs from the top
    # and interrupt() returns the reviewer's answer. Everything above is
    # deterministic, so re-running it is safe.
    answer = interrupt(request)

    chosen = _resolve(answer, pending, gap_output)
    entries = list(gap_output.entries)
    for index, resolution in chosen.items():
        entry = entries[index]
        entries[index] = entry.model_copy(update={
            "conflict_info": entry.conflict_info.model_copy(update={"resolution": resolution}),
        })

    summary = ", ".join(f"[{i}] {r}" for i, r in chosen.items())
    logger.info("Conflict review: %s", summary)
    return {
        "current_step": "conflict_review",
        "gap_output": gap_output.model_copy(update={"entries": entries}),
        "react_notes": [{
            "node": "conflict_review",
            "action": "conflicts_resolved",
            "note": f"Reviewer settled {len(chosen)} conflicts: {summary}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }],
    }


def _pending_conflicts(gap_output: GapReport | None) -> list[int]:
    if gap_output is None:
        return []
    return [
        i for i, e in enumerate(gap_output.entries)
        if e.classification == "conflict" and e.conflict_info is not None
        and e.conflict_info.resolution is None
    ]


def _describe(index: int, entry, tickets: dict) -> dict:
    info = entry.conflict_info
    ticket = tickets.get(info.item_b_ticket_id)
    return {
        "index": index,
        "item": entry.item.text,
        "ticket_id": info.item_b_ticket_id,
        "ticket_title": ticket.title if ticket else "",
        "ticket_description": ticket.description if ticket else "",
        "ticket_status": ticket.status if ticket else "",
        "reason": info.contradiction_description,
        "suggestion": info.suggested_resolution,
    }


def _resolve(answer, pending: list[int], gap_output: GapReport) -> dict[int, str]:
    """Each pending conflict's resolution: the reviewer's override, else the suggestion."""
    overrides: dict = {}
    if isinstance(answer, dict):
        overrides = answer.get("resolutions") or {}
    chosen = {}
    for index in pending:
        override = overrides.get(str(index), overrides.get(index))
        suggestion = gap_output.entries[index].conflict_info.suggested_resolution
        chosen[index] = override if override in RESOLUTIONS else suggestion
    return chosen
