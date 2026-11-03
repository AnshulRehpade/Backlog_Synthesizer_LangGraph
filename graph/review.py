"""Helpers for pausing at human review and resuming the graph.

human_review_node calls interrupt(), which stops the run and saves state in
the checkpointer. Callers (CLI, UI, eval) use these helpers to see whether a
run is waiting on a person and to resume it with their decision.

The same compiled graph (and so the same checkpointer) must be used to resume,
because the default MemorySaver only lives in memory.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

REVIEW_OPTIONS = ("approve", "reject", "modify_threshold")


def thread_config(session_id: str) -> dict:
    """LangGraph config that identifies a run's checkpoint thread."""
    return {"configurable": {"thread_id": session_id}}


def pending_review(state: dict) -> dict | None:
    """Return the review request a run is paused on, or None if it isn't paused.

    The payload is what human_review_node passed to interrupt(): message,
    current_step, summary and options.
    """
    interrupts = state.get("__interrupt__") or ()
    if not interrupts:
        return None
    return getattr(interrupts[0], "value", None) or {}


async def current_state(graph: Any, session_id: str) -> dict:
    """Full state of a run from its latest checkpoint, including any pending review.

    astream() yields per-node updates rather than the whole state, so streaming
    callers use this to get the final result.
    """
    snapshot = await graph.aget_state(thread_config(session_id))
    state = dict(snapshot.values)
    if snapshot.interrupts:
        state["__interrupt__"] = snapshot.interrupts
    return state


async def resume_review(graph: Any, session_id: str, decision: str | dict) -> dict:
    """Resume a paused run with the reviewer's decision and return the new state.

    Args:
        graph: The compiled graph the run was started with.
        session_id: The run's session ID (its checkpoint thread).
        decision: "approve", "reject", "modify_threshold", or a dict such as
            {"decision": "modify_threshold", "duplicate_threshold": 0.85}.
    """
    return await graph.ainvoke(Command(resume=decision), config=thread_config(session_id))
