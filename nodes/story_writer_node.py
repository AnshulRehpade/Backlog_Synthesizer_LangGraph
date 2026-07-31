"""Story Writer node — wraps StoryWriterAgent as a LangGraph node function.

This node:
1. Reads gap_output from state
2. Filters to NEW and CONFLICT items only
3. Calls StoryWriterAgent.generate_stories() → group_into_epics() → serialize_output()
4. Writes story_output back to state
5. Records OTel span and audit log entry

Never raises exceptions — always returns state gracefully.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from graph.state import PipelineState
from backlog_synthesizer_langgraph.agents.story_writer import StoryWriterAgent
from backlog_synthesizer_langgraph.config import PipelineConfig
from backlog_synthesizer_langgraph.observability import get_tracer
from backlog_synthesizer_langgraph.tools.errors import (
    PermanentToolError,
    TransientToolError,
)

logger = logging.getLogger(__name__)

_FEW_SHOT_EXAMPLES_PATH = Path("data/few_shot_examples.json")


def _get_few_shot_store(config: PipelineConfig):
    """Create a FewShotStore if few-shot is enabled and examples exist.

    Returns None if disabled or if loading fails (non-fatal).
    """
    if not config.few_shot_enabled:
        return None

    if not _FEW_SHOT_EXAMPLES_PATH.exists():
        return None

    try:
        from backlog_synthesizer_langgraph.tools.few_shot_store import FewShotStore

        embedding_tool = config.create_embedding_tool()
        vector_search_tool = config.create_vector_search_tool()

        store = FewShotStore(
            embedding_tool=embedding_tool,
            vector_search_tool=vector_search_tool,
            collection_prefix="few_shot_",
        )

        data = json.loads(_FEW_SHOT_EXAMPLES_PATH.read_text(encoding="utf-8"))
        loaded = store.load_from_json(data)
        if loaded > 0:
            logger.info("Loaded %d few-shot examples for story generation", loaded)
            return store
    except Exception as e:
        logger.warning("Failed to initialize few-shot store: %s", e)

    return None


async def story_writer_node(state: PipelineState) -> PipelineState:
    """LangGraph node that generates user stories from gap detection output.

    Reads gap_output from state, filters to eligible items (NEW + CONFLICT),
    invokes the StoryWriterAgent pipeline (generate → group → serialize),
    and writes the result back to state.

    Error handling:
        - TransientToolError → appends error, sets status to "partial_failure"
        - PermanentToolError → appends error, sets status to "permanent_failure"
        - Any other exception → appends error, sets status to "partial_failure"
        - Never raises out of the node — always returns state.

    Args:
        state: The current PipelineState.

    Returns:
        Updated PipelineState with story_output populated (or errors appended).
    """
    tracer = get_tracer()
    start_time = time.time()

    # Start OTel span if available
    span = None
    if tracer is not None:
        span = tracer.start_span("story_writer_node")
        span.set_attribute("session_id", state["session_id"])

    try:
        return await _execute_story_writer(state, start_time, span)
    except Exception as e:
        # Catch-all safety net — should never reach here due to inner handling,
        # but guarantees we never raise out of the node.
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("Unexpected error in story_writer_node: %s", e)

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.end()

        return {
            "current_step": "story_writer",
            "status": "partial_failure",
            "errors": [
                {
                    "node": "story_writer",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }


async def _execute_story_writer(
    state: PipelineState, start_time: float, span: object | None
) -> PipelineState:
    """Core logic for the story writer node, separated for clarity.

    Args:
        state: The current pipeline state.
        start_time: Timestamp when the node started.
        span: Optional OTel span for tracing.

    Returns:
        Updated state dict with story_output or error information.
    """
    # 1. Check gap_output exists
    gap_output = state.get("gap_output")
    if gap_output is None:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "story_writer",
            "error_type": "MissingInput",
            "message": "gap_output is None — cannot generate stories without gap detection results",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "story_writer",
            "status": "partial_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    # 2. Create StoryWriterAgent with config
    try:
        config = PipelineConfig.from_env()
        llm_tool = config.create_llm_tool()
    except PermanentToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "story_writer",
            "error_type": "PermanentToolError",
            "message": f"Failed to create LLM tool: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "story_writer",
            "status": "permanent_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    agent = StoryWriterAgent(llm_tool=llm_tool, few_shot_store=_get_few_shot_store(config))

    # 3. Generate stories from gap report entries
    try:
        stories = await agent.generate_stories(gap_output.entries)
    except TransientToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "story_writer",
            "error_type": "TransientToolError",
            "message": f"Transient error during story generation: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.warning(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "story_writer",
            "status": "partial_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }
    except PermanentToolError as e:
        duration_ms = int((time.time() - start_time) * 1000)
        error_info = {
            "node": "story_writer",
            "error_type": "PermanentToolError",
            "message": f"Permanent error during story generation: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(error_info["message"])

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", error_info["message"])
            span.end()

        return {
            "current_step": "story_writer",
            "status": "permanent_failure",
            "errors": [error_info],
            "react_notes": [],
            "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
        }

    # 4. Group stories into epics
    epics = agent.group_into_epics(stories)

    # 5. Serialize output
    session_id = state["session_id"]
    serialization_result = agent.serialize_output(epics, session_id)

    duration_ms = int((time.time() - start_time) * 1000)

    # 6. Record span attributes
    if span is not None:
        span.set_attribute("stories_generated", len(stories))
        span.set_attribute("epics_generated", len(epics))
        span.set_attribute("duration_ms", duration_ms)
        if serialization_result.output is not None:
            needs_refinement_count = sum(
                1 for epic in serialization_result.output.epics
                for story in epic.stories
                if story.needs_refinement
            )
            span.set_attribute("stories_needing_refinement", needs_refinement_count)
        span.end()

    # 7. Handle serialization errors
    errors_to_append = []
    if serialization_result.errors:
        for ser_err in serialization_result.errors:
            errors_to_append.append({
                "node": "story_writer",
                "error_type": "SerializationError",
                "message": f"{ser_err.item_title}: {ser_err.failing_field} — {ser_err.description}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # 8. Determine final status for this node
    if serialization_result.output is None:
        status = "partial_failure"
        errors_to_append.append({
            "node": "story_writer",
            "error_type": "SerializationFailed",
            "message": "Story serialization produced no output",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    else:
        status = "completed"

    # 9. Build react note for observability
    react_notes = []
    if stories:
        needs_refinement_count = sum(1 for s in stories if s.needs_refinement)
        react_notes.append({
            "node": "story_writer",
            "note": (
                f"Generated {len(stories)} stories, grouped into {len(epics)} epics. "
                f"{needs_refinement_count} stories need refinement."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    logger.info(
        "Story writer completed: %d stories, %d epics, %dms",
        len(stories),
        len(epics),
        duration_ms,
    )

    return {
        "story_output": serialization_result.output,
        "current_step": "story_writer",
        "status": status,
        "errors": errors_to_append,
        "react_notes": react_notes,
        "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
    }
