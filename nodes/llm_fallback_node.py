"""LLM Fallback node — Level 2 of the three-tier decision pattern.

Called when conditional edges (Level 1) cannot determine the correct next action.
Uses an LLM to reason about the current pipeline state and choose from a
predefined set of actions.

This node:
1. Builds an observation summary from current state
2. Defines available actions based on current_step
3. Prompts the LLM to choose an action
4. Executes the chosen action (retry, return partial, or halt)
5. Logs everything to audit log and records OTel span

Never raises exceptions — always returns state gracefully.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

from graph.state import PipelineState
from backlog_synthesizer_langgraph.config import PipelineConfig
from backlog_synthesizer_langgraph.memory.audit_log import AuditLog
from backlog_synthesizer_langgraph.models.memory import AuditEntry
from backlog_synthesizer_langgraph.observability import get_tracer

logger = logging.getLogger(__name__)

# Module-level audit log instance (shared across invocations)
_audit_log = AuditLog()


def get_audit_log() -> AuditLog:
    """Return the module-level audit log for external access (e.g., testing)."""
    return _audit_log


# ─── Action Definitions ───────────────────────────────────────────

_STORY_WRITER_ACTIONS = {
    "retry_story_writer": "Call story_writer_node again with the same inputs",
    "return_partial": "Return current results with a warning — pipeline completes with partial output",
    "halt": "Stop the pipeline and report permanent failure",
}

_PARSER_ACTIONS = {
    "retry_parser": "Call parser_node again with the same documents",
    "request_better_transcript": "Halt pipeline — ask user to provide clearer transcript",
    "proceed_with_low_confidence": "Continue to gap_detection despite low confidence — add warning",
    "return_partial": "Return current results with a warning",
    "halt": "Stop the pipeline and report permanent failure",
}

_GAP_DETECTION_ACTIONS = {
    "lower_threshold": "Reduce duplicate threshold by 0.05 and retry gap detection — items may be genuinely new",
    "return_nothing_new": "Halt gracefully — all items already exist in backlog",
    "proceed_with_conflicts_only": "Continue to story_writer with only conflict items",
    "request_review": "Flag for human review — set needs_human_review=True",
    "return_partial": "Return current results with a warning",
    "halt": "Stop the pipeline and report permanent failure",
}


# ─── Main Node ────────────────────────────────────────────────────


async def llm_fallback_node(state: PipelineState) -> PipelineState:
    """LangGraph node implementing Level 2 LLM-based fallback decisions.

    When conditional routing cannot determine the next action, this node
    uses an LLM to reason about the current state and choose from a set
    of predefined actions appropriate to the current pipeline step.

    Args:
        state: The current PipelineState.

    Returns:
        Updated PipelineState reflecting the chosen action.
    """
    tracer = get_tracer()
    start_time = time.time()

    # Start OTel span
    span = None
    if tracer is not None:
        span = tracer.start_span("llm_fallback")
        span.set_attribute("session_id", state["session_id"])

    current_step = state.get("current_step", "unknown")
    retry_count = state.get("retry_count", 0)

    try:
        result = await _execute_fallback(state, current_step, retry_count, start_time, span)
        return result
    except Exception as e:
        # Safety net — LLM fallback must never crash the pipeline
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("Unexpected error in llm_fallback_node: %s", e)

        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.end()

        # Default to return_partial on any unexpected error
        return _build_return_partial(
            state,
            reason=f"LLM fallback node encountered unexpected error: {e}",
            duration_ms=duration_ms,
            confidence=0.0,
        )


async def _execute_fallback(
    state: PipelineState,
    current_step: str,
    retry_count: int,
    start_time: float,
    span: object | None,
) -> PipelineState:
    """Core fallback logic — separated for clarity.

    Args:
        state: Current pipeline state.
        current_step: The step that triggered the fallback.
        retry_count: How many retries have already been attempted.
        start_time: When the node started executing.
        span: Optional OTel span.

    Returns:
        Updated state dict.
    """
    # 1. Build observation from current state
    observation = _build_observation(state, current_step)

    # 2. Get available actions for current step
    available_actions = _get_available_actions(current_step)

    # 3. Call LLM to choose action
    action, reason, confidence = await _ask_llm_for_decision(
        observation, available_actions, current_step
    )

    # 4. Record span attributes
    if span is not None:
        span.set_attribute("action_chosen", action)
        span.set_attribute("confidence", confidence)
        span.set_attribute("current_step", current_step)
        span.set_attribute("retry_count", retry_count)
        span.set_attribute("fallback_triggered_by", current_step)

    duration_ms = int((time.time() - start_time) * 1000)
    is_retry = action in ("retry_story_writer", "retry_parser", "lower_threshold")

    # 5. Log to audit log
    _log_to_audit(
        session_id=state["session_id"],
        observation=observation,
        action=action,
        reason=reason,
        confidence=confidence,
        is_retry=is_retry,
        duration_ms=duration_ms,
    )

    # 6. Build result based on chosen action
    #    The node records the decision; graph edges handle routing to the next node.
    if action == "retry_story_writer":
        if retry_count >= 1:
            result = _build_terminal(
                state, action="return_partial",
                reason="Retry limit reached (max 1 retry). Returning partial results.",
                duration_ms=duration_ms, confidence=0.5,
                status="partial_failure",
            )
        else:
            result = _build_terminal(
                state, action=action, reason=reason,
                duration_ms=duration_ms, confidence=confidence,
                status=state.get("status", "in_progress"),
                increment_retry=True,
            )
    elif action == "retry_parser":
        if retry_count >= 1:
            result = _build_terminal(
                state, action="return_partial",
                reason="Retry limit reached (max 1 retry). Returning partial results.",
                duration_ms=duration_ms, confidence=0.5,
                status="partial_failure",
            )
        else:
            result = _build_terminal(
                state, action=action, reason=reason,
                duration_ms=duration_ms, confidence=confidence,
                status=state.get("status", "in_progress"),
                increment_retry=True,
            )
    elif action == "lower_threshold":
        if retry_count >= 1:
            result = _build_terminal(
                state, action="return_partial",
                reason="Retry limit reached (max 1 retry). Returning partial results.",
                duration_ms=duration_ms, confidence=0.5,
                status="partial_failure",
            )
        else:
            result = _build_terminal(
                state, action=action, reason=reason,
                duration_ms=duration_ms, confidence=confidence,
                status=state.get("status", "in_progress"),
                increment_retry=True,
            )
    elif action == "proceed_with_low_confidence":
        result = _build_terminal(
            state, action=action, reason=reason,
            duration_ms=duration_ms, confidence=confidence,
            status=state.get("status", "in_progress"),
        )
    elif action == "proceed_with_conflicts_only":
        result = _build_terminal(
            state, action=action,
            reason="Proceeding with conflict items only — duplicates filtered",
            duration_ms=duration_ms, confidence=confidence,
            status=state.get("status", "in_progress"),
        )
    elif action == "request_review":
        result = _build_terminal(
            state, action=action, reason=reason,
            duration_ms=duration_ms, confidence=confidence,
            status="partial_failure",
            needs_human_review=True,
        )
    elif action == "request_better_transcript":
        result = _build_terminal(
            state, action=action,
            reason="User should provide a clearer transcript for better extraction",
            duration_ms=duration_ms, confidence=confidence,
            status="permanent_failure",
        )
    elif action == "return_nothing_new":
        result = _build_terminal(
            state, action=action,
            reason="All items already exist in backlog — nothing new to process",
            duration_ms=duration_ms, confidence=confidence,
            status="permanent_failure",
        )
    elif action == "halt":
        result = _build_terminal(
            state, action=action, reason=reason,
            duration_ms=duration_ms, confidence=confidence,
            status="permanent_failure",
        )
    elif action == "return_partial":
        result = _build_terminal(
            state, action=action, reason=reason,
            duration_ms=duration_ms, confidence=confidence,
            status="partial_failure",
        )
    else:
        # Unknown action — default to return_partial
        result = _build_terminal(
            state, action="return_partial",
            reason=f"Unknown action '{action}' — defaulting to return_partial",
            duration_ms=duration_ms, confidence=0.0,
            status="partial_failure",
        )

    if span is not None:
        span.end()

    return result


# ─── Observation Builder ──────────────────────────────────────────


def _build_observation(state: PipelineState, current_step: str) -> str:
    """Build a concise observation string from current pipeline state.

    Args:
        state: The current PipelineState.
        current_step: Name of the current step.

    Returns:
        A multi-line summary of the pipeline state for the LLM.
    """
    lines = [f"current_step: {current_step}"]

    # Story and epic counts
    story_output = state.get("story_output")
    if story_output is not None:
        epic_count = len(story_output.epics)
        story_count = sum(len(e.stories) for e in story_output.epics)
        needs_refinement_count = sum(
            1 for e in story_output.epics for s in e.stories if s.needs_refinement
        )
        pct = (needs_refinement_count / story_count * 100) if story_count > 0 else 0
        lines.append(f"stories: {story_count}, epics: {epic_count}")
        lines.append(f"needs_refinement: {needs_refinement_count} ({pct:.0f}%)")
    else:
        lines.append("story_output: None")

    # Error summary
    errors = state.get("errors", [])
    if errors:
        lines.append(f"errors: {len(errors)} total")
        # Show last 2 error messages for context
        for err in errors[-2:]:
            if isinstance(err, dict):
                lines.append(f"  - [{err.get('node', '?')}] {err.get('message', 'unknown')}")

    # Gap report summary
    gap_output = state.get("gap_output")
    if gap_output is not None:
        lines.append(
            f"gap_report: new={gap_output.total_new}, "
            f"duplicates={gap_output.total_duplicates}, "
            f"conflicts={gap_output.total_conflicts}"
        )

    # Parser output summary
    parser_output = state.get("parser_output")
    if parser_output is not None:
        items = parser_output.items if hasattr(parser_output, "items") else []
        item_count = len(items)
        if item_count > 0 and hasattr(items[0], "confidence"):
            avg_conf = sum(i.confidence for i in items) / item_count
            lines.append(f"parser_output: {item_count} items, avg_confidence={avg_conf:.2f}")
        else:
            lines.append(f"parser_output: {item_count} items")
    else:
        lines.append("parser_output: None")

    # Retry count
    retry_count = state.get("retry_count", 0)
    lines.append(f"retry_count: {retry_count}")

    return "\n".join(lines)


# ─── Action Resolution ────────────────────────────────────────────


def _get_available_actions(current_step: str) -> dict[str, str]:
    """Get the set of available actions for the given step.

    Args:
        current_step: The pipeline step that triggered fallback.

    Returns:
        Dict mapping action_name → description.
    """
    if current_step == "story_writer":
        return _STORY_WRITER_ACTIONS

    if current_step == "parser":
        return _PARSER_ACTIONS

    if current_step == "gap_detection":
        return _GAP_DETECTION_ACTIONS

    # Default fallback actions for unknown steps:
    return {
        "return_partial": "Return current results with a warning",
        "halt": "Stop the pipeline and report permanent failure",
    }


# ─── LLM Decision ────────────────────────────────────────────────


async def _ask_llm_for_decision(
    observation: str,
    available_actions: dict[str, str],
    current_step: str,
) -> tuple[str, str, float]:
    """Ask the LLM to choose an action given the observation.

    Args:
        observation: Pipeline state summary.
        available_actions: Dict of action_name → description.
        current_step: Current pipeline step.

    Returns:
        Tuple of (action_name, reason, confidence).
        On any failure, defaults to ("return_partial", <reason>, 0.0).
    """
    actions_text = "\n".join(
        f"- {name}: {desc}" for name, desc in available_actions.items()
    )

    prompt = (
        f"Given this pipeline state:\n{observation}\n\n"
        f"The conditional routing could not determine the correct next action.\n\n"
        f"Available actions:\n{actions_text}\n\n"
        f"Choose the most appropriate action.\n"
        f"Return only JSON:\n"
        f'{{"action": "action_name", "reason": "one sentence explanation", "confidence": 0.0}}'
    )

    system_prompt = (
        "You are a pipeline decision engine. Analyze the state and choose the best action. "
        "Be concise. Return only valid JSON with action, reason, and confidence (0.0-1.0)."
    )

    try:
        config = PipelineConfig.from_env()
        llm_tool = config.create_llm_tool()

        # LLMGenerationTool.generate() is sync — run in thread
        response = await asyncio.to_thread(
            llm_tool.generate, prompt, system_prompt
        )

        # Parse response
        action, reason, confidence = _parse_llm_response(response, available_actions)
        return action, reason, confidence

    except Exception as e:
        logger.warning("LLM call failed in llm_fallback_node: %s", e)
        # Log warning to audit log — handled by caller
        return "return_partial", f"LLM call failed: {e}", 0.0


def _parse_llm_response(
    response: str,
    available_actions: dict[str, str],
) -> tuple[str, str, float]:
    """Parse the LLM JSON response and validate the action.

    Args:
        response: Raw LLM text response.
        available_actions: Valid action names.

    Returns:
        Tuple of (action, reason, confidence).
        Defaults to ("return_partial", ..., 0.0) on parse failure.
    """
    # Strip markdown code fences if present
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse LLM fallback response: %s", e)
        return "return_partial", f"Could not parse LLM response: {e}", 0.0

    action = data.get("action", "return_partial")
    reason = data.get("reason", "No reason provided")
    confidence = data.get("confidence", 0.0)

    # Validate action is in available set
    if action not in available_actions:
        logger.warning(
            "LLM chose invalid action '%s', defaulting to return_partial", action
        )
        return "return_partial", f"Invalid action '{action}' — defaulting to return_partial", 0.0

    # Clamp confidence to [0.0, 1.0]
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    return action, reason, confidence


# ─── Result Builder ───────────────────────────────────────────────


def _build_terminal(
    state: PipelineState,
    action: str,
    reason: str,
    duration_ms: int,
    confidence: float,
    status: str,
    increment_retry: bool = False,
    needs_human_review: bool = False,
) -> PipelineState:
    """Build the unified result dict for any LLM fallback action.

    Records the action in react_notes with an 'action' field so that
    route_after_llm_fallback can read it and route accordingly.

    Args:
        state: Current pipeline state.
        action: The action name chosen by the LLM.
        reason: One-sentence explanation from the LLM.
        duration_ms: Time spent in fallback node.
        confidence: LLM confidence in the decision.
        status: Pipeline status to set.
        increment_retry: Whether to increment retry_count.
        needs_human_review: Whether to set the human review flag.

    Returns:
        State update dict with action recorded in react_notes.
    """
    retry_count = state.get("retry_count", 0)
    if increment_retry:
        retry_count += 1

    result: dict = {
        "current_step": "llm_fallback",
        "status": status,
        "retry_count": retry_count,
        "errors": [],
        "react_notes": [
            {
                "node": "llm_fallback",
                "action": action,
                "note": f"{action}: {reason} (confidence: {confidence:.2f})",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "total_duration_ms": state.get("total_duration_ms", 0) + duration_ms,
    }

    if needs_human_review:
        result["needs_human_review"] = True

    return result


# ─── Audit Logging ────────────────────────────────────────────────


def _log_to_audit(
    session_id: str,
    observation: str,
    action: str,
    reason: str,
    confidence: float,
    is_retry: bool,
    duration_ms: int,
) -> None:
    """Record the fallback decision to the audit log.

    Args:
        session_id: Pipeline session ID.
        observation: State observation summary.
        action: The action that was chosen.
        reason: The LLM's reasoning.
        confidence: Confidence score.
        is_retry: Whether this is a retry attempt.
        duration_ms: Time spent making the decision.
    """
    input_summary = f"observation={observation[:200]}"
    output_summary = (
        f"action={action}, reason={reason[:150]}, "
        f"confidence={confidence:.2f}, is_retry={is_retry}"
    )

    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        agent_name="llm_fallback",
        input_summary=input_summary,
        output_summary=output_summary,
        duration_ms=duration_ms,
    )

    try:
        _audit_log.record(session_id, entry)
    except Exception as e:
        # Audit logging failure must never crash the node
        logger.warning("Failed to record audit log entry: %s", e)
