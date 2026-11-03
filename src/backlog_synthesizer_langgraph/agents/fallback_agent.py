"""Bounded tool-calling agent for the LLM fallback node.

Instead of one LLM call that picks a recovery action from a summary, the agent
investigates first: Claude calls tools from FallbackToolbox (list matches, read
tickets, ask the judge, correct labels), reads the results, and ends by calling
`finish` with an action from the same per-step menu the old fallback used.

The loop is bounded so it can't run away:
    - at most MAX_TURNS model calls
    - at most TOKEN_BUDGET input + output tokens across those calls
    - one nudge if the model stops without calling `finish`
Any other way out (API error, refusal, truncated reply) ends the loop with no
decision. The caller then falls back to the single menu call, so the worst case
is the pre-agent behavior.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from backlog_synthesizer_langgraph.agents.fallback_tools import Decision, FallbackToolbox
from backlog_synthesizer_langgraph.models.gap_detection import GapReport

logger = logging.getLogger(__name__)

MAX_TURNS = 6
TOKEN_BUDGET = 30_000
MAX_TOKENS_PER_TURN = 2_048
_RESULT_PREVIEW = 200

SYSTEM_PROMPT = (
    "You are the recovery agent for a pipeline that turns meeting transcripts into "
    "backlog stories. Normal runs never reach you; you are called when a step "
    "produced an unexpected result, such as almost everything matching existing "
    "tickets, very few items extracted, or no stories written.\n"
    "Investigate with the tools before deciding. Similarity scores measure topic, "
    "not agreement, so a high score can still be a conflict or unrelated work. "
    "Look at the matches you doubt, read the tickets, and use compare when the "
    "text alone doesn't settle it.\n"
    "Reclassify an item only when the evidence shows its label is wrong, and say why. "
    "You have a few turns, so batch independent tool calls in one turn.\n"
    "End by calling finish exactly once. If you can't tell what is right and "
    "request_review is offered, choose it so a person decides."
)


@dataclass
class AgentStep:
    """One tool call the agent made, for the audit trail."""

    tool: str
    input: dict
    result: str
    is_error: bool


@dataclass
class AgentOutcome:
    """What the agent decided and how it got there.

    stopped_because is one of: finished, turn_limit, budget, no_finish,
    api_error, max_tokens, refusal.
    """

    decision: Decision | None
    updated_gap_output: GapReport | None
    steps: list[AgentStep] = field(default_factory=list)
    turns: int = 0
    tokens_used: int = 0
    stopped_because: str = ""


def supports_tool_calling(llm_tool: Any) -> bool:
    """Whether the LLM tool can send tool definitions (only the Anthropic tool can)."""
    return callable(getattr(llm_tool, "create_message", None))


async def run_fallback_agent(
    toolbox: FallbackToolbox,
    llm_tool: Any,
    observation: str,
    current_step: str,
) -> AgentOutcome:
    """Run the investigation loop until the agent calls finish or a limit is hit.

    Args:
        toolbox: Tools over the pipeline state, with this step's action menu.
        llm_tool: An LLM tool with create_message (see supports_tool_calling).
        observation: Summary of the pipeline state, as the menu fallback sees it.
        current_step: The step that triggered the fallback.
    """
    messages: list[dict] = [{
        "role": "user",
        "content": (
            f"The pipeline reached an unexpected state after the '{current_step}' step.\n\n"
            f"{observation}\n\n"
            "Investigate with the tools, then call finish."
        ),
    }]
    steps: list[AgentStep] = []
    tokens_used = 0
    nudged = False

    def outcome(turns: int, stopped_because: str) -> AgentOutcome:
        logger.info(
            "Fallback agent stopped (%s) after %d turns, %d tool calls, %d tokens",
            stopped_because, turns, len(steps), tokens_used,
        )
        return AgentOutcome(
            decision=toolbox.decision,
            updated_gap_output=toolbox.updated_gap_output(),
            steps=steps,
            turns=turns,
            tokens_used=tokens_used,
            stopped_because=stopped_because,
        )

    for turn in range(1, MAX_TURNS + 1):
        if tokens_used >= TOKEN_BUDGET:
            return outcome(turn - 1, "budget")

        try:
            response = await asyncio.to_thread(
                llm_tool.create_message,
                list(messages),  # snapshot: the loop keeps appending to messages
                SYSTEM_PROMPT,
                toolbox.definitions(),
                MAX_TOKENS_PER_TURN,
            )
        except Exception as e:
            logger.warning("Fallback agent API call failed: %s", e)
            return outcome(turn - 1, "api_error")

        usage = getattr(response, "usage", None)
        tokens_used += getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)

        if response.stop_reason in ("max_tokens", "refusal"):
            return outcome(turn, response.stop_reason)

        if response.content:
            messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            # The model answered in text instead of calling finish. Ask once.
            if nudged:
                return outcome(turn, "no_finish")
            nudged = True
            messages.append({
                "role": "user",
                "content": "Call the finish tool now with your decision.",
            })
            continue

        # Run every requested tool and return all results in one message.
        results = []
        for block in tool_uses:
            result = await toolbox.run(block.name, block.input or {})
            steps.append(AgentStep(block.name, dict(block.input or {}), _preview(result.content), result.is_error))
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result.content,
                "is_error": result.is_error,
            })

        if toolbox.decision is not None:
            return outcome(turn, "finished")
        messages.append({"role": "user", "content": results})

    return outcome(MAX_TURNS, "turn_limit")


def _preview(text: str) -> str:
    return text if len(text) <= _RESULT_PREVIEW else text[: _RESULT_PREVIEW - 1] + "…"
