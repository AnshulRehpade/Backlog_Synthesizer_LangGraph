"""Tools the fallback agent can call while investigating an unexpected pipeline state.

The fallback agent (agents/fallback_agent.py) replaces a single pick-from-a-menu
LLM call with a short loop: look, call a tool, read the result, decide again.
These are the tools it can call.

FallbackToolbox wraps a snapshot of the pipeline state. Read-only tools describe
it; `reclassify` records corrections to the gap report without touching the
real state; `finish` records the final decision and ends the loop. The node
applies the corrections and the decision afterwards, so everything the agent
changed is visible in one place.

Tool definitions use the Claude Messages API tool format (name, description,
input_schema). Each tool returns plain text, because that is what the model reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backlog_synthesizer_langgraph.agents.gap_detection import build_entry, verify_match
from backlog_synthesizer_langgraph.models.extraction import ExtractedItem
from backlog_synthesizer_langgraph.models.gap_detection import GapReport, GapReportEntry
from backlog_synthesizer_langgraph.tools.interfaces import LLMGenerationTool

# Caps that keep one fallback run cheap and its tool output small.
MAX_COMPARES = 5
MAX_LISTED = 40
_TEXT_PREVIEW = 160

_CLASSIFICATIONS = ("new", "duplicate", "conflict")


@dataclass
class ToolResult:
    """Text sent back to the model, flagged when the call failed."""

    content: str
    is_error: bool = False


@dataclass
class Reclassification:
    index: int
    classification: str
    ticket_id: str | None
    reason: str


@dataclass
class Decision:
    action: str
    reason: str
    confidence: float


@dataclass
class FallbackToolbox:
    """Tools over one snapshot of pipeline state.

    Args:
        state: The PipelineState the fallback node received.
        actions: Valid final actions for this step (name → description).
        llm_tool: LLM used by `compare`; without it, `compare` is not offered.
    """

    state: dict
    actions: dict[str, str]
    llm_tool: LLMGenerationTool | None = None
    reclassifications: dict[int, Reclassification] = field(default_factory=dict)
    decision: Decision | None = None
    compares_used: int = 0

    # ─── Tool definitions ──────────────────────────────────────

    def definitions(self) -> list[dict]:
        """Tool definitions for the tools that make sense in this state."""
        defs: list[dict] = []
        if self._parser_items():
            defs.append({
                "name": "list_items",
                "description": (
                    "List the requirements the parser extracted, with their index, type "
                    "and confidence (0-1). Use it to judge whether extraction went wrong."
                ),
                "input_schema": {"type": "object", "properties": {}},
            })
        if self._gap_entries():
            defs.append({
                "name": "list_matches",
                "description": (
                    "List each gap-detection result: index, label (new/duplicate/conflict), "
                    "similarity score, matched ticket, and the requirement text. Indexes "
                    "here are what compare and reclassify take."
                ),
                "input_schema": {"type": "object", "properties": {}},
            })
        if self._tickets():
            defs.append({
                "name": "get_ticket",
                "description": "Show the full title, description, status and tags of one backlog ticket.",
                "input_schema": {
                    "type": "object",
                    "properties": {"ticket_id": {"type": "string"}},
                    "required": ["ticket_id"],
                },
            })
        if self.llm_tool is not None and self._gap_entries() and self._tickets():
            defs.append({
                "name": "compare",
                "description": (
                    "Ask a careful judge whether a requirement duplicates, conflicts with, "
                    f"or is unrelated to a ticket. Costs one LLM call; at most {MAX_COMPARES} "
                    "per run, so compare only matches you doubt."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Index from list_matches"},
                        "ticket_id": {"type": "string"},
                    },
                    "required": ["index", "ticket_id"],
                },
            })
        if self._gap_entries():
            defs.append({
                "name": "reclassify",
                "description": (
                    "Correct one gap-detection label when the evidence shows it is wrong. "
                    "duplicate and conflict need the ticket_id it matches. Give a one-sentence reason."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "classification": {"type": "string", "enum": list(_CLASSIFICATIONS)},
                        "ticket_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["index", "classification", "reason"],
                },
            })
        defs.append({
            "name": "finish",
            "description": (
                "End the investigation with what the pipeline should do next. Call this "
                "exactly once, when you are confident. Choose request_review (if offered) "
                "when a person should decide.\n"
                + "\n".join(f"- {name}: {desc}" for name, desc in self.actions.items())
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(self.actions)},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["action", "reason", "confidence"],
            },
        })
        return defs

    # ─── Dispatch ──────────────────────────────────────────────

    async def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Run a tool by name. Bad input comes back as an error result, never an exception."""
        offered = {d["name"] for d in self.definitions()}
        if name not in offered:
            return ToolResult(f"Tool '{name}' is not available here.", is_error=True)
        try:
            if name == "list_items":
                return self._list_items()
            if name == "list_matches":
                return self._list_matches()
            if name == "get_ticket":
                return self._get_ticket(str(args["ticket_id"]))
            if name == "compare":
                return await self._compare(int(args["index"]), str(args["ticket_id"]))
            if name == "reclassify":
                return self._reclassify(
                    int(args["index"]),
                    str(args["classification"]),
                    args.get("ticket_id"),
                    str(args.get("reason", "")),
                )
            if name == "finish":
                return self._finish(
                    str(args["action"]), str(args.get("reason", "")), args.get("confidence", 0.5)
                )
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(f"Invalid input for {name}: {e}", is_error=True)
        return ToolResult(f"Tool '{name}' is not implemented.", is_error=True)

    # ─── Read-only tools ───────────────────────────────────────

    def _list_items(self) -> ToolResult:
        items = self._parser_items()
        lines = [
            f"[{i}] {item.item_type}, confidence {item.confidence:.2f}: {_preview(item.text)}"
            for i, item in enumerate(items[:MAX_LISTED])
        ]
        return ToolResult(_with_overflow(lines, len(items)))

    def _list_matches(self) -> ToolResult:
        entries = self._current_entries()
        lines = []
        for i, entry in enumerate(entries[:MAX_LISTED]):
            ticket = entry.similar_ticket_id or "-"
            edited = " (reclassified)" if i in self.reclassifications else ""
            lines.append(
                f"[{i}] {entry.classification}{edited}, score {entry.similarity_score:.2f}, "
                f"ticket {ticket}: {_preview(entry.item.text)}"
            )
        return ToolResult(_with_overflow(lines, len(entries)))

    def _get_ticket(self, ticket_id: str) -> ToolResult:
        ticket = self._ticket(ticket_id)
        if ticket is None:
            return ToolResult(f"No ticket '{ticket_id}' in this backlog.", is_error=True)
        tags = ", ".join(ticket.tags) if ticket.tags else "none"
        return ToolResult(
            f"{ticket.id}: {ticket.title}\nStatus: {ticket.status}\nTags: {tags}\n"
            f"Description: {ticket.description}"
        )

    async def _compare(self, index: int, ticket_id: str) -> ToolResult:
        if self.compares_used >= MAX_COMPARES:
            return ToolResult(
                f"Compare limit reached ({MAX_COMPARES}). Decide with what you have.",
                is_error=True,
            )
        entry = self._entry(index)
        ticket = self._ticket(ticket_id)
        if ticket is None:
            return ToolResult(f"No ticket '{ticket_id}' in this backlog.", is_error=True)

        self.compares_used += 1
        verdict, reason = await verify_match(
            self.llm_tool, ticket.id, f"{ticket.title} {ticket.description}", entry.item.text
        )
        if verdict is None:
            return ToolResult("The judge gave no usable answer.", is_error=True)
        return ToolResult(f"Verdict: {verdict}. {reason}")

    # ─── Tools that change things ──────────────────────────────

    def _reclassify(
        self, index: int, classification: str, ticket_id: str | None, reason: str
    ) -> ToolResult:
        entry = self._entry(index)
        if classification not in _CLASSIFICATIONS:
            raise ValueError(f"classification must be one of {_CLASSIFICATIONS}")
        if classification != "new":
            if not ticket_id:
                raise ValueError(f"'{classification}' needs a ticket_id")
            if self._ticket(ticket_id) is None:
                return ToolResult(f"No ticket '{ticket_id}' in this backlog.", is_error=True)
        if not reason.strip():
            raise ValueError("reason is required")

        self.reclassifications[index] = Reclassification(
            index, classification, ticket_id if classification != "new" else None, reason
        )
        return ToolResult(
            f"Item [{index}] is now {classification} (was {entry.classification})."
        )

    def _finish(self, action: str, reason: str, confidence: Any) -> ToolResult:
        if action not in self.actions:
            raise ValueError(f"action must be one of {list(self.actions)}")
        self.decision = Decision(action, reason, min(max(float(confidence), 0.0), 1.0))
        return ToolResult(f"Recorded: {action}.")

    # ─── Results for the node ──────────────────────────────────

    def updated_gap_output(self) -> GapReport | None:
        """The gap report with reclassifications applied, or None if nothing changed."""
        gap_output = self.state.get("gap_output")
        if gap_output is None or not self.reclassifications:
            return None
        entries = self._current_entries()
        return GapReport(
            entries=entries,
            total_new=sum(e.classification == "new" for e in entries),
            total_duplicates=sum(e.classification == "duplicate" for e in entries),
            total_conflicts=sum(e.classification == "conflict" for e in entries),
            total_unprocessed=sum(e.classification == "unprocessed" for e in entries),
        )

    # ─── Helpers ───────────────────────────────────────────────

    def _parser_items(self) -> list[ExtractedItem]:
        parser_output = self.state.get("parser_output")
        return list(parser_output.items) if parser_output is not None else []

    def _gap_entries(self) -> list[GapReportEntry]:
        gap_output = self.state.get("gap_output")
        return list(gap_output.entries) if gap_output is not None else []

    def _current_entries(self) -> list[GapReportEntry]:
        entries = self._gap_entries()
        for index, change in self.reclassifications.items():
            original = entries[index]
            entries[index] = build_entry(
                original.item,
                change.classification,
                original.similarity_score,
                change.ticket_id,
                change.reason,
            )
        return entries

    def _entry(self, index: int) -> GapReportEntry:
        entries = self._current_entries()
        if not 0 <= index < len(entries):
            raise ValueError(f"index must be between 0 and {len(entries) - 1}")
        return entries[index]

    def _tickets(self) -> list:
        inputs = self.state.get("inputs")
        return list(inputs.backlog_tickets) if inputs is not None else []

    def _ticket(self, ticket_id: str):
        return next((t for t in self._tickets() if t.id == ticket_id), None)


def _preview(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _TEXT_PREVIEW else text[: _TEXT_PREVIEW - 1] + "…"


def _with_overflow(lines: list[str], total: int) -> str:
    if total > len(lines):
        lines.append(f"… and {total - len(lines)} more")
    return "\n".join(lines) if lines else "(none)"
