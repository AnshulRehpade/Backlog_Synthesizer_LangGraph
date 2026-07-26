"""LangGraph state definition for the Backlog Synthesizer pipeline.

Defines PipelineState as a TypedDict with nested Pydantic models for type safety.
Uses LangGraph's Annotated[list, add] pattern for append-only fields (errors, react_notes).

Design decision: Nested state over flattened
    We preserve the original Pydantic model hierarchy (ExtractionResult, GapReport,
    StoryOutput) inside the state rather than flattening into primitive fields. This
    keeps the state readable, validates structure at each node boundary, and matches
    the original project's convention of "Pydantic models for all data contracts."
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from operator import add

from backlog_synthesizer_langgraph.models.extraction import ExtractionResult
from backlog_synthesizer_langgraph.models.gap_detection import GapReport
from backlog_synthesizer_langgraph.models.output import StoryOutput
from backlog_synthesizer_langgraph.models.inputs import SessionInputs


class PipelineState(TypedDict):
    """Complete pipeline state passed between LangGraph nodes.

    Fields:
        inputs: Session inputs (documents + backlog tickets).
        session_id: Unique identifier for this pipeline execution.

        parser_output: Result from the Parser Agent (extracted items + errors).
        gap_output: Result from the Gap Detection Agent (classified items).
        story_output: Result from the Story Writer Agent (epics + stories).

        current_step: Name of the currently executing step.
        status: Pipeline status — one of:
            "in_progress", "completed", "partial_failure", "permanent_failure"

        errors: Append-only list of error dicts accumulated across nodes.
            Uses LangGraph's `add` reducer so each node can append without
            overwriting previous errors.

        react_notes: Append-only list of ReAct reasoning notes.
            Captures LLM advisory decisions made at pipeline decision points.

        retry_count: Number of retries attempted by the LLM fallback node.
            Prevents infinite retry loops — max 1 retry allowed.

        needs_human_review: Flag set by LLM fallback node when a decision
            requires human intervention (Level 3). Defaults to False.

        trace_id: OpenTelemetry trace ID for distributed tracing.
        total_duration_ms: Cumulative pipeline execution time in milliseconds.
    """

    # Inputs
    inputs: SessionInputs
    session_id: str

    # Agent outputs (nested Pydantic models)
    parser_output: Optional[ExtractionResult]
    gap_output: Optional[GapReport]
    story_output: Optional[StoryOutput]

    # Control flow
    current_step: str
    status: str  # in_progress, completed, partial_failure, permanent_failure

    # Error tracking (append-only using reducer)
    errors: Annotated[list, add]

    # ReAct notes (append-only)
    react_notes: Annotated[list, add]

    # Retry control (prevents infinite loops in LLM fallback)
    retry_count: int

    # Human review flag (set by LLM fallback when human decision needed)
    needs_human_review: bool

    # Observability
    trace_id: str
    total_duration_ms: int
