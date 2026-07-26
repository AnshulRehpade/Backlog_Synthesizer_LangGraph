"""Memory and audit data models for the Backlog Synthesizer system."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from backlog_synthesizer_langgraph.models.extraction import ExtractionResult
from backlog_synthesizer_langgraph.models.gap_detection import GapReport
from backlog_synthesizer_langgraph.models.output import StoryOutput


class AuditEntry(BaseModel):
    """A single audit log entry recording a sub-agent invocation."""

    timestamp: datetime
    agent_name: str
    input_summary: str = Field(max_length=500)
    output_summary: str = Field(max_length=500)
    duration_ms: int


class SessionState(BaseModel):
    """Complete session state including all intermediate and final results."""

    session_id: str
    created_at: datetime
    status: str  # "in_progress", "completed", "partial_failure", "permanent_failure"
    extraction_result: ExtractionResult | None = None
    gap_report: GapReport | None = None
    story_output: StoryOutput | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
