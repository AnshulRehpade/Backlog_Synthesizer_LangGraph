"""Gap detection data models for the Backlog Synthesizer system."""

from pydantic import BaseModel, Field

from backlog_synthesizer_langgraph.models.extraction import ExtractedItem


class DuplicateFlag(BaseModel):
    """Flag indicating an extracted item is a potential duplicate of an existing ticket."""

    extracted_item: ExtractedItem
    matching_ticket_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class ConflictFlag(BaseModel):
    """Flag indicating a conflict between an extracted item and an existing ticket."""

    item_a: ExtractedItem
    item_b_ticket_id: str
    similarity_score: float = Field(ge=0.50, le=0.85)
    contradiction_description: str


class GapReportEntry(BaseModel):
    """A single entry in the gap report classifying an extracted item."""

    item: ExtractedItem
    classification: str  # "new", "duplicate", "conflict", "unprocessed"
    gap_type: str = "new"  # "DUPLICATE", "CONFLICT", "NEW", "UNPROCESSED"
    confidence: float = Field(ge=0.0, le=1.0)
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    similar_ticket_id: str | None = None
    duplicate_info: DuplicateFlag | None = None
    conflict_info: ConflictFlag | None = None
    error_reason: str | None = None  # for unprocessed items


class GapReport(BaseModel):
    """Complete gap analysis report with classified entries and summary counts."""

    entries: list[GapReportEntry]
    total_new: int
    total_duplicates: int
    total_conflicts: int
    total_unprocessed: int
