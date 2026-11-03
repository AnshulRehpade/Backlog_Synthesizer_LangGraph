"""Gap detection data models for the Backlog Synthesizer system."""

from pydantic import BaseModel, Field

from backlog_synthesizer_langgraph.models.extraction import ExtractedItem


class DuplicateFlag(BaseModel):
    """Flag indicating an extracted item is a potential duplicate of an existing ticket."""

    extracted_item: ExtractedItem
    matching_ticket_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)


# How a person can settle a conflict between a meeting item and a ticket.
RESOLUTIONS = {
    "meeting_wins": "The meeting decision replaces the ticket — write a story that updates it",
    "keep_ticket": "Keep the existing ticket as it is — write no story for this item",
    "keep_both": "Both stand — write a story for the item alongside the ticket",
}


class ConflictFlag(BaseModel):
    """Flag indicating a conflict between an extracted item and an existing ticket."""

    item_a: ExtractedItem
    item_b_ticket_id: str
    similarity_score: float = Field(ge=0.50, le=0.85)
    contradiction_description: str
    # What the judge suggests, and what a person decided (None until reviewed).
    suggested_resolution: str = "meeting_wins"
    resolution: str | None = None


class GapReportEntry(BaseModel):
    """A single entry in the gap report classifying an extracted item."""

    item: ExtractedItem
    classification: str  # "new", "duplicate", "conflict", "related", "unprocessed"
    gap_type: str = "new"  # "DUPLICATE", "CONFLICT", "RELATED", "NEW", "UNPROCESSED"
    confidence: float = Field(ge=0.0, le=1.0)
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    similar_ticket_id: str | None = None
    duplicate_info: DuplicateFlag | None = None
    conflict_info: ConflictFlag | None = None
    error_reason: str | None = None  # for unprocessed items
    # For related items: how the item relates to similar_ticket_id (follow-on, addition, bug).
    relation: str | None = None
    # Entries with the same group describe the same piece of work and become one story.
    group: int | None = None
    # The existing ticket the group's story is about, chosen with the group (None: decide by rank).
    group_ticket: str | None = None


class GapReport(BaseModel):
    """Complete gap analysis report with classified entries and summary counts."""

    entries: list[GapReportEntry]
    total_new: int
    total_duplicates: int
    total_conflicts: int
    total_unprocessed: int
    # Items in the same area as a ticket that don't contradict it (no review needed).
    total_related: int = 0
