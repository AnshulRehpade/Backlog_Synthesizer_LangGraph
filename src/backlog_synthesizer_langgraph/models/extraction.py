"""Extraction data models for the Backlog Synthesizer system."""

from typing import Any

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """A segment of text from a document after chunking."""

    index: int
    text: str
    token_count: int


class ExtractedItem(BaseModel):
    """A single extracted item (decision, pain point, feature request, or constraint)."""

    item_type: str  # "decision", "pain_point", "feature_request", "constraint"
    text: str
    source_chunk_index: int
    char_offset: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    stakeholder: str | None = None
    section_heading: str | None = None
    type_classification: str | None = None  # for architecture items
    tags: list[str] = Field(default_factory=list)  # extracted topic/feature tags


class DocumentError(BaseModel):
    """Error information for a malformed or unreadable document."""

    filename: str
    reason: str
    byte_offset: int | None = None
    line_number: int | None = None


class ExtractionResult(BaseModel):
    """Result of document extraction containing items and any errors."""

    items: list[ExtractedItem]
    errors: list[DocumentError] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
