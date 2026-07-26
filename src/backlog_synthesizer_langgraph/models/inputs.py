"""Input data models for the Backlog Synthesizer system."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types for ingestion."""

    TRANSCRIPT_TXT = "transcript_txt"
    TRANSCRIPT_MD = "transcript_md"
    TRANSCRIPT_PDF = "transcript_pdf"
    ARCHITECTURE_HTML = "architecture_html"
    BACKLOG_JSON = "backlog_json"


class InputDocument(BaseModel):
    """Represents a single input document provided for processing."""

    filename: str
    document_type: DocumentType
    content: bytes
    size_bytes: int


class BacklogTicket(BaseModel):
    """Represents an existing backlog ticket from JIRA or GitHub Issues."""

    id: str
    title: str
    description: str
    status: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class SessionInputs(BaseModel):
    """Represents all inputs for a single processing session."""

    session_id: str
    documents: list[InputDocument]
    backlog_tickets: list[BacklogTicket] = Field(default_factory=list)
