"""Output data models for the Backlog Synthesizer system."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AcceptanceCriterion(BaseModel):
    """A single acceptance criterion for a user story."""

    description: str


class UserStory(BaseModel):
    """A structured user story with acceptance criteria and tags."""

    title: str
    user_story: str  # "As a [role], I want [goal], so that [benefit]"
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=2, max_length=10)
    tags: list[str] = Field(min_length=1, max_length=5)
    needs_refinement: bool = False


class Epic(BaseModel):
    """A grouping of related user stories under a common theme."""

    epic_title: str = Field(max_length=60)
    stories: list[UserStory]


class OutputMetadata(BaseModel):
    """Metadata for the output including session and timestamp information."""

    session_id: str
    timestamp: datetime  # ISO 8601


class StoryOutput(BaseModel):
    """Complete output from the Story Writer Agent."""

    index: list[dict[str, Any]]  # [{epic_title: str, story_count: int}]
    epics: list[Epic]
    metadata: OutputMetadata
