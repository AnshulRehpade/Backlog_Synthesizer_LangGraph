"""Story Writer Agent for the Backlog Synthesizer system.

Produces structured user stories with acceptance criteria from deduplicated
gap report entries using LLM generation.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 10.1, 10.2, 10.3, 10.4, 10.5

Copied from original project — only import paths updated to new namespace.
"""

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from backlog_synthesizer_langgraph.models.extraction import ExtractedItem
from backlog_synthesizer_langgraph.models.gap_detection import GapReportEntry
from backlog_synthesizer_langgraph.models.output import (
    AcceptanceCriterion,
    Epic,
    OutputMetadata,
    StoryOutput,
    UserStory,
)
from backlog_synthesizer_langgraph.tools.interfaces import LLMGenerationTool

logger = logging.getLogger(__name__)


@dataclass
class SerializationError:
    """Describes a serialization failure for a specific item.

    Requirements: 10.5
    """

    item_title: str
    failing_field: str
    description: str


@dataclass
class SerializationResult:
    """Result of output serialization, containing the output and any errors.

    When serialization succeeds for all items, `output` is a valid StoryOutput
    and `errors` is empty. When some items fail validation, successfully serialized
    items are still included in the output, and failures are captured in `errors`.

    Requirements: 10.5
    """

    output: StoryOutput | None = None
    errors: list[SerializationError] = field(default_factory=list)


# Classifications eligible for story generation
ELIGIBLE_CLASSIFICATIONS = ("new", "conflict")

SYSTEM_PROMPT = (
    "You are a product manager assistant that converts extracted information "
    "into well-structured user stories. You produce JSON output only, with no "
    "additional commentary or markdown formatting."
)

STORY_GENERATION_PROMPT_TEMPLATE = """Convert the following extracted item into a structured user story.

Item type: {item_type}
Item text: {item_text}
Stakeholder: {stakeholder}
Section heading: {section_heading}
Extracted tags: {extracted_tags}

Generate a JSON object with these fields:
- "title": A concise title for the story (max 80 characters)
- "user_story": In the format "As a [role], I want [goal], so that [benefit]"
- "acceptance_criteria": An array of 2-10 objects, each with a "description" field describing a single testable condition
- "tags": An array of 1-5 keyword tags relevant to this story. Start with the extracted tags provided above and add more if needed.

Rules:
- The user_story MUST follow the format "As a [role], I want [goal], so that [benefit]"
- Generate between 2 and 10 acceptance criteria (inclusive)
- Generate between 1 and 5 tags (inclusive)
- Tags should be lowercase, hyphenated keywords (e.g., "user-auth", "data-export")
- Prefer using the extracted tags above when they are relevant
- If the item text is too vague to determine a specific role, goal, or benefit, use "[placeholder]" for the missing part

Respond with ONLY the JSON object, no other text."""


def _build_prompt(item: ExtractedItem) -> str:
    """Build the LLM prompt for a single extracted item.

    Args:
        item: The extracted item to generate a story from.

    Returns:
        A formatted prompt string.
    """
    extracted_tags = ", ".join(item.tags) if item.tags else "none"
    return STORY_GENERATION_PROMPT_TEMPLATE.format(
        item_type=item.item_type,
        item_text=item.text,
        stakeholder=item.stakeholder or "Unknown",
        section_heading=item.section_heading or "N/A",
        extracted_tags=extracted_tags,
    )


def _has_insufficient_detail(item: ExtractedItem) -> bool:
    """Determine if an extracted item has insufficient detail for story generation.

    An item is considered insufficient if its text is very short (less than 10 characters)
    or has very low confidence.

    Args:
        item: The extracted item to evaluate.

    Returns:
        True if the item has insufficient detail.
    """
    return len(item.text.strip()) < 10 or item.confidence < 0.3


def _create_placeholder_story(item: ExtractedItem) -> UserStory:
    """Create a placeholder user story for items with insufficient detail.

    Produces a story with placeholder text and the "needs-refinement" tag
    as required by Requirement 5.6.

    Args:
        item: The extracted item with insufficient detail.

    Returns:
        A UserStory with placeholder content and needs_refinement=True.
    """
    return UserStory(
        title=f"[Needs Refinement] {item.text[:50]}",
        user_story="As a [placeholder], I want [placeholder], so that [placeholder]",
        acceptance_criteria=[
            AcceptanceCriterion(description="Define the specific user role for this story"),
            AcceptanceCriterion(description="Clarify the goal and expected behavior"),
        ],
        tags=["needs-refinement"],
        needs_refinement=True,
    )


def _parse_llm_response(response: str, item: ExtractedItem) -> UserStory:
    """Parse the LLM JSON response into a UserStory object.

    Handles common LLM output issues like markdown code fences and
    invalid JSON. Falls back to a placeholder story if parsing fails.

    Args:
        response: The raw LLM response string.
        item: The original extracted item (used for fallback).

    Returns:
        A validated UserStory instance.
    """
    # Strip markdown code fences if present
    cleaned = response.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (possibly with language tag)
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse LLM response as JSON for item: %s", item.text[:50]
        )
        return _create_placeholder_story(item)

    # Validate and clamp acceptance criteria count
    acceptance_criteria = data.get("acceptance_criteria", [])
    if not isinstance(acceptance_criteria, list):
        acceptance_criteria = []

    # Ensure at least 2 criteria
    if len(acceptance_criteria) < 2:
        acceptance_criteria = [
            {"description": "Define the expected behavior"},
            {"description": "Verify the feature works as intended"},
        ]
    # Cap at 10 criteria
    acceptance_criteria = acceptance_criteria[:10]

    # Validate and clamp tags count
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    # Filter to strings only
    tags = [t for t in tags if isinstance(t, str) and t.strip()]
    # Ensure at least 1 tag
    if len(tags) < 1:
        tags = [item.item_type.replace("_", "-")]
    # Cap at 5 tags
    tags = tags[:5]

    # Extract user_story field
    user_story = data.get("user_story", "")
    if not isinstance(user_story, str) or not user_story.strip():
        user_story = "As a [placeholder], I want [placeholder], so that [placeholder]"

    # Check if the story has placeholder content indicating insufficient detail
    needs_refinement = "[placeholder]" in user_story

    # Add "needs-refinement" tag if the story needs refinement
    if needs_refinement and "needs-refinement" not in tags:
        if len(tags) >= 5:
            tags[-1] = "needs-refinement"
        else:
            tags.append("needs-refinement")

    # Build AcceptanceCriterion objects
    criteria_objects = []
    for ac in acceptance_criteria:
        if isinstance(ac, dict) and "description" in ac:
            criteria_objects.append(
                AcceptanceCriterion(description=str(ac["description"]))
            )
        elif isinstance(ac, str):
            criteria_objects.append(AcceptanceCriterion(description=ac))

    # Final safety: ensure we have 2-10 criteria
    if len(criteria_objects) < 2:
        criteria_objects = [
            AcceptanceCriterion(description="Define the expected behavior"),
            AcceptanceCriterion(description="Verify the feature works as intended"),
        ]

    title = data.get("title", item.text[:80])
    if not isinstance(title, str) or not title.strip():
        title = item.text[:80]

    return UserStory(
        title=title,
        user_story=user_story,
        acceptance_criteria=criteria_objects,
        tags=tags,
        needs_refinement=needs_refinement,
    )


class StoryWriterAgent:
    """Agent that produces structured user stories from deduplicated gap report entries.

    Uses LLMGenerationTool to generate user stories in the "As a [role], I want [goal],
    so that [benefit]" format with acceptance criteria and feature tags.

    Requirements: 5.1, 5.2, 5.3, 5.6
    """

    def __init__(self, llm_tool: LLMGenerationTool, few_shot_store: Any | None = None) -> None:
        """Initialize StoryWriterAgent with tool dependencies.

        Args:
            llm_tool: Tool for LLM text generation.
            few_shot_store: Optional FewShotStore for dynamic few-shot prompting.
        """
        self._llm_tool = llm_tool
        self._few_shot_store = few_shot_store

    async def generate_stories(
        self, items: list[GapReportEntry]
    ) -> list[UserStory]:
        """Generate user stories from deduplicated gap report entries.

        Filters entries to only "new" and "conflict" classifications, then
        generates structured user stories for all eligible items concurrently
        using asyncio.gather() for improved throughput.

        Args:
            items: List of GapReportEntry items from the gap detection phase.

        Returns:
            A list of UserStory objects generated from eligible items.
        """
        # Filter to eligible classifications only
        eligible_entries = [
            entry for entry in items
            if entry.classification in ELIGIBLE_CLASSIFICATIONS
        ]

        if not eligible_entries:
            logger.info("No eligible items for story generation.")
            return []

        # Generate all stories concurrently
        coros = [self._generate_single_story(entry) for entry in eligible_entries]
        stories = await asyncio.gather(*coros)

        return list(stories)

    async def _generate_single_story(self, entry: GapReportEntry) -> UserStory:
        """Generate a single user story from a gap report entry.

        Handles insufficient detail by producing placeholder stories.
        Handles LLM failures gracefully by falling back to placeholder stories.

        Args:
            entry: A single GapReportEntry to convert into a user story.

        Returns:
            A validated UserStory instance.
        """
        item = entry.item

        # Check for insufficient detail before calling LLM
        if _has_insufficient_detail(item):
            logger.info(
                "Item has insufficient detail, producing placeholder: %s",
                item.text[:50],
            )
            return _create_placeholder_story(item)

        # Build prompt and call LLM
        prompt = _build_prompt(item)

        # Inject conflict context if this is a CONFLICT entry
        if entry.classification == "conflict" and entry.conflict_info is not None:
            conflict_section = (
                "\n\n## Conflict Context:\n"
                f"This item CONFLICTS with existing ticket '{entry.conflict_info.item_b_ticket_id}'.\n"
                f"Similarity score: {entry.conflict_info.similarity_score:.2f}\n"
                f"What changed: {entry.conflict_info.contradiction_description}\n\n"
                "The user story should clearly describe what is DIFFERENT from the "
                "existing ticket. Frame it as an update, migration, or replacement. "
                "Include acceptance criteria that validate the new approach works "
                "AND that the old behavior is properly deprecated or migrated.\n"
            )
            prompt = conflict_section + prompt

        # Inject few-shot examples if available
        if self._few_shot_store is not None:
            try:
                examples = self._few_shot_store.get_similar_story_examples(
                    item.text, top_k=2
                )
                if examples:
                    examples_section = "\n\n## Examples of high quality user stories:\n"
                    for i, ex in enumerate(examples, 1):
                        examples_section += (
                            f"\nExample {i} (topic: {ex.description}):\n"
                            f'Input item: "{ex.transcript[:150]}"\n'
                            f'Output: {{"title": "...", "user_story": "As a...", '
                            f'"acceptance_criteria": [...], "tags": [...]}}\n'
                        )
                    examples_section += "\n## Now generate a story for the item above:\n"
                    prompt = examples_section + prompt
            except Exception as e:
                logger.warning("Story Writer few-shot retrieval failed: %s", e)

        try:
            response = await asyncio.to_thread(
                self._llm_tool.generate,
                prompt,
                SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error(
                "LLM generation failed for item '%s': %s", item.text[:50], e
            )
            return _create_placeholder_story(item)

        # Parse LLM response into a UserStory
        return _parse_llm_response(response, item)

    def group_into_epics(self, stories: list[UserStory]) -> list[Epic]:
        """Group user stories into epics based on shared tags.

        Uses a union-find (disjoint set) approach to find connected components
        of stories that share at least one tag. Stories sharing a tag are grouped
        together transitively: if A shares a tag with B, and B shares a tag with C,
        all three end up in the same epic.

        Stories that share no tags with any other story get their own single-story epic.

        Args:
            stories: List of UserStory objects to group.

        Returns:
            A list of Epic objects, each containing related stories.

        Requirements: 5.4
        """
        return group_stories_into_epics(stories)

    def serialize_output(
        self, epics: list[Epic], session_id: str
    ) -> SerializationResult:
        """Serialize epics into a StoryOutput JSON-conformant structure.

        Generates the top-level index array with epic titles and story counts,
        creates output metadata with session_id and current ISO 8601 timestamp,
        and assembles the final StoryOutput. Handles serialization failures by
        capturing errors per item while still including successfully serialized items.

        Args:
            epics: List of Epic objects to serialize.
            session_id: The session identifier for metadata.

        Returns:
            A SerializationResult containing the output and any errors encountered.

        Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
        """
        errors: list[SerializationError] = []
        valid_epics: list[Epic] = []

        for epic in epics:
            try:
                # Validate the epic by re-constructing it through Pydantic
                validated_epic = Epic.model_validate(epic.model_dump())
                valid_epics.append(validated_epic)
            except ValidationError as e:
                # Extract error details for reporting
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error["loc"])
                    errors.append(
                        SerializationError(
                            item_title=epic.epic_title if hasattr(epic, "epic_title") else "Unknown",
                            failing_field=field_path,
                            description=error["msg"],
                        )
                    )
            except Exception as e:
                errors.append(
                    SerializationError(
                        item_title=getattr(epic, "epic_title", "Unknown"),
                        failing_field="unknown",
                        description=str(e),
                    )
                )

        # Generate the index array from valid epics
        index = [
            {
                "epic_title": epic.epic_title,
                "story_count": len(epic.stories),
            }
            for epic in valid_epics
        ]

        # Create metadata with current timestamp
        metadata = OutputMetadata(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
        )

        # Assemble the StoryOutput
        try:
            output = StoryOutput(
                index=index,
                epics=valid_epics,
                metadata=metadata,
            )
        except ValidationError as e:
            # If the overall StoryOutput assembly fails, report the error
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                errors.append(
                    SerializationError(
                        item_title="StoryOutput",
                        failing_field=field_path,
                        description=error["msg"],
                    )
                )
            return SerializationResult(output=None, errors=errors)

        return SerializationResult(output=output, errors=errors)


# --- Epic Grouping Logic (Requirement 5.4) ---


class _UnionFind:
    """Disjoint set (union-find) data structure for grouping stories by shared tags."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        """Find the root representative of the set containing x (with path compression)."""
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing x and y (union by rank)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def _generate_epic_title(stories: list[UserStory]) -> str:
    """Generate a concise epic title from the shared tags of grouped stories.

    Uses a heuristic approach: picks the most common tags shared across the
    stories, capitalizes them, and joins them. Ensures the title is at most
    60 characters.

    Args:
        stories: The stories in this epic group.

    Returns:
        An epic title string of at most 60 characters.
    """
    # Collect all tags from all stories in this group
    tag_counter: Counter[str] = Counter()
    for story in stories:
        for tag in story.tags:
            tag_counter[tag] += 1

    # Pick tags that appear in more than one story first (shared tags),
    # then fall back to most common tags overall
    shared_tags = [tag for tag, count in tag_counter.most_common() if count > 1]
    if not shared_tags:
        # Single-story epic or no repeated tags — use most common tags
        shared_tags = [tag for tag, _ in tag_counter.most_common(3)]

    # Build title from tags: capitalize, replace hyphens with spaces
    title_parts = []
    for tag in shared_tags:
        formatted = tag.replace("-", " ").replace("_", " ").title()
        title_parts.append(formatted)

    title = " & ".join(title_parts)

    # Truncate to 60 characters
    if len(title) > 60:
        title = title[:57] + "..."

    # If empty for some reason, provide a fallback
    if not title:
        title = "General Stories"

    return title


def group_stories_into_epics(stories: list[UserStory]) -> list[Epic]:
    """Group user stories into epics based on shared tags using union-find.

    Stories that share at least one tag are grouped together transitively.
    If story A shares a tag with story B, and story B shares a tag with story C,
    all three will appear in the same epic.

    Stories that share no tags with any other story get their own single-story epic.

    Epic titles are generated heuristically from the most common shared tags,
    enforcing a maximum of 60 characters.

    Args:
        stories: List of UserStory objects to group.

    Returns:
        A list of Epic objects, each containing related stories grouped by shared tags.

    Requirements: 5.4
    """
    if not stories:
        return []

    n = len(stories)
    uf = _UnionFind(n)

    # Build a mapping from tag -> list of story indices
    tag_to_indices: dict[str, list[int]] = {}
    for i, story in enumerate(stories):
        for tag in story.tags:
            if tag not in tag_to_indices:
                tag_to_indices[tag] = []
            tag_to_indices[tag].append(i)

    # Union all stories that share a tag
    for indices in tag_to_indices.values():
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                uf.union(first, other)

    # Collect stories by their root representative
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Build Epic objects from groups
    epics: list[Epic] = []
    for indices in groups.values():
        group_stories = [stories[i] for i in indices]
        title = _generate_epic_title(group_stories)
        epics.append(Epic(epic_title=title, stories=group_stories))

    return epics
