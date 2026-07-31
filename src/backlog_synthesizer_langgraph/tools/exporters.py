"""Export utilities for converting StoryOutput to various formats.

Supports:
- JIRA CSV: importable into JIRA via CSV import
- GitHub Issues Markdown: one file per story, ready for gh CLI
- JSON: native Pydantic serialization (already handled elsewhere)
"""

from __future__ import annotations

import csv
import io
from typing import Any

from backlog_synthesizer_langgraph.models.output import StoryOutput


def export_jira_csv(story_output: StoryOutput) -> str:
    """Export stories as JIRA-compatible CSV.

    Columns: Summary, Description, Issue Type, Labels, Epic Link, Acceptance Criteria

    Args:
        story_output: The StoryOutput to export.

    Returns:
        CSV string ready for JIRA import.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # JIRA CSV header
    writer.writerow([
        "Summary",
        "Description",
        "Issue Type",
        "Labels",
        "Epic Name",
        "Acceptance Criteria",
    ])

    for epic in story_output.epics:
        for story in epic.stories:
            # Build description from user story
            description = story.user_story

            # Build acceptance criteria as bullet list
            ac_text = "\n".join(
                f"* {ac.description}" for ac in story.acceptance_criteria
            )

            # Labels from tags
            labels = " ".join(story.tags)

            writer.writerow([
                story.title,
                description,
                "Story",
                labels,
                epic.epic_title,
                ac_text,
            ])

    return output.getvalue()


def export_github_markdown(story_output: StoryOutput) -> str:
    """Export stories as GitHub Issues markdown.

    Each story becomes a separate issue block with frontmatter-style
    metadata, suitable for use with `gh issue create` or batch import.

    Args:
        story_output: The StoryOutput to export.

    Returns:
        Markdown string with all stories formatted as GitHub issues.
    """
    sections = []

    for epic in story_output.epics:
        sections.append(f"# Epic: {epic.epic_title}\n")

        for story in epic.stories:
            lines = []
            lines.append(f"## {story.title}\n")

            # Labels line
            labels = ", ".join(f"`{tag}`" for tag in story.tags)
            lines.append(f"**Labels:** {labels}")
            lines.append(f"**Epic:** {epic.epic_title}")
            if story.needs_refinement:
                lines.append("**Status:** Needs Refinement")
            lines.append("")

            # User story
            lines.append(f"> {story.user_story}")
            lines.append("")

            # Acceptance criteria as task list
            lines.append("### Acceptance Criteria\n")
            for ac in story.acceptance_criteria:
                lines.append(f"- [ ] {ac.description}")
            lines.append("")

            sections.append("\n".join(lines))

    return "\n---\n\n".join(sections)


def export_stories_summary(story_output: StoryOutput) -> list[dict[str, Any]]:
    """Export a flat summary list for easy programmatic consumption.

    Args:
        story_output: The StoryOutput to export.

    Returns:
        List of dicts with key fields per story.
    """
    results = []
    for epic in story_output.epics:
        for story in epic.stories:
            results.append({
                "epic": epic.epic_title,
                "title": story.title,
                "user_story": story.user_story,
                "tags": story.tags,
                "acceptance_criteria_count": len(story.acceptance_criteria),
                "needs_refinement": story.needs_refinement,
            })
    return results
