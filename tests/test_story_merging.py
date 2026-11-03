"""Tests for merging same-work items into one story, and for ticket links on stories."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from backlog_synthesizer_langgraph.agents.gap_detection import GapDetectionAgent
from backlog_synthesizer_langgraph.agents.story_writer import StoryWriterAgent
from backlog_synthesizer_langgraph.agents.work_grouping import _parse_groups, assign_groups
from backlog_synthesizer_langgraph.models.gap_detection import GapReport
from backlog_synthesizer_langgraph.tools.exporters import export_github_markdown, export_jira_csv
from backlog_synthesizer_langgraph.tools.interfaces import SearchResult
from tests.test_full_pipeline import _make_extracted_item, _make_story_output


def _related(text: str, ticket: str = "PLAT-104"):
    return GapDetectionAgent._related_entry(_make_extracted_item(text), SearchResult(ticket, 0.5), 0.5, "bug")


def _conflict(text: str, ticket: str = "PLAT-104", suggestion: str = "meeting_wins"):
    return GapDetectionAgent._conflict_entry(
        _make_extracted_item(text), SearchResult(ticket, 0.5), 0.5, "changes it", suggestion
    )


def _new(text: str):
    return GapDetectionAgent._new_entry(_make_extracted_item(text), 0.1)


def _report(*entries) -> GapReport:
    return GapReport(entries=list(entries), total_new=0, total_duplicates=0, total_conflicts=0, total_unprocessed=0)


def _llm(reply) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = reply
    return llm


# ─── Grouping ─────────────────────────────────────────────────────


class TestParseGroups:
    def test_valid_partition(self):
        assert _parse_groups('{"groups": [[2, 0], [1]]}', [0, 1, 2]) == [[0, 2], [1]]

    @pytest.mark.parametrize("reply", [
        '{"groups": [[0, 1]]}',            # misses an index
        '{"groups": [[0, 1], [1, 2]]}',    # repeats an index
        '{"groups": [[0, 1, 2, 3]]}',      # invents an index
        '{"groups": [[]]}',                # empty group
        "no json",
    ])
    def test_unusable_replies_are_rejected(self, reply):
        assert _parse_groups(reply, [0, 1, 2]) is None


async def test_groups_assigned_and_duplicates_left_out():
    dup = GapDetectionAgent._duplicate_entry(_make_extracted_item("dup"), SearchResult("T", 0.9), 0.9)
    report = _report(_related("CSV times out"), dup, _conflict("CSV async queue"), _new("Dark mode"))
    llm = _llm('{"groups": [[0, 2], [3]]}')

    grouped = await assign_groups(report, llm)

    assert [e.group for e in grouped.entries] == [0, None, 0, 1]
    prompt = llm.generate.call_args.args[0]
    assert "[0] (related vs PLAT-104) CSV times out" in prompt and "[1]" not in prompt


async def test_no_llm_or_bad_answer_means_one_story_per_item():
    report = _report(_related("a"), _new("b"))
    for llm in (None, _llm("garbage")):
        grouped = await assign_groups(report, llm)
        assert [e.group for e in grouped.entries] == [0, 1]


# ─── Story writing ────────────────────────────────────────────────


def _writer():
    llm = _llm(json.dumps({
        "title": "Async CSV export", "user_story": "As a user, I want exports, so that I can work",
        "acceptance_criteria": ["Exports 1M rows", "Runs in background"], "tags": ["export"],
    }))
    return StoryWriterAgent(llm), llm


async def test_same_group_becomes_one_story_covering_every_point():
    agent, llm = _writer()
    entries = [_related("CSV export times out"), _conflict("Move CSV export to async queue")]
    entries = [e.model_copy(update={"group": 0}) for e in entries]

    stories = await agent.generate_stories(entries)

    assert len(stories) == 1
    prompt = llm.generate.call_args.args[0]
    # The conflict frames the story; the related point is included.
    assert "Move CSV export to async queue" in prompt and "- CSV export times out" in prompt
    assert "CONFLICTS with existing ticket 'PLAT-104'" in prompt
    assert (stories[0].linked_ticket, stories[0].link_type) == ("PLAT-104", "replaces")


async def test_ungrouped_entries_still_get_one_story_each():
    agent, _ = _writer()
    stories = await agent.generate_stories([_new("a thing to build"), _new("another thing")])
    assert len(stories) == 2


@pytest.mark.parametrize("entry, link", [
    (_related("Fix CSV timeout"), ("PLAT-104", "relates_to")),
    (_conflict("Async CSV", suggestion="keep_both"), ("PLAT-104", "alongside")),
    (_conflict("Async CSV", suggestion="meeting_wins"), ("PLAT-104", "replaces")),
    (_new("Dark mode for the dashboard"), (None, None)),
])
async def test_link_type_follows_classification_and_resolution(entry, link):
    agent, _ = _writer()
    story = (await agent.generate_stories([entry]))[0]
    assert (story.linked_ticket, story.link_type) == link


async def test_kept_ticket_conflict_leaves_the_rest_of_its_group():
    """Reviewer keeps the ticket for the conflict; the bug fix in its group still gets a story."""
    agent, llm = _writer()
    kept = _conflict("Move CSV export to async queue")
    kept = kept.model_copy(update={
        "group": 0, "conflict_info": kept.conflict_info.model_copy(update={"resolution": "keep_ticket"}),
    })
    bug = _related("CSV export times out").model_copy(update={"group": 0})

    stories = await agent.generate_stories([kept, bug])

    assert len(stories) == 1 and stories[0].link_type == "relates_to"
    assert "async queue" not in llm.generate.call_args.args[0]


# ─── Exports ──────────────────────────────────────────────────────


def _linked_output():
    output = _make_story_output()
    story = output.epics[0].stories[0].model_copy(update={"linked_ticket": "PLAT-104", "link_type": "replaces"})
    epic = output.epics[0].model_copy(update={"stories": [story]})
    return output.model_copy(update={"epics": [epic]})


def test_jira_csv_has_link_columns():
    import csv, io
    rows = list(csv.DictReader(io.StringIO(export_jira_csv(_linked_output()))))
    assert (rows[0]["Linked Issue"], rows[0]["Link Type"]) == ("PLAT-104", "Replaces")
    assert "Replaces PLAT-104." in rows[0]["Description"]


def test_jira_csv_unlinked_story_has_empty_link_columns():
    import csv, io
    rows = list(csv.DictReader(io.StringIO(export_jira_csv(_make_story_output()))))
    assert (rows[0]["Linked Issue"], rows[0]["Link Type"]) == ("", "")


def test_github_markdown_shows_link():
    assert "**Linked ticket:** Replaces PLAT-104" in export_github_markdown(_linked_output())


# ─── Which ticket and which item lead a merged story ──────────────


async def test_group_ticket_is_the_best_matching_items_ticket():
    """A weaker (likelier wrong) match doesn't decide the link."""
    good = GapDetectionAgent._related_entry(
        _make_extracted_item("Add rate limiting to login"), SearchResult("PLAT-103", 0.46), 0.46, "adds a limit"
    )
    false_alarm = GapDetectionAgent._conflict_entry(
        _make_extracted_item("Login allows 1000 attempts/sec"), SearchResult("PLAT-108", 0.41), 0.41, "x", "keep_both"
    )
    grouped = await assign_groups(_report(good, false_alarm), _llm('{"groups": [[0, 1]]}'))
    assert {e.group_ticket for e in grouped.entries} == {"PLAT-103"}

    agent, _ = _writer()
    story = (await agent.generate_stories(grouped.entries))[0]
    assert (story.linked_ticket, story.link_type) == ("PLAT-103", "relates_to")


async def test_product_work_leads_a_merged_story_over_a_planning_decision():
    agent, llm = _writer()
    decision = _conflict("Start with SAML first for Acme's Q1 rollout", ticket="PLAT-109")
    decision = decision.model_copy(update={
        "item": decision.item.model_copy(update={"item_type": "decision"}), "group": 0, "group_ticket": "PLAT-109",
    })
    feature = _related("Add SAML 2.0 and OpenID Connect support", ticket="PLAT-109")
    feature = feature.model_copy(update={"group": 0, "group_ticket": "PLAT-109"})

    story = (await agent.generate_stories([decision, feature]))[0]

    prompt = llm.generate.call_args.args[0]
    lead, rest = prompt.split("This story also covers these points from the meeting:", 1)
    assert "Add SAML 2.0 and OpenID Connect support" in lead
    assert "- Start with SAML first" in rest
    # The decisive conflict still sets the link.
    assert (story.linked_ticket, story.link_type) == ("PLAT-109", "replaces")
