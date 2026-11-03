"""UI test: the Streamlit human-review buttons resume a paused run.

Runs the real app.py with Streamlit's AppTest. The agent nodes are faked and
the fallback is steered to request_review, so the run pauses without API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_human_review_resume import _FakeNodes


@pytest.fixture
def paused_app():
    """App after clicking Generate, paused at human review; yields (app, fakes)."""
    fakes = _FakeNodes()
    with patch("graph.builder.parser_node", fakes.parser), \
         patch("graph.builder.gap_detection_node", fakes.gap_detection), \
         patch("graph.builder.story_writer_node", fakes.story_writer), \
         patch(
             "nodes.llm_fallback_node._ask_llm_for_decision",
             AsyncMock(return_value=("request_review", "Mostly duplicates", 0.9)),
         ):
        app = AppTest.from_file("app.py", default_timeout=30)
        app.run()
        next(b for b in app.button if "Generate Stories" in b.label).click().run()
        yield app, fakes


def _review_banner_shown(app) -> bool:
    return any("Human review required" in info.value for info in app.info)


def test_paused_run_shows_review_controls(paused_app):
    app, fakes = paused_app
    assert not app.exception
    assert _review_banner_shown(app)
    assert {"hr_approve", "hr_reject", "hr_rerun"} <= {b.key for b in app.button}
    assert fakes.story_writer_calls == 0


def test_approve_resumes_and_writes_stories(paused_app):
    app, fakes = paused_app
    app.button(key="hr_approve").click().run()

    assert not app.exception
    assert not _review_banner_shown(app)
    assert fakes.story_writer_calls == 1
    assert app.session_state["result"]["status"] == "completed"


def test_reject_ends_the_run_with_a_clear_message(paused_app):
    app, fakes = paused_app
    app.button(key="hr_reject").click().run()

    # The review banner isn't asserted gone here: AppTest keeps elements from the
    # click run after st.rerun(). Checked in a real browser that it disappears.
    assert app.session_state["result"]["status"] == "permanent_failure"
    assert any("Rejected in human review" in w.value for w in app.warning)
    assert not any("API key" in e.value for e in app.error)
    assert fakes.story_writer_calls == 0


def test_paused_run_is_not_reported_as_finished_or_failed(paused_app):
    app, _ = paused_app
    assert not any("completed with warnings" in w.value for w in app.warning)
    assert not any("No stories generated" in i.value for i in app.info)


def test_rerun_uses_the_threshold_from_the_input(paused_app):
    app, fakes = paused_app
    app.number_input(key="hr_threshold").set_value(0.9)
    app.button(key="hr_rerun").click().run()

    assert fakes.gap_thresholds[:2] == [None, 0.9]
