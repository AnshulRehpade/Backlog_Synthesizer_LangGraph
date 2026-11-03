"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _fallback_menu_mode(monkeypatch):
    """Keep the LLM fallback on the single-call menu unless a test opts into the agent.

    Agent mode would make real API calls whenever a key is in the environment, and
    MagicMock LLM tools look tool-capable, so tests must choose agent mode explicitly.
    """
    monkeypatch.setenv("FALLBACK_MODE", "menu")
