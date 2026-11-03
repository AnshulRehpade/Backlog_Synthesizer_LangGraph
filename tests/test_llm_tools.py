"""Tests that LLM tools pass the configured temperature to the provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from backlog_synthesizer_langgraph.tools.anthropic_generation import AnthropicGenerationTool


def _anthropic_tool(**kwargs) -> tuple[AnthropicGenerationTool, MagicMock]:
    with patch("anthropic.Anthropic") as client_cls:
        tool = AnthropicGenerationTool(api_key="test-key", **kwargs)
    client = client_cls.return_value
    block = MagicMock(type="text", text="ok")
    client.messages.create.return_value = MagicMock(content=[block], usage=MagicMock())
    return tool, client


def test_temperature_sent_when_set():
    tool, client = _anthropic_tool(temperature=0.0)
    tool.generate("hi")
    assert client.messages.create.call_args.kwargs["temperature"] == 0.0


def test_temperature_omitted_by_default():
    tool, client = _anthropic_tool()
    tool.generate("hi")
    assert "temperature" not in client.messages.create.call_args.kwargs
