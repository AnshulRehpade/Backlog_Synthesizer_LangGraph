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


def test_usage_is_recorded_from_every_response():
    from backlog_synthesizer_langgraph.tools.anthropic_generation import usage_snapshot

    tool, client = _anthropic_tool()
    client.messages.create.return_value.usage = MagicMock(input_tokens=120, output_tokens=30)
    before = usage_snapshot()
    tool.generate("hi")
    tool.generate("again")
    after = usage_snapshot()
    assert after["input_tokens"] - before["input_tokens"] == 240
    assert after["output_tokens"] - before["output_tokens"] == 60
    assert after["calls"] - before["calls"] == 2
