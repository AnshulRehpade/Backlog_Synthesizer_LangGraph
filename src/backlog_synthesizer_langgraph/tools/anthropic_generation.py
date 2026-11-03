"""Concrete LLM generation tool implementation using the Anthropic SDK.

Implements the LLMGenerationTool protocol, translating Anthropic-specific
exceptions into interface-defined ToolError subtypes.
"""

from __future__ import annotations

import os

import anthropic

from backlog_synthesizer_langgraph.tools.errors import (
    PermanentToolError,
    ToolError,
    TransientToolError,
)


# HTTP status codes classified as transient (retryable)
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})

# HTTP status codes classified as permanent (non-retryable)
_PERMANENT_STATUS_CODES = frozenset({401, 403, 404})


class AnthropicGenerationTool:
    """LLM text generation tool using the Anthropic Python SDK.

    Satisfies the LLMGenerationTool protocol. Translates all Anthropic-specific
    exceptions into ToolError subtypes before propagation.

    Args:
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var if None.
        model: Model identifier to use for generation. Defaults to "claude-haiku-4-20250414".
        max_tokens: Maximum tokens in the response. Defaults to 4096.
        timeout: Request timeout in seconds. Defaults to 60.
        temperature: Sampling temperature. None uses the API default (1.0);
            use 0 for classification calls that should be repeatable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-20250414",
        max_tokens: int = 4096,
        timeout: float = 60.0,
        temperature: float | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise PermanentToolError(
                "No API key provided and ANTHROPIC_API_KEY environment variable is not set."
            )

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = anthropic.Anthropic(
            api_key=resolved_key,
            timeout=timeout,
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using Claude.

        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: Optional system prompt for context/instruction.

        Returns:
            The generated text response.

        Raises:
            TransientToolError: For retryable failures.
            PermanentToolError: For non-retryable failures.
            ToolError: For any other unexpected errors.
        """
        text, _ = self._call_api(prompt, system_prompt)
        return text

    def generate_with_usage(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, dict]:
        """Generate text using Claude and return token usage.

        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: Optional system prompt for context/instruction.

        Returns:
            Tuple of (generated_text, usage_dict) where usage_dict contains
            'input_tokens' and 'output_tokens' keys.

        Raises:
            TransientToolError: For retryable failures.
            PermanentToolError: For non-retryable failures.
            ToolError: For any other unexpected errors.
        """
        return self._call_api(prompt, system_prompt)

    def _call_api(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, dict]:
        """Internal: call the Anthropic API and return (text, usage).

        Returns:
            Tuple of (response_text, {"input_tokens": int, "output_tokens": int}).
        """
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt is not None:
            kwargs["system"] = system_prompt
        # SDK 1.x removed this parameter (pyproject.toml pins anthropic<1.0); drop it
        # if the SDK is upgraded.
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        response = self._send(kwargs)
        content = "".join(block.text for block in response.content if block.type == "text")
        if not content:
            raise ToolError("LLM returned empty content in response.")

        # Extract token usage from response
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        }
        return content, usage

    def create_message(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ):
        """Send a full Messages API request and return the SDK Message.

        Used for tool calling, where the caller needs the content blocks
        (tool_use, text) and stop_reason rather than just the text.

        Raises:
            TransientToolError, PermanentToolError, ToolError: as for generate().
        """
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return self._send(kwargs)

    def _send(self, kwargs: dict):
        """Call messages.create, translating SDK exceptions into ToolError subtypes."""
        try:
            return self._client.messages.create(**kwargs)

        except anthropic.AuthenticationError as e:
            raise PermanentToolError(
                f"Authentication failed: {e.message}", original_error=e
            ) from e

        except anthropic.RateLimitError as e:
            raise TransientToolError(
                f"Rate limit exceeded: {e.message}", original_error=e
            ) from e

        except anthropic.APIConnectionError as e:
            raise TransientToolError(
                f"Connection error: {e}", original_error=e
            ) from e

        except anthropic.APITimeoutError as e:
            raise TransientToolError(
                f"Request timed out: {e}", original_error=e
            ) from e

        except anthropic.APIStatusError as e:
            if e.status_code in _TRANSIENT_STATUS_CODES:
                raise TransientToolError(
                    f"Transient API error (HTTP {e.status_code}): {e.message}",
                    original_error=e,
                ) from e
            elif e.status_code in _PERMANENT_STATUS_CODES:
                raise PermanentToolError(
                    f"Permanent API error (HTTP {e.status_code}): {e.message}",
                    original_error=e,
                ) from e
            else:
                raise ToolError(
                    f"API error (HTTP {e.status_code}): {e.message}",
                    original_error=e,
                ) from e

        except Exception as e:
            raise ToolError(
                f"Unexpected error during LLM generation: {e}",
                original_error=e,
            ) from e
