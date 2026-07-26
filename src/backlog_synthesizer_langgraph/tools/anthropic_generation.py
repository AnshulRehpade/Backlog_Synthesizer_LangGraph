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
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-20250414",
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise PermanentToolError(
                "No API key provided and ANTHROPIC_API_KEY environment variable is not set."
            )

        self._model = model
        self._max_tokens = max_tokens
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
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt is not None:
            kwargs["system"] = system_prompt

        try:
            response = self._client.messages.create(**kwargs)
            text_parts = [
                block.text
                for block in response.content
                if block.type == "text"
            ]
            content = "".join(text_parts)
            if not content:
                raise ToolError("LLM returned empty content in response.")
            return content

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

        except (TransientToolError, PermanentToolError, ToolError):
            raise

        except Exception as e:
            raise ToolError(
                f"Unexpected error during LLM generation: {e}",
                original_error=e,
            ) from e
