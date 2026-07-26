"""Concrete LLM generation tool implementation using OpenAI SDK.

Implements the LLMGenerationTool protocol, translating OpenAI-specific
exceptions into interface-defined ToolError subtypes.
"""

from __future__ import annotations

import os

import openai

from backlog_synthesizer_langgraph.tools.errors import (
    PermanentToolError,
    ToolError,
    TransientToolError,
)


# HTTP status codes classified as transient (retryable)
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# HTTP status codes classified as permanent (non-retryable)
_PERMANENT_STATUS_CODES = frozenset({401, 403, 404})


class OpenAIGenerationTool:
    """LLM text generation tool using the OpenAI Python SDK.

    Satisfies the LLMGenerationTool protocol. Translates all OpenAI-specific
    exceptions into ToolError subtypes before propagation.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var if None.
        model: Model identifier to use for generation. Defaults to "gpt-4o-mini".
        base_url: Optional base URL for alternative API endpoints.
        timeout: Request timeout in seconds. Defaults to 60.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise PermanentToolError(
                "No API key provided and OPENAI_API_KEY environment variable is not set."
            )

        self._model = model
        self._client = openai.OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using an LLM.

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
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
            )
            content = response.choices[0].message.content
            if content is None:
                raise ToolError("LLM returned empty content in response.")
            return content

        except openai.AuthenticationError as e:
            raise PermanentToolError(
                f"Authentication failed: {e.message}", original_error=e
            ) from e

        except openai.RateLimitError as e:
            raise TransientToolError(
                f"Rate limit exceeded: {e.message}", original_error=e
            ) from e

        except openai.APIConnectionError as e:
            raise TransientToolError(
                f"Connection error: {e.message}", original_error=e
            ) from e

        except openai.APITimeoutError as e:
            raise TransientToolError(
                f"Request timed out: {e.message}", original_error=e
            ) from e

        except openai.APIStatusError as e:
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
