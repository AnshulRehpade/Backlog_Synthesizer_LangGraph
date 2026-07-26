"""Error types for tool interfaces.

All tool implementations must translate implementation-specific exceptions
into these interface-defined error types before propagating to calling agents.
"""


class ToolError(Exception):
    """Base error type for all tool interfaces."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error


class TransientToolError(ToolError):
    """Raised for transient/retryable failures (e.g., HTTP 429, 500, 502, 503, 504, network timeouts)."""

    pass


class PermanentToolError(ToolError):
    """Raised for permanent/non-retryable failures (e.g., HTTP 401, 403, 404, auth failures, schema violations)."""

    pass
