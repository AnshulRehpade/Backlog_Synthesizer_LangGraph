"""LangGraph checkpointer setup for the Backlog Synthesizer pipeline.

Replaces Short-Term Memory from the original project.

Supports two backends:
    - MemorySaver (default): In-memory, suitable for development/testing
    - SqliteSaver: Persistent, suitable for production — enables session
      resume after restarts and multi-user support

Switch between backends via the CHECKPOINTER_BACKEND environment variable:
    CHECKPOINTER_BACKEND=memory   (default)
    CHECKPOINTER_BACKEND=sqlite   (persists to .checkpoints/sessions.db)

PRODUCTION SWAP:
    For production beyond SQLite:
        - PostgresSaver: langgraph-checkpoint-postgres
        - RedisSaver: community implementations
"""

from __future__ import annotations

import os
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

_BACKEND = os.environ.get("CHECKPOINTER_BACKEND", "memory").lower()
_SQLITE_PATH = os.environ.get(
    "CHECKPOINTER_SQLITE_PATH", ".checkpoints/sessions.db"
)


def create_checkpointer():
    """Create a LangGraph checkpointer based on environment configuration.

    Uses CHECKPOINTER_BACKEND env var to select the backend:
        - "memory" (default): MemorySaver — fast, ephemeral
        - "sqlite": SqliteSaver — persistent, supports resume

    Returns:
        A checkpointer instance compatible with LangGraph's compile().
    """
    if _BACKEND == "sqlite":
        return _create_sqlite_checkpointer()
    return MemorySaver()


def _create_sqlite_checkpointer():
    """Create a SqliteSaver checkpointer with auto-created directory.

    Returns:
        A SqliteSaver connected to the configured database path.
    """
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            # Fallback if sqlite checkpoint package not installed
            import logging
            logging.getLogger(__name__).warning(
                "SqliteSaver not available (install langgraph-checkpoint-sqlite). "
                "Falling back to MemorySaver."
            )
            return MemorySaver()

    # Ensure directory exists
    db_path = Path(_SQLITE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Try async version first, fall back to sync
    try:
        return AsyncSqliteSaver.from_conn_string(str(db_path))
    except Exception:
        try:
            return SqliteSaver.from_conn_string(str(db_path))
        except Exception:
            return MemorySaver()
