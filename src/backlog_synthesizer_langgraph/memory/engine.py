"""Memory Engine facade wiring Long-Term Memory and Audit Log.

Provides a unified interface for semantic search indexing and audit logging.

NOTE: Short-Term Memory delegation has been removed in this LangGraph migration.
LangGraph's Checkpointer (graph/checkpointer.py) replaces Short-Term Memory entirely.
The store_intermediate/retrieve_intermediate methods are no longer needed — graph state
is automatically persisted after every node execution by the checkpointer.
"""

from __future__ import annotations

import logging
from typing import Any

from backlog_synthesizer_langgraph.memory.audit_log import AuditLog
from backlog_synthesizer_langgraph.memory.long_term import LongTermMemory
from backlog_synthesizer_langgraph.models.memory import AuditEntry
from backlog_synthesizer_langgraph.tools.interfaces import SearchResult

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Facade coordinating Long-Term Memory and Audit Log.

    Provides a single entry point for:
    - Indexing items for semantic search (Long-Term Memory)
    - Recording and querying audit log entries (Audit Log)

    Short-Term Memory responsibilities are handled by LangGraph's Checkpointer.

    Args:
        long_term: LongTermMemory instance for vector-backed semantic search.
        audit_log: AuditLog instance for recording sub-agent invocations.
    """

    def __init__(
        self,
        long_term: LongTermMemory,
        audit_log: AuditLog,
    ) -> None:
        self._long_term = long_term
        self._audit_log = audit_log

    # ──────────────────────────────────────────────
    # Long-Term Memory delegation
    # ──────────────────────────────────────────────

    def store_for_search(self, session_id: str, items: list[dict[str, Any]]) -> None:
        """Index items in long-term memory for semantic search.

        Each item dict must contain at least `item_id` and `content` keys.
        The session_id is added to the item metadata for traceability.
        """
        for item in items:
            item_id = item["item_id"]
            content = item["content"]
            metadata: dict[str, Any] = {
                k: v for k, v in item.items() if k not in ("item_id", "content")
            }
            metadata["session_id"] = session_id
            self._long_term.store_item(item_id, content, metadata)

    def search_similar(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search for items semantically similar to the query."""
        return self._long_term.search_similar(query, top_k)

    def purge_expired(self) -> list[str]:
        """Purge expired entries from long-term memory."""
        return self._long_term.purge_expired()

    # ──────────────────────────────────────────────
    # Audit Log delegation
    # ──────────────────────────────────────────────

    def log_action(self, session_id: str, entry: AuditEntry) -> None:
        """Record a sub-agent invocation in the audit log."""
        self._audit_log.record(session_id, entry)

    def get_audit_log(self, session_id: str) -> list[AuditEntry]:
        """Retrieve all audit log entries for a session in chronological order."""
        return self._audit_log.get_entries(session_id)
