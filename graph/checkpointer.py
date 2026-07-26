"""LangGraph checkpointer setup for the Backlog Synthesizer pipeline.

Replaces Short-Term Memory from the original project.

WHY SHORT-TERM MEMORY WAS REMOVED:
    The original project's ShortTermMemory (memory/short_term.py) provided
    in-session state keyed by session ID — storing intermediate results between
    agent steps. LangGraph's Checkpointer does exactly this job:

    - Persists full graph state after every node execution
    - Keyed by thread_id (equivalent to session_id)
    - Supports time-travel / replay of intermediate states
    - Built-in fallback from persistent → in-memory (same pattern as the
      original's primary_store → dict fallback)

    Using both would be redundant. The Checkpointer is the canonical LangGraph
    way to handle this, and it integrates with the graph execution lifecycle
    automatically rather than requiring manual store/retrieve calls.

PRODUCTION SWAP:
    For production, MemorySaver can be replaced with zero code changes:
        - SqliteSaver: langgraph.checkpoint.sqlite.SqliteSaver
        - PostgresSaver: langgraph-checkpoint-postgres
        - RedisSaver: community implementations

    This follows the same Protocol-based swap pattern used in the original
    project for tool implementations.
"""

from langgraph.checkpoint.memory import MemorySaver


def create_checkpointer() -> MemorySaver:
    """Create a LangGraph checkpointer for development use.

    Returns a MemorySaver (in-memory) checkpointer suitable for development
    and testing. For production, swap with SqliteSaver or a persistent backend.

    Returns:
        A MemorySaver instance.
    """
    return MemorySaver()
