"""Graph construction for the Backlog Synthesizer LangGraph pipeline.

Builds the StateGraph wiring together nodes and conditional edges.

Full pipeline topology:
    parser → [conditional] → gap_detection | llm_fallback_node | error_node
    gap_detection → [conditional] → story_writer | llm_fallback_node | error_node
    story_writer → [conditional] → END | llm_fallback_node | error_node
    llm_fallback_node → [conditional] → parser | gap_detection | story_writer | human_review | END
    human_review → [conditional] → gap_detection | story_writer | END
    error_node → END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.checkpointer import create_checkpointer
from graph.routing import (
    route_after_gap_detection,
    route_after_human_review,
    route_after_llm_fallback,
    route_after_parser,
    route_after_story_writer,
)
from graph.state import PipelineState
from nodes.error_node import error_node
from nodes.gap_detection_node import gap_detection_node
from nodes.human_review_node import human_review_node
from nodes.llm_fallback_node import llm_fallback_node
from nodes.parser_node import parser_node
from nodes.story_writer_node import story_writer_node


def build_graph() -> StateGraph:
    """Build and return the pipeline StateGraph (uncompiled).

    Full graph topology:
        START → parser
        parser → [route_after_parser] → gap_detection | llm_fallback_node | error_node
        gap_detection → [route_after_gap_detection] → story_writer | llm_fallback_node | error_node
        story_writer → [route_after_story_writer] → END | llm_fallback_node | error_node
        llm_fallback_node → [route_after_llm_fallback] → parser | gap_detection | story_writer | human_review | END
        human_review → [route_after_human_review] → gap_detection | story_writer | END
        error_node → END

    Returns:
        A configured StateGraph ready for compilation.
    """
    builder = StateGraph(PipelineState)

    # Add nodes
    builder.add_node("parser", parser_node)
    builder.add_node("gap_detection", gap_detection_node)
    builder.add_node("story_writer", story_writer_node)
    builder.add_node("error_node", error_node)
    builder.add_node("llm_fallback_node", llm_fallback_node)
    builder.add_node("human_review", human_review_node)

    # Set entry point
    builder.set_entry_point("parser")

    # Conditional edges after parser
    builder.add_conditional_edges(
        "parser",
        route_after_parser,
        {
            "error_node": "error_node",
            "llm_fallback_node": "llm_fallback_node",
            "gap_detection": "gap_detection",
        },
    )

    # Conditional edges after gap detection
    builder.add_conditional_edges(
        "gap_detection",
        route_after_gap_detection,
        {
            "error_node": "error_node",
            "llm_fallback_node": "llm_fallback_node",
            "story_writer": "story_writer",
        },
    )

    # Conditional edges after story writer
    builder.add_conditional_edges(
        "story_writer",
        route_after_story_writer,
        {
            "error_node": "error_node",
            "llm_fallback_node": "llm_fallback_node",
            END: END,
        },
    )

    # Conditional edges after LLM fallback (the key fix — no longer just END)
    builder.add_conditional_edges(
        "llm_fallback_node",
        route_after_llm_fallback,
        {
            "parser": "parser",
            "gap_detection": "gap_detection",
            "story_writer": "story_writer",
            "human_review": "human_review",
            END: END,
        },
    )

    # Conditional edges after human review
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "gap_detection": "gap_detection",
            "story_writer": "story_writer",
            END: END,
        },
    )

    # error_node is terminal
    builder.add_edge("error_node", END)

    return builder


def compile_graph():
    """Build, compile, and return the runnable graph with checkpointer.

    The checkpointer enables:
    - State persistence between nodes
    - Resume after human_review interrupt
    - Session replay capability

    Returns:
        A compiled LangGraph runnable ready for invocation.
    """
    builder = build_graph()
    checkpointer = create_checkpointer()
    return builder.compile(checkpointer=checkpointer)
