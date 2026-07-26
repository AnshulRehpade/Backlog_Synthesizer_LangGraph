"""Main entry point for the Backlog Synthesizer LangGraph pipeline.

Usage:
    python main.py transcript.txt
    python main.py transcript.txt --backlog backlog.json
    python main.py transcript.pdf --backlog backlog.json

Loads documents and optional backlog tickets, builds initial PipelineState,
runs the graph with session_id as thread_id for the checkpointer, and
prints the final result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env file before any config reads

sys.path.insert(0, ".")

from graph.builder import compile_graph
from graph.state import PipelineState
from backlog_synthesizer_langgraph.models.inputs import (
    BacklogTicket,
    DocumentType,
    InputDocument,
    SessionInputs,
)


def _detect_document_type(filepath: Path) -> DocumentType:
    """Detect document type from file extension.

    Args:
        filepath: Path to the input file.

    Returns:
        The appropriate DocumentType enum value.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = filepath.suffix.lower()
    mapping = {
        ".txt": DocumentType.TRANSCRIPT_TXT,
        ".md": DocumentType.TRANSCRIPT_MD,
        ".pdf": DocumentType.TRANSCRIPT_PDF,
        ".html": DocumentType.ARCHITECTURE_HTML,
        ".json": DocumentType.BACKLOG_JSON,
    }
    if suffix not in mapping:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported: {', '.join(mapping.keys())}"
        )
    return mapping[suffix]


def _load_document(filepath: Path) -> InputDocument:
    """Load a document from disk into an InputDocument model.

    Args:
        filepath: Path to the document file.

    Returns:
        An InputDocument instance with content loaded.
    """
    content = filepath.read_bytes()
    doc_type = _detect_document_type(filepath)
    return InputDocument(
        filename=filepath.name,
        document_type=doc_type,
        content=content,
        size_bytes=len(content),
    )


def _load_backlog(filepath: Path) -> list[BacklogTicket]:
    """Load backlog tickets from a JSON file.

    Expected format: a JSON array of objects with fields:
    id, title, description, status, tags (optional).

    Args:
        filepath: Path to the backlog JSON file.

    Returns:
        List of BacklogTicket instances.
    """
    content = filepath.read_text(encoding="utf-8")
    data = json.loads(content)

    if not isinstance(data, list):
        raise ValueError("Backlog JSON must be a JSON array of ticket objects")

    tickets = []
    for item in data:
        tickets.append(BacklogTicket(
            id=item["id"],
            title=item["title"],
            description=item.get("description", ""),
            status=item.get("status", "open"),
            tags=item.get("tags", []),
        ))
    return tickets


async def run_pipeline(
    transcript_path: str,
    backlog_path: str | None = None,
) -> dict:
    """Run the full Backlog Synthesizer pipeline.

    Args:
        transcript_path: Path to the transcript/document to process.
        backlog_path: Optional path to existing backlog tickets (JSON).

    Returns:
        Final pipeline state as a dict.
    """
    # Load documents
    doc_path = Path(transcript_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {transcript_path}")

    document = _load_document(doc_path)

    # Load optional backlog
    backlog_tickets = []
    if backlog_path:
        bl_path = Path(backlog_path)
        if not bl_path.exists():
            raise FileNotFoundError(f"Backlog file not found: {backlog_path}")
        backlog_tickets = _load_backlog(bl_path)

    # Build session
    session_id = str(uuid.uuid4())

    inputs = SessionInputs(
        session_id=session_id,
        documents=[document],
        backlog_tickets=backlog_tickets,
    )

    # Build initial state
    initial_state: PipelineState = {
        "inputs": inputs,
        "session_id": session_id,
        "parser_output": None,
        "gap_output": None,
        "story_output": None,
        "current_step": "start",
        "status": "in_progress",
        "errors": [],
        "react_notes": [],
        "retry_count": 0,
        "needs_human_review": False,
        "trace_id": str(uuid.uuid4()),
        "total_duration_ms": 0,
    }

    # Compile and run graph
    graph = compile_graph()
    config = {"configurable": {"thread_id": session_id}}

    final_state = await graph.ainvoke(initial_state, config=config)

    return final_state


async def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backlog Synthesizer — extract user stories from transcripts"
    )
    parser.add_argument(
        "transcript",
        help="Path to transcript file (.txt, .md, .pdf)",
    )
    parser.add_argument(
        "--backlog",
        help="Path to existing backlog tickets (.json)",
        default=None,
    )
    args = parser.parse_args()

    try:
        result = await run_pipeline(args.transcript, args.backlog)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    status = result.get("status", "unknown")
    print(f"\nPipeline completed with status: {status}")
    print(f"Session ID: {result.get('session_id', 'unknown')}")
    print(f"Total duration: {result.get('total_duration_ms', 0)}ms")

    # Print story output summary if available
    story_output = result.get("story_output")
    if story_output is not None:
        epic_count = len(story_output.epics)
        story_count = sum(len(e.stories) for e in story_output.epics)
        print(f"\nGenerated {story_count} stories in {epic_count} epics:")
        for epic in story_output.epics:
            print(f"  Epic: {epic.epic_title} ({len(epic.stories)} stories)")
            for story in epic.stories:
                refinement = " [needs refinement]" if story.needs_refinement else ""
                print(f"    - {story.title}{refinement}")
    else:
        print("\nNo stories generated.")

    # Print errors if any
    errors = result.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            if isinstance(err, dict):
                print(f"  [{err.get('node', '?')}] {err.get('message', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
