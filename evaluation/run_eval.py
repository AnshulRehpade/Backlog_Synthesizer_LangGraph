"""Evaluation CLI — runs the pipeline against golden datasets and reports quality metrics.

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --dataset evaluation/golden_dataset.json
    python -m evaluation.run_eval --verbose

Evaluates each golden entry by:
1. Running the full pipeline with the entry's transcript and backlog
2. Comparing generated output against expected counts
3. Computing keyword overlap between generated stories and transcript
4. Reporting per-entry results and aggregate statistics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from graph.builder import compile_graph
from graph.state import PipelineState
from backlog_synthesizer_langgraph.models.inputs import (
    BacklogTicket,
    DocumentType,
    InputDocument,
    SessionInputs,
)
from backlog_synthesizer_langgraph.models.evaluation import (
    EvalRunResult,
    EvalSummary,
    GoldenDatasetEntry,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path("evaluation/golden_dataset.json")


def _load_golden_dataset(path: Path) -> list[GoldenDatasetEntry]:
    """Load golden dataset from JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenDatasetEntry(**entry) for entry in data]


def _compute_keyword_overlap(transcript: str, stories_text: str) -> float:
    """Compute keyword overlap between transcript and generated stories.

    Uses simple word-level Jaccard similarity on significant words
    (length > 3, lowercased, deduplicated).
    """
    def _extract_keywords(text: str) -> set[str]:
        words = text.lower().split()
        return {w.strip(".,;:!?\"'()[]") for w in words if len(w) > 3}

    transcript_kw = _extract_keywords(transcript)
    stories_kw = _extract_keywords(stories_text)

    if not transcript_kw:
        return 0.0

    intersection = transcript_kw & stories_kw
    union = transcript_kw | stories_kw

    return len(intersection) / len(union) if union else 0.0


async def _run_single_entry(
    entry: GoldenDatasetEntry,
    graph,
    verbose: bool = False,
) -> EvalRunResult:
    """Run pipeline for a single golden entry and evaluate results."""
    # Build inputs
    content = entry.transcript.encode("utf-8")
    document = InputDocument(
        filename=f"eval_{entry.id}.txt",
        document_type=DocumentType.TRANSCRIPT_TXT,
        content=content,
        size_bytes=len(content),
    )

    backlog_tickets = [
        BacklogTicket(
            id=t["id"],
            title=t["title"],
            description=t.get("description", ""),
            status=t.get("status", "open"),
            tags=t.get("tags", []),
        )
        for t in entry.existing_backlog
    ]

    session_id = str(uuid.uuid4())
    inputs = SessionInputs(
        session_id=session_id,
        documents=[document],
        backlog_tickets=backlog_tickets,
    )

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

    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        return EvalRunResult(
            golden_id=entry.id,
            keyword_overlap=0.0,
            stories_generated=0,
            passed=False,
            failure_reason=f"Pipeline crashed: {e}",
        )

    # Extract results
    story_output = final_state.get("story_output")
    gap_output = final_state.get("gap_output")
    status = final_state.get("status", "unknown")

    stories_generated = 0
    stories_text = ""
    if story_output is not None:
        stories_generated = sum(len(e.stories) for e in story_output.epics)
        stories_text = " ".join(
            f"{s.title} {s.user_story}"
            for e in story_output.epics
            for s in e.stories
        )

    # Compute keyword overlap
    keyword_overlap = _compute_keyword_overlap(entry.transcript, stories_text)

    # Check pass/fail
    passed = True
    failure_reason = None

    if status in ("permanent_failure",):
        passed = False
        failure_reason = f"Pipeline ended with {status}"
    elif entry.expected_stories_count > 0 and stories_generated == 0:
        passed = False
        failure_reason = f"Expected {entry.expected_stories_count} stories, got 0"

    if verbose:
        status_icon = "✅" if passed else "❌"
        print(
            f"  {status_icon} {entry.id}: {stories_generated} stories, "
            f"overlap={keyword_overlap:.2f}, status={status}"
        )
        if failure_reason:
            print(f"     Reason: {failure_reason}")

    return EvalRunResult(
        golden_id=entry.id,
        keyword_overlap=keyword_overlap,
        stories_generated=stories_generated,
        passed=passed,
        failure_reason=failure_reason,
    )


async def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    verbose: bool = False,
) -> EvalSummary:
    """Run evaluation against the golden dataset.

    Args:
        dataset_path: Path to the golden dataset JSON file.
        verbose: Print per-entry results.

    Returns:
        EvalSummary with aggregate metrics.
    """
    entries = _load_golden_dataset(dataset_path)
    graph = compile_graph()

    if verbose:
        print(f"\nRunning evaluation: {len(entries)} golden entries")
        print("-" * 50)

    results: list[EvalRunResult] = []
    for entry in entries:
        result = await _run_single_entry(entry, graph, verbose=verbose)
        results.append(result)

    # Compute aggregate metrics
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    overlaps = [r.keyword_overlap for r in results]

    overlap_mean = sum(overlaps) / len(overlaps) if overlaps else 0.0
    overlap_min = min(overlaps) if overlaps else 0.0
    overlap_max = max(overlaps) if overlaps else 0.0

    if len(overlaps) > 1:
        variance = sum((x - overlap_mean) ** 2 for x in overlaps) / len(overlaps)
        overlap_std = math.sqrt(variance)
    else:
        overlap_std = 0.0

    success_rate = passed / len(results) if results else 0.0

    summary = EvalSummary(
        total_entries=len(results),
        passed=passed,
        failed=failed,
        keyword_overlap_mean=overlap_mean,
        keyword_overlap_std=overlap_std,
        keyword_overlap_min=overlap_min,
        keyword_overlap_max=overlap_max,
        success_rate=success_rate,
        results=results,
    )

    if verbose:
        print("-" * 50)
        print(f"\nResults: {passed}/{len(results)} passed ({success_rate:.0%})")
        print(f"Keyword overlap: mean={overlap_mean:.3f}, min={overlap_min:.3f}, max={overlap_max:.3f}")
        if failed > 0:
            print(f"\nFailed entries:")
            for r in results:
                if not r.passed:
                    print(f"  - {r.golden_id}: {r.failure_reason}")

    return summary


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Run evaluation against golden dataset"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to golden dataset JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-entry results",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write results as JSON to this path",
    )
    args = parser.parse_args()

    summary = asyncio.run(run_evaluation(args.dataset, verbose=args.verbose))

    if args.json_output:
        args.json_output.write_text(
            summary.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"\nResults written to {args.json_output}")

    # Exit with non-zero if any failures
    if summary.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
