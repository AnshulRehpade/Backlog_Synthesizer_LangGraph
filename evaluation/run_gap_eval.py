"""Score gap detection item by item against labeled items from the sample meeting.

Usage:
    python evaluation/run_gap_eval.py

Runs the real gap_detection_node (embeddings, vector store, LLM judge) on the
items in evaluation/gap_labels.json against data/sample_backlog.json and reports:
    - verdicts: items whose verdict is one of the acceptable ones
    - conflicts caught: items that must be conflicts and were flagged as conflicts
    - false alarms: items sent to conflict review that aren't conflicts
    - ticket: items linked to a ticket that point to the right one
    - resolution: conflicts whose suggested resolution is acceptable
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import graph.builder  # noqa: F401  (nodes.* must be imported after graph)
from nodes.gap_detection_node import gap_detection_node
from backlog_synthesizer_langgraph.models.extraction import ExtractedItem, ExtractionResult
from backlog_synthesizer_langgraph.models.inputs import BacklogTicket, SessionInputs

LABELS = Path("evaluation/gap_labels.json")
BACKLOG = Path("data/sample_backlog.json")


def _tickets() -> list[BacklogTicket]:
    raw = json.loads(BACKLOG.read_text(encoding="utf-8"))
    raw = raw if isinstance(raw, list) else raw.get("tickets", raw)
    return [BacklogTicket(**{k: t[k] for k in ("id", "title", "description", "status", "tags") if k in t}) for t in raw]


async def main() -> None:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["items"]
    os.environ["CHROMA_COLLECTION"] = f"gap_eval_{uuid.uuid4().hex[:8]}"
    items = [
        ExtractedItem(item_type="feature_request", text=l["text"], source_chunk_index=0, confidence=0.9)
        for l in labels
    ]
    state = {
        "inputs": SessionInputs(session_id="gap-eval", documents=[], backlog_tickets=_tickets()),
        "session_id": "gap-eval", "react_notes": [], "errors": [], "total_duration_ms": 0,
        "parser_output": ExtractionResult(items=items, errors=[], metadata={}),
    }
    entries = (await gap_detection_node(state))["gap_output"].entries

    verdict_ok = caught = must_conflict = false_alarms = ticket_ok = ticket_total = 0
    res_ok = res_total = 0
    for label, entry in zip(labels, entries):
        got = entry.classification
        ok = got in label["verdicts"]
        verdict_ok += ok
        if label["verdicts"] == ["conflict"]:
            must_conflict += 1
            caught += got == "conflict"
        if got == "conflict" and "conflict" not in label["verdicts"]:
            false_alarms += 1
        if got in ("duplicate", "conflict", "related") and label["ticket"]:
            ticket_total += 1
            ticket_ok += entry.similar_ticket_id == label["ticket"]
        resolution = ""
        if got == "conflict" and label.get("resolution"):
            res_total += 1
            resolution = entry.conflict_info.suggested_resolution
            res_ok += resolution in label["resolution"]
        print(
            f"{'✅' if ok else '❌'} {got:9} {entry.similar_ticket_id or '-':8} "
            f"{entry.similarity_score:.2f} {resolution:12} {label['text'][:70]}"
            + ("" if ok else f"   (want {'/'.join(label['verdicts'])})")
        )

    review = sum(e.classification == "conflict" for e in entries)
    print(
        f"\nverdicts {verdict_ok}/{len(labels)} | conflicts caught {caught}/{must_conflict} | "
        f"false alarms {false_alarms} | sent to review {review} | ticket {ticket_ok}/{ticket_total} | "
        f"resolution {res_ok}/{res_total}"
    )


if __name__ == "__main__":
    asyncio.run(main())
