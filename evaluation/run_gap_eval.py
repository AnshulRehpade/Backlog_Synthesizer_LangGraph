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
    - grouping: items merged into one story vs. the labeled pieces of work
      (pair precision: merged pairs that are the same work; pair recall: same-work
      pairs that were merged)
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
    print(_grouping_summary(labels, entries))
    print(
        f"\nverdicts {verdict_ok}/{len(labels)} | conflicts caught {caught}/{must_conflict} | "
        f"false alarms {false_alarms} | sent to review {review} | ticket {ticket_ok}/{ticket_total} | "
        f"resolution {res_ok}/{res_total}"
    )


def _grouping_summary(labels: list[dict], entries) -> str:
    from itertools import combinations

    story_items = [i for i, e in enumerate(entries) if e.group is not None]
    groups: dict[int, list[int]] = {}
    for i in story_items:
        groups.setdefault(entries[i].group, []).append(i)
    lines = ["\ngroups:"]
    for members in groups.values():
        works = {labels[i]["work"] for i in members}
        mark = "✅" if len(works) == 1 else "❌"
        want = sorted({labels[i]["ticket"] for i in members} - {None}) or ["none"]
        lines.append(
            f"  {mark} [ticket {entries[members[0]].group_ticket or 'none'}, want {'/'.join(want)}] "
            f"{', '.join(labels[i]['text'][:38] for i in members)}"
        )
    # A group's story ticket is right if it's one its items are labeled with. With no
    # labeled ticket, none is right, and so is any ticket if an item may be "related".
    ticket_ok = 0
    for members in groups.values():
        want = {labels[i]["ticket"] for i in members} - {None}
        got = entries[members[0]].group_ticket
        if want:
            ticket_ok += got in want
        else:
            ticket_ok += got is None or any("related" in labels[i]["verdicts"] for i in members)
    merged = {frozenset(p) for g in groups.values() for p in combinations(g, 2)}
    same = {frozenset(p) for p in combinations(story_items, 2) if labels[p[0]]["work"] == labels[p[1]]["work"]}
    precision = f"{len(merged & same)}/{len(merged)}" if merged else "-"
    recall = f"{len(merged & same)}/{len(same)}" if same else "-"
    work_units = len({labels[i]["work"] for i in story_items})
    lines.append(
        f"stories {len(groups)} (labeled pieces of work {work_units}) | "
        f"pair precision {precision} | pair recall {recall} | group ticket {ticket_ok}/{len(groups)}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
