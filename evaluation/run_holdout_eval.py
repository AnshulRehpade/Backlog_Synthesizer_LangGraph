"""Score the full pipeline on the held-out meetings against the frozen answer key.

Usage:
    python evaluation/run_holdout_eval.py                 # this checkout
    python evaluation/run_holdout_eval.py --repo PATH     # another checkout (e.g. an older commit)

For each meeting in evaluation/holdout/answer_key.json the real pipeline runs
end to end (conflict suggestions are accepted automatically). One LLM call
(temperature 0) then maps every story, and every item flagged as a conflict,
to the answer key's pieces of work. Every mapping is printed so it can be
checked by eye.

Scores, summed over meetings:
    covered        expected stories that got at least one story
    one story each covered work with exactly one story (not split across several)
    extra          stories for work that shouldn't be a story, or for nothing
    over-merged    stories covering two or more pieces of work
    link           covered work whose story links the right ticket the right way
                   (n/a for versions without story links)
    conflicts      must-review work flagged as a conflict
    false alarms   conflict items for work that shouldn't reach review
    resolution     must-review conflicts whose suggested resolution is acceptable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path

KEY = Path(__file__).resolve().parent / "holdout" / "answer_key.json"

_MATCH_SYSTEM = (
    "You check a generated backlog against an answer key. For each numbered text "
    "(a user story S#, or a requirement flagged as a conflict C#), list the ids of "
    "the pieces of work it covers. A text can cover several pieces of work, or none "
    "if it matches nothing in the list. Judge by what the text is about, not wording. "
    'Answer with JSON only: {"S0": ["work-id", ...], "C0": [...], ...} with every '
    "number present."
)


def _load_pipeline(repo: Path):
    """Import run_pipeline and PipelineConfig from the given checkout."""
    os.chdir(repo)
    sys.path[:0] = [str(repo / "src"), str(repo)]
    from dotenv import load_dotenv

    load_dotenv(repo / ".env") if (repo / ".env").exists() else load_dotenv(Path.cwd() / ".env")
    from main import run_pipeline
    from backlog_synthesizer_langgraph.config import PipelineConfig
    import backlog_synthesizer_langgraph

    print(f"code: {Path(backlog_synthesizer_langgraph.__file__).parent}")
    return run_pipeline, PipelineConfig


def _reviewer(review: dict) -> str:
    """Accept conflict suggestions; approve a fallback's request for a person."""
    return "accept_all" if review.get("kind") == "conflict_review" else "approve"


async def _match(llm, work: list[dict], stories: list[str], conflicts: list[str]) -> dict[str, list[str]]:
    ids = {w["id"] for w in work}
    lines = ["Pieces of work:"] + [f"- {w['id']}: {w['what']}" for w in work]
    lines += ["", "Texts:"] + [f"S{i}: {s}" for i, s in enumerate(stories)]
    lines += [f"C{i}: {c}" for i, c in enumerate(conflicts)]
    raw = await asyncio.to_thread(llm.generate, "\n".join(lines), _MATCH_SYSTEM)
    found = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(found.group(0)) if found else {}
    return {k: [w for w in v if w in ids] for k, v in data.items() if isinstance(v, list)}


def _link_ok(work: dict, story) -> bool:
    ticket = getattr(story, "linked_ticket", None)
    link = getattr(story, "link_type", None)
    return ticket in work["tickets"] and link in work["link_types"]


async def _score_meeting(meeting: dict, run_pipeline, PipelineConfig) -> dict:
    os.environ["CHROMA_COLLECTION"] = f"holdout_{meeting['id']}_{uuid.uuid4().hex[:8]}"
    key_dir = KEY.parent.parent.parent
    state = await run_pipeline(
        str(key_dir / meeting["transcript"]), str(key_dir / meeting["backlog"]), ask_reviewer=_reviewer
    )
    story_output = state.get("story_output")
    stories = [s for e in story_output.epics for s in e.stories] if story_output else []
    gap = state.get("gap_output")
    conflict_entries = [e for e in (gap.entries if gap else []) if e.classification == "conflict"]

    llm = PipelineConfig.from_env().create_llm_tool(temperature=0.0)
    matches = await _match(
        llm,
        meeting["work"],
        [f"{s.title} — {s.user_story}" for s in stories],
        [e.item.text for e in conflict_entries],
    )

    work = {w["id"]: w for w in meeting["work"]}
    expected = [w for w in meeting["work"] if w["story"]]
    story_work = [matches.get(f"S{i}", []) for i in range(len(stories))]
    conflict_work = [matches.get(f"C{i}", []) for i in range(len(conflict_entries))]

    print(f"\n── {meeting['id']}: {len(stories)} stories, {len(conflict_entries)} conflicts flagged")
    for s, ws in zip(stories, story_work):
        link = f"{getattr(s, 'link_type', None) or '-'} {getattr(s, 'linked_ticket', None) or ''}".strip()
        print(f"   S {s.title[:60]:60} [{link:22}] → {', '.join(ws) or 'nothing'}")
    for e, ws in zip(conflict_entries, conflict_work):
        suggestion = getattr(e.conflict_info, "suggested_resolution", None) or "-"
        print(f"   C {e.item.text[:60]:60} [{suggestion:12}] → {', '.join(ws) or 'nothing'}")

    stories_per_work = {w["id"]: [s for s, ws in zip(stories, story_work) if w["id"] in ws] for w in expected}
    must = [w for w in expected if w.get("conflict") == "must"]
    flagged = {w for ws in conflict_work for w in ws}
    resolution_ok = 0
    for w in must:
        for e, ws in zip(conflict_entries, conflict_work):
            if w["id"] in ws:
                resolution_ok += getattr(e.conflict_info, "suggested_resolution", None) in w["resolutions"]
                break
    return {
        "expected": len(expected),
        "covered": sum(1 for w in expected if stories_per_work[w["id"]]),
        "one_each": sum(1 for w in expected if len(stories_per_work[w["id"]]) == 1),
        "extra": sum(1 for ws in story_work if not any(work[w]["story"] for w in ws)),
        "over_merged": sum(1 for ws in story_work if sum(work[w]["story"] for w in ws) >= 2),
        "link": sum(1 for w in expected if stories_per_work[w["id"]] and _link_ok(w, stories_per_work[w["id"]][0])),
        "has_links": any(hasattr(s, "linked_ticket") for s in stories),
        "must": len(must),
        "caught": sum(1 for w in must if w["id"] in flagged),
        "false_alarms": sum(1 for ws in conflict_work if not ws or all(work[w].get("conflict", "no") == "no" for w in ws)),
        "resolution_ok": resolution_ok,
        "stories": len(stories),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=KEY.parent.parent.parent)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    key = json.loads(KEY.read_text(encoding="utf-8"))
    run_pipeline, PipelineConfig = _load_pipeline(args.repo.resolve())

    results = {m["id"]: await _score_meeting(m, run_pipeline, PipelineConfig) for m in key["meetings"]}
    total = {k: sum(r[k] for r in results.values()) for k in results[next(iter(results))] if k != "has_links"}
    links = f"{total['link']}/{total['expected']}" if any(r["has_links"] for r in results.values()) else "n/a"
    print(
        f"\nTOTAL stories {total['stories']} | covered {total['covered']}/{total['expected']} | "
        f"one story each {total['one_each']}/{total['expected']} | extra {total['extra']} | "
        f"over-merged {total['over_merged']} | link {links} | conflicts {total['caught']}/{total['must']} | "
        f"false alarms {total['false_alarms']} | resolution {total['resolution_ok']}/{total['must']}"
    )
    if args.json_output:
        args.json_output.write_text(json.dumps({"meetings": results, "total": total}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
