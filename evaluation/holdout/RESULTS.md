# Held-out evaluation: before vs. after

Date: 2026-11-03 · Model: `claude-haiku-4-5` · Scorer: `evaluation/run_holdout_eval.py`

## Setup

Three meetings in domains the pipeline was never tuned on (e-commerce checkout, a fitness app, an internal data platform), each with a 10-ticket backlog. The answer key (`answer_key.json`) lists 20 pieces of work that should become stories, 10 of which must go to conflict review, plus 10 traps that should not become stories (5 duplicates, 4 bits of meeting talk or praise, 1 tentative idea). The labels were reviewed and committed (4c57090) before any run.

- **Before:** `4bcdb3f`, the code before the parser, gap detection, conflict review and story-merging changes.
- **After:** `main` with those changes.
- Each version ran twice. An LLM matcher (temperature 0) mapped every story and flagged conflict to the answer key; every mapping is printed in the run output and was checked by eye.

## Results (two runs each)

| | Before | After |
|---|---|---|
| Stories written (20 expected) | 40, 41 | **20, 20** |
| Pieces of work covered | 20/20, 20/20 | 20/20, 20/20 |
| Covered by exactly one story | 7/20, 6/20 | **20/20, 20/20** |
| Extra stories (meeting talk, tentative ideas) | 6, 5 | **0, 0** |
| Over-merged stories | 1, 1 | 0, 0 |
| Correct ticket link | n/a (no links) | **18/20, 18/20** |
| Required conflicts flagged | 7/10, 9/10 | **10/10, 10/10** |
| False alarms | 2, 1 | 1, 1 |
| Suggested resolution right | n/a (no suggestions) | **10/10, 10/10** |

Per meeting (second run): checkout 15 → 7 stories, fitness 13 → 7, data platform 13 → 6.

## What changed, in the output

- **Before**, most pieces of work became two or three stories. A problem and its fix were written separately ("Prevent overselling…" and "Replace post-payment stock reservation…"), and so was a request said twice. Meeting talk became stories ("Estimate and prioritize key sprint items", "Prioritize Linear items and share sprint plan"), and "Maybe evaluate Snowflake someday" became two stories.
- **Before**, the data platform meeting flagged only 2 of its 4 required conflicts. Email hashing and finance dashboard access were missed, because the contradicted detail ("email in plain text", "all employees can view") is a small part of the ticket text. And flagged conflicts went straight to stories: there was no review.
- **After**, every piece of work is one story linked to its ticket, all 10 required conflicts reach one review screen per meeting, and every suggested resolution is right.

## Remaining errors (after)

- **Two wrong ticket links, in both runs:** Klarna links to CHK-204 (Apple Pay) instead of no link or CHK-202 (card payments), and the chart timezone bug links to FIT-301 (workout logging) instead of FIT-307 (progress charts).
- **One false alarm, in both runs:** "Maybe evaluate Snowflake someday" reaches conflict review. Its suggestion is *keep the ticket*, so accepting it writes no story, but it is an unnecessary decision.

## Caveats

- Three meetings and 20 pieces of work is a small sample, and the meetings and labels were written by the same assistant that built the changes (the labels were reviewed by the project owner before any run).
- The before version's failures are mostly in areas the changes targeted, so this shows the changes work on new meetings. It doesn't show how they'd do on messier real transcripts.
