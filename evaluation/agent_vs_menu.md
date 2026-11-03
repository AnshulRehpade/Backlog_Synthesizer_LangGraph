# Fallback agent vs. menu: evaluation results

Date: 2026-11-03 · Model: `claude-haiku-4-5` · Branch: `feat/autonomous-fallback-agent`

## Question

The LLM fallback node can pick a recovery action in two ways:

- **menu**: one LLM call picks an action from a state summary.
- **agent**: a bounded tool-calling loop investigates first (lists items and matches, reads tickets, asks the judge, can correct labels), then picks from the same menu.

Is the agent worth its extra cost?

## Method

- Golden dataset: the original 5 cases plus 3 added for this experiment (eval-006 to eval-008), designed to reach the fallback. The new cases were written after the agent was built, so they are reported separately.
- Each mode was run twice (`python evaluation/run_eval.py --fallback-mode menu|agent`), because LLM output varies between runs.
- Success criteria were set before running:

| Criterion | Pass if |
|---|---|
| Original 5 | Agent passes at least as many as menu |
| Hidden conflict (eval-007) | Agent passes both runs, menu fails |
| Restraint | Agent makes no wrong reclassifications |
| Cost | Under ~15k agent tokens per fallback on average, never above the 30k cap |

## Results

`*` = the run reached the fallback.

| Case | menu 1 | menu 2 | agent 1 | agent 2 |
|---|---|---|---|---|
| eval-001 | PASS | PASS | PASS | PASS |
| eval-002 | PASS* | PASS* | PASS* | PASS* |
| eval-003 | PASS* | PASS* | PASS* | **FAIL*** |
| eval-004 | PASS* | PASS* | PASS* | PASS* |
| eval-005 | PASS | PASS | PASS | PASS |
| eval-006 (new) | PASS | PASS | PASS* | PASS* |
| eval-007 (new) | FAIL | FAIL | FAIL | FAIL |
| eval-008 (new) | PASS* | PASS* | PASS* | PASS* |
| **Total** | 7/8 | 7/8 | 7/8 | 6/8 |

| | menu | agent |
|---|---|---|
| Fallbacks per run | 5 | 6 |
| Agent tokens per fallback | n/a | mean ~5k, max 9.3k |
| Reclassifications | n/a | 0 |
| Eval wall time per run | ~70 s | ~102 s (+45%) |

## Against the criteria

| Criterion | Result |
|---|---|
| Original 5 | **Not met.** Menu 10/10, agent 9/10 |
| Hidden conflict | **Not testable.** eval-007 never reached the fallback (see below) |
| Restraint | **Met.** 0 reclassifications. On eval-008 the agent read every ticket, asked the judge twice, and left the labels alone |
| Cost | **Met.** Mean ~5k tokens, max 9.3k |

## What the runs showed

1. **When both modes reached the same fallback, they chose the same action every time except once.** Parser fallbacks went to `proceed_with_low_confidence` and all-duplicate cases to `return_nothing_new`. The agent's investigation confirmed what the summary already implied.
2. **The one difference was worse for the agent.** On eval-003 run 2 it chose `retry_parser` on a clear 2-item transcript. The retry again produced 2 items, and with the retry already used the routing rule sends the run to the error node. **On a legitimately short transcript, `retry_parser` always ends in failure**, whichever mode picks it.
3. **The parser rule "fewer than 3 items → fallback" fires on normal short transcripts.** It triggered in 3 of the 5 original cases, all of which had clear, high-confidence items. Most fallback traffic is this rule, not real trouble.
4. **eval-007 was written wrong, not answered wrong.** The parser splits "run in the background and handle up to one million rows" into a feature plus a constraint, and both are flagged as conflicts. The expected counts (3 duplicates, 1 conflict) don't match how the parser extracts. The case never reached the fallback in any run.
5. **Whether eval-006 reaches the fallback varies.** It depends on how many items the parser extracts: 6 items is 83% duplicates (fallback), 7 items is 71% (no fallback). Both modes still passed.
6. **The gap detector's LLM check already fixes most labels**, so the agent found nothing to correct. Its main reason to exist, fixing labels, rarely comes up after that check.

## Recommendation

- **Make `menu` the default and `agent` opt-in.** On this data the agent adds about 45% runtime and about 5k tokens per fallback, with no accuracy gain and one regression.
- **Fix the retry trap** for both modes: don't offer `retry_parser` when the extracted items have high confidence, or stop sending clear short transcripts to the fallback at all.
- **Fix eval-007** so the CSV requirement is one atomic statement, then re-run the comparison. That's the case where the agent's ability to correct labels should matter.
- With 8 cases and 2 runs each, this is a small sample. It shows the agent isn't clearly better here, not that it never is.

## Changes made after this evaluation

- `FALLBACK_MODE` now defaults to `menu`. Run with `--fallback-mode agent` (or `FALLBACK_MODE=agent`) to use the agent.
- Retry trap fixed in `route_after_parser`: after a parser retry, fewer than 3 items continue to `gap_detection` unless all are low confidence. Zero items, or all low-confidence items, still go to `error_node`. Covered by four routing tests.
- Re-run after the changes (one run per mode): menu 7/8, agent 7/8, with eval-007 failing in both as before. The agent didn't choose `retry_parser` in this run, so the retry fix is verified by the tests, not by this run.
- eval-007 has since been rewritten so its CSV requirement is one statement ("run as a background job instead of inside the request"); expected counts unchanged.
- A second retry trap, in `route_after_gap_detection`, was found when merging into main: after a retry, a run that was over 80% duplicates but still had new items went to `error_node` (eval-006 failed when the menu chose `lower_threshold`). It now continues to the story writer.
- After merging with the later gap detection and story changes, both modes pass 8/8.
