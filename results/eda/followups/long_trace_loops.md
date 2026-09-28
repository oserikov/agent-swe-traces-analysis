# Long model-delta/run-12 traces: model looping vs. duplicated log entries

Targets are the 5 longest delta/run-12 traces. Baseline is rows 72 and 304, both at the job's median of 110 assistant turns.
The numbers come from `long_trace_loops.csv`. They were computed with Python aggregates only; no raw transcripts were read.

| row | task_id | status | asst turns | proxy reqs | identical consec. msgs | repeated call/result ids | bash repeat | longest identical-call run | tool-output repeat | onset (call idx, frac) |
|---|---|---|---|---|---|---|---|---|---|---|
| 369 | zope.interface-335 | OK | 861 | 863 | 0 | 0 / 0 | 0.439 | 2 | 0.441 | 338 (0.39) |
| 115 | django-axes-885 | OK | 812 | 814 | 0 | 0 / 0 | 0.636 | 2 | 0.382 | 23 (0.03) |
| 57 | curator-560 | TL | 845 | 845 | 0 | 0 / 0 | 0.394 | 5 | 0.235 | 523 (0.62) |
| 168 | icontract-236 | OK | 1213 | 1213 | 0 | 0 / 0 | 0.515 | 4 | 0.440 | 130 (0.11) |
| 263 | pygfx-121 | TL | 1786 | 1786 | 0 | 0 / 0 | 0.994 | 5 | 0.996 | 0 (0.00) |
| 72 (base) | feature_engine-531 | OK | 110 | 110 | 0 | 0 / 0 | 0.274 | 1 | 0.193 | never |
| 304 (base) | pyquil-119 | WA | 110 | 110 | 0 | 0 / 0 | 0.250 | 2 | 0.358 | never |

- **bash repeat:** the share of bash commands that exactly repeat an earlier command in the same trace.
- **onset:** the first call index where more than 50% of a 50-call sliding window repeats an earlier call (exact name and arguments).
- **reasoning:** delta writes non-empty `reasoning_content` on only 2–8 turns per trace, so repeated reasoning can't be tested. 0 of those strings repeat, except one repeat in row 369.

## Duplicated log entries: none, in all 7 traces
- There are 0 consecutive identical messages, 0 repeated tool-call IDs, and 0 tool results with a repeated `tool_call_id`.
- `n_proxy_requests` matches the number of assistant turns exactly, or is 2 higher (rows 369 and 115).
- So every logged assistant turn corresponds to a real model call. The length is real, not an artifact of logging.

## Verdict per row
- **263 (pygfx-121, TL): degenerate loop.** Two commands make up 1,777 of the 1,786 calls: `cat pygfx/materials/_base.py` ×947 and
  `python -c "print(open('pygfx/materials/_base.py').read())"` ×830. 99.6% of tool outputs repeat an earlier output.
  Repetition starts at call 0. The longest identical run is only 5, so the model *alternates* between two ways of reading the same file.
  It ran until the TL cutoff with an open tool call. This is the only true loop in the set.
- **115 (django-axes-885, OK): heavy re-running, starting early, but it converged.** The top commands are `pytest tests/` ×43, `git status` ×39,
  a pytest `-k` filter ×37, `cat -n ... | sed -n '75,85p'` ×34, and `git diff HEAD~1 HEAD` ×30. 64% of bash commands repeat, yet only 38% of outputs do.
  The same checks give changing results, which looks like an edit–test cycle rather than a stuck loop. It ended with a final answer and was graded OK.
- **168 (icontract-236, OK): mixed.** Repetition dominates from call 130 onward (`python /tmp/spam.py` ×71, `/tmp/test.py` ×35, mypy ×34).
  It still makes 132 edits and ends graded OK, though the log ends with an open tool call.
- **369 (zope.interface-335, OK): mixed, drifting late.** `pytest src/zope/interface/` ×73 and `git status` ×40. Repetition dominates after
  call 338 (39% of the way through). It ends cleanly with a final answer and was graded OK.
- **57 (curator-560, TL): late drift.** Test commands repeat ×28 and ×27, and there are Docker cleanup commands (`docker stop ... && docker rm ...` ×19).
  Repetition dominates only after call 523 (62%). It hit TL with an open tool call.

## Summary and uncertainty
- None of the long traces is a logging artifact. Only row 263 is a pure two-command loop.
- The other four are long edit–test cycles whose repeat rate (39–64% of bash commands) is 1.5–2.5× the baseline (25–27%).
  Three of the four were still graded OK.
- The "onset" threshold (>50% repeats in a 50-call window) is arbitrary. Exact-string matching misses near-duplicates, so repeat rates are lower bounds.
- An identical command with a different output (e.g. `pytest` after an edit) counts as a repeat here. That's why bash repeat is higher than tool-output repeat for rows 115 and 369.
