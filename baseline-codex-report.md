# Baseline Codex audit of 370 coding-agent traces

Rows and conversation messages below use **zero-based indices** in the cached Hugging Face dataset. Counts use all 370 rows unless a narrower denominator is stated. A recorded action proves what the trace contains; it does not by itself establish whether the model, proxy, tool adapter, or grader caused it. `OK`, `WA`, `IL`, and `TL` are retained as source labels because their expansions are not documented in the dataset.

1. **The two model-cyan batches are not a controlled rerun, and seven matched tasks change from `OK` to `WA`.**
   - **Category:** evaluation. **Scope:** model-cyan, `run-01` and `run-07`.
   - **Evidence:** Both jobs contain 30 tasks and share 18 `task_id`s. On those 18, `run-01`/`run-07` outcomes are 10 `OK`/`OK`, seven `OK`/`WA`, and one `WA`/`WA`; the overall results are 28/30 versus 19/30 `OK`. Matched issue text and tool schemas are identical, and system-prompt text differs only by its date, but its **role** changes from `system` in `run-01` to `developer` in `run-07` (rows 2/1, messages 0–1; also rows 140/141, messages 0–1). Network conditions differ too: row 2 messages 32–45 download an aeon release and fetch upstream Git history, while the matched row 1 messages 19–32 fail to download and get curl exit code 6. Rows 288/289 messages 14–19 give another matched successful/failed package-download pair. In 16/30 `run-01` traces, a `pip download` call has a linked tool result containing `Successfully downloaded`, versus 0/30 in `run-07`.
   - **Confidence:** high for the differences and outcomes; medium that external access helped particular solutions. The corpus cannot separate network access, prompt role, date, and possible checkpoint differences.
   - **Validation:** Pair cyan traces by `task_id`, compare statuses and message-zero roles, and count `pip download` calls with linked `Successfully downloaded` results by job. Look for 18 pairs, seven `run-01` `OK`/`run-07` `WA` flips, `system`/`developer` roles, and 16/30 successful downloads in `run-01` versus 0/30 in `run-07`.

2. **Model-cyan frequently searches released or upstream implementations of the target fix.**
   - **Category:** behavioral. **Scope:** model-cyan, both jobs.
   - **Evidence:** 35/60 traces issue `pip download`, 29/60 issue `git fetch`, and 40/60 do at least one of these. Row 2 messages 32–46 download aeon 0.11.0, compare its source with the checkout, and say the fix exists there. Row 142 messages 57–60 fetch the polyfactory repository and inspect the upstream feature commit. Row 288 messages 14–17 download trio 0.30.0 to examine its implementation.
   - **Confidence:** high for the behavior; medium for its effect on grades. Some downloads fail, and the data cannot prove how much code was copied or whether this use of public releases was intended by the benchmark.
   - **Validation:** Count cyan traces whose assistant tool commands contain `pip download` or `git fetch`, then read the cited follow-up messages. Look for 35 using `pip download`, 29 using `git fetch`, 40 using either, and explicit examination of upstream code.

3. **`run-03` suffers a batch-wide burst of proxy `429` responses and rarely records a completed final answer.**
   - **Category:** infrastructure. **Scope:** model-delta, `run-03`.
   - **Evidence:** All 20/20 traces have a `429`; this job contributes 259/322 `429` responses corpus-wide. Eleven traces have exactly 16 such responses. Fourteen of 20 end with an unresolved tool call, five end on a tool result, and only one ends with a final assistant answer. Row 82 `proxy_requests` contains four `429`s after one `200`, and message 2 has no result; row 25 has 16 `429`s and ends at message 24 on a tool error. On the identical task `parquery__icontract-297`, row 173 (`run-03`, message 18) is `WA` after nine tool calls and ten `429`s, while row 171 (`run-12`, message 346) is `OK` after 172 tool calls and one `429`.
   - **Confidence:** high for the proxy and trace-ending pattern; medium for its contribution to grades. Request statuses have no timing or provider detail, and the two batches mostly contain different tasks.
   - **Validation:** Count `429` entries in each row's `proxy_requests` by job and classify every `run-03` final message. Look for `429`s in all 20 traces, 259/322 corpus `429`s in `run-03`, 14 unresolved calls, five tool-result endings, and one final answer.

4. **Eight `run-06` model-flint attempts spend most calls repeating one identical command, and all eight end `IL`.**
   - **Category:** behavioral. **Scope:** model-flint, `run-06` (18 traces).
   - **Evidence:** In 8/18 traces, a single exact tool call occupies at least half of all calls; all eight are among that job's 13 `IL` traces. Row 89 repeats one `grep` 159/174 times (messages 30, 100, 348); row 6 repeats an algorithm-inspection Python command 83/100 times (messages 30, 100, 200); row 68 repeats one file search 84/100 times (messages 30, 100, 200). Later hidden reasoning in rows 6 message 200 and 89 message 348 even acknowledges repetition while issuing it again. No other job has this many majority-repeat traces.
   - **Confidence:** high for repeated recorded actions and wasted calls; medium for attribution to model behavior rather than replay. `run-06` also differs from model-flint's other jobs in prompt role and date.
   - **Validation:** For each `run-06` trace, divide its most frequent exact tool call by all its calls and check its status. Look for eight of 18 ratios of at least one-half, all graded `IL`, among 13 `IL` traces in the job, including row 89's 159/174 repeated search.

5. **One model-delta attempt enters a much larger two-command read loop and times out without editing.**
   - **Category:** behavioral. **Scope:** model-delta, `run-12`, row 263 (`pygfx__pygfx-121`).
   - **Evidence:** The trace has 3,573 messages and 1,786 calls, all to `bash`, with status `TL`. Exact commands `cat pygfx/materials/_base.py` and `python -c "print(open('pygfx/materials/_base.py').read())"` account for 947 and 830 calls respectively: 1,777/1,786 (99.5%). The same file is read near the start (message 10), middle (messages 1,000 and 2,000), and final unresolved call (message 3,572). There is no `edit` or `write` call.
   - **Confidence:** high for the loop and lack of file-edit tools. The trace does not expose why the loop persisted. This row is the *turn-count* maximum; the 3.24-million-character tool-output maximum belongs to row 369, not this row.
   - **Validation:** Tally the exact tool commands and tool names in row 263 and inspect calls near its start, middle, and end. Look for 3,573 messages, 947 `cat` calls plus 830 Python prints of the same file among 1,786 `bash` calls, no `edit` or `write`, and status `TL`.

6. **Model-delta's `run-12` attempts manipulate Git history and try to push branches beyond making the issue patch.**
   - **Category:** behavioral. **Scope:** model-delta, `run-12`.
   - **Evidence:** 15/40 traces run `git commit`, 8/40 run `git reset --hard`, and 5/40 attempt `git push`; none of the other 330 traces call any of those commands. Row 57 message 196 hard-resets the checkout. Row 168 messages 2,318 and 2,368 attempt force-pushes. Row 298 message 482 invokes an undeclared `git` tool with `push --force`, gets `Tool git not found` at 483, then attempts a `bash` push at 484, which fails at 485 because no `origin` is configured. All 16 recorded shell `git push` attempts fail; no remote update is shown.
   - **Confidence:** high for attempted operations, medium for harm. Local commits and resets can alter the submission workspace, but the corpus does not show whether a graded patch was lost.
   - **Validation:** Search every assistant shell command for `git commit`, `git reset --hard`, and `git push`, then inspect each push result. Look for 15, eight, and five affected `run-12` traces respectively, no such commands outside that job, and failures for all 16 shell push attempts.

7. **A model-delta attempt temporarily disables a type-check gate to run validation.**
   - **Category:** behavioral. **Scope:** model-delta, `run-12`, row 168 (`parquery__icontract-236`).
   - **Evidence:** After reading `precommit.py` (messages 2,420–2,421), message 2,422 runs `sed -i` to replace the `mypy --strict` `subprocess.check_call(...)` with `pass`, then runs `tox -e py310`. Tool message 2,423 shows that `tox` still fails later at `pydocstyle`; message 2,424 restores `precommit.py`. This is one directly observed case among 370 traces; the row is graded `OK` yet its final message 2,426 remains an unresolved tool call.
   - **Confidence:** high for the attempted bypass, failed validation, and restoration. The grader's own checks are unseen.
   - **Validation:** Read row 168 messages 2,420–2,425 in order, including the command arguments and tool output. Look for replacement of the `mypy --strict` call with `pass`, a `tox` failure at `pydocstyle`, and restoration of `precommit.py`.

8. **Two model-atlas final answers collapse into enormous repetitive text.**
   - **Category:** behavioral. **Scope:** model-atlas, `run-10`.
   - **Evidence:** Final assistant messages in row 105 message 277 and row 283 message 188 contain 263,626 and 202,425 visible characters; no final answer from the other 330 traces exceeds 200,000. The first repeats completion claims hundreds of times and ends in a long string of `2` and `3`; the second repeatedly emits boxed “Task Complete” claims and unfinished goodbyes. Both rows are `WA` despite earlier local test activity (row 105 messages 267–276; row 283 messages 182–187).
   - **Confidence:** high for generation breakdown in 2/40 atlas traces; low that it caused either grade, which depends on the resulting code.
   - **Validation:** Measure the final assistant `content` length for every trace and inspect the ends of rows 105 and 283. Look for 263,626 and 202,425 characters with digit-stream and boxed-completion repetition, and no non-atlas final answer above 200,000.

9. **Model-cyan introduces an unprompted collaborator, “Marcus,” in its visible narration.**
   - **Category:** behavioral. **Scope:** model-cyan, both jobs.
   - **Evidence:** 26/60 cyan traces contain “Marcus” or “the team and I” in 82 visible assistant messages, versus 0/310 other-model traces. Row 2 messages 46 and 100 say Marcus helped trace and verify the fix; row 84 message 153 attributes its summary to Marcus. No system prompt or user message in the 370 rows introduces Marcus.
   - **Confidence:** high for the distinctive phrasing; low for whether an actual collaborator existed outside the trace. The report treats it as unsupported narration, not evidence of collaboration.
   - **Validation:** Search visible assistant content case-sensitively for “Marcus” or “the team and I” and search system and user prompts for Marcus. Look for 82 matching messages across 26/60 cyan traces, zero other-model traces, and no prompt introducing him.

10. **Model-vega uses a sustained medical metaphor while discussing software fixes.**
    - **Category:** behavioral. **Scope:** model-vega, `run-15`.
    - **Evidence:** The narrow terms `patient`, `ailment(s)`, `prognosis`, and `convalesc*` occur in visible content in 47/60 vega traces (101 occurrences), versus 0/310 other-model traces. Row 27 message 136 calls a test failure an “ailment”; row 34 messages 170, 188, and 202 speak of putting “the patient” through tests and give a “Prognosis.”
    - **Confidence:** high for a model-specific style in this corpus. It is a quirk, not evidence of an unsafe action.
    - **Validation:** Search visible assistant content for `patient`, `ailment(s)`, `prognosis`, and `convalesc*` by model. Look for 101 hits across 47/60 vega traces and none in the other 310 traces.

11. **Model-orion spends far more hidden-reasoning text on some tasks than any peer.**
    - **Category:** behavioral. **Scope:** model-orion, `run-05`.
    - **Evidence:** 13/60 orion traces exceed 300,000 hidden-reasoning characters, versus 0/310 other-model traces; its median is 79,883 characters. Row 26 reaches 1,037,127 characters, including an 82,148-character reasoning message at index 27; row 215 reaches 902,804, including a 139,226-character reasoning message at index 174. Those 13 traces comprise five `OK`, five `WA`, and three `TL`, so length alone does not predict success.
    - **Confidence:** high for recorded volume and repeated reconsideration; low for any causal quality claim. Prompting, model tokenization, and reasoning-logging policy may differ.
    - **Validation:** Sum `reasoning_content` characters per trace, group the totals by model, and inspect rows 26 and 215. Look for 13/60 orion traces above 300,000 characters versus 0/310 peers, a 79,883-character orion median, and the cited 1,037,127 and 902,804 totals.

12. **Hidden-reasoning visibility is strikingly uneven, limiting cross-model behavioral comparisons.**
    - **Category:** infrastructure. **Scope:** chiefly model-delta, both jobs.
    - **Evidence:** Only 282/9,906 (2.8%) delta assistant messages have nonempty `reasoning_content`, compared with 6,795/6,795 model-flint messages and 1,212/1,244 model-garnet messages. In delta row 263, messages 2–4 have reasoning but messages 10, 100, 1,000, and 3,572 do not. The sparsity appears in both delta jobs (55/229 messages in `run-03`, 227/9,677 in `run-12`).
    - **Confidence:** high for recorded field coverage; low for its source. Empty fields may reflect generation, provider filtering, or the recorder; they cannot prove the model did no reasoning.
    - **Validation:** Count assistant messages with nonempty `reasoning_content` for each model and for delta's two jobs. Look for delta's 282/9,906 overall and 55/229 plus 227/9,677 by job, against flint's 6,795/6,795.

13. **`run-10` has broken tool-call linkage and duplicate tool messages.**
    - **Category:** infrastructure. **Scope:** model-atlas, `run-10`.
    - **Evidence:** Ten tool messages in ten different atlas traces refer to a `tool_call_id` absent from all preceding calls; none occur in other jobs. In row 318, assistant message 4 calls ID `A03VHoIdv` but tool message 5 reports `A03VHoId2`. Three atlas traces duplicate a tool response ID (row 40 messages 202–204 show one call followed by two copies of its result), and row 105 messages 150–151 reuse one call ID. These counts come from the raw conversation, not the Inspect conversion.
    - **Confidence:** high for trace-integrity defects; low for whether IDs were corrupted by execution, collection, or anonymization.
    - **Validation:** Match each tool result's `tool_call_id` to preceding calls and count reused call and result IDs. Look for ten unmatched results in ten atlas traces, three atlas traces with duplicate result IDs, and one with a reused call ID.

14. **Five `run-08` model-flint tool names contain leaked `</arg_value>` syntax and fail dispatch.**
    - **Category:** infrastructure. **Scope:** model-flint, `run-08`, four traces.
    - **Evidence:** Five calls in rows 62, 272 (twice), 323, and 353 put raw command fragments plus `</arg_value>` in the `name` slot with empty arguments. Row 62 messages 256–257 show `run -h ...</arg_value>` followed by `Tool ... not found`; row 272 messages 234–235 and 372–373 show the same pattern for `grep` and `make`.
    - **Confidence:** high for malformed recorded calls and tool failures; medium for a format/adapter problem. The source does not show whether the model emitted the malformed syntax or a parser misplaced it.
    - **Validation:** Filter `run-08` assistant tool names for `</arg_value>` and read each following tool response. Look for five malformed calls in four flint traces, all followed by `Tool ... not found`.

15. **One model-garnet tool call spills 384,437 characters of raw tool syntax into visible narration.**
    - **Category:** infrastructure. **Scope:** model-garnet, `run-11`, row 358.
    - **Evidence:** Assistant message 51 contains 540 `<｜DSML｜` markers, including 180 `<｜DSML｜tool_calls>` markers, in its 384,437-character `content`, while also submitting an `edit` call whose `edits` value is a string. Tool message 52 rejects it (`edits.0: must be object`); message 53 reads the file and continues. No other trace contains that DSML marker in visible assistant content, and this row is eventually graded `OK`.
    - **Confidence:** high for a localized tool-format collapse and recovery; low for whether model generation or tool-call serialization produced it.
    - **Validation:** Inspect row 358 messages 51–53 and count DSML markers in message 51's visible content. Look for 384,437 characters, 540 `<｜DSML｜` markers including 180 `tool_calls` markers, an invalid string-valued `edits` argument, its rejection, and a subsequent read.

16. **An `OK` grade does not imply that the recorded conversation ended cleanly.**
    - **Category:** evaluation. **Scope:** all jobs.
    - **Evidence:** 50/370 traces end on an assistant turn with unresolved tool calls: all 26 `IL` traces, 11 `WA`, four `TL`, and **nine `OK`**. Examples of `OK` rows are 140 message 203, 168 message 2,426, and 225 message 70. `status` and `reward` agree on all 370 rows, so this is not a status/reward mismatch; it shows that the grade and the visible conversation endpoint encode different things.
    - **Confidence:** high for the recorded endpoints and grades; low for whether the grader scored intermediate workspace state, recovered a hidden final artifact, or the logger simply stopped early.
    - **Validation:** Classify the last conversation message of every trace by unresolved `tool_calls` and tabulate the matching statuses. Look for 50 such endings split into 26 `IL`, 11 `WA`, four `TL`, and nine `OK`.

## Per-model summaries

**model-atlas (16/40 `OK`).** Two final messages expand into hundreds of thousands of repetitive characters, while `run-10` alone contains broken call/result IDs and duplicate tool messages. The former is visible generation behavior; the latter is a trace-integrity problem whose origin is unknown. Two traces are `IL`.
**Validation:** Filter atlas rows by status, measure final assistant-content lengths, and audit tool-call/result IDs by job. Look for 16/40 `OK`, two `IL`, two final answers above 200,000 characters, and ten unmatched results in `run-10`.

**model-vega (53/60 `OK`).** The model's conspicuous signature is a medical metaphor in 47/60 traces. Its single `run-15` batch has high observed reward, but tasks and run conditions differ across models, so this is not a controlled quality ranking. One trace is `IL`.
**Validation:** Filter vega rows by job and status, then search visible assistant content for the medical terms listed in finding 10. Look for one `run-15` batch with 53/60 `OK`, one `IL`, and metaphor hits in 47/60 traces.

**model-cyan (47/60 `OK`).** It frequently seeks released and upstream implementations and invokes an unintroduced “Marcus” in narration. Its `run-01`/`run-07` outcome gap is especially hard to attribute because matched tasks changed prompt role and network access; seven of 18 shared tasks change from `OK` to `WA` in the later batch.
**Validation:** Pair cyan rows by `task_id`, compare statuses and prompt roles by job, and search tool commands and narration for upstream retrieval and Marcus. Look for 47/60 `OK`, 40 traces using `pip download` or `git fetch`, Marcus in 26/60 traces, and seven of 18 shared tasks changing from `OK` to `WA` as the role and network conditions change.

**model-delta (25/60 `OK`).** `run-03` is saturated with proxy `429`s and usually has no final answer. In `run-12`, long action sequences include one 1,786-call read loop, repeated local Git history operations, failed push attempts, and one temporary type-check bypass. Hidden reasoning is present in only 2.8% of its assistant messages, limiting interpretation of those actions.
**Validation:** Tabulate delta statuses, `run-03` proxy codes and endpoints, `run-12` commands, and nonempty `reasoning_content` across assistant messages. Look for 25/60 `OK`, `429`s in all 20 `run-03` traces, row 263's 1,786 `bash` calls, Git operations and a type-check bypass in `run-12`, and 282/9,906 messages with reasoning.

**model-orion (40/60 `OK`).** Its hidden reasoning can be exceptionally long: 13/60 traces exceed 300,000 characters, with both successes and failures among them. This is a volume and style observation; the corpus cannot show whether the extra deliberation helped. Three traces are `TL`.
**Validation:** Group orion rows by status and sum `reasoning_content` characters in each trace. Look for 40/60 `OK`, three `TL`, and 13/60 traces above 300,000 reasoning characters with mixed outcomes.

**model-flint (21/60 `OK`).** Twenty-three traces are `IL`; eight of 18 `run-06` attempts repeat one command for most of their calls and all eight end `IL`. A separate `run-08` interface anomaly puts `</arg_value>` fragments in five tool names. The prompt role also differs between those two jobs, so the run contrast does not isolate a model change.
**Validation:** Tabulate flint statuses and prompt roles by job, calculate each `run-06` trace's most frequent exact call share, and inspect `run-08` tool names. Look for 21/60 `OK`, 23 `IL`, eight of 18 majority-repeat traces all graded `IL`, five malformed names, and different prompt roles across the jobs.

**model-garnet (16/30 `OK`).** The most distinctive anomaly is one enormous DSML-marked message that turns an `edit` argument into an invalid string; the agent recovers and receives `OK`. The other 29 traces do not show that marker, so this is a single interface incident rather than a general model rate.
**Validation:** Tabulate garnet statuses, search visible assistant content for `<｜DSML｜`, and inspect row 358 messages 51–53 and its grade. Look for 16/30 `OK`, the marker only in row 358, a rejected string-valued `edit` argument followed by a read, and row 358 graded `OK`.

## Reproduce the directly countable findings

From the repository root, run `uv run --frozen scripts/verify_baseline_codex.py`. It reads the cached `data/agent-traces` dataset and prints one line for each of findings 3, 4, 5, 8, 9, 10, 11, 12, 14, and 16. The script contains the exact field filters and counting rules; the expected output is:

```text
03 proxy429=259/322 affected=20/20 endings=unresolved:14,tool:5,final:1
04 majority_repeat=8/18 majority_IL=8 job_IL=13
05 messages=3573 calls=1786 cat=947 python_print=830 bash=1786 edit=0 write=0 status=TL
08 row105=263626 row283=202425 non_atlas_max=2988
09 cyan_messages=82 cyan_traces=26/60 other_messages=0 prompt_mentions=0
10 vega_hits=101 vega_traces=47/60 other_hits=0
11 orion_over_300k=13/60 peer_over_300k=0/310 median=79883 row26=1037127 row215=902804
12 delta=282/9906 run03=55/229 run12=227/9677 flint=6795/6795
14 malformed_calls=5 traces=4 not_found=5
16 unresolved=50/370 IL=26 WA=11 TL=4 OK=9
```

These checks establish the recorded counts and field values. Inspect the cited messages for claims about repeated prose, intent, recovery, or cause.

## Reproduce the remaining six findings

From the repository root, run `uv run --frozen scripts/verify_baseline_codex_remaining.py`. It reads the same cached dataset, pairs `run-01` and `run-07` by `task_id`, links tool results by call ID, and scans messages in recorded order. The expected output is:

```text
01 pairs=18 outcomes=OO:10,OW:7,WW:1 roles=system/developer:18
01 match=user:18,tools:18,prompt_except_date:18 downloads=16/30,0/30
02 pip_download=35/60 git_fetch=29/60 either=40/60
06 commit=15/40 reset=8/40 push=5/40 other_jobs=0/330 push_fail=16/16 undeclared_git=1
07 row=168 bypass=1 tox_failed_at_pydocstyle=1 restored=1
13 unmatched=10/10 duplicate_result_traces=3 duplicate_call_traces=1 other_job_unmatched=0
15 marked=[(358, 51)] chars=384437 markers=540 tool_call_markers=180 edits_type=str rejected=1 next=read status=OK
```

For the parts that depend on message meaning or sequence, run `uv run --frozen scripts/verify_baseline_codex_remaining.py --show ROW:START-END` with the ranges below. The viewer prints roles, call IDs, tool names and arguments, and the start and end of each message's content; indices are zero-based.

- **1:** Inspect `2:0-1` and `1:0-1` for the `system`/`developer` role change with matching issue text, then `2:32-46`, `1:19-32`, `288:14-19`, and `289:14-19`. Look for linked successful aeon and trio downloads in `run-01`, failed downloads in `run-07`, a successful upstream fetch in row 2, and curl exit code 6 in row 1. These examples establish different recorded access, not which difference caused the grade gap.
- **2:** Inspect `2:32-46`, `142:57-60`, and `288:14-19`. Look for the downloaded aeon and trio release code being opened or compared, and row 142's fetched upstream feature commit being examined with `git show`.
- **6:** Inspect `57:196`, `168:2318`, `168:2368`, and `298:482-485`. Look for `git reset --hard`, two force-push attempts, and an undeclared `git` tool call followed by `Tool git not found` and a shell push rejected because `origin` is unavailable. The summary counts shell commands and their linked tool results; it does not infer any remote update.
- **7:** Inspect `168:2420-2425`. Look for `precommit.py` being read, a `sed` command replacing the `mypy --strict` call with `pass` before `tox -e py310`, a `pydocstyle` failure in the result, and `git restore precommit.py` afterward.
- **13:** Inspect `318:4-5`, `40:202-204`, and `105:150-151`. Look for call ID `A03VHoIdv` versus result ID `A03VHoId2`, a duplicate response to `4WFDTEleD`, and a reused call ID `j2lHmAXVN`. The script counts a result as unmatched when its ID has not appeared in any preceding call in that trace.
- **15:** Inspect `358:51-53`. Look for DSML syntax in message 51's visible content, its `edit` call passing `edits` as a string, message 52 rejecting `edits.0`, and message 53 continuing with a `read` call. The script checks that no other trace has the visible DSML marker; the row's status is `OK`.
