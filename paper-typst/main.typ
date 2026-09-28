#import "@preview/drafting:0.2.2": margin-note

#let page-left-margin = 2.5cm
#let page-right-margin = 2.5cm
#let note-col-width = 21cm - page-left-margin - page-right-margin

#let todooleg(body) = margin-note(stroke: rgb("#AAAEEE"), margin-right: page-right-margin, page-width: note-col-width)[#par(
  [#text(body, size: 5pt)],
  leading: 0.1em,
)]

#let todoai(body) = margin-note(stroke: rgb("#CC22AA"), margin-right: page-right-margin, page-width: note-col-width)[#par(
  [#text(body, size: 5pt, fill: rgb("#CC22AA"))],
  leading: 0.1em,
)]

#set document(title: "Model Behavior & Infrastructure Investigation")
#set page(margin: (top: 2.501cm, bottom: 2.5cm, left: page-left-margin, right: page-right-margin))
#set text(size: 11pt)

#align(center)[
  #text(size: 18pt, weight: "bold")[Model Behavior \& Infrastructure Investigation]

  Take-home report

  #datetime.today().display("[month repr:long] [day], [year]")
]

= Task

== Context

Our autoresearch agent trains and evaluates coding models. It recently trained and ran a number of
models on a suite of coding-agent benchmarks --- SWE-style tasks where an agent works inside a repo
sandbox with four tools (read, bash, edit, write) and tries to resolve a real issue, ending in a
graded pass/fail.

The runs were not clean. Going through the results, we found a number of problems --- some in the
models themselves and some in the infrastructure around them. Some are obvious; some are subtle;
some only show up when you look across many traces rather than at any single one.

We've packaged a sample of the resulting agent traces and anonymized them. We'd like you to
investigate.

== The dataset

370 coding-agent traces across 7 models (referred to by codename --- model-atlas, model-vega,
model-cyan, model-delta, model-orion, model-flint, model-garnet) and 15 runs (run-01 ... run-15).
Each trace contains:

- *conversation* --- the full run: assistant turns (with visible content, hidden
  `reasoning_content`, and `tool_calls`), and the tool results that came back
- *tools* --- the tool schemas the agent was given
- *task_id* --- the coding task (a repo + issue)
- *status* (OK / WA / IL / TL) and *reward* (0/1) --- the final grade
- *proxy_requests* --- one request status per model call (e.g. 200, 429)
- *model* (codename) and *job* (run code)

Get the data --- load it directly with the HuggingFace `datasets` library (~100 MB, 370 rows):

```python
from datasets import load_dataset

traces = load_dataset("mfmVNfpt2q/agent-traces", split="train")
```

=== Verified schema (from the live dataset)

The description above names the fields at a glance; the columns actually present on all 370 rows,
confirmed by loading the dataset directly (`ds.column_names`), are:

- *model* --- codename, one of the 7 listed above
- *job* --- run code, `run-01` ... `run-15`; each job used exactly one model (15 distinct
  `(job, model)` pairs, not a 15x7 cross product)
- *trial* --- per-attempt identifier, e.g. `academysoftwarefoundation__rez-1__v9DA9Mr`; unique
  across the entire dataset, not just within a job
- *task_id* --- the coding task (repo + issue), e.g. `academysoftwarefoundation__rez-1863`
- *status* --- `OK` / `WA` / `IL` / `TL`
- *reward* --- `0` or `1`; pairs with status as `{OK:1, WA:0, IL:0, TL:0}` with no exceptions
- *n_proxy_requests* --- integer count; equals `len(proxy_requests)` on every row
- *proxy_requests* --- list of `{status: int}` dicts, one per model call (e.g. `200`, `429`)
- *system_prompt* --- the harness system prompt as a standalone string; byte-identical to the
  first message in `conversation` on every row (see below)
- *tools* --- tool schemas; identical 4-tool set (`bash`, `edit`, `read`, `write`) across all rows
- *conversation* --- list of message dicts with keys `role`, `content`, and, for assistant
  messages, `tool_calls` and `reasoning_content`. Roles seen: `system` (217 rows), `developer`
  (153 rows --- models delta/atlas/cyan/flint/garnet only), `user`, `assistant`, `tool`. Whichever
  of `system`/`developer` a row uses, its content matches `system_prompt` exactly. Tool messages
  carry `tool_call_id` linking back to the originating `tool_calls` entry.

This was checked programmatically against all 370 rows, not sampled.

== The task

=== Part 1 --- Investigate

Find out as much as you can about the models and the runs, and give us evidence. We're interested
in both:

- *Behavioral findings* --- failure modes, unsafe or undesirable behavior, and quirks of any kind.
  A quirk need not be a safety problem --- a model might simply act, phrase things, or generate
  text in a way that's distinctly its own. Look in what a model does, in its visible narration
  (`content`), and in its hidden reasoning (`reasoning_content`); anything that sets one model or
  one run apart from another is worth reporting.
- *Infrastructure / evaluation findings* --- problems in how the runs were executed, recorded, or
  graded.

Most of the models here have at least one distinctive characteristic worth reporting. If a model
looks unremarkable from its pass rate and tool-call graph, look closer --- read widely across its
narration and reasoning text, not just its actions.

For every claim, cite the traces (and messages) it rests on, and quantify it where you can --- a
rate across many traces, a comparison between models or runs. Be explicit about what you're unsure
of and what the data doesn't let you determine.

#todooleg[A note on judgment: not everything unusual is a finding, and not every finding is the
  model's fault. Some oddities are the environment, some are the grader, some are just noise that
  looks like signal. Part of the exercise is telling those apart.]

=== Part 2 --- Automate

Design a discovery pipeline that could surface findings like these automatically from a corpus like
this, so the next batch of runs doesn't have to be audited by hand. Describe the architecture;
prototype as much as time allows. We're interested in how you'd keep it from drowning in false
positives, and how you'd handle the scale --- the traces are long and there are many of them, so
most won't fit in a single context window. How your pipeline reads, reduces, and aggregates is part
of the design.

== Resources

- An OpenRouter API key --- use whatever models you find useful (\$100 credit budget; key expires
  2026-10-25).
- Use any external tools or agents you like --- Claude Code, Cursor, your own agent frameworks,
  whatever you work best with. Model calls are not a cost concern; you can use your own agent
  limits together with the key.
- Build whatever tooling helps. There's no restriction on approach; the length and volume of the
  traces are the real constraint to design around.

== What we're evaluating

- *Breadth, importance, and difficulty* of what you find --- how much of what's really in the data
  you surface, weighted toward the findings that are most consequential and the hardest to reach.
  We benchmark against a naive one-shot agent baseline (see Logistics): the things that baseline
  already catches count for little; the consequential, non-obvious findings that take real
  investigation --- and that it misses --- are what we weigh most.
- *Evidence and calibration* --- findings that are cited, quantified, and honest about uncertainty,
  and a clear sense of a genuine finding vs. an artifact of the setup.
- *Pipeline design* --- how you'd scale this from a manual read to an automated audit.

== Deliverable

Two things:

- *Part 1 --- report.* A written report of findings, with evidence (a markdown file is fine).
  Surface as much of the real signal as possible, but keep it calibrated: a cited, quantified
  finding beats a pile of hunches.
- *Part 2 --- pipeline.* An overview of the discovery-pipeline design --- the architecture, and how
  it reads, reduces, and aggregates across long traces at scale --- alongside an implemented
  prototype, however partial.

== Logistics

- Time limit: 4 hours. Exhaustive coverage in that window isn't expected --- what's evaluated is how
  work is prioritized, what gets gone deep on, and how reasoning proceeds under the constraint.
- For calibration: there's more here than a first pass suggests. A naive agentic baseline --- a
  Kimi-K3 swarm, or an Opus 4.8 workflow --- surfaces only a small fraction of the quirks actually
  present.

#todooleg[If anything seems broken or unclear --- the data, the key, the task --- email
  dmitriy\@whitecircle.ai. Don't lose time to a problem on their end.]

= Introduction
== Oleg's initial plan
Key decisions:
- I will transform the logs into inspect files if feasible, to ease analysis and match the default convention
- I will ask codex to do the task at its best (sol, high effort (not sure about astra -- may not have enough credits)). It will be the proxy of Opus 4.8 and Kimi swarm performance which I will aim on superceding.
- In the meanwhile I will be getting to know the data: close-reading and EDA, will put here as a data description setting. At the first glance looks like pi agent being ran in the Harbor harness

== Inspect AI conversion

Converted the 370-row source table into Inspect AI `.eval` logs (`scripts/convert_to_inspect.py`,
spec in `inspect-conversion-spec.txt`) so traces can be browsed in Inspect View instead of raw
table rows. Principles followed:

- *Source is authoritative, conversion is lossless.* No rerunning or regrading --- every source
  field is preserved verbatim, either as the natural Inspect field it maps to (`reward` $arrow$
  score value, `conversation` $arrow$ messages) or, when Inspect has no native slot for it
  (`system_prompt`, `tools`, `proxy_requests`, `job`, `trial`, `task_id`), under
  `EvalSample.metadata["source_*"]`.
- *The atomic unit is the sample, not the log.* Every source row becomes exactly one
  `EvalSample`, keyed by `trial` (already unique across the full table) --- 370 rows in, 370
  samples out, none merged or dropped. Samples are then filed into logs by `(job, model)`, matching
  the dataset's 15 distinct pairs, purely so each log's top-level model/run identity is truthful;
  this filing is a grouping choice, not a reduction in trace count.
- *Never collapse distinctions the source keeps separate.* Visible `content` and hidden
  `reasoning_content` become separate content blocks (text vs. `ContentReasoning`) even when one
  side is empty; `developer`-role messages display as Inspect `system` messages but keep
  `source_role: "developer"` in metadata so the substitution is visible, not silent; `status` and
  `reward` are both kept (status text lives in `scores["grade"].metadata`, not folded into the
  numeric grade).
- *Flag, don't fix, data anomalies.* Anything the converter can't cleanly represent is recorded in
  `metadata["conversion_exceptions"]` on the sample rather than repaired or dropped.

Result: 15 logs, 370 samples, written via `write_eval_log()` to `results/inspect_logs/`
(`{job}_{model}.eval`, e.g. `run-05_model-orion.eval`; gitignored as a rebuildable artifact ---
regenerate with `uv run scripts/convert_to_inspect.py`). The full-table smoke check (read
every log back with `read_eval_log()`, compare sample counts, `(job, model, trial)` uniqueness,
score/status/message-role/proxy-count fidelity against the source) passed in full, ran in under 4
seconds. The converter surfaced one genuine data anomaly worth carrying into Part 1: 10 tool
messages across the table (all in `run-10` / model-atlas) carry a `tool_call_id` that doesn't match
any preceding tool call in that trace --- including one off-by-one-character ID
(`A03VHoId2` vs. the call's actual `A03VHoIdv`) --- which the converter kept verbatim and flagged
rather than silently repairing.

=== What's inside each run

A `job` is a batch, not a single task: each one bundles many distinct `task_id`s (repo + issue
pairs) run under one fixed model, so `job` $eq.not$ `task`. One `(job, model)` log holds one row
per trial in that batch, and the batch sizes vary widely (5 to 60 traces per job):

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, left, right, right, right),
  table.header([*job*], [*model*], [*traces*], [*unique tasks*], [*reruns*]),
  [run-01], [model-cyan], [30], [30], [0],
  [run-02], [model-garnet], [5], [5], [0],
  [run-03], [model-delta], [20], [20], [0],
  [run-04], [model-flint], [5], [5], [0],
  [run-05], [model-orion], [60], [60], [0],
  [run-06], [model-flint], [18], [18], [0],
  [run-07], [model-cyan], [30], [30], [0],
  [run-08], [model-flint], [30], [29], [1],
  [run-09], [model-garnet], [9], [9], [0],
  [run-10], [model-atlas], [40], [40], [0],
  [run-11], [model-garnet], [8], [8], [0],
  [run-12], [model-delta], [40], [40], [0],
  [run-13], [model-garnet], [8], [8], [0],
  [run-14], [model-flint], [7], [7], [0],
  [run-15], [model-vega], [60], [60], [0],
)

*reruns* = traces minus unique tasks, i.e. how many trials in that job repeat a `task_id` already
attempted elsewhere in the same job. Only `run-08` has one: 30 trials over 29 distinct tasks, so
one task was attempted twice under `model-flint` within that batch. Every other job runs each of
its tasks exactly once. Note this counts reruns *within* a job only --- the same `task_id` can
still recur *across* different `(job, model)` pairs (e.g. under a different model), which this
table does not show.

== EDA

Descriptive pass over all 370 traces before hunting for quirks or infra bugs (spec:
`spec-eda.md`; script: `uv run scripts/eda.py`; raw outputs: `results/eda/*.csv|json|txt`, one
file per block below, every number here traceable to one). Regex-only, no LLM calls, no causal
claims --- that's Part 1's job.

*Counts.* 7 models, 15 jobs (`run-01`.."run-15"), 370 traces total
(`counts_per_model.csv`, `counts_per_job.csv`). Coverage is uneven and each model $times$ job cell
is (almost) a partition, not a grid: model-orion is only in `run-05` (60), model-vega only in
`run-15` (60), model-atlas only in `run-10` (40), model-cyan splits `run-01`/`run-07` (30/30), and
model-garnet is spread thin across four jobs (`run-02`/`run-09`/`run-11`/`run-13`, 5/9/8/8 = 30).
model-delta and model-flint are the only models that reappear across jobs with different traffic
patterns (delta: `run-03`=20, `run-12`=40; flint: `run-04`=5, `run-06`=18, `run-08`=30, `run-14`=7)
--- see `crosstab_model_job.csv`. Every `task_id` appears at most twice for a given model
(`reruns_model_task.csv`: 26 (model, task_id) pairs have 2 traces, none have more).
#todoai[bb8efc78-f4bc-4eb5-b9dc-673e39fec042 how many individual tasks are there?]

*Lengths.* Per-trace assistant-turn/tool-call/char counts by model and job are in
`lengths_by_model.csv`, `lengths_by_job.csv`, `lengths_per_trace.csv`;
per-tool-name call counts in `tool_call_counts_per_trace.csv`. Aggregate tool-call mix: 23,923
`bash`, 3,282 `read`, 2,805 `edit`, 728 `write`, out of 30,757 total --- plus 18 calls whose `name`
field is not one of the four schema tools at all (see Malformed below). Mean tool-output chars per
trace ranges from ~53k (model-vega) to ~372k (model-delta) (`lengths_by_model.csv`), driven by a
small number of very long traces rather than a uniform shift: the single longest trace
(row 168 (0-indexed), model-delta/run-12/`parquery__icontract-297`) alone hits 3,573 conversation
turns and 3.24M tool-output chars (`envelope_max_by_model_job.csv`), two orders of magnitude past
the next-longest `run-12` trace.

*Grade family.* Status counts: OK 218, WA 121, IL 26, TL 5 (`grade_status_by_model_job.csv`).
Reward mean by model ranges from 0.35 (model-flint) to 0.883 (model-vega)
(`grade_reward_by_model.csv`). Consistency check (OK with reward 0, or non-OK with reward 1) found
*zero* violations across all 370 rows (`grade_consistency_anomalies.json` is `[]`) --- status and
reward agree everywhere.

*Proxy family.* 30,444 requests at 200, 322 at 429, 1 at 503 (`proxy_status_code_counts.csv`).
Non-200 traffic is not spread evenly: it is almost entirely `run-03` (mean non-200 rate 0.564 over
that job's 20 traces, vs. 0.0 for 12 of the other 14 jobs) and `run-14` (0.098) --- see
`proxy_rate_by_job.csv`. Since `run-03` is exclusively model-delta (`crosstab_model_job.csv`), this
surfaces as a model-level number too: model-delta's per-trace non-200 rate averages 0.194 vs. 0.0
for five of the other six models (`proxy_rate_by_model.csv`). Worst single trace: row 225
(model-flint/run-14/`pybamm-team__pybamm-602`), 28/63 non-200 (`proxy_per_trace.csv`).

*Tool-result errors.* Taxonomy hit counts across all tool messages (30,708 total):
non-zero exit 3,482, traceback 1,600, syntax/import error 566, file-not-found 288, timeout 137,
command-not-found 73, edit-no-match 27, permission 9 (`tool_error_taxonomy_summary.csv`,
broken out by model in `tool_error_taxonomy_by_model.csv`). 3,205 tool outputs looked
error-adjacent (matched `/error|fail(ed|ure)?|exception/i`) but hit none of the eight regexes; 200
are sampled verbatim in `tool_error_unmatched_sample.txt` for manual triage.

*Malformed turns.* 69 issues total (`malformed_turns.csv`, `malformed_turns_summary.csv`): 50
traces end with an assistant turn that still has open `tool_calls` (no matching tool result ever
arrives), and 19 tool-call `name` fields fall outside the four declared tools (`bash`/`read`/`write`/`edit`)
--- values like `"grep -n \"sys\" ..."`, `"run -h 2>&1 | head -30\n</arg_value>"`, or
`"task_complete"` (rows 42, 62, 155, 158, 160, 182, 272, 298, 307, 323, 353, 366 ---
full list in `malformed_turns.csv`), which look like raw command/argument text leaking into the
`name` slot rather than a model calling an undeclared tool. Zero unparsable JSON tool-call
arguments found. Zero task_ids have more than one distinct task-statement hash
(`task_statement_variants.csv` is empty) --- the underlying issue text is stable per task_id.

*Reruns and configurations.* Scaffold fingerprint = sha256(system prompt + tools schema + first
user message with everything from `"## Issue"` onward masked out), 16 hex chars
(`scripts/eda.py:scaffold_fingerprint`). Every one of the 15 jobs contains multiple distinct
fingerprints (`fingerprint_job_disagreement.json`, `jobs_with_multiple_fingerprints`) --- expected,
since fingerprint also depends on the fixed pre-`"## Issue"` preamble, which embeds the repo name
and (per `scaffold_diffs_sample.txt`) both `system_prompt` and, less often, the `tools` schema
itself vary trace-to-trace for the *same* task_id, evidently per-model rather than per-run. 35
fingerprints recur across 2 different jobs each (`fingerprints_spanning_multiple_jobs`). Under the
stricter rerun-rerun definitions, only a handful of (model, task_id) pairs repeat under an
*identical* fingerprint (`rerun_rerun_by_fingerprint.csv`: model-delta/`parquery__icontract-297`,
model-flint/`pybamm-team__pybamm-612`, model-garnet/`encode__django-rest-framework-9455`,
model-garnet/`geopandas__geopandas-2286`) or an identical job
(`rerun_rerun_by_job.csv`: only model-flint/`pybamm-team__pybamm-612`/`run-08`) --- of the 26
(model, task_id) reruns, most are *not* same-fingerprint, same-job repeats.

*Envelope.* Max turns/chars per (model, job) in `envelope_max_by_model_job.csv`. Cross-tabbing
status against how a trace ends (`envelope_end_status.csv`) gives a clean signal: *all* 26 IL
traces end with an assistant turn carrying unresolved `tool_calls` (the "IL" grade looks entirely
consistent with an input/output-limit cutoff hitting mid-tool-call). But this ending is not
exclusive to IL: 9 traces graded OK and 4 graded TL also end the same way (rows 36, 140, 168, 225,
275, 281, 282, 333, 343 for the OK case --- `envelope_end_status.csv` / cross-reference with
`malformed_turns.csv`'s `trace_ends_mid_tool_call` rows) --- a trace can be graded OK for what it
already produced even though its last tool call was never resolved in the log.

*Leads for Part 1* (descriptive only, no causal claims):
- model-delta's `run-03` proxy traffic is 56% non-200 (12.95 non-200 / 24.4 requests per trace on
  average) --- isolate whether this is rate-limiting the model faced, or an infra-side proxy issue
  specific to that job, and whether it correlates with model-delta's below-median reward (0.417).
- 19 tool calls across 12 traces carry a non-schema `name` --- looks like harness-side
  parsing spillover (raw shell/argument text landing in the `name` field) rather than the model
  inventing tools; worth checking whether these traces' bash-tool JSON was malformed upstream.
- 9 OK-graded and 4 TL-graded traces end mid-tool-call like the 26 IL traces do --- worth checking
  whether these are genuine IL cases mislabeled, or a logging cutoff that doesn't affect grading.
- row 168 (model-delta/run-12/`parquery__icontract-297`) is a 3,573-turn, 3.24M-char outlier, about
  two orders of magnitude beyond any other trace in the dataset --- worth checking whether this is
  a genuine runaway/looping episode or a duplication artifact in logging.
- `system_prompt` and occasionally `tools` schema differ across models solving the *same* task_id
  (`scaffold_diffs_sample.txt`) --- confirms scaffolding is model-conditioned, not just
  task-conditioned; useful context before attributing any behavioral difference to the model alone.
- 3,205 tool outputs look error-like but matched none of the 8 regexes
  (`tool_error_unmatched_sample.txt`, 200 sampled) --- worth a manual pass to see whether they hide
  additional error families (assertion failures, lint failures, custom harness errors) the current
  taxonomy misses.