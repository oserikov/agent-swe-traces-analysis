# Spec: Descriptive EDA of the agent-traces dataset

## Problem

Before hunting for model quirks and infra bugs (task Part 1), we need a descriptive map of the
370-trace dataset: what's in it, how it's distributed, where errors occur, and whether repeated
attempts at the same task were run under the same setup. Results feed a new EDA section in
`paper-typst/main.typ`.

## Scope

**In:**
- Counts: traces per model, per task, per job; model × task and model × job cross-tabs.
- Lengths: assistant turns, tool calls (total and per tool name); characters split by channel —
  `content`, `reasoning_content`, tool outputs. Distributions per model and per job.
- Four error families, each reported separately, never pooled:
  1. **Grade**: status (OK/WA/IL/TL) × reward, per model/job; consistency (e.g. OK with reward 0).
  2. **Proxy**: `proxy_requests` status-code counts and rates per model/job; per-trace non-200 count.
  3. **Tool-result errors**: regex taxonomy (traceback, non-zero exit, command not found,
     file not found, edit no-match, timeout, permission, syntax/import error, …). Error-looking
     outputs no regex matched → sampled into a TXT for manual review.
  4. **Malformed turns**: unparsable tool_call arguments, unknown tool names, empty assistant turns
     (no content, no reasoning, no tool call), trace ending mid-tool-call.
- Reruns and configurations:
  - **Rerun** = >1 trace for the same (model, task_id).
  - **Configuration** computed two ways: (a) scaffold fingerprint = hash of tools schema + system
    prompt + first user prompt, with the task statement text masked out so it is comparable across
    tasks; (b) job code. The task statement is hashed separately to catch same-task_id/different-text.
    Envelope is inferred post hoc, so it is not part of the fingerprint.
  - **Rerun-rerun** = >1 trace for the same (model, task_id, configuration), under each definition.
  - Report where fingerprint and job disagree (one job with several fingerprints; one fingerprint
    across several jobs).
- Scaffolding diffs across reruns: which components differ (tools / system prompt / first user
  prompt / task statement for the same task_id), with a short text diff sample.
- Runtime envelope, inferred: max turns and max total chars per job/model, how traces end
  (final role, last tool call, empty final turn) vs status TL/IL.

**Out:**
- No LLM calls (no classification, no judging). Regexes only.
- No behavioral/quirk analysis, no causal claims about infra vs model.
- No pipeline design (Part 2).
- No token counts.

## Design

- Entry point: `uv run scripts/eda.py` — one script, loads data via `scripts/fetch_dataset.py:get_traces()`.
- Writes raw outputs to `results/eda/` (CSV for tables, JSON for summaries, TXT for samples/diffs).
  Every number cited in prose must be traceable to a file there.
- Every anomaly listed with trace identifiers (row index + model + job + task_id).
- Agent then writes an **EDA** section in `paper-typst/main.typ`: descriptive numbers plus a
  "Leads for Part 1" list of anomalies with trace IDs, no causal claims.

## Quick-failure checkpoints

1. `uv run scripts/fetch_dataset.py` loads 370 rows (cached).
2. `uv run scripts/eda.py --limit 20` finishes in seconds and writes all output files.
3. Inspect one trace's `conversation` schema before writing regexes (field names, tool-result role).

## Done criteria

- `uv run scripts/eda.py` runs end to end on all 370 rows and populates `results/eda/`.
- All five blocks computed: counts/cross-counts, lengths, 4 error families, reruns/configs,
  scaffolding/envelope.
- `ruff check scripts/` clean.
- `paper-typst/main.typ` has an EDA section citing those outputs; `ninja` compiles it.
