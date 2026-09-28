# Spec: NLP contrast analysis of agent traces (lexical + topic + embedding), run on Colab

## Problem

The report on the 370-trace dataset (`mfmVNfpt2q/agent-traces`, see `paper-typst/main.typ` §Task) is weakest on:

1. **Breadth beyond the naive baseline.** The known behavioral quirks (model-cyan's "Marcus", model-vega's
   medical metaphor, grader talk, the garnet DSML leak) are baseline-level. model-garnet, model-flint,
   model-atlas and model-orion have no model-level text quirk yet.
2. **Attribution.** Findings are not labeled as caused by the *model*, the *run* (job/environment), or the
   *task* (repo/issue).
3. **Part 2 pipeline.** No prototype exists. This job is the pipeline's first corpus-wide stage (text contrast
   → candidate findings → verification) and must be rerunnable with one command.

The job reads the model-written text across all traces, surfaces terms, topics and semantic clusters that are
specific to a condition (model, job, task), labels each candidate MODEL / RUN / TASK with a mechanical rule,
and has the executing agent (Claude) validate the candidates by reading the cited messages.

## Timebox

**40 minutes wall clock total**: about 20 min build and local debug, ≤15 min on the Colab GPU, 5 min pull,
validation and write-up. Cut scope, not the timebox. Order of cuts if late: the classifier first, then the
embedding clusters. The lexical stage (with masking, style features and the shared-task column) and the gates are never cut.

## Data facts the executor needs

- 370 rows. Row index = zero-based position in the dataset. Message index = zero-based position in `conversation`.
- Columns used: `model`, `job`, `task_id`, `trial`, `status`, `conversation` (assistant messages carry
  `content`, `reasoning_content`, `tool_calls`; a key may be missing, so use `.get`).
- Model ≡ job for model-atlas (`run-10`), model-orion (`run-05`) and model-vega (`run-15`). Run vs model can be
  separated **only** within model-cyan (`run-01`/`run-07`), model-delta (`run-03`/`run-12`), model-flint
  (`run-04`/`06`/`08`/`14`) and model-garnet (`run-02`/`09`/`11`/`13`). For single-job models the verdict
  can be at most "MODEL-or-RUN (confounded)".
- model-delta has almost no `reasoning_content` (≈2.8% of messages), so its reasoning results are
  underpowered. Report that; don't treat it as a finding.
- 253 distinct tasks; 79 are attempted by ≥2 models.
- Baseline-known quirks, used only as the recall check and not reported as new: "Marcus" / "the team and I"
  (cyan), patient / ailment / prognosis / convalesc* (vega), grader / hidden tests (cyan, orion, vega),
  `<｜DSML｜` (garnet row 358).

## Design

Single self-contained script: `tasks/nlp_analysis.py`. It loads the dataset with
`datasets.load_dataset("mfmVNfpt2q/agent-traces", split="train")` directly, so it works on Colab without the
repo. Arguments: `--limit N` (first N rows, stratified by model if easy, else the first N), `--out DIR`,
`--stages lexical,topics,embed,classify` (default all), `--device cuda|cpu`.

### 1. Units (text extraction)

From assistant messages only. Tool outputs are **excluded** (environment text).

| Field | Unit | Caps |
|---|---|---|
| `reasoning` | `reasoning_content` split on blank lines; merge pieces <200 chars into the next | ≤2000 chars per unit (truncate), ≤150 units per trace (uniform sample, seed 0) |
| `content` | one visible `content` message | same caps |
| `bash` | each `bash` tool call's `command`, reduced to a skeleton: strip leading `cd … &&`; per pipeline/`&&` segment keep the first 2 tokens (`git apply`, `pip download`, `python -m`) | none |

Each unit row: `row, trial, model, job, task_id, status, msg_idx, field, text` (raw) and `text_masked`.

**Masking** before any lexical or topic step (the embedding stage uses raw `text`). Purpose: task and repo
vocabulary must not look like a model trait. Masked spans are **replaced by placeholder tokens, never
deleted**, so the *rate* of code, paths or task echo survives as a style signal (e.g. `cherry pick HEX`,
`NUM passed`). Apply these rules in this order:

| Order | What | Regex / rule | Placeholder |
|---|---|---|---|
| 1 | fenced code | ```` ```.*?``` ```` (DOTALL) | `CODE` |
| 2 | inline code | `` `[^`\n]+` `` | `CODE` |
| 3 | URLs | `https?://\S+` | `URL` |
| 4 | paths | tokens with a file extension `\b[\w./-]+\.(py\|pyi\|txt\|toml\|cfg\|ini\|md\|rst\|json\|ya?ml\|sh\|c\|h\|js\|ts)\b`, or ≥2 slashes `\b[\w.-]+(/[\w.-]+){2,}` (so "and/or", "pass/fail" survive) | `PATH` |
| 5 | hex | `\b[0-9a-f]{7,40}\b` | `HEX` |
| 6 | identifiers | snake_case `\b\w*[a-z0-9]_\w+\b`, camelCase/PascalCase `\b[a-z]+[A-Z]\w*\b` or `\b[A-Z][a-z]+[A-Z]\w*\b`. **All-caps words (`DSML`, `IMPORTANT`, `OK`) are not masked.** | `IDENT` |
| 7 | numbers | `\b\d+(\.\d+)?\b` | `NUM` |
| 8 | task echo | see below | `TASKWORD` |

Casing: lowercase everything **except** that the all-caps words are recorded in the style features (§2b) first.

**Task-echo mask (rule 8).**
- The task statement of a trace is the content of its first `user` message.
- The mask set of a trace = the unigrams of its own task statement (same tokenizer, after rules 1–7) whose
  document frequency across the 253 distinct task statements is < 5%.
- Any word from that set in the trace's units becomes `TASKWORD`.
- Common English ("fix", "test", "function") and all boilerplate are never masked. The boilerplate includes
  "The following interfaces are expected by the test suite", which is in 370/370 prompts, and the system prompt,
  which is identical apart from its date line.
- Explicitly exclude from the mask set every word of the system prompt and of that boilerplate line, even if the
  threshold would allow it. Model reactions to the boilerplate (grader awareness) are a model property.
- Words that appear only in tool outputs are **not** masked. That extension is named as a next step, not built.

**Tokenizer** (the same for the lexical and topic stages; do **not** use sklearn's default `token_pattern`,
which drops the style characters). Tokens are, in priority order:
- placeholder tokens;
- words `[^\W\d_]{2,}` restricted to Latin script;
- each run of a non-Latin script as one token tagged by script, e.g. `SCRIPT_HAN`, `SCRIPT_CYRILLIC`, detected
  from the character's Unicode name (`unicodedata.name`);
- every emoji codepoint as its own token;
- whitelisted punctuation tokens: `—`, `…`, `!`, `?!`, `##`, `**`, `->`, `✅`, `❌`, `｜`.

### 2. Lexical stage (CPU): TF-IDF as the feature space, contrasted by condition

For each field separately:

- A **trace-level binary** document-term matrix: one document per trace = concatenation of its masked units,
  presence only (so one runaway trace cannot dominate). Vocabulary: words plus 2–3-grams, English stop words
  kept for n-grams (style lives in them) and removed for unigrams, term must occur in ≥5 traces.
- For a grouping `G ∈ {model, job-within-model, task_id}` and each group `g`:
  `rate[g,t]` = share of g's traces containing t. The ranking score is class-based TF-IDF on those rates:
  `W[g,t] = rate[g,t] · log(1 + |G| / Σ_g' rate[g',t])`.
  Keep the term as a candidate for `g` if `rate[g,t] ≥ 0.2`, and a one-sided Fisher exact test (g vs rest,
  trace counts) gives p < 1e-3.
- Attach to every (term, model) candidate:
  - `n_tasks`: distinct task_ids among the traces containing it;
  - `task_score`: max `W[task,t]`;
  - per-job rates for the multi-job models;
  - `rate_rest`: rate in all other models;
  - `survives_shared_tasks` (yes/no): rerun the same contrast restricted to traces of the 79 tasks that ≥2
    models attempted. "yes" if the term is still a candidate for the same model there. Task mix is the
    confound that masking cannot remove, and this is its only direct control.

**Attribution rule** (applies to terms, topics and clusters alike):
- **TASK**: `n_tasks < 3`, or ≥50% of its hits come from tasks where another model also has it (task-bound
  wording);
- **RUN**: model has several jobs and the per-job rates differ with Fisher p < 0.01;
- **MODEL**: `n_tasks ≥ 5` and present in every job of a multi-job model; for single-job models it is labeled
  `MODEL-or-RUN (confounded)`;
- anything else is `UNCLEAR`.

For the `bash` field, the same contrast at trace level over skeletons (bigrams of skeletons optional).

### 2b. Style features (CPU, part of the lexical stage)

Per trace and per field (`reasoning`, `content`), computed on raw text:
- share of non-Latin-script letters, overall and per script;
- emoji per 1k chars;
- markdown headers, bullet lines and `**bold**` per 1k chars;
- share of all-caps words (≥2 letters);
- `!` per 1k chars;
- `masked_share`: the share of tokens that became placeholders, split by placeholder type.

Contrast each feature by model with a Mann–Whitney U test (model vs rest, p < 1e-3), with the effect reported
as the median ratio. Candidates go through the same attribution rule and into `candidates.csv` with
`source = style`.

Masking stops task vocabulary from looking like a model trait, but it cannot see traits that live outside
words: switching language mid-reasoning, emoji, formatting habits. This table covers those.

### 3. Topic stage (CPU): topic modeling in the same TF-IDF space

Per field (`reasoning`, `content`): unit-level TF-IDF on masked text (same vocabulary rules, fit on units),
then NMF with k=25 (seed 0). For each topic:
- top 12 terms;
- distribution over model, over job-within-model, and over task_id;
- adjusted mutual information (sklearn `adjusted_mutual_info_score`) between the unit's argmax topic and each
  of model, job and task_id, reported for the whole field;
- per-topic: its dominant model, that model's share of the topic's units vs the model's share of all units
  (lift), and `n_tasks`.

A topic whose lift is ≥3 for one model and whose `n_tasks ≥ 5` becomes a candidate, attributed with the rule
above.

### 4. Embedding stage (GPU, Colab L4)

- `sentence-transformers` with `Qwen/Qwen3-Embedding-0.6B`, fp16, `max_seq_length=512`, batch 64,
  normalized, on **raw** (unmasked) `reasoning` and `content` units.
- Cache the embeddings to `--out/embeddings_{field}.npy` plus a row-aligned `units_{field}.parquet`.
- MiniBatchKMeans k=60 (seed 0) per field. Same outputs as the topic stage: AMI with model, job and task_id;
  per-cluster lift, `n_tasks`, attribution; 5 closest-to-centroid units per cluster.
- Optional cross-check only if the GPU time remains: `intfloat/e5-small-v2`, same k, report the AMI only.

### 5. Classifier stage (CPU, cut first if late)

All stages run in the single Colab invocation. The CPU stages add seconds there, and there's no second
handoff.


Logistic regression on the embeddings predicting `model`, `GroupKFold(5)` grouped by `task_id`.
- Report macro-F1 vs the majority-class baseline.
- Report the same with unit length (chars) as the only feature, so a win that is really a length effect shows
  up as one.
- Write the 10 most-confident correctly-classified units per model as reading material.

### 6. Outputs (`results/nlp_analysis/`)

- `units_summary.csv`: unit counts per model × field.
- `masking_report.csv`: `masked_share` per model × field × placeholder type, and the recall-term hits lost to
  `TASKWORD` (see checkpoint 2).
- `style_features.csv` (per trace) and `style_contrast.csv` (per model × feature).
- `lexical_terms_{field}.csv`: every candidate with the columns above plus `verdict_rule`.
- `topics_{field}.csv`, `clusters_{field}.csv`, `ami.json`: AMI for every stage × field × grouping.
- `classifier.json`.
- `candidates.csv`: one unified table.
  - Columns: `candidate_id, source (term|topic|cluster|style), field, model, label` (the term, the top terms,
    or the feature name), `rate_model, rate_rest, n_tasks, per_job_rates, survives_shared_tasks, verdict_rule`.
    `survives_shared_tasks` is `n/a` for topics and clusters.
  - Filled in by Claude during validation: `verdict_claude (confirmed|rejected|artifact), note`.
- `examples/{candidate_id}.md`: ≤5 example units per candidate, each with `row, msg_idx, field` and a verbatim
  excerpt ≤300 chars. The script asserts every excerpt is a substring of the source message; if one isn't, it
  fails loudly.

### 7. Validation by the executing agent (Claude)

After pulling the results, Claude reads `candidates.csv` top-down: the top 5 per model across all sources,
ranked by lift, capped at 35 candidates overall (the timebox allows no more). For each one, Claude opens its examples file (and, when needed, the cited source
message) and sets `verdict_claude`:
- **confirmed**: a genuine, distinctive trait of that condition;
- **artifact**: masking leak, repo vocabulary, a scaffold or prompt string, or one trace repeated;
- **rejected**: not actually distinctive on reading.

Record the reason in one line.

Claude also checks for an attribution disagreement: `verdict_rule` says MODEL but the examples all come from
one job or task. If so, it overrides the verdict and notes why. Claude must not invent quotes; it cites only
`row`/`msg_idx` values that appear in the examples files.

### 8. Colab execution (follows the Colab GPU loop in `CLAUDE.md`)

1. Local debug: `uv run --with sentence-transformers --with scikit-learn tasks/nlp_analysis.py --limit 40 --device cpu --out results/nlp_analysis_debug`.
2. colab-mcp `change_runtime("L4")`.
3. Push the script: `uv run --with ~/src/ats05-colab-mcp scripts/colab_runtime.py push tasks/nlp_analysis.py /content/tasks/nlp_analysis.py`.
4. Run with `run_code_cell`:
   - `pip install -q sentence-transformers datasets scikit-learn pyarrow`;
   - then `python /content/tasks/nlp_analysis.py --device cuda --out /content/results/nlp_analysis`;
   - poll `get_code_execution`.
5. Pull the results: `scripts/colab_runtime.py pull /content/results/nlp_analysis results/nlp_analysis`.
6. Release the runtime: `scripts/colab_runtime.py release`. **Always**, including after a failure.

Paths are absolute (`/content/...`). No Colab Secrets are needed: the dataset is public.

## Quick-failure checkpoints (each ≤1 min)

1. Local: dataset load plus unit extraction on `--limit 40`, printing unit counts per model × field. Fail if any field is empty for every model.
2. Local, **masking check**, before any stage runs on the full data:
   - print 5 masked units per model with the raw text next to them;
   - print `masked_share` per model;
   - print how many hits of the recall terms (marcus, patient, prognosis, ailment) became `TASKWORD`.

   Fail and fix the regexes if:
   - a recall term is masked outside the task-echo rule;
   - `masked_share` exceeds 0.5 for any model;
   - the sample shows ordinary English replaced.
3. Local: lexical, style and topic stages on `--limit 40`. Fail if the vocabulary is empty or NMF raises an error.
4. Local: embed 50 units on CPU with the 0.6B model, to prove the model loads and the dimensions match.
5. Colab: a first cell prints `torch.cuda.is_available()` and the GPU name. Fail fast if there's no GPU.
6. Colab: a full run with `--stages lexical` first (seconds), then the rest.

## Done criteria (gates, all required)

1. **Recall.**
   - On the full data, in `lexical_terms_content.csv`, "marcus" (or a bigram containing it) ranks in model-cyan's top 20 by `W`.
   - At least one of patient / prognosis / ailment ranks in model-vega's top 20.
   - If either fails, the job is broken: fix the masking or the vocabulary before anything else.
2. **Beyond the baseline.** Each of model-garnet, model-flint, model-atlas and model-orion has ≥1 candidate with `verdict_claude = confirmed` that is not in the baseline-known list. Otherwise it gets an explicit line "no distinctive text trait above threshold", listing its top 5 terms and why each was rejected.
3. **Attribution per candidate.** Every confirmed candidate has a final verdict of MODEL, RUN, TASK or MODEL-or-RUN (confounded), with the per-job rates, `n_tasks` and `survives_shared_tasks` that justify it. A MODEL verdict on a lexical
   candidate with `survives_shared_tasks = no` is downgraded to UNCLEAR unless Claude's validation note
   explains why. `ami.json` reports AMI with model, job and task for every stage.
4. **One-command rerun.** `tasks/nlp_analysis.py` runs end to end both locally (`--limit 40`, CPU) and on Colab (full, L4) with no manual steps between stages, and writes the files in §6.
5. **Paper.**
   - A new section in `paper-typst/main.typ`, "NLP contrast of model text (pipeline stage 1)", containing:
     - a table of the confirmed candidates (model, candidate, rate vs rest, `n_tasks`, attribution, cited row/message);
     - the AMI summary in one sentence;
     - one paragraph on how this stage feeds verification in the Part 2 pipeline, and its false-positive controls: masking, trace-level presence, the Fisher filter, the task/run attribution rule, verbatim-quote checks and agent validation.
   - `ninja` compiles.
6. Add the items above to `VERIFY.md` (Read first, then Edit; never Write) with this session's id, and check them off.

## Out of scope

- Tool outputs as text units, and masking words that come from tool outputs.
- OpenRouter LLM labeling or judging. Claude validates instead.
- LLM summarization of traces before embedding (Clio-style).
- 4B/8B embedders, hyperparameter sweeps over k, UMAP plots and figures.
- Reward-model or critic hidden states.
- Reworking earlier report sections.
- Any causal claim beyond the attribution rule.

## Open questions

- The task-echo threshold (document frequency < 5% across task statements) is a first guess. If checkpoint 2
  shows domain words still leaking into the top terms, try 10%. Don't tune it further within the timebox.
- The next step after this job, not built here: extend the task-echo mask to rare words the model saw in earlier
  tool outputs of the same trace.
