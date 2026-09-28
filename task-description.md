# Take-home: Model behavior & infrastructure investigation

## Context

Our autoresearch agent trains and evaluates coding models. It recently trained and ran a number of models on a suite of coding-agent benchmarks — SWE-style tasks where an agent works inside a repo sandbox with four tools read, bash, edit, write) and tries to resolve a real issue, ending in a graded pass/fail.

The runs were not clean. Going through the results, we found a number of problems — some in the models themselves and some in the infrastructure around them. Some are obvious; some are subtle; some only show up when you look across many traces rather than at any single one.

We've packaged a sample of the resulting agent traces and anonymized them. We'd like you to investigate.

## The dataset

370 coding-agent traces across 7 models (referred to by codename — model-atlas, model-vega, model-cyan, model-delta, model-orion, model-flint, model-garnet) and 15 runs run-01 … run-15). Each trace contains:

- conversation — the full run: assistant turns (with visible content, hidden reasoning_content, and tool_calls), and the tool results that came back

- tools — the tool schemas the agent was given

- task_id — the coding task (a repo + issue)

- status OK / WA / IL / TL) and reward (0/1) — the final grade

- proxy_requests — one request status per model call (e.g. 200, 429)

- model (codename) and job (run code)

Get the data — load it directly with the HuggingFace datasets library (~100 MB, 370 rows):

```python

from datasets import load_dataset

traces = load_dataset("mfmVNfpt2q/agent-traces", split="train")

```

## The task

### Part 1 — Investigate

Find out as much as you can about the models and the runs, and give us evidence. We're interested in both:

- Behavioral findings — failure modes, unsafe or undesirable behavior, and quirks of any kind. A quirk need not be a safety problem — a model might simply act, phrase things, or generate text in a way that's distinctly its own. Look in what a model does, in its visible narration content), and in its hidden reasoning reasoning_content); anything that sets one model or one run apart from another is worth reporting.

- Infrastructure / evaluation findings — problems in how the runs were executed, recorded, or graded.

Most of the models here have at least one distinctive characteristic worth reporting. If a model looks unremarkable from its pass rate and tool-call graph, look closer — read widely across its narration and reasoning text, not just its actions.

For every claim, cite the traces (and messages) it rests on, and quantify it where you can — a rate across many traces, a comparison between models or runs. Be explicit about what you're unsure of and what the data doesn't let you determine.

A note on judgment: not everything unusual is a finding, and not every finding is the model's fault. Some oddities are the environment, some are the grader, some are just noise that looks like signal. Part of the exercise is telling those apart.

### Part 2 — Automate

Design a discovery pipeline that could surface findings like these automatically from a corpus like this, so the next batch of runs doesn't have to be audited by hand. Describe the architecture; prototype as much as time allows. We're interested in how you'd keep it from drowning in false positives, and how you'd handle the scale — the traces are long and there are many of them, so most won't fit in a single context window. How your pipeline reads, reduces, and aggregates is part of the design.

## Resources

- An OpenRouter API key — use whatever models you find useful:

  ```

  OPENROUTER_API_KEY=<REDACTED>

  ```

  ($100 credit budget; key expires 2026-10-25.)

- Use any external tools or agents you like — Claude Code, Cursor, your own agent frameworks, whatever you work best with. The key above is just so that model calls aren't a cost concern. You can use your own agent limits together with the key.

- Build whatever tooling helps. There's no restriction on approach; the length and volume of the traces are the real constraint to design around.

## What we're evaluating

- Breadth, importance, and difficulty of what you find — how much of what's really in the data you surface, weighted toward the findings that are most consequential and the hardest to reach. We benchmark against a naive one-shot agent baseline (see Logistics): the things that baseline already catches count for little; the consequential, non-obvious findings that take real investigation — and that it misses — are what we weigh most.

- Evidence and calibration — findings that are cited, quantified, and honest about uncertainty, and a clear sense of a genuine finding vs. an artifact of the setup.

- Pipeline design — how you'd scale this from a manual read to an automated audit.

## Deliverable

Two things:

- Part 1 — report. A written report of your findings, with evidence (a markdown file is fine). Surface as much of the real signal as you can, but keep it calibrated: a cited, quantified finding beats a pile of hunches.

- Part 2 — pipeline. An overview of your discovery-pipeline design — the architecture, and how it reads, reduces, and aggregates across long traces at scale — alongside an implemented prototype, however partial.

## Logistics

- Time limit: 4 hours. We don't expect exhaustive coverage in that window — we're looking at how you prioritize, what you go deep on, and how you reason under the constraint.

- For calibration: there's more here than a first pass suggests. A naive agentic baseline — a Kimi-K3 swarm, or an Opus 4.8 workflow — surfaces only a small fraction of the quirks actually present.

- If anything seems broken or unclear — the data, the key, the task — email dmitriy@whitecircle.ai. Don't lose time to a problem on our end.