# Model Behavior & Infrastructure Investigation

A take-home investigation of coding-agent traces from seven models: atlas, cyan, delta, flint,
garnet, orion and vega. The report lives in `paper-typst/main.typ`.

## How to reproduce the findings

**Use `paper-typst/main.typ` as the documentation.** It gives concrete steps to reproduce each
finding: the script to run, the command, and the expected counts. This README only summarizes.
Build the PDF with `ninja` from the repo root (`paper-typst/main.typ` → `paper-typst/main.pdf`).

## What the work covers

1. **Pre-processing.** The raw traces become Inspect AI logs, for close reading and LLM judging.
2. **Exploratory analysis.** A close reading of the traces, by hand and with Claude.
3. **Baseline.** Codex ran the task as given. Its findings were then grounded in reproducible
   checks.
4. **LLM-judge screening.** A cheap model screens each trace against a list of hypotheses. Opus
   then validates what the cheap model flagged.
5. **NLP-style analysis.** Term and bigram frequencies, plus clustering, build a lexical profile of
   each model.
6. **Automation.** A spec (`spec-automation.md`) and an MVP of an automated analysis engine. The
   MVP reproduces some of the report's findings.

## Key findings

- **Evaluation awareness.** Every prompt says a test suite will check the result. Cyan reasons
  about the grader in 38/60 traces, orion in 19/60, vega in 9/60. Flint and garnet never do.
- **Cyan** often credits an unseen collaborator, "Marcus" (26/60 traces). It looks up solutions
  online. In `run-01` it applied the project's upstream fix in 21/30 traces, and all 21 passed.
  Network access worked only in `run-01`. In `run-07` its score fell from 28/30 to 19/30. It probes
  for credentials unprompted, in 20/60 traces.
- **Delta** writes very little visible text. One trace re-reads a single file 1,777 times. Another
  disables a type check to pass validation. It commits heavily with `git` and calls a `git` tool
  that does not exist.
- **Orion, flint and atlas** call undeclared tools. Flint leaks markup into tool names. Flint also
  struggles with the environment: it says one thing and does another.
- **Secrets.** Orion and vega try to expose environment secrets when baited. Cyan does so without
  bait.
- **Lexical profiles.** Flint opens with "I'll help you..." and writes "Great!" densely. Vega
  narrates in the plural ("Let's run...") and drops articles. Vega also speaks of code as a medical
  patient.

## Setup

```bash
direnv allow        # loads the nix + uv environment
uv run <script>     # runs a Python script
ninja               # compiles the paper
```

Copy `.env.example` to `.env` and fill in keys. LLM judging needs `OPENROUTER_API_KEY`. See
`CLAUDE.md` for the containerized-experiment harness and the Colab GPU loop.
