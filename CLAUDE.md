# CLAUDE.md or AGENTS.md whatever you call it.

Ensure `direnv` is active before working — run `direnv allow` if the shell env looks wrong.
Use `uv` for Python dependency management; run scripts with `uv run <script>`.
Run `ninja` from the repo root to compile the Typst paper (`paper-typst/main.typ` → `paper-typst/main.pdf`).
The paper lives in `paper-typst/main.typ`; rebuild with `ninja` after editing to check it compiles.
`paper-typst/main.pdf` is gitignored — it's a rebuildable artifact, not committed. Don't add it back
to source control: a tracked compiled PDF tends to conflict with users' workflow.
Use `ruff` for linting/formatting Python code.
Do not commit `.direnv/`, `.venv/`, `.cursor/`, `.ninja_log`, or `paper-typst/main.pdf`.

## Running Containerized Experiments

The `experiment/` directory contains a harness for running Claude Code agents in isolated devcontainers.

### Setup

1. Place task IDs in `experiment/tasks.txt` (one per line)
2. Place condition-gated data in `experiment/data/{condition}/`
3. Customize `experiment/AGENT_PROMPT.md.template` (uses `{{TASK_ID}}` and `{{CONDITION}}` placeholders)
4. Run: `./experiment/run_experiment.sh --conditions cond1,cond2`

### Key files

- `experiment/run_experiment.sh` — orchestrator (worktrees, containers, credentials, cost tracking)
- `experiment/AGENT_PROMPT.md.template` — agent prompt with `{{TASK_ID}}` and `{{CONDITION}}` placeholders
- `experiment/.devcontainer/` — Dockerfile, firewall, permission bypass
- `experiment/data/{condition}/` — resource files mounted per condition
- `experiment/results/{condition}/{task_id}.jsonl` — full agent session logs
- `experiment/logs/` — per-run logs, devcontainer logs, rendered prompts

### Critical gotchas

- **CLAUDECODE env var**: cleared via `-e CLAUDECODE=` in docker exec so agents can run inside Claude Code sessions.
- **Workspace permissions**: orchestrator runs `chown -R node:node /workspace` after copying data.
- **State files**: per-condition with `fcntl` locking for concurrent safety.
- **Time budget**: agents must write `/workspace/result.json` by turn 12. Without this, agents research indefinitely.
- **Docker Hub rate limits**: pre-pull `node:20-bookworm` before long runs.

### Customization points

Override these bash functions before the main loop to customize behavior:
- `load_tasks` — return task IDs (default: reads `experiment/tasks.txt`)
- `render_prompt` — render the prompt for a given condition + task_id
- `get_data_dir` — return the data directory for a given condition

## Trace re-grading

`scripts/rescore.py <log.eval>` appends LLM-judge verdicts (default: 3 Claude models via OpenRouter,
needs `OPENROUTER_API_KEY` in `.env`) next to the log's original score and writes `<log>-rejudged.eval`.
Each judge is registered as its own named scorer (`judge_<model>`); passing bare `model_graded_qa`
several times would yield indistinguishable `model_graded_qa`, `model_graded_qa1`, ... columns.
Test offline with `--judge mockllm/model` (no key; grades come out `nan` because the mock never emits `GRADE:`).

## Colab GPU loop (headless, no git, no browser tab)

Debug locally on a sample (`uv run --with sentence-transformers tasks/<task>.py --limit 200`), then:
1. colab-mcp `change_runtime("T4"|"L4"|"A100")` assigns a runtime.
2. `uv run --with ~/src/ats05-colab-mcp scripts/colab_runtime.py push tasks/<task>.py /content/tasks/<task>.py`
3. colab-mcp `run_code_cell(code=...)` runs `python /content/tasks/<task>.py --out /content/results/<task>`, then poll `get_code_execution`.
4. `scripts/colab_runtime.py pull /content/results/<task> results/<task>`
5. `scripts/colab_runtime.py release` — an idle assigned GPU keeps spending Colab compute units.
Direct-mode kernels start in `/`, so use absolute `/content/...` paths. Colab Secrets don't work headless.
