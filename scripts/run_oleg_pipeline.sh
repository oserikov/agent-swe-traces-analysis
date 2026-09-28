#!/usr/bin/env bash
# Orchestrates the full pilot -> stage1 -> stage2 -> merge run for oleg_judge.py.
# This is the exact sequence run for oleg-hypothesis-screen-spec.md (2026-09-28); see
# paper-typst/main.typ's "Hypothesis screening" chapter for the results it produced and
# VERIFY.md for the real spend ($35.90 against a $35 cap -- see NOTE below).
#
# Usage: ./scripts/run_oleg_pipeline.sh [budget_remaining_for_stage2] [cost_per_call]
#   both positional and optional; defaults below match what the 2026-09-28 run used.
#
# Costs real money against OPENROUTER_API_KEY (in .env). Each stage can also be run manually
# with the underlying `uv run scripts/oleg_judge.py <subcommand>` calls shown below.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

SCREENER="openrouter/openai/gpt-5.6-luna"
SILVER="openrouter/anthropic/claude-opus-5.5"
BUDGET_STAGE2="${1:-16.37}"      # remaining $ for stage 2; recompute from GET /api/v1/key usage
COST_PER_CALL="${2:-0.1506}"     # from results/oleg/pilot/report.md after the pilot step below
                                  # NOTE: this real run underestimated real Stage 2 cost by ~10%
                                  # ($35.90 actual vs $35 cap) -- pass a lower budget or add a
                                  # safety margin (e.g. multiply COST_PER_CALL by 1.15) to avoid
                                  # repeating the overage.

echo "=== Step 1: pilot-select (deterministic, seeded, free) ==="
uv run scripts/oleg_judge.py pilot-select

echo "=== Step 2: pilot -- opus silver + 2 screener candidates, all 10 pilot logs ==="
for judge in "$SILVER" "openrouter/openai/gpt-5.6-luna" "openrouter/openai/gpt-5.1-codex-mini"; do
  for f in results/oleg/pilot/*.ids.txt; do
    stem=$(basename "$f" .ids.txt)
    log="results/inspect_logs/${stem}.eval"
    uv run scripts/oleg_judge.py run --stage pilot --judge "$judge" --log "$log" --samples "$f" --max-connections 8
  done
done

echo "=== Step 3: pilot-report (chooses screener, writes results/oleg/pilot/report.md) ==="
uv run scripts/oleg_judge.py pilot-report

echo "=== Step 4: stage 1 -- screen all non-pilot traces with the chosen screener ==="
for log in results/inspect_logs/*.eval; do
  stem=$(basename "$log" .eval)
  excl="results/oleg/pilot/${stem}.ids.txt"
  if [ -f "$excl" ]; then
    uv run scripts/oleg_judge.py run --stage screen --judge "$SCREENER" --log "$log" --exclude "$excl" --max-connections 16
  else
    uv run scripts/oleg_judge.py run --stage screen --judge "$SCREENER" --log "$log" --max-connections 16
  fi
done

echo "=== Step 5: select-confirm -- budget governor builds results/oleg/confirm/manifest.json ==="
uv run scripts/oleg_judge.py select-confirm --budget "$BUDGET_STAGE2" --cost-per-call "$COST_PER_CALL"

echo "=== Step 6: stage 2 -- opus confirms the manifest's (trace, group) pairs ==="
for log in results/inspect_logs/*.eval; do
  uv run scripts/oleg_judge.py run --stage confirm --judge "$SILVER" --log "$log" --manifest results/oleg/confirm/manifest.json --max-connections 8
done

echo "=== Step 7: merge -- final logs, results/oleg_hits.jsonl, results/oleg_prevalence.csv ==="
uv run scripts/oleg_judge.py merge

echo "done. Check spend: curl -s https://openrouter.ai/api/v1/key -H \"Authorization: Bearer \$OPENROUTER_API_KEY\""
