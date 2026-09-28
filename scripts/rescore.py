"""Add several LLM-judge verdicts to an existing Inspect .eval log, keeping the original score.

Usage:
    uv run scripts/rescore.py logs/run.eval
    uv run scripts/rescore.py logs/run.eval --judge openrouter/anthropic/claude-sonnet-5 --limit 5

Judges are Inspect model strings. `openrouter/...` judges need OPENROUTER_API_KEY in .env.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Scorer, accuracy, model_graded_qa, scorer, stderr

DEFAULT_JUDGES = [
    "openrouter/anthropic/claude-opus-5.5",
    "openrouter/anthropic/claude-sonnet-5",
    "openrouter/anthropic/claude-haiku-4.5",
]

# model_graded_qa's default template compares an answer to a target; SWE traces often have an
# empty target, so ask the judge about the whole transcript instead.
TEMPLATE = """You are grading a transcript of a software-engineering agent.

[BEGIN DATA]
[Task and transcript]: {question}
[Agent's final answer]: {answer}
[Reference / criterion, may be empty]: {criterion}
[END DATA]

Did the agent correctly and completely solve the task? Judge the actual work in the
transcript (edits, commands, test results), not the agent's own claims of success.

{instructions}
"""

JUDGE_MAX_TOKENS = 4096


def judge(model: str, max_tokens: int = JUDGE_MAX_TOKENS) -> Scorer:
    """Build a transcript-grading scorer named `judge_<model>`.

    `max_tokens` caps the judge's output. Without it, Inspect requests the model's full output
    limit (64k tokens for Claude), and OpenRouter reserves credit for that whole ceiling before
    the call, so a key with a spending limit fails with HTTP 402 even though the verdict itself
    is short. 4096 leaves room for step-by-step reasoning plus the final "GRADE: X" line; if
    reasoning gets cut off before the grade, the sample is left unscored, so raise it then.
    """
    name = "judge_" + model.split("/")[-1].replace(".", "_").replace("-", "_")

    @scorer(metrics=[accuracy(), stderr()], name=name)
    def _judge() -> Scorer:
        return model_graded_qa(
            template=TEMPLATE,
            include_history=True,
            partial_credit=True,
            model=get_model(model, config=GenerateConfig(max_tokens=max_tokens)),
        )

    return _judge()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", type=Path, help="input .eval log")
    parser.add_argument(
        "--judge",
        action="append",
        help="judge model (repeatable); default: 3 Claude models via OpenRouter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="only re-grade the first N samples (the output keeps only those)",
    )
    parser.add_argument(
        "--out", type=Path, help="output log (default: <log>-rejudged.eval)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=JUDGE_MAX_TOKENS,
        help="cap on each judge's output tokens (default: %(default)s; see judge())",
    )
    args = parser.parse_args()
    judges = args.judge or DEFAULT_JUDGES

    load_dotenv()
    if any(j.startswith("openrouter/") for j in judges) and not os.getenv(
        "OPENROUTER_API_KEY"
    ):
        sys.exit(
            "OPENROUTER_API_KEY missing: add it to .env (see .env.example) or pass non-OpenRouter --judge models."
        )

    log = read_eval_log(str(args.log))
    if not log.samples:
        sys.exit(f"{args.log} has no samples to re-grade.")
    if args.limit:
        log.samples = log.samples[: args.limit]

    rescored = score(log, [judge(j, args.max_tokens) for j in judges], action="append")

    out = args.out or args.log.with_name(f"{args.log.stem}-rejudged.eval")
    write_eval_log(rescored, str(out))

    assert rescored.results is not None
    for s in rescored.results.scores:
        metrics = ", ".join(f"{k}={m.value:.3f}" for k, m in s.metrics.items())
        print(f"{s.name:40s} {metrics}")
    print(f"wrote {out}  (view with: uv run inspect view --log-dir {out.parent})")


if __name__ == "__main__":
    main()
