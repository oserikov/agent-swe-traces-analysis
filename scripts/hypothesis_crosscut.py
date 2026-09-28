"""Cross-model, cross-job and cross-task breakdown of the 13 oleg-* hypothesis verdicts.

Usage:
    uv run scripts/hypothesis_crosscut.py [--perms 5000] [--out results/oleg_crosscut]

Group comparisons use the screener (gpt-5.6-luna) verdict, the only judge that saw all 370 traces
with the same prompt and rendering: Stage 1 (`results/oleg/screen`) plus the 16 pilot traces
(`results/oleg/pilot/*gpt-5_6-luna.eval`). The merged `-oleg.eval` scores are not used for this,
because they mix Opus and screener verdicts depending on which traces were sent to confirmation.
Opus verdicts (confirm + pilot) are used only to estimate the screener's precision per model.

Tests are permutation tests (seeded), each over one family of 13 hypotheses with Benjamini-Hochberg
q-values:
- model: shuffle model labels across all traces; statistic = chi-square of hit rate by model;
- job within model: shuffle job labels within each multi-job model (cyan, delta, flint, garnet);
- task / repo: shuffle task (or repo) labels within each job, which keeps every job's hit rate;
  statistic = number of trace pairs sharing a task (repo) where both are hits.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_dataset import get_traces  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCREEN_LOGS = [
    *sorted(glob.glob(str(ROOT / "results/oleg/screen/*.eval"))),
    *sorted(glob.glob(str(ROOT / "results/oleg/pilot/*gpt-5_6-luna.eval"))),
]
OPUS_LOGS = [
    *sorted(glob.glob(str(ROOT / "results/oleg/confirm/*.eval"))),
    *sorted(glob.glob(str(ROOT / "results/oleg/pilot/*claude-opus-5_5.eval"))),
]
SCORE_RE = re.compile(r"^(oleg-[a-z-]+)__(screen|pilot|confirm)$")


def load_verdicts(paths: list[str], column: str) -> pd.DataFrame:
    rows = []
    for path in paths:
        for sample in read_eval_log(path).samples:
            for key, score in (sample.scores or {}).items():
                m = SCORE_RE.match(key)
                if not m:
                    continue
                value = score.value
                if not isinstance(value, int | float) or np.isnan(value):
                    continue
                rows.append(
                    {
                        "trial": sample.id,
                        "hypothesis": m.group(1),
                        column: int(value),
                        f"{column}_quote": (score.metadata or {}).get("quote", ""),
                        f"{column}_msg": (score.metadata or {}).get("message_index"),
                    }
                )
    return pd.DataFrame(rows)


def ending(conv: list[dict]) -> str:
    last = conv[-1] if conv else {}
    if last.get("tool_calls"):
        return "open call"
    return "final message" if last.get("role") == "assistant" else "tool result"


def trace_table() -> pd.DataFrame:
    ds = get_traces()
    return pd.DataFrame(
        {
            "row": range(len(ds)),
            "trial": ds["trial"],
            "model": ds["model"],
            "job": ds["job"],
            "task_id": ds["task_id"],
            "repo": [re.sub(r"-\d+$", "", t) for t in ds["task_id"]],
            "reward": ds["reward"],
            "turns": [
                sum(m.get("role") == "assistant" for m in c) for c in ds["conversation"]
            ],
            # Step cap, proxy kill and TL all end on an unanswered tool call (EDA "Why the 50
            # end mid call"); such traces have no final message to claim success in.
            "ending": [ending(c) for c in ds["conversation"]],
        }
    )


def bh(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    ranked = order * len(order) / np.arange(1, len(order) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(upper=1)
    return q.reindex(p.index)


def chi2_stat(hits: np.ndarray, groups: np.ndarray, n_groups: int) -> float:
    n = np.bincount(groups, minlength=n_groups)
    k = np.bincount(groups, weights=hits, minlength=n_groups)
    p = hits.mean()
    if p in (0, 1):
        return 0.0
    mask = n > 0
    return float((((k[mask] - n[mask] * p) ** 2) / (n[mask] * p * (1 - p))).sum())


def pair_stat(hits: np.ndarray, groups: np.ndarray, n_groups: int) -> float:
    k = np.bincount(groups, weights=hits, minlength=n_groups)
    return float((k * (k - 1) / 2).sum())


def shuffle_within(labels: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    out = labels.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = labels[rng.permutation(idx)]
    return out


def perm_test(hits, labels, strata, stat, perms, rng) -> tuple[float, float, float]:
    codes, uniq = pd.factorize(labels)
    n_groups = len(uniq)
    observed = stat(hits, codes, n_groups)
    null = np.array(
        [
            stat(
                hits,
                shuffle_within(codes, strata, rng)
                if strata is not None
                else rng.permutation(codes),
                n_groups,
            )
            for _ in range(perms)
        ]
    )
    p = (1 + (null >= observed).sum()) / (perms + 1)
    return observed, float(null.mean()), float(p)


def length_test(hits, log_turns, strata, perms, rng) -> tuple[float, float, float]:
    """Within-stratum shuffle of verdicts; statistic = mean log-turns of hits minus misses."""

    def stat(h):
        if h.sum() in (0, len(h)):
            return 0.0
        return float(log_turns[h == 1].mean() - log_turns[h == 0].mean())

    observed = stat(hits)
    null = np.array([stat(shuffle_within(hits, strata, rng)) for _ in range(perms)])
    p = (1 + (null >= observed).sum()) / (perms + 1)
    return observed, float(null.mean()), float(p)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--perms", type=int, default=5000)
    parser.add_argument("--out", default=str(ROOT / "results/oleg_crosscut"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    traces = trace_table()
    screen = load_verdicts(SCREEN_LOGS, "screen")
    opus = load_verdicts(OPUS_LOGS, "opus")
    df = screen.merge(traces, on="trial", how="left").merge(
        opus, on=["trial", "hypothesis"], how="left"
    )
    assert df["row"].notna().all(), "screener sample not in dataset"
    per_hyp = df.groupby("hypothesis")["trial"].nunique()
    print(
        f"screener verdicts: {len(df)} rows; traces per hypothesis {sorted(set(per_hyp))}"
    )
    df.to_csv(out / "verdicts.csv", index=False)

    models = sorted(traces["model"].unique())
    rates = df.pivot_table(
        index="hypothesis", columns="model", values="screen", aggfunc="mean"
    )
    hits = df.pivot_table(
        index="hypothesis", columns="model", values="screen", aggfunc="sum"
    )
    rates.round(3).to_csv(out / "screen_rate_by_model.csv")
    hits.astype(int).to_csv(out / "screen_hits_by_model.csv")

    # Screener precision against Opus, among screen hits Opus also judged.
    judged = df[(df["screen"] == 1) & df["opus"].notna()]
    precision = (
        judged.groupby(["hypothesis", "model"])["opus"]
        .agg(["size", "sum"])
        .rename(columns={"size": "opus_judged", "sum": "opus_hits"})
        .reset_index()
    )
    precision.to_csv(out / "screen_precision_by_model.csv", index=False)
    pooled_precision = judged.groupby("model")["opus"].agg(["size", "sum"])
    pooled_precision["precision"] = (
        pooled_precision["sum"] / pooled_precision["size"]
    ).round(2)
    pooled_precision.to_csv(out / "screen_precision_pooled_by_model.csv")
    opus_hits = (
        df[df["opus"] == 1]
        .groupby(["hypothesis", "model"])["trial"]
        .nunique()
        .unstack(fill_value=0)
    )
    opus_hits.to_csv(out / "opus_confirmed_hits_by_model.csv")

    multi_job = traces.groupby("model")["job"].nunique()
    multi_job = multi_job[multi_job > 1].index.tolist()
    tests, job_rows, task_top = [], [], []
    for hyp, g in df.groupby("hypothesis"):
        g = g.sort_values("row")
        h = g["screen"].to_numpy(float)
        obs, null, p = perm_test(
            h, g["model"].to_numpy(), None, chi2_stat, args.perms, rng
        )
        tests.append(
            {
                "hypothesis": hyp,
                "family": "model",
                "stat": obs,
                "null_mean": null,
                "p": p,
            }
        )

        gm = g[g["model"].isin(multi_job)]
        obs, null, p = perm_test(
            gm["screen"].to_numpy(float),
            gm["job"].to_numpy(),
            gm["model"].to_numpy(),
            chi2_stat,
            args.perms,
            rng,
        )
        tests.append(
            {
                "hypothesis": hyp,
                "family": "job|model",
                "stat": obs,
                "null_mean": null,
                "p": p,
            }
        )
        gc = gm[gm["ending"] == "final message"]
        obs, null, p = perm_test(
            gc["screen"].to_numpy(float),
            gc["job"].to_numpy(),
            gc["model"].to_numpy(),
            chi2_stat,
            args.perms,
            rng,
        )
        tests.append(
            {
                "hypothesis": hyp,
                "family": "job|model,final-msg",
                "stat": obs,
                "null_mean": null,
                "p": p,
            }
        )
        obs, null, p = length_test(
            h, np.log(g["turns"].to_numpy(float)), g["job"].to_numpy(), args.perms, rng
        )
        tests.append(
            {
                "hypothesis": hyp,
                "family": "length|job",
                "stat": obs,
                "null_mean": null,
                "p": p,
            }
        )
        for (model, job), gj in gm.groupby(["model", "job"]):
            job_rows.append(
                {
                    "hypothesis": hyp,
                    "model": model,
                    "job": job,
                    "n": len(gj),
                    "hits": int(gj["screen"].sum()),
                }
            )

        for family, col in (("task|job", "task_id"), ("repo|job", "repo")):
            obs, null, p = perm_test(
                h, g[col].to_numpy(), g["job"].to_numpy(), pair_stat, args.perms, rng
            )
            tests.append(
                {
                    "hypothesis": hyp,
                    "family": family,
                    "stat": obs,
                    "null_mean": null,
                    "p": p,
                }
            )

        by_task = g.groupby("task_id").agg(
            n=("screen", "size"), hits=("screen", "sum"), models=("model", "nunique")
        )
        hit_rows = g[g["screen"] == 1]
        by_task["hit_models"] = hit_rows.groupby("task_id")["model"].agg(
            lambda s: ",".join(sorted(s))
        )
        by_task["hit_rows"] = hit_rows.groupby("task_id")["row"].agg(
            lambda s: ",".join(str(int(r)) for r in sorted(s))
        )
        for task, r in by_task[
            (by_task["hits"] >= 2) & (by_task["models"] >= 2)
        ].iterrows():
            task_top.append({"hypothesis": hyp, "task_id": task, **r.to_dict()})

    tests = pd.DataFrame(tests)
    tests["q"] = tests.groupby("family", group_keys=False)["p"].apply(bh)
    tests["ratio"] = (tests["stat"] / tests["null_mean"].replace(0, np.nan)).round(2)
    tests.round(4).to_csv(out / "permutation_tests.csv", index=False)
    pd.DataFrame(job_rows).to_csv(out / "rate_by_job.csv", index=False)
    pd.DataFrame(task_top).to_csv(out / "cross_model_task_hits.csv", index=False)

    # Rerun pairs: same model, same task, two traces.
    pairs = []
    for (hyp, model, task), g in df.groupby(["hypothesis", "model", "task_id"]):
        if len(g) == 2:
            a, b = g.sort_values("job")["screen"].tolist()
            jobs = "/".join(g.sort_values("job")["job"])
            pairs.append(
                {
                    "hypothesis": hyp,
                    "model": model,
                    "task_id": task,
                    "jobs": jobs,
                    "a": a,
                    "b": b,
                }
            )
    pd.DataFrame(pairs).to_csv(out / "rerun_pairs.csv", index=False)

    claims = df[df["hypothesis"] == "oleg-claims-success"]
    claims_table = claims.groupby(["model", "ending"]).apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "claims": int(g["screen"].sum()),
                "failed": int((g["reward"] == 0).sum()),
                "claims_and_failed": int(
                    ((g["screen"] == 1) & (g["reward"] == 0)).sum()
                ),
            }
        ),
        include_groups=False,
    )
    claims_table.to_csv(out / "claims_success_by_ending.csv")

    lines = ["# Hypothesis verdicts across model, job and task (screener, n=370)", ""]
    lines += [
        "## Screen hits by model",
        "",
        (
            hits.astype(int).astype(str)
            + "/"
            + df.groupby("model")["trial"].nunique().reindex(models).astype(str)
        ).to_string(),
        "",
    ]
    lines += [
        "## Screener precision vs Opus, pooled over hypotheses",
        "",
        pooled_precision.to_string(),
        "",
    ]
    lines += [
        "## Permutation tests (BH q within family)",
        "",
        tests.pivot(index="hypothesis", columns="family", values="q")
        .round(4)
        .to_string(),
        "",
        "Observed / null-mean statistic:",
        "",
        tests.pivot(index="hypothesis", columns="family", values="ratio").to_string(),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
