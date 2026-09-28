"""Descriptive EDA over the agent-traces dataset. See spec-eda.md.

Usage:
    uv run scripts/eda.py [--limit N]

Writes all outputs (CSV/JSON/TXT) to results/eda/. No LLM calls, no classification —
regexes and counting only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_dataset import get_traces  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "eda"

ERROR_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "nonzero_exit": re.compile(
        r"exit code[: ]+[1-9]\d*|returned non-zero exit status|exited with (code |status )?[1-9]\d*",
        re.IGNORECASE,
    ),
    "cmd_not_found": re.compile(r"command not found|/bin/(ba)?sh:.*: not found", re.IGNORECASE),
    "file_not_found": re.compile(r"No such file or directory|FileNotFoundError", re.IGNORECASE),
    "edit_no_match": re.compile(
        r"did not match|no match found for|old_string not found|string not found in file"
        r"|could not find the (text|string|pattern)",
        re.IGNORECASE,
    ),
    "timeout": re.compile(r"\btimed out\b|TimeoutExpired|TimeoutError|Command timed out", re.IGNORECASE),
    "permission": re.compile(r"Permission denied|PermissionError", re.IGNORECASE),
    "syntax_import": re.compile(r"SyntaxError|ImportError|ModuleNotFoundError"),
}
LOOKS_ERROR = re.compile(r"\berror\b|\bfail(ed|ure)?\b|\bexception\b", re.IGNORECASE)
KNOWN_TOOLS = {"bash", "read", "write", "edit"}


def load(limit: int | None) -> pd.DataFrame:
    ds = get_traces()
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def char_lengths(conversation: list[dict]) -> dict:
    content_chars = 0
    reasoning_chars = 0
    tool_output_chars = 0
    for turn in conversation:
        role = turn.get("role")
        if role == "assistant":
            content_chars += len(turn.get("content") or "")
            reasoning_chars += len(turn.get("reasoning_content") or "")
        elif role == "tool":
            tool_output_chars += len(turn.get("content") or "")
    return {
        "content_chars": content_chars,
        "reasoning_chars": reasoning_chars,
        "tool_output_chars": tool_output_chars,
    }


def tool_call_counts(conversation: list[dict]) -> Counter:
    c = Counter()
    for turn in conversation:
        if turn.get("role") != "assistant":
            continue
        for tc in turn.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name", "<missing>")
            c[name] += 1
    return c


TASK_STATEMENT_MARKER = "## Issue"


def mask_task_statement(first_user_content: str) -> str:
    idx = first_user_content.find(TASK_STATEMENT_MARKER)
    if idx == -1:
        return first_user_content
    return first_user_content[:idx] + "<TASK_STATEMENT_MASKED>"


def scaffold_fingerprint(system_prompt: str, tools: list, first_user_content: str) -> str:
    masked_user = mask_task_statement(first_user_content)
    payload = json.dumps({"system_prompt": system_prompt, "tools": tools, "first_user": masked_user}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def first_user_message(conversation: list[dict]) -> str:
    for turn in conversation:
        if turn.get("role") == "user":
            return turn.get("content") or ""
    return ""


def build_row_records(ds) -> list[dict]:
    records = []
    for i, row in enumerate(ds):
        conv = row["conversation"]
        lengths = char_lengths(conv)
        tc_counts = tool_call_counts(conv)
        n_assistant_turns = sum(1 for t in conv if t.get("role") == "assistant")
        n_tool_calls = sum(tc_counts.values())
        first_user = first_user_message(conv)
        idx = first_user.find(TASK_STATEMENT_MARKER)
        task_statement = first_user[idx:] if idx != -1 else first_user
        task_statement_hash = hashlib.sha256(task_statement.encode()).hexdigest()[:16]
        fp = scaffold_fingerprint(row["system_prompt"], row["tools"], first_user)
        last_turn = conv[-1] if conv else {}
        ends_mid_tool_call = bool(last_turn.get("tool_calls"))
        records.append(
            {
                "row_idx": i,
                "model": row["model"],
                "job": row["job"],
                "trial": row["trial"],
                "task_id": row["task_id"],
                "status": row["status"],
                "reward": row["reward"],
                "n_proxy_requests": row["n_proxy_requests"],
                "n_assistant_turns": n_assistant_turns,
                "n_tool_calls": n_tool_calls,
                "n_conv_turns": len(conv),
                "content_chars": lengths["content_chars"],
                "reasoning_chars": lengths["reasoning_chars"],
                "tool_output_chars": lengths["tool_output_chars"],
                "tool_calls_by_name": dict(tc_counts),
                "fingerprint": fp,
                "task_statement_hash": task_statement_hash,
                "last_role": last_turn.get("role"),
                "ends_mid_tool_call": ends_mid_tool_call,
                "final_turn_empty": (
                    last_turn.get("role") == "assistant"
                    and not (last_turn.get("content") or "").strip()
                    and not (last_turn.get("reasoning_content") or "").strip()
                    and not last_turn.get("tool_calls")
                ),
            }
        )
    return records


def write_counts(df: pd.DataFrame):
    df["model"].value_counts().rename_axis("model").reset_index(name="count").to_csv(
        OUT_DIR / "counts_per_model.csv", index=False
    )
    df["task_id"].value_counts().rename_axis("task_id").reset_index(name="count").to_csv(
        OUT_DIR / "counts_per_task.csv", index=False
    )
    df["job"].value_counts().rename_axis("job").reset_index(name="count").to_csv(
        OUT_DIR / "counts_per_job.csv", index=False
    )
    pd.crosstab(df["model"], df["task_id"]).to_csv(OUT_DIR / "crosstab_model_task.csv")
    pd.crosstab(df["model"], df["job"]).to_csv(OUT_DIR / "crosstab_model_job.csv")


def write_lengths(df: pd.DataFrame):
    tool_rows = []
    for _, r in df.iterrows():
        for tool_name, cnt in r["tool_calls_by_name"].items():
            tool_rows.append({"row_idx": r["row_idx"], "model": r["model"], "job": r["job"], "tool": tool_name, "count": cnt})
    pd.DataFrame(tool_rows).to_csv(OUT_DIR / "tool_call_counts_per_trace.csv", index=False)

    cols = ["n_assistant_turns", "n_tool_calls", "content_chars", "reasoning_chars", "tool_output_chars"]
    df.groupby("model")[cols].describe().to_csv(OUT_DIR / "lengths_by_model.csv")
    df.groupby("job")[cols].describe().to_csv(OUT_DIR / "lengths_by_job.csv")
    df[["row_idx", "model", "job", "task_id", *cols]].to_csv(OUT_DIR / "lengths_per_trace.csv", index=False)


def write_grade_family(df: pd.DataFrame):
    df.groupby(["model", "job"])["status"].value_counts().unstack(fill_value=0).to_csv(
        OUT_DIR / "grade_status_by_model_job.csv"
    )
    df.groupby("model")["reward"].agg(["mean", "sum", "count"]).to_csv(OUT_DIR / "grade_reward_by_model.csv")
    df.groupby("job")["reward"].agg(["mean", "sum", "count"]).to_csv(OUT_DIR / "grade_reward_by_job.csv")

    inconsistent = df[((df["status"] == "OK") & (df["reward"] == 0)) | ((df["status"] != "OK") & (df["reward"] == 1))]
    inconsistent_records = inconsistent[["row_idx", "model", "job", "task_id", "status", "reward"]].to_dict(
        orient="records"
    )
    (OUT_DIR / "grade_consistency_anomalies.json").write_text(json.dumps(inconsistent_records, indent=2))


def write_proxy_family(df: pd.DataFrame, ds):
    status_counter = Counter()
    per_trace_rows = []
    for i, row in enumerate(ds):
        reqs = row["proxy_requests"] or []
        codes = Counter(r.get("status") for r in reqs)
        status_counter.update(codes)
        non200 = sum(v for k, v in codes.items() if k != 200)
        per_trace_rows.append(
            {
                "row_idx": i,
                "model": row["model"],
                "job": row["job"],
                "task_id": row["task_id"],
                "n_proxy_requests": len(reqs),
                "n_non200": non200,
                "non200_rate": (non200 / len(reqs)) if reqs else 0.0,
            }
        )
    pd.DataFrame({"status_code": list(status_counter.keys()), "count": list(status_counter.values())}).sort_values(
        "status_code"
    ).to_csv(OUT_DIR / "proxy_status_code_counts.csv", index=False)

    per_trace_df = pd.DataFrame(per_trace_rows)
    per_trace_df.to_csv(OUT_DIR / "proxy_per_trace.csv", index=False)
    per_trace_df.groupby("model")[["n_proxy_requests", "n_non200", "non200_rate"]].mean().to_csv(
        OUT_DIR / "proxy_rate_by_model.csv"
    )
    per_trace_df.groupby("job")[["n_proxy_requests", "n_non200", "non200_rate"]].mean().to_csv(
        OUT_DIR / "proxy_rate_by_job.csv"
    )


def write_tool_result_errors(ds, seed: int = 0):
    taxonomy_rows = []
    unmatched_samples = []
    for i, row in enumerate(ds):
        for msg_idx, turn in enumerate(row["conversation"]):
            if turn.get("role") != "tool":
                continue
            content = turn.get("content") or ""
            matched_any = False
            for name, pattern in ERROR_PATTERNS.items():
                if pattern.search(content):
                    matched_any = True
                    taxonomy_rows.append(
                        {
                            "row_idx": i,
                            "model": row["model"],
                            "job": row["job"],
                            "task_id": row["task_id"],
                            "msg_idx": msg_idx,
                            "error_type": name,
                        }
                    )
            if not matched_any and LOOKS_ERROR.search(content):
                unmatched_samples.append(
                    f"row_idx={i} model={row['model']} job={row['job']} task_id={row['task_id']} msg_idx={msg_idx}\n"
                    + content[:500]
                )

    tax_df = pd.DataFrame(taxonomy_rows)
    tax_df.to_csv(OUT_DIR / "tool_error_taxonomy_hits.csv", index=False)
    if not tax_df.empty:
        tax_df["error_type"].value_counts().rename_axis("error_type").reset_index(name="count").to_csv(
            OUT_DIR / "tool_error_taxonomy_summary.csv", index=False
        )
        tax_df.groupby(["model", "error_type"]).size().unstack(fill_value=0).to_csv(
            OUT_DIR / "tool_error_taxonomy_by_model.csv"
        )

    rng = random.Random(seed)
    sample_size = min(200, len(unmatched_samples))
    sample = rng.sample(unmatched_samples, sample_size) if unmatched_samples else []
    (OUT_DIR / "tool_error_unmatched_sample.txt").write_text(
        f"Total unmatched error-looking tool outputs: {len(unmatched_samples)}\n"
        f"Sampled {sample_size} for manual review.\n\n" + ("\n\n---\n\n".join(sample))
    )


def write_malformed_turns(ds):
    rows = []
    for i, row in enumerate(ds):
        conv = row["conversation"]
        for msg_idx, turn in enumerate(conv):
            role = turn.get("role")
            if role == "assistant":
                has_content = bool((turn.get("content") or "").strip())
                has_reasoning = bool((turn.get("reasoning_content") or "").strip())
                has_tool_calls = bool(turn.get("tool_calls"))
                if not has_content and not has_reasoning and not has_tool_calls:
                    rows.append(
                        {"row_idx": i, "model": row["model"], "job": row["job"], "task_id": row["task_id"], "msg_idx": msg_idx, "issue": "empty_assistant_turn"}
                    )
                for tc in turn.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments")
                    if name not in KNOWN_TOOLS:
                        rows.append(
                            {"row_idx": i, "model": row["model"], "job": row["job"], "task_id": row["task_id"], "msg_idx": msg_idx, "issue": f"unknown_tool:{name}"}
                        )
                    if isinstance(args, str):
                        try:
                            json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            rows.append(
                                {"row_idx": i, "model": row["model"], "job": row["job"], "task_id": row["task_id"], "msg_idx": msg_idx, "issue": "unparsable_tool_call_arguments"}
                            )
        if conv and conv[-1].get("role") == "assistant" and conv[-1].get("tool_calls"):
            rows.append(
                {"row_idx": i, "model": row["model"], "job": row["job"], "task_id": row["task_id"], "msg_idx": len(conv) - 1, "issue": "trace_ends_mid_tool_call"}
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "malformed_turns.csv", index=False)
    if not df.empty:
        df["issue"].value_counts().rename_axis("issue").reset_index(name="count").to_csv(
            OUT_DIR / "malformed_turns_summary.csv", index=False
        )


def write_reruns_and_configs(df: pd.DataFrame):
    rerun_groups = df.groupby(["model", "task_id"]).size().reset_index(name="n_traces")
    reruns = rerun_groups[rerun_groups["n_traces"] > 1]
    reruns.to_csv(OUT_DIR / "reruns_model_task.csv", index=False)

    rerun_rerun_fp = df.groupby(["model", "task_id", "fingerprint"]).size().reset_index(name="n_traces")
    rerun_rerun_fp[rerun_rerun_fp["n_traces"] > 1].to_csv(OUT_DIR / "rerun_rerun_by_fingerprint.csv", index=False)

    rerun_rerun_job = df.groupby(["model", "task_id", "job"]).size().reset_index(name="n_traces")
    rerun_rerun_job[rerun_rerun_job["n_traces"] > 1].to_csv(OUT_DIR / "rerun_rerun_by_job.csv", index=False)

    fp_per_job = df.groupby("job")["fingerprint"].nunique().reset_index(name="n_distinct_fingerprints")
    jobs_multi_fp = fp_per_job[fp_per_job["n_distinct_fingerprints"] > 1]
    job_per_fp = df.groupby("fingerprint")["job"].nunique().reset_index(name="n_distinct_jobs")
    fp_multi_job = job_per_fp[job_per_fp["n_distinct_jobs"] > 1]
    disagreement = {
        "jobs_with_multiple_fingerprints": jobs_multi_fp.to_dict(orient="records"),
        "fingerprints_spanning_multiple_jobs": fp_multi_job.to_dict(orient="records"),
    }
    (OUT_DIR / "fingerprint_job_disagreement.json").write_text(json.dumps(disagreement, indent=2))

    task_stmt_variants = df.groupby("task_id")["task_statement_hash"].nunique().reset_index(name="n_distinct_statements")
    task_stmt_variants[task_stmt_variants["n_distinct_statements"] > 1].to_csv(
        OUT_DIR / "task_statement_variants.csv", index=False
    )


def write_scaffold_diffs(ds, df: pd.DataFrame):
    lines = []
    for task_id, group in df.groupby("task_id"):
        if group["fingerprint"].nunique() <= 1:
            continue
        row_indices = group["row_idx"].tolist()[:2]
        rows = [ds[int(idx)] for idx in row_indices]
        a, b = rows[0], rows[1]
        diffs = []
        if a["system_prompt"] != b["system_prompt"]:
            diffs.append("system_prompt differs")
        if json.dumps(a["tools"], sort_keys=True) != json.dumps(b["tools"], sort_keys=True):
            diffs.append("tools schema differs")
        fu_a, fu_b = first_user_message(a["conversation"]), first_user_message(b["conversation"])
        if fu_a != fu_b:
            diffs.append("first_user_message differs")
        lines.append(
            f"task_id={task_id} row_a={row_indices[0]} row_b={row_indices[1]} diffs={diffs}\n"
            f"  first_user_a[:200]={fu_a[:200]!r}\n  first_user_b[:200]={fu_b[:200]!r}\n"
        )
    (OUT_DIR / "scaffold_diffs_sample.txt").write_text("\n".join(lines) if lines else "No scaffold diffs found across reruns.\n")


def write_envelope(df: pd.DataFrame):
    df.groupby(["model", "job"])[["n_conv_turns", "content_chars", "reasoning_chars", "tool_output_chars"]].max().to_csv(
        OUT_DIR / "envelope_max_by_model_job.csv"
    )
    end_status = df.groupby(["status", "last_role", "ends_mid_tool_call", "final_turn_empty"]).size().reset_index(
        name="count"
    )
    end_status.to_csv(OUT_DIR / "envelope_end_status.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load(args.limit)
    records = build_row_records(ds)
    df = pd.DataFrame(records)

    write_counts(df)
    write_lengths(df)
    write_grade_family(df)
    write_proxy_family(df, ds)
    write_tool_result_errors(ds)
    write_malformed_turns(ds)
    write_reruns_and_configs(df)
    write_scaffold_diffs(ds, df)
    write_envelope(df)

    print(f"Wrote EDA outputs for {len(df)} traces to {OUT_DIR}")


if __name__ == "__main__":
    main()
