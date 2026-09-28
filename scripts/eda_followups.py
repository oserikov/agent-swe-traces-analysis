"""Follow-up tables answering review notes on the EDA section. Regex/counting only, no LLM calls.

Usage:
    uv run scripts/eda_followups.py

Writes to results/eda/followups/.
"""

from __future__ import annotations

import difflib
import itertools
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda import (  # noqa: E402
    ERROR_PATTERNS,
    KNOWN_TOOLS,
    LOOKS_ERROR,
    TASK_STATEMENT_MARKER,
    build_row_records,
    first_user_message,
    mask_task_statement,
)
from fetch_dataset import get_traces  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "eda" / "followups"


def run_length(codes: list) -> str:
    return " ".join(f"{code}x{len(list(group))}" for code, group in itertools.groupby(codes))


def proxy_vs_turns(ds, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in enumerate(ds):
        codes = [r.get("status") for r in row["proxy_requests"] or []]
        n200 = sum(1 for c in codes if c == 200)
        trailing_non200 = len(codes) - len(list(itertools.dropwhile(lambda c: c != 200, reversed(codes))))
        rows.append(
            {
                "row_idx": i,
                "model": row["model"],
                "job": row["job"],
                "task_id": row["task_id"],
                "status": row["status"],
                "n_proxy_requests": len(codes),
                "n_200": n200,
                "n_non200": len(codes) - n200,
                "n_assistant_turns": int(df.loc[i, "n_assistant_turns"]),
                "n200_equals_assistant_turns": n200 == int(df.loc[i, "n_assistant_turns"]),
                "trailing_non200": trailing_non200,
                "ends_mid_tool_call": bool(df.loc[i, "ends_mid_tool_call"]),
                "sequence_rle": run_length(codes),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "proxy_vs_turns.csv", index=False)
    pd.crosstab(
        [out["status"], out["ends_mid_tool_call"]], out["trailing_non200"].clip(upper=4).rename("trailing_non200 (4 = 4+)")
    ).to_csv(OUT_DIR / "proxy_tail_vs_ending.csv")
    return out


def write_job_profile(df: pd.DataFrame, proxy: pd.DataFrame):
    merged = df.merge(proxy[["row_idx", "n_non200", "n200_equals_assistant_turns"]], on="row_idx")
    merged["non200_rate"] = merged["n_non200"] / merged["n_proxy_requests"].where(merged["n_proxy_requests"] > 0)
    grouped = merged.groupby(["model", "job"])
    profile = pd.DataFrame(
        {
            "n_traces": grouped.size(),
            "reward_mean": grouped["reward"].mean().round(3),
            "n_OK": grouped["status"].apply(lambda s: (s == "OK").sum()),
            "n_WA": grouped["status"].apply(lambda s: (s == "WA").sum()),
            "n_IL": grouped["status"].apply(lambda s: (s == "IL").sum()),
            "n_TL": grouped["status"].apply(lambda s: (s == "TL").sum()),
            "non200_rate_mean": grouped["non200_rate"].mean().round(3),
            "traces_with_any_non200": grouped["n_non200"].apply(lambda s: (s > 0).sum()),
            "asst_turns_median": grouped["n_assistant_turns"].median(),
            "asst_turns_max": grouped["n_assistant_turns"].max(),
            "tool_output_chars_median": grouped["tool_output_chars"].median(),
            "tool_output_chars_max": grouped["tool_output_chars"].max(),
            "reasoning_chars_median": grouped["reasoning_chars"].median(),
            "content_chars_median": grouped["content_chars"].median(),
            "n_fingerprints": grouped["fingerprint"].nunique(),
            "ends_mid_tool_call": grouped["ends_mid_tool_call"].sum(),
            "n200_equals_asst_turns": grouped["n200_equals_assistant_turns"].sum(),
        }
    ).reset_index()
    profile.to_csv(OUT_DIR / "job_profile.csv", index=False)


def write_proxy_row(proxy: pd.DataFrame, ds, row_idx: int):
    focus = proxy.loc[row_idx]
    conv = ds[row_idx]["conversation"]
    last = conv[-1]
    lines = [
        f"row_idx={row_idx} model={focus['model']} job={focus['job']} task_id={focus['task_id']} status={focus['status']}",
        f"n_proxy_requests={focus['n_proxy_requests']} n_200={focus['n_200']} n_non200={focus['n_non200']} "
        f"n_assistant_turns={focus['n_assistant_turns']}",
        f"request sequence (run-length): {focus['sequence_rle']}",
        f"last message role={last.get('role')} has_tool_calls={bool(last.get('tool_calls'))}",
        "",
        f"All traces of job {focus['job']} (same columns):",
    ]
    same_job = proxy[proxy["job"] == focus["job"]]
    for _, r in same_job.iterrows():
        lines.append(
            f"  row {r['row_idx']:>3} {r['model']} {r['status']} requests={r['n_proxy_requests']} "
            f"non200={r['n_non200']} asst_turns={r['n_assistant_turns']} rle={r['sequence_rle']}"
        )
    run03 = proxy[proxy["job"] == "run-03"]
    lines += ["", "All traces of run-03 (for comparison):"]
    for _, r in run03.iterrows():
        lines.append(
            f"  row {r['row_idx']:>3} {r['model']} {r['status']} requests={r['n_proxy_requests']} "
            f"non200={r['n_non200']} asst_turns={r['n_assistant_turns']} rle={r['sequence_rle']}"
        )
    (OUT_DIR / f"proxy_row{row_idx}.txt").write_text("\n".join(lines) + "\n")


def write_nonschema_tool_names(ds):
    rows = []
    for i, row in enumerate(ds):
        for msg_idx, turn in enumerate(row["conversation"]):
            if turn.get("role") != "assistant":
                continue
            for tc in turn.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name")
                if name in KNOWN_TOOLS:
                    continue
                rows.append(
                    {
                        "row_idx": i,
                        "model": row["model"],
                        "job": row["job"],
                        "task_id": row["task_id"],
                        "status": row["status"],
                        "msg_idx": msg_idx,
                        "name": repr(name),
                        "arguments": json.dumps(fn.get("arguments"))[:150],
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "nonschema_tool_names.csv", index=False)
    out.groupby(["model", "job"]).agg(n_calls=("row_idx", "size"), n_traces=("row_idx", "nunique")).reset_index().to_csv(
        OUT_DIR / "nonschema_tool_names_by_model_job.csv", index=False
    )


def scaffold_parts(row) -> dict:
    first_user = first_user_message(row["conversation"])
    idx = first_user.find(TASK_STATEMENT_MARKER)
    return {
        "system_prompt": row["system_prompt"],
        "prompt_role": row["conversation"][0]["role"],
        "tools": json.dumps(row["tools"], sort_keys=True, indent=1),
        "preamble": mask_task_statement(first_user),
        "statement": first_user[idx:] if idx != -1 else first_user,
    }


def write_reruns(ds, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task_id), group in df.groupby(["model", "task_id"]):
        if len(group) < 2:
            continue
        a_idx, b_idx = (int(x) for x in group["row_idx"].tolist()[:2])
        a, b = scaffold_parts(ds[a_idx]), scaffold_parts(ds[b_idx])
        ra, rb = df.loc[a_idx], df.loc[b_idx]
        diff = [part for part in ("system_prompt", "prompt_role", "tools", "preamble", "statement") if a[part] != b[part]]
        rows.append(
            {
                "model": model,
                "task_id": task_id,
                "row_a": a_idx,
                "row_b": b_idx,
                "job_a": ra["job"],
                "job_b": rb["job"],
                "same_job": ra["job"] == rb["job"],
                "same_fingerprint": ra["fingerprint"] == rb["fingerprint"],
                "differing_parts": "+".join(diff) or "none",
                "status_a": ra["status"],
                "status_b": rb["status"],
                "reward_a": ra["reward"],
                "reward_b": rb["reward"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reruns_explained.csv", index=False)

    per_model = pd.DataFrame(
        {
            "n_traces": df.groupby("model").size(),
            "n_distinct_tasks": df.groupby("model")["task_id"].nunique(),
            "n_rerun_pairs": out.groupby("model").size(),
            "rerun_pairs_same_job": out[out["same_job"]].groupby("model").size(),
            "rerun_pairs_same_fingerprint": out[out["same_fingerprint"]].groupby("model").size(),
            "rerun_pairs_reward_differs": out[out["reward_a"] != out["reward_b"]].groupby("model").size(),
        }
    ).fillna(0).astype(int)
    per_model.reset_index().to_csv(OUT_DIR / "reruns_by_model.csv", index=False)

    traces_per_task = df.groupby("task_id").agg(n_traces=("row_idx", "size"), n_models=("model", "nunique"))
    traces_per_task.value_counts().reset_index(name="n_tasks").sort_values(["n_traces", "n_models"]).to_csv(
        OUT_DIR / "tasks_by_trace_and_model_count.csv", index=False
    )
    return out


def unified(a: str, b: str, label_a: str, label_b: str) -> str:
    return "\n".join(
        difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile=label_a, tofile=label_b, n=1, lineterm="")
    )


DATE_LINE = re.compile(r"Current date: (\S+)")


def write_scaffold_variants(ds) -> pd.DataFrame:
    tools_ids: dict[str, str] = {}
    rows = []
    for i, row in enumerate(ds):
        tools_json = json.dumps(row["tools"], sort_keys=True, indent=1)
        tools_ids.setdefault(tools_json, f"T{len(tools_ids) + 1}")
        date = DATE_LINE.search(row["system_prompt"])
        rows.append(
            {
                "row_idx": i,
                "model": row["model"],
                "job": row["job"],
                "date": date.group(1) if date else None,
                "prompt_role": row["conversation"][0]["role"],
                "system_prompt_date_masked": DATE_LINE.sub("Current date: X", row["system_prompt"]),
                "tools_variant": tools_ids[tools_json],
                "tools_json": tools_json,
            }
        )
    out = pd.DataFrame(rows)
    out.groupby(["model", "job"]).agg(
        n_traces=("row_idx", "size"),
        dates=("date", lambda s: " ".join(sorted(set(s)))),
        prompt_roles=("prompt_role", lambda s: " ".join(sorted(set(s)))),
        n_system_prompts_date_masked=("system_prompt_date_masked", "nunique"),
        tools_variants=("tools_variant", lambda s: " ".join(sorted(set(s)))),
    ).reset_index().to_csv(OUT_DIR / "scaffold_variants_by_model_job.csv", index=False)
    summary = {
        "distinct_system_prompts_raw": int(pd.Series([ds[i]["system_prompt"] for i in range(len(ds))]).nunique()),
        "distinct_system_prompts_date_masked": int(out["system_prompt_date_masked"].nunique()),
        "distinct_tools_schemas": int(out["tools_variant"].nunique()),
        "traces_per_tools_variant": out["tools_variant"].value_counts().to_dict(),
    }
    (OUT_DIR / "scaffold_variants_summary.json").write_text(json.dumps(summary, indent=2))
    return out


def write_scaffold_diff_examples(ds, reruns: pd.DataFrame, df: pd.DataFrame, variants: pd.DataFrame):
    lines = []
    only_sp = reruns[reruns["differing_parts"] == "system_prompt"]
    lines.append(
        f"===== system_prompt: {len(only_sp)} same-model rerun pairs differ ONLY in system_prompt; example below"
    )
    if not only_sp.empty:
        pick = only_sp.iloc[0]
        a_idx, b_idx = int(pick["row_a"]), int(pick["row_b"])
        ra, rb = df.loc[a_idx], df.loc[b_idx]
        lines.append(
            unified(
                ds[a_idx]["system_prompt"],
                ds[b_idx]["system_prompt"],
                f"row {a_idx} {ra['model']} {ra['job']} {ra['task_id']}",
                f"row {b_idx} {rb['model']} {rb['job']} {rb['task_id']}",
            )
        )
    lines.append("")
    only_tools = reruns[reruns["differing_parts"] == "tools"]
    lines.append(f"===== tools: {len(only_tools)} same-model rerun pairs differ ONLY in tools.")
    lines.append("Diff between the dataset's distinct tools schemas (first trace of each variant):")
    firsts = variants.drop_duplicates("tools_variant")
    for (_, a), (_, b) in itertools.combinations(firsts.iterrows(), 2):
        users_a = sorted(set(variants.loc[variants["tools_variant"] == a["tools_variant"], "model"] + "/" + variants.loc[variants["tools_variant"] == a["tools_variant"], "job"]))
        users_b = sorted(set(variants.loc[variants["tools_variant"] == b["tools_variant"], "model"] + "/" + variants.loc[variants["tools_variant"] == b["tools_variant"], "job"]))
        lines.append(f"{a['tools_variant']} used by: {', '.join(users_a)}")
        lines.append(f"{b['tools_variant']} used by: {', '.join(users_b)}")
        lines.append(unified(a["tools_json"], b["tools_json"], a["tools_variant"], b["tools_variant"]))
    (OUT_DIR / "scaffold_diff_examples.txt").write_text("\n".join(lines) + "\n")


VALIDATION_FAILED = re.compile(r'Validation failed for tool "(\w+)"')


def write_validation_failures(ds):
    rows = []
    for i, row in enumerate(ds):
        for msg_idx, turn in enumerate(row["conversation"]):
            if turn.get("role") != "tool":
                continue
            match = VALIDATION_FAILED.search(turn.get("content") or "")
            if match:
                rows.append({"row_idx": i, "model": row["model"], "job": row["job"], "msg_idx": msg_idx, "tool": match.group(1)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "validation_failures.csv", index=False)
    out.groupby(["model", "job", "tool"]).agg(n=("row_idx", "size"), n_traces=("row_idx", "nunique")).reset_index().to_csv(
        OUT_DIR / "validation_failures_by_model_job.csv", index=False
    )


def write_unmatched_breakdown(ds):
    rows = []
    examples: dict[tuple, list[str]] = {}
    for i, row in enumerate(ds):
        call_names = {}
        for turn in row["conversation"]:
            for tc in turn.get("tool_calls") or []:
                call_names[tc.get("id")] = (tc.get("function") or {}).get("name")
        for msg_idx, turn in enumerate(row["conversation"]):
            if turn.get("role") != "tool":
                continue
            content = turn.get("content") or ""
            if any(p.search(content) for p in ERROR_PATTERNS.values()):
                continue
            match = LOOKS_ERROR.search(content)
            if not match:
                continue
            tool = call_names.get(turn.get("tool_call_id"), "<unmapped>")
            keyword = match.group(0).lower()
            rows.append({"row_idx": i, "model": row["model"], "msg_idx": msg_idx, "tool": tool, "keyword": keyword})
            bucket = examples.setdefault((tool, keyword), [])
            if len(bucket) < 3:
                start = max(0, match.start() - 80)
                snippet = content[start : match.end() + 80].replace("\n", "\\n")
                bucket.append(f"row {i} msg {msg_idx} {row['model']}: ...{snippet}...")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "unmatched_hits.csv", index=False)
    out.groupby(["tool", "keyword"]).size().unstack(fill_value=0).to_csv(OUT_DIR / "unmatched_breakdown.csv")
    text = []
    for key, snippets in sorted(examples.items(), key=lambda kv: -len(out[(out["tool"] == kv[0][0]) & (out["keyword"] == kv[0][1])])):
        text.append(f"== tool={key[0]} keyword={key[1]}")
        text.extend(snippets)
        text.append("")
    (OUT_DIR / "unmatched_examples.txt").write_text("\n".join(text))


def message_chars(turn: dict) -> int:
    calls = turn.get("tool_calls")
    return (
        len(turn.get("content") or "")
        + len(turn.get("reasoning_content") or "")
        + (len(json.dumps(calls)) if calls else 0)
    )


def write_il_check(ds):
    rows = []
    for i, row in enumerate(ds):
        conv = row["conversation"]
        assistant = [t for t in conv if t.get("role") == "assistant"]
        out_lengths = pd.Series([message_chars(t) for t in assistant])
        rows.append(
            {
                "row_idx": i,
                "model": row["model"],
                "job": row["job"],
                "status": row["status"],
                "asst_turns": len(assistant),
                "conversation_chars": sum(message_chars(t) for t in conv),
                "last_asst_chars": int(out_lengths.iloc[-1]) if len(out_lengths) else 0,
                "median_asst_chars": float(out_lengths.median()) if len(out_lengths) else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "il_check_per_trace.csv", index=False)
    summary = []
    for (model, job), group in df.groupby(["model", "job"]):
        il, other = group[group["status"] == "IL"], group[group["status"] != "IL"]
        if il.empty:
            continue
        summary.append(
            {
                "model": model,
                "job": job,
                "n_traces": len(group),
                "n_IL": len(il),
                "IL_asst_turns": " ".join(str(t) for t in sorted(il["asst_turns"])),
                "nonIL_max_asst_turns": int(other["asst_turns"].max()),
                "nonIL_at_or_above_200_or_100_cap": int(other["asst_turns"].isin([100, 200]).sum()),
                "IL_chars_min_k": int(il["conversation_chars"].min() // 1000),
                "IL_chars_max_k": int(il["conversation_chars"].max() // 1000),
                "nonIL_chars_max_k": int(other["conversation_chars"].max() // 1000),
                "nonIL_larger_than_smallest_IL": int((other["conversation_chars"] > il["conversation_chars"].min()).sum()),
                "IL_last_turn_over_median_max": round(float((il["last_asst_chars"] / il["median_asst_chars"]).max()), 2),
            }
        )
    pd.DataFrame(summary).to_csv(OUT_DIR / "il_check_by_model_job.csv", index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = get_traces()
    df = pd.DataFrame(build_row_records(ds))
    proxy = proxy_vs_turns(ds, df)
    write_job_profile(df, proxy)
    write_proxy_row(proxy, ds, 225)
    write_nonschema_tool_names(ds)
    reruns = write_reruns(ds, df)
    variants = write_scaffold_variants(ds)
    write_scaffold_diff_examples(ds, reruns, df, variants)
    write_unmatched_breakdown(ds)
    write_validation_failures(ds)
    write_il_check(ds)
    print(f"Wrote follow-up outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
