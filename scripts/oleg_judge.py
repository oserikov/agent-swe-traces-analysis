"""Two-stage LLM-judge screening of agent traces for Oleg's 13 hypotheses.

Subcommands:
    run          Run one judge, one stage, over one log's samples (or a --samples subset).
    merge        Fold per-stage outputs into merged logs + results/oleg_hits.jsonl +
                 results/oleg_prevalence.csv.
    pilot-report Build results/oleg/pilot/report.md from pilot-stage outputs.

See oleg-hypothesis-screen-spec.md for the design.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.log import EvalSample, read_eval_log, write_eval_log
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import Score
from oleg_hypotheses import GROUPS, HYPOTHESES, group_judge_prompt

RESULTS_DIR = Path("results/oleg")
INSPECT_LOGS_DIR = Path("results/inspect_logs")
OUTPUT_TRUNC_CHARS = 500  # head + tail, per tool message, stage 2 / pilot silver only
# Corpus has extreme outliers: traces with up to 1786 tool calls (p50 is 54) and traces with up
# to 270K no-output tokens of thinking alone. Per-message truncation alone doesn't bound total
# transcript size for a many-tool-call trace (1786 calls x 1000 chars = 1.79M chars). These are
# running-total budgets across the WHOLE transcript, applied in message order: once exhausted,
# further content of that kind is omitted with a note rather than included. Found live during
# the pilot run (2026-09-28) after an Opus confirm call hit ~460K cache-write tokens on one trace
# -- see WORKLOG / oleg-hypothesis-screen-spec.md known-limitations for the corpus stat.
OUTPUT_TOTAL_BUDGET_CHARS = 20_000  # ~5K tokens; stage 2 / pilot silver only (screen omits outputs entirely)
THINKING_TOTAL_BUDGET_CHARS = 80_000  # ~20K tokens; all stages, protects the cheap screener too


def render_transcript(sample: EvalSample, include_outputs: bool) -> str:
    """Render one sample's messages as text, stripped of the original grade.

    Message layout is fixed across all 13 hypotheses so the OpenRouter cache prefix
    matches call to call.
    """
    lines: list[str] = []
    output_budget = OUTPUT_TOTAL_BUDGET_CHARS
    thinking_budget = THINKING_TOTAL_BUDGET_CHARS
    for m in sample.messages:
        idx = m.metadata.get("source_index") if m.metadata else None
        tag = f"[msg {idx if idx is not None else '?'}]"
        if isinstance(m, ChatMessageSystem):
            lines.append(f"{tag} SYSTEM: {_text_of(m)}")
        elif isinstance(m, ChatMessageUser):
            lines.append(f"{tag} USER: {_text_of(m)}")
        elif isinstance(m, ChatMessageAssistant):
            content = m.content
            if isinstance(content, str):
                if content:
                    lines.append(f"{tag} ASSISTANT TEXT: {content}")
            else:
                for block in content:
                    if block.type == "reasoning" and block.reasoning:
                        if thinking_budget <= 0:
                            lines.append(f"{tag} ASSISTANT THINKING: [budget exhausted, {len(block.reasoning)} chars omitted]")
                        else:
                            text = block.reasoning[:thinking_budget]
                            thinking_budget -= len(block.reasoning)
                            suffix = "" if thinking_budget >= 0 else f" [...{-thinking_budget} chars over budget, cut]"
                            lines.append(f"{tag} ASSISTANT THINKING: {text}{suffix}")
                    elif block.type == "text" and block.text:
                        lines.append(f"{tag} ASSISTANT TEXT: {block.text}")
            for tc in m.tool_calls or []:
                lines.append(f"{tag} ASSISTANT CALL: {tc.function}({json.dumps(tc.arguments)})")
        elif isinstance(m, ChatMessageTool):
            if not include_outputs:
                out = f"[output omitted: {len(m.text)} chars]"
            elif output_budget <= 0:
                out = f"[output budget exhausted, {len(m.text)} chars omitted]"
            elif len(m.text) > 2 * OUTPUT_TRUNC_CHARS:
                head = m.text[:OUTPUT_TRUNC_CHARS]
                tail = m.text[-OUTPUT_TRUNC_CHARS:]
                out = f"{head}\n...[{len(m.text) - 2 * OUTPUT_TRUNC_CHARS} chars omitted]...\n{tail}"
                output_budget -= 2 * OUTPUT_TRUNC_CHARS
            else:
                out = m.text
                output_budget -= len(m.text)
            if m.error:
                out = f"[TOOL ERROR: {m.error.message}] " + out
            lines.append(f"{tag} TOOL RESULT ({m.function}): {out}")
    return "\n".join(lines)


def _text_of(m) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    return "".join(b.text for b in c if getattr(b, "type", None) == "text")


def build_judge_messages(transcript: str, group: str) -> list:
    # The transcript message uses list-of-blocks content (not a plain string) so OpenRouter's
    # Anthropic cache-marker injection (see openrouter.py:_add_anthropic_cache_markers) can
    # attach cache_control to it: it only marks blocks inside a list-content message, and its
    # fallback path marks the *previous* message's last block when the final message is short
    # (which is always true here, since the hypothesis-group instructions are the last message).
    return [
        ChatMessageUser(content=[ContentText(text=f"[TRACE]\n{transcript}\n[END TRACE]")]),
        ChatMessageUser(content=group_judge_prompt(group)),
    ]


def parse_group_verdict(text: str, names: list[str]) -> dict | None:
    """Extract the trailing JSON object mapping each hypothesis name to a verdict."""
    text = text.strip()
    start = text.rfind("{")
    while start != -1:
        candidate = text[start:]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and any(n in obj for n in names):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.rfind("{", 0, start)
    return None


@dataclass
class RunStats:
    calls: int = 0
    parsed: int = 0
    hits: int = 0
    cache_read: int = 0
    cache_write: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


async def score_sample(
    model, sample: EvalSample, stage: str, cache_events: dict, only_groups: set[str] | None = None
) -> dict[str, Score]:
    """Run hypotheses over one sample as grouped calls (see oleg_hypotheses.GROUPS).

    Grouping cuts the per-trace call count from 13 to 4, which matters because OpenRouter only
    injects Anthropic cache_control markers for openrouter/anthropic/* models — a non-Anthropic
    screener gets no cache benefit at all, so the call count itself is the cost lever. For an
    Anthropic-family judge (confirm stage), the cache warm-up barrier still helps across the
    group calls that share the transcript prefix.

    `only_groups`, when given (confirm stage under the budget governor), restricts the call to
    just the groups containing a hypothesis Stage 1 flagged for this sample — paying for a
    hypothesis Stage 1 never flagged wastes the confirm budget, since almost every trace has at
    least one screen-stage hit somewhere across 13 hypotheses (~98% at the pilot's ~27% per-row
    screen rate), so confirming every group of every selected trace would blow through the cap.
    """
    include_outputs = stage != "screen"
    transcript = render_transcript(sample, include_outputs)
    scores: dict[str, Score] = {}

    key = sample.id
    event = cache_events.setdefault(key, asyncio.Event())
    is_first = not event.is_set()

    group_names = [g for g in GROUPS if only_groups is None or g in only_groups]
    if not group_names:
        return scores

    async def run_group(group: str) -> dict[str, Score]:
        names = GROUPS[group]
        messages = build_judge_messages(transcript, group)
        config = GenerateConfig(
            max_tokens=8192 if stage == "confirm" else 4096,
            reasoning_effort="medium",
        )
        output = await model.generate(input=messages, config=config)
        raw = output.completion or ""
        verdict = parse_group_verdict(raw, names)
        usage = output.usage
        base_meta = {
            "stage": stage,
            "judge_model": model.name,
            "group": group,
            "raw_len": len(raw),
            "usage": usage.model_dump() if usage else None,
        }
        if verdict is None:
            meta = dict(base_meta, parse_failed=True)
            return {n: Score(value=float("nan"), explanation="judge output unparseable", metadata=dict(meta)) for n in names}
        out: dict[str, Score] = {}
        for name in names:
            entry = verdict.get(name)
            meta = dict(base_meta)
            if not isinstance(entry, dict):
                meta["parse_failed"] = True
                out[name] = Score(value=float("nan"), explanation="hypothesis missing from judge JSON", metadata=meta)
                continue
            quote = str(entry.get("quote") or "")
            meta.update(entry)
            meta["quote_verified"] = bool(quote) and quote in transcript
            out[name] = Score(
                value=1 if entry.get("hit") else 0,
                answer=quote[:300],
                explanation=str(entry.get("how") or ""),
                metadata=meta,
            )
        return out

    if is_first:
        first = await run_group(group_names[0])
        scores.update(first)
        event.set()
        rest = await asyncio.gather(*(run_group(g) for g in group_names[1:]))
    else:
        await event.wait()
        rest = await asyncio.gather(*(run_group(g) for g in group_names))
    for group_scores in rest:
        scores.update(group_scores)
    return scores


async def run_log(
    log_path: Path,
    stage: str,
    judge: str,
    out_dir: Path,
    sample_ids: list[str] | None,
    max_connections: int,
    groups_by_sample: dict[str, set[str]] | None = None,
) -> None:
    log = read_eval_log(str(log_path))
    samples = log.samples or []
    if sample_ids:
        wanted = set(sample_ids)
        samples = [s for s in samples if str(s.id) in wanted]
    if groups_by_sample is not None:
        samples = [s for s in samples if str(s.id) in groups_by_sample]
    if not samples:
        print(f"{log_path}: no samples selected, skipping")
        return

    model = get_model(judge, config=GenerateConfig(max_connections=max_connections))

    out_dir.mkdir(parents=True, exist_ok=True)
    # Pilot stage runs 3 judges (opus silver + 2 screener candidates) against the same logs;
    # key filenames by judge slug there so they don't overwrite each other. Screen/confirm each
    # use exactly one judge per stage, so the plain stem is unambiguous and kept for those.
    judge_slug = judge.rsplit("/", 1)[-1].replace(".", "_")
    stem = f"{log_path.stem}.{judge_slug}" if stage == "pilot" else log_path.stem
    hits_path = out_dir / f"{stem}.hits.jsonl"
    cache_events: dict = {}

    sem = asyncio.Semaphore(max_connections)

    async def do_sample(s: EvalSample):
        only_groups = groups_by_sample.get(str(s.id)) if groups_by_sample is not None else None
        async with sem:
            return s, await score_sample(model, s, stage, cache_events, only_groups)

    results = await asyncio.gather(*(do_sample(s) for s in samples))

    with hits_path.open("w") as f:
        for s, scores in results:
            for name, sc in scores.items():
                f.write(json.dumps({
                    "log": log_path.name,
                    "sample_id": s.id,
                    "source_model": s.metadata.get("source_model") if s.metadata else None,
                    "task_id": s.metadata.get("source_task_id") if s.metadata else None,
                    "reward": s.scores["grade"].value if s.scores and "grade" in s.scores else None,
                    "hypothesis": name,
                    "value": sc.value,
                    "metadata": sc.metadata,
                }) + "\n")

    # attach scores onto a copy of the log's samples and write it out
    by_id = {s.id: scores for s, scores in results}
    for s in log.samples or []:
        if s.id in by_id:
            s.scores = dict(s.scores or {})
            for name, sc in by_id[s.id].items():
                s.scores[f"{name}__{stage}"] = sc
    out_log_path = out_dir / f"{stem}.eval"
    write_eval_log(log, str(out_log_path))
    print(f"wrote {out_log_path} and {hits_path} ({len(samples)} samples x {len(HYPOTHESES)} hypotheses)")


def cmd_run(args: argparse.Namespace) -> None:
    load_dotenv()
    if args.judge.startswith("openrouter/") and not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY missing")
    sample_ids = None
    if args.samples:
        sample_ids = Path(args.samples).read_text().split() if Path(args.samples).exists() else args.samples.split(",")
    if args.exclude:
        excluded = set(Path(args.exclude).read_text().split() if Path(args.exclude).exists() else args.exclude.split(","))
        log = read_eval_log(str(args.log))
        all_ids = [str(s.id) for s in (log.samples or [])]
        sample_ids = [sid for sid in (sample_ids or all_ids) if sid not in excluded]
    groups_by_sample = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        log_entry = manifest.get(Path(args.log).name, {})
        groups_by_sample = {sid: set(groups) for sid, groups in log_entry.items()}
    out_dir = RESULTS_DIR / args.stage
    asyncio.run(run_log(Path(args.log), args.stage, args.judge, out_dir, sample_ids, args.max_connections, groups_by_sample))


def cmd_render_check(args: argparse.Namespace) -> None:
    """Offline checkpoint: render all logs, print token estimates."""
    no_out = 0
    trunc = 0
    for p in sorted(glob.glob(str(INSPECT_LOGS_DIR / "*.eval"))):
        log = read_eval_log(p)
        for s in log.samples or []:
            no_out += len(render_transcript(s, include_outputs=False)) // 4
            trunc += len(render_transcript(s, include_outputs=True)) // 4
    print(f"no-outputs tokens ~{no_out/1e6:.2f}M (spec: 8.6M)")
    print(f"truncated-outputs tokens ~{trunc/1e6:.2f}M (spec: ~8.6M + 3.85M = 12.45M)")


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - margin) / denom, (center + margin) / denom)


def cmd_select_confirm(args: argparse.Namespace) -> None:
    """Build the Stage 2 budget-governor manifest from Stage 1 screen hits.

    Per hypothesis: if it has <= k screen hits, confirm all of them; otherwise take a seeded
    random sample of k. k is one value applied to every hypothesis (per spec), chosen as the
    largest value that keeps projected cost within --budget given --cost-per-call (from the
    pilot report's real, deduplicated $/call for the Anthropic confirmer). Selected
    (hypothesis, trace) pairs are converted to a manifest of {log: {sample_id: [groups]}} so
    `run --stage confirm --manifest ...` only pays for the groups actually needed.
    """
    rows = []
    for p in RESULTS_DIR.glob("screen/*.hits.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    by_hyp = defaultdict(list)
    for r in rows:
        if r.get("value") == 1:
            by_hyp[r["hypothesis"]].append(r)

    # search for the largest k that fits the budget (monotonic in k, so a simple scan is fine —
    # there are at most a few hundred hits per hypothesis). Cost is billed per (trace, group)
    # call, not per trace — a trace needing 2 groups pays for 2 calls — so count group-level
    # pairs here, matching the manifest built below exactly.
    name_to_group_ = {name: g for g, names in GROUPS.items() for name in names}
    max_hits = max((len(v) for v in by_hyp.values()), default=0)
    best_k = 0
    for k in range(0, max_hits + 1):
        rng_trial = random.Random(20260928)
        group_pairs = set()
        for hyp, hits in sorted(by_hyp.items()):
            chosen = hits if len(hits) <= k else rng_trial.sample(hits, k)
            group = name_to_group_[hyp]
            for r in chosen:
                group_pairs.add((r["log"], str(r["sample_id"]), group))
        cost = len(group_pairs) * args.cost_per_call
        if cost <= args.budget:
            best_k = k
        else:
            break

    k = best_k
    rng = random.Random(20260928)
    manifest: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(set))
    name_to_group = {name: g for g, names in GROUPS.items() for name in names}
    selected_pairs = 0
    for hyp, hits in sorted(by_hyp.items()):
        chosen = hits if len(hits) <= k else rng.sample(hits, k)
        selected_pairs += len(chosen)
        group = name_to_group[hyp]
        for r in chosen:
            manifest[r["log"]][str(r["sample_id"])].add(group)

    out = {log: {sid: sorted(groups) for sid, groups in samples.items()} for log, samples in manifest.items()}
    n_traces = sum(len(v) for v in out.values())
    n_calls = sum(len(groups) for samples in out.values() for groups in samples.values())
    out_path = RESULTS_DIR / "confirm" / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"k={k}, selected {selected_pairs} (hypothesis,trace) pairs across {n_traces} traces, {n_calls} group calls")
    print(f"projected cost: ${n_calls * args.cost_per_call:.2f} (budget ${args.budget:.2f}, ${args.cost_per_call:.4f}/call)")
    print(f"wrote {out_path}")


def cmd_merge(args: argparse.Namespace) -> None:
    screen_hits = list(RESULTS_DIR.glob("screen/*.hits.jsonl"))
    confirm_hits = list(RESULTS_DIR.glob("confirm/*.hits.jsonl"))
    pilot_hits = list(RESULTS_DIR.glob("pilot/*.hits.jsonl"))

    def load(paths):
        rows = []
        for p in paths:
            for line in p.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    screen = load(screen_hits)
    confirm = load(confirm_hits)
    pilot = load(pilot_hits)  # pilot judge rows include both the silver (opus) and screener candidates

    # index by (log, sample_id, hypothesis)
    def key(r):
        return (r["log"], str(r["sample_id"]), r["hypothesis"])

    screen_by_key = {key(r): r for r in screen}
    confirm_by_key = {key(r): r for r in confirm}
    # pilot rows tagged by judge in metadata["judge_model"]; opus rows serve as confirm, others as screen
    for r in pilot:
        jm = (r.get("metadata") or {}).get("judge_model", "")
        if "opus" in jm.lower():
            confirm_by_key.setdefault(key(r), r)
        else:
            screen_by_key.setdefault(key(r), r)

    all_keys = set(screen_by_key) | set(confirm_by_key)
    hits_out = RESULTS_DIR.parent / "oleg_hits.jsonl" if False else Path("results/oleg_hits.jsonl")
    per_hyp_model = defaultdict(lambda: {"n": 0, "screen_hits": 0, "confirmed_n": 0, "confirmed_hits": 0})

    with hits_out.open("w") as f:
        for k in sorted(all_keys):
            log, sid, hyp = k
            sr = screen_by_key.get(k)
            cr = confirm_by_key.get(k)
            screen_hit = bool(sr and sr.get("value") == 1)
            confirm_hit = None if cr is None else bool(cr.get("value") == 1)
            src_model = (sr or cr or {}).get("source_model")
            task_id = (sr or cr or {}).get("task_id")
            bucket = per_hyp_model[(hyp, src_model)]
            bucket["n"] += 1
            if screen_hit:
                bucket["screen_hits"] += 1
            if cr is not None:
                bucket["confirmed_n"] += 1
                if confirm_hit:
                    bucket["confirmed_hits"] += 1
            if not (screen_hit or confirm_hit):
                continue
            meta = (cr or sr or {}).get("metadata") or {}
            f.write(json.dumps({
                "log": log, "sample_id": sid, "source_model": src_model, "task_id": task_id,
                "hypothesis": hyp, "screen_hit": screen_hit, "confirm_hit": confirm_hit,
                "message_index": meta.get("message_index"), "quote": meta.get("quote"),
                "quote_verified": meta.get("quote_verified"), "how": meta.get("how"),
                "judge_models": {"screen": (sr or {}).get("metadata", {}).get("judge_model"),
                                  "confirm": (cr or {}).get("metadata", {}).get("judge_model")},
            }) + "\n")
    print(f"wrote {hits_out}")

    # prevalence table
    prev_out = Path("results/oleg_prevalence.csv")
    with prev_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hypothesis", "source_model", "n", "screen_hits", "screen_rate",
                     "confirmed_n", "confirmed_hits", "precision", "precision_corrected_rate",
                     "ci_low", "ci_high"])
        agg = defaultdict(lambda: {"n": 0, "screen_hits": 0, "confirmed_n": 0, "confirmed_hits": 0})
        for (hyp, model), b in sorted(per_hyp_model.items()):
            precision = b["confirmed_hits"] / b["confirmed_n"] if b["confirmed_n"] else float("nan")
            screen_rate = b["screen_hits"] / b["n"] if b["n"] else 0.0
            corrected = screen_rate * precision if b["confirmed_n"] else float("nan")
            lo, hi = wilson_ci(b["screen_hits"], b["n"])
            w.writerow([hyp, model, b["n"], b["screen_hits"], f"{screen_rate:.3f}",
                        b["confirmed_n"], b["confirmed_hits"],
                        f"{precision:.3f}" if b["confirmed_n"] else "", f"{corrected:.3f}" if b["confirmed_n"] else "",
                        f"{lo:.3f}", f"{hi:.3f}"])
            a = agg[hyp]
            for kk in ("n", "screen_hits", "confirmed_n", "confirmed_hits"):
                a[kk] += b[kk]
        for hyp, b in sorted(agg.items()):
            precision = b["confirmed_hits"] / b["confirmed_n"] if b["confirmed_n"] else float("nan")
            screen_rate = b["screen_hits"] / b["n"] if b["n"] else 0.0
            corrected = screen_rate * precision if b["confirmed_n"] else float("nan")
            lo, hi = wilson_ci(b["screen_hits"], b["n"])
            w.writerow([hyp, "ALL", b["n"], b["screen_hits"], f"{screen_rate:.3f}",
                        b["confirmed_n"], b["confirmed_hits"],
                        f"{precision:.3f}" if b["confirmed_n"] else "", f"{corrected:.3f}" if b["confirmed_n"] else "",
                        f"{lo:.3f}", f"{hi:.3f}"])
    print(f"wrote {prev_out}")

    # derived: claims-success & reward=0
    cs_rows = [r for r in screen + confirm + pilot if r["hypothesis"] == "oleg-claims-success"]
    cs_fail = [r for r in cs_rows if r.get("value") == 1 and r.get("reward") == 0]
    print(f"claims-success & reward=0: {len(cs_fail)} / {len(cs_rows)} scored rows")

    # merged eval logs: fold stage scores onto original logs
    for p in sorted(glob.glob(str(INSPECT_LOGS_DIR / "*.eval"))):
        log = read_eval_log(p)
        stem = Path(p).stem
        stage_logs = {}
        for stage, d in [("screen", RESULTS_DIR / "screen"), ("confirm", RESULTS_DIR / "confirm")]:
            sp = d / f"{stem}.eval"
            if sp.exists():
                stage_logs[stage] = {s.id: s for s in (read_eval_log(str(sp)).samples or [])}
        # pilot dir holds 3 judges per log (opus silver + 2 screener candidates), filenames
        # judge-qualified as f"{stem}.{judge_slug}.eval"; only the opus file is silver/confirm-grade.
        opus_pilot_candidates = sorted((RESULTS_DIR / "pilot").glob(f"{stem}.*opus*.eval"))
        if opus_pilot_candidates:
            stage_logs["pilot"] = {s.id: s for s in (read_eval_log(str(opus_pilot_candidates[0])).samples or [])}
        if not stage_logs:
            continue
        for s in log.samples or []:
            merged_scores = dict(s.scores or {})
            for name in HYPOTHESES:
                confirm_s = stage_logs.get("confirm", {}).get(s.id)
                pilot_s = stage_logs.get("pilot", {}).get(s.id)
                screen_s = stage_logs.get("screen", {}).get(s.id)
                chosen = None
                stage_tag = None
                for cand, tag in [(confirm_s, "confirm"), (pilot_s, "pilot"), (screen_s, "screen")]:
                    if cand and cand.scores and f"{name}__{tag}" in cand.scores:
                        sc = cand.scores[f"{name}__{tag}"]
                        if tag in ("confirm", "pilot") and (sc.metadata or {}).get("judge_model", "").lower().find("opus") == -1:
                            continue
                        chosen = sc
                        stage_tag = tag
                        break
                if chosen is None and screen_s and screen_s.scores and f"{name}__screen" in screen_s.scores:
                    chosen = screen_s.scores[f"{name}__screen"]
                    stage_tag = "screen-only"
                if chosen is not None:
                    meta = dict(chosen.metadata or {})
                    meta["stage"] = stage_tag
                    merged_scores[name] = Score(value=chosen.value, answer=chosen.answer, explanation=chosen.explanation, metadata=meta)
            s.scores = merged_scores
        out_path = RESULTS_DIR / f"{stem}-oleg.eval"
        write_eval_log(log, str(out_path))
    print(f"wrote merged logs to {RESULTS_DIR}/*-oleg.eval")


def cmd_pilot_report(args: argparse.Namespace) -> None:
    pilot_hits = list((RESULTS_DIR / "pilot").glob("*.hits.jsonl"))
    rows = []
    for p in pilot_hits:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    by_judge_hyp = defaultdict(list)
    for r in rows:
        jm = (r.get("metadata") or {}).get("judge_model", "unknown")
        by_judge_hyp[(jm, r["hypothesis"])].append(r)

    opus_rows = {(r["sample_id"], r["hypothesis"]): r for r in rows if "opus" in (r.get("metadata") or {}).get("judge_model", "").lower()}
    screeners = sorted({(r.get("metadata") or {}).get("judge_model") for r in rows if "opus" not in (r.get("metadata") or {}).get("judge_model", "").lower()})

    lines = ["# Pilot report\n", f"Pilot rows: {len(rows)}\n"]
    summary = {}
    for screener in screeners:
        tp = fp = fn = tn = 0
        for r in rows:
            if (r.get("metadata") or {}).get("judge_model") != screener:
                continue
            gold = opus_rows.get((r["sample_id"], r["hypothesis"]))
            if gold is None:
                continue
            g = gold.get("value") == 1
            s = r.get("value") == 1
            if g and s:
                tp += 1
            elif g and not s:
                fn += 1
            elif not g and s:
                fp += 1
            else:
                tn += 1
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        summary[screener] = (recall, precision, tp, fp, fn, tn)
        lines.append(f"## {screener}\nrecall={recall:.2f} precision={precision:.2f} tp={tp} fp={fp} fn={fn} tn={tn}\n")

    chosen = None
    best_recall = -1
    for screener, (recall, _precision, *_rest) in summary.items():
        if not math.isnan(recall) and recall > best_recall:
            best_recall = recall
            chosen = screener
    lines.append(f"\n**Chosen screener: {chosen}** (pooled recall {best_recall:.2f})\n")

    # Cost per trace from usage metadata. Usage is per GROUP CALL but this JSONL has one row per
    # HYPOTHESIS (3-4 rows share one call's usage) -- dedup by (sample_id, group) or every group's
    # cost gets counted once per hypothesis in it (was a real bug: inflated pilot Opus cost ~3.25x
    # before this fix; ground truth is the OpenRouter account's own GET /api/v1/key usage).
    PRICES = {  # $/M tokens: (input, cache_read, cache_write, output)
        "anthropic/claude-opus-5.5": (4.00, 0.20, 5.00, 20.00),
        "openai/gpt-5.6-luna": (0.20, 0.02, 0.20, 1.20),
        "openai/gpt-5.1-codex-mini": (0.25, 0.03, 0.25, 2.00),
    }
    for jm in sorted({r.get("metadata", {}).get("judge_model") for r in rows}):
        by_call = {}
        for r in rows:
            m = r.get("metadata") or {}
            if m.get("judge_model") != jm or not m.get("usage"):
                continue
            by_call[(r["sample_id"], m.get("group"))] = m["usage"]
        if not by_call:
            continue
        n_traces = len({k[0] for k in by_call})
        cache_read = sum(u.get("input_tokens_cache_read") or 0 for u in by_call.values())
        cache_write = sum(u.get("input_tokens_cache_write") or 0 for u in by_call.values())
        in_tok = sum(u.get("input_tokens") or 0 for u in by_call.values())
        out_tok = sum(u.get("output_tokens") or 0 for u in by_call.values())
        price = PRICES.get(jm)
        cost_line = ""
        per_call_cost = None
        if price:
            p_in, p_cr, p_cw, p_out = price
            cost = in_tok * p_in / 1e6 + cache_read * p_cr / 1e6 + cache_write * p_cw / 1e6 + out_tok * p_out / 1e6
            per_call_cost = cost / len(by_call)
            cost_line = f", est_cost=${cost:.3f} (${cost / n_traces:.4f}/trace, ${per_call_cost:.4f}/call)"
        lines.append(f"{jm}: {n_traces} traces, {len(by_call)} distinct calls, in={in_tok} cache_read={cache_read} cache_write={cache_write} out={out_tok}{cost_line}\n")

    parse_fail = sum(1 for r in rows if (r.get("metadata") or {}).get("parse_failed"))
    lines.append(f"\nParse failures: {parse_fail}/{len(rows)} ({parse_fail/len(rows)*100 if rows else 0:.1f}%)\n")

    out_path = RESULTS_DIR / "pilot" / "report.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print(f"chosen screener: {chosen}")


def cmd_pilot_select(args: argparse.Namespace) -> None:
    """Deterministic stratified pilot sample: 16 ids by source_model x pass/fail + 2 longest-thinking."""
    logs = sorted(glob.glob(str(INSPECT_LOGS_DIR / "*.eval")))
    all_samples = []
    for p in logs:
        log = read_eval_log(p)
        for s in log.samples or []:
            reward = s.scores["grade"].value if s.scores and "grade" in s.scores else None
            think_len = sum(len(b.reasoning) for m in s.messages if isinstance(m, ChatMessageAssistant)
                             and not isinstance(m.content, str) for b in m.content if b.type == "reasoning" and b.reasoning)
            all_samples.append((p, s.id, s.metadata.get("source_model"), reward, think_len))

    rng = random.Random(20260928)
    by_bucket = defaultdict(list)
    for row in all_samples:
        by_bucket[(row[2], row[3] == 1)].append(row)
    picked = []
    for _bucket, items in sorted(by_bucket.items()):
        rng.shuffle(items)
        if items:
            picked.append(items[0])
    picked = picked[:16]
    remaining_slots = 16 - len(picked)
    longest = sorted(all_samples, key=lambda r: -r[4])
    for row in longest:
        if remaining_slots <= 0:
            break
        if row not in picked:
            picked.append(row)
            remaining_slots -= 1

    out_dir = RESULTS_DIR / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_ids.txt").write_text("\n".join(str(r[1]) for r in picked))
    by_log = defaultdict(list)
    for p, sid, *_ in picked:
        by_log[p].append(str(sid))
    for p, ids in by_log.items():
        idfile = out_dir / f"{Path(p).stem}.ids.txt"
        idfile.write_text("\n".join(ids))
    print(f"picked {len(picked)} pilot samples across {len(by_log)} logs")
    for p, ids in by_log.items():
        print(f"  {p} -> {out_dir / (Path(p).stem + '.ids.txt')} ({len(ids)} ids)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--stage", choices=["pilot", "screen", "confirm"], required=True)
    p_run.add_argument("--judge", required=True)
    p_run.add_argument("--log", required=True)
    p_run.add_argument("--samples", help="comma-separated ids or a file with one id per line")
    p_run.add_argument("--exclude", help="comma-separated ids or a file with one id per line to exclude (e.g. pilot ids from stage 1)")
    p_run.add_argument("--max-connections", type=int, default=16)
    p_run.add_argument("--manifest", help="JSON {log: {sample_id: [group, ...]}} restricting confirm-stage groups")
    p_run.set_defaults(func=cmd_run)

    p_render = sub.add_parser("render-check")
    p_render.set_defaults(func=cmd_render_check)

    p_pilot_select = sub.add_parser("pilot-select")
    p_pilot_select.set_defaults(func=cmd_pilot_select)

    p_select_confirm = sub.add_parser("select-confirm")
    p_select_confirm.add_argument("--budget", type=float, required=True, help="remaining $ available for stage 2")
    p_select_confirm.add_argument("--cost-per-call", type=float, required=True, help="real $/confirm group-call from the pilot report")
    p_select_confirm.set_defaults(func=cmd_select_confirm)

    p_merge = sub.add_parser("merge")
    p_merge.set_defaults(func=cmd_merge)

    p_report = sub.add_parser("pilot-report")
    p_report.set_defaults(func=cmd_pilot_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
