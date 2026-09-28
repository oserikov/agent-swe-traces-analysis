"""Check the live 429 warning against actual proxy kills, and find mid-trace runs of 4+ non-200s.

Usage:
    uv run scripts/check_proxy_runs.py

Prints PASS/FAIL per claim in the paper's "Analysis of Attempting to Implement" section and exits 1
on any failure. Row indices are zero-based positions in the HF dataset.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_dataset import get_traces  # noqa: E402

results: list[bool] = []


def check(label: str, expected, actual) -> None:
    ok = expected == actual
    results.append(ok)
    print(
        f"  {'PASS' if ok else 'FAIL'}  {label}: paper={expected!r} recomputed={actual!r}"
    )


def main() -> None:
    ds = get_traces()
    warned, killed, warning_events, mid_runs = set(), set(), 0, {}
    for i, row in enumerate(ds):
        codes = [r["status"] for r in row["proxy_requests"] or []]
        runs = [
            (ok, len(list(g)))
            for ok, g in itertools.groupby(codes, key=lambda c: c == 200)
        ]
        non200_runs = [n for ok, n in runs if not ok]
        events = sum(n >= 3 for n in non200_runs)
        warning_events += events
        if events:
            warned.add(i)
        tail = runs[-1][1] if runs and not runs[-1][0] else 0
        if tail >= 4:
            killed.add(i)
        body = runs[:-1] if tail else runs
        long_mid = [n for ok, n in body if not ok and n >= 4]
        if long_mid:
            trailing_200s = runs[-1][1] if runs[-1][0] else 0
            mid_runs[i] = (row["job"], max(long_mid), trailing_200s)

    check("traces where the live 3-in-a-row warning fires", 25, len(warned))
    check("warning events", 87, warning_events)
    check("traces killed (end on 4+ non-200s)", 12, len(killed))
    check("killed traces that were warned first", 12, len(killed & warned))
    check("warned traces not killed", 13, len(warned - killed))
    check("traces with a run of 4+ non-200s that is not the ending", 8, len(mid_runs))
    check("... all in run-03", {"run-03"}, {job for job, _, _ in mid_runs.values()})
    check("... longest such run", 4, max(n for _, n, _ in mid_runs.values()))
    check(
        "... successful responses after it before the trace ends",
        {1, 2},
        {t for _, _, t in mid_runs.values()},
    )
    print(f"  INFO  rows: {sorted(mid_runs)}")
    print(f"{results.count(True)} passed, {results.count(False)} failed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
