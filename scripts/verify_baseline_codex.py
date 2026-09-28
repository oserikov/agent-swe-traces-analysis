"""Recompute the directly countable baseline Codex findings from cached traces."""

import json
import re
from collections import Counter
from statistics import median
from typing import Any, cast

from datasets import load_from_disk


def assistant_calls(row: dict[str, Any]):
    for message in row["conversation"]:
        if message["role"] == "assistant":
            yield from message.get("tool_calls") or []


def call_key(call: dict[str, Any]) -> tuple[str, str]:
    function = call["function"]
    return function["name"], json.dumps(function["arguments"], sort_keys=True)


def ending(row: dict[str, Any]) -> str:
    last = row["conversation"][-1]
    if last["role"] == "tool":
        return "tool"
    if last["role"] == "assistant" and last.get("tool_calls"):
        return "unresolved"
    return "final"


def main() -> None:
    rows = cast(list[dict[str, Any]], list(load_from_disk("data/agent-traces")))

    run03 = [row for row in rows if row["job"] == "run-03"]
    all_429 = sum(p["status"] == 429 for row in rows for p in row["proxy_requests"])
    run03_429 = sum(p["status"] == 429 for row in run03 for p in row["proxy_requests"])
    affected = sum(
        any(p["status"] == 429 for p in row["proxy_requests"]) for row in run03
    )
    endings = Counter(ending(row) for row in run03)
    print(
        f"03 proxy429={run03_429}/{all_429} affected={affected}/{len(run03)} "
        f"endings=unresolved:{endings['unresolved']},tool:{endings['tool']},final:{endings['final']}"
    )

    run06 = [row for row in rows if row["job"] == "run-06"]
    majority = []
    for row in run06:
        counts = Counter(call_key(call) for call in assistant_calls(row))
        if counts and counts.most_common(1)[0][1] * 2 >= counts.total():
            majority.append(row)
    print(
        f"04 majority_repeat={len(majority)}/{len(run06)} "
        f"majority_IL={sum(row['status'] == 'IL' for row in majority)} "
        f"job_IL={sum(row['status'] == 'IL' for row in run06)}"
    )

    row = rows[263]
    calls = list(assistant_calls(row))
    commands = Counter(call["function"]["arguments"].get("command") for call in calls)
    names = Counter(call["function"]["name"] for call in calls)
    print(
        f"05 messages={len(row['conversation'])} calls={len(calls)} "
        f"cat={commands['cat pygfx/materials/_base.py']} "
        f"python_print={commands['python -c "print(open(\'pygfx/materials/_base.py\').read())"']} "
        f"bash={names['bash']} edit={names['edit']} write={names['write']} status={row['status']}"
    )

    final_lengths = {
        i: len(row["conversation"][-1].get("content") or "")
        for i, row in enumerate(rows)
        if ending(row) == "final" and row["conversation"][-1]["role"] == "assistant"
    }
    other_max = max(
        length
        for i, length in final_lengths.items()
        if rows[i]["model"] != "model-atlas"
    )
    print(
        f"08 row105={final_lengths[105]} row283={final_lengths[283]} "
        f"non_atlas_max={other_max}"
    )

    marcus = re.compile(r"Marcus|the team and I")
    phrase_counts = Counter()
    phrase_traces = Counter()
    prompt_mentions = 0
    for row in rows:
        hits = sum(
            bool(marcus.search(message.get("content") or ""))
            for message in row["conversation"]
            if message["role"] == "assistant"
        )
        phrase_counts[row["model"]] += hits
        phrase_traces[row["model"]] += hits > 0
        prompt_mentions += sum(
            "marcus" in (message.get("content") or "").lower()
            for message in row["conversation"]
            if message["role"] in {"system", "developer", "user"}
        )
    print(
        f"09 cyan_messages={phrase_counts['model-cyan']} "
        f"cyan_traces={phrase_traces['model-cyan']}/60 "
        f"other_messages={sum(v for k, v in phrase_counts.items() if k != 'model-cyan')} "
        f"prompt_mentions={prompt_mentions}"
    )

    medical = re.compile(
        r"\b(?:patient|ailments?|prognosis|convalesc\w*)\b", re.IGNORECASE
    )
    medical_hits = Counter()
    medical_traces = Counter()
    for row in rows:
        hits = sum(
            len(medical.findall(message.get("content") or ""))
            for message in row["conversation"]
            if message["role"] == "assistant"
        )
        medical_hits[row["model"]] += hits
        medical_traces[row["model"]] += hits > 0
    print(
        f"10 vega_hits={medical_hits['model-vega']} "
        f"vega_traces={medical_traces['model-vega']}/60 "
        f"other_hits={sum(v for k, v in medical_hits.items() if k != 'model-vega')}"
    )

    reasoning_lengths = [
        sum(
            len(message.get("reasoning_content") or "")
            for message in row["conversation"]
        )
        for row in rows
    ]
    orion = [
        reasoning_lengths[i]
        for i, row in enumerate(rows)
        if row["model"] == "model-orion"
    ]
    peer_high = sum(
        reasoning_lengths[i] > 300_000
        for i, row in enumerate(rows)
        if row["model"] != "model-orion"
    )
    print(
        f"11 orion_over_300k={sum(n > 300_000 for n in orion)}/60 "
        f"peer_over_300k={peer_high}/310 median={median(orion):g} "
        f"row26={reasoning_lengths[26]} row215={reasoning_lengths[215]}"
    )

    visibility = Counter()
    for row in rows:
        key = row["model"] if row["model"] != "model-delta" else row["job"]
        for message in row["conversation"]:
            if message["role"] == "assistant":
                visibility[key, "total"] += 1
                visibility[key, "nonempty"] += bool(message.get("reasoning_content"))
    delta_nonempty = visibility["run-03", "nonempty"] + visibility["run-12", "nonempty"]
    delta_total = visibility["run-03", "total"] + visibility["run-12", "total"]
    print(
        f"12 delta={delta_nonempty}/{delta_total} "
        f"run03={visibility['run-03', 'nonempty']}/{visibility['run-03', 'total']} "
        f"run12={visibility['run-12', 'nonempty']}/{visibility['run-12', 'total']} "
        f"flint={visibility['model-flint', 'nonempty']}/{visibility['model-flint', 'total']}"
    )

    malformed = []
    failed = 0
    for i, row in enumerate(rows):
        if row["job"] != "run-08":
            continue
        for j, message in enumerate(row["conversation"]):
            for call in message.get("tool_calls") or []:
                if "</arg_value>" in call["function"]["name"]:
                    malformed.append(i)
                    failed += j + 1 < len(row["conversation"]) and (
                        row["conversation"][j + 1]["role"] == "tool"
                        and "not found"
                        in (row["conversation"][j + 1].get("content") or "")
                    )
    print(
        f"14 malformed_calls={len(malformed)} traces={len(set(malformed))} not_found={failed}"
    )

    unresolved = [row for row in rows if ending(row) == "unresolved"]
    grades = Counter(row["status"] for row in unresolved)
    print(
        f"16 unresolved={len(unresolved)}/{len(rows)} "
        f"IL={grades['IL']} WA={grades['WA']} TL={grades['TL']} OK={grades['OK']}"
    )


if __name__ == "__main__":
    main()
