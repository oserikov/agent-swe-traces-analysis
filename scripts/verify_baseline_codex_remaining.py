"""Recompute the six baseline findings that also need targeted message review."""

import argparse
import re
from collections import Counter
from typing import Any, cast

from datasets import load_from_disk


def calls(row: dict[str, Any]):
    for message in row["conversation"]:
        if message["role"] == "assistant":
            yield from message.get("tool_calls") or []


def shell_commands(row: dict[str, Any]):
    for call in calls(row):
        function = call["function"]
        if function["name"] == "bash":
            yield str(function["arguments"].get("command", ""))


def successful_download(row: dict[str, Any]) -> bool:
    download_ids = {
        call["id"]
        for call in calls(row)
        if "pip download" in str(call["function"]["arguments"].get("command", ""))
    }
    return any(
        message["role"] == "tool"
        and message.get("tool_call_id") in download_ids
        and "Successfully downloaded" in (message.get("content") or "")
        for message in row["conversation"]
    )


def show_messages(rows: list[dict[str, Any]], selector: str) -> None:
    match = re.fullmatch(r"(\d+):(\d+)(?:-(\d+))?", selector)
    if not match:
        raise ValueError("--show expects ROW:MESSAGE or ROW:START-END")
    row_index, start = map(int, match.group(1, 2))
    end = int(match.group(3) or start)
    if (
        row_index >= len(rows)
        or start > end
        or end >= len(rows[row_index]["conversation"])
    ):
        raise ValueError("--show indices are outside the cached dataset")
    for message_index in range(start, end + 1):
        message = rows[row_index]["conversation"][message_index]
        print(
            f"row={row_index} message={message_index} role={message['role']} "
            f"result_id={message.get('tool_call_id')}"
        )
        for call in message.get("tool_calls") or []:
            function = call["function"]
            print(
                f"  call_id={call['id']} name={function['name']} "
                f"arguments={str(function['arguments'])[:350]}"
            )
        content = message.get("content") or ""
        print(f"  content_head={content[:250]!r}")
        if len(content) > 250:
            print(f"  content_tail={content[-450:]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", metavar="ROW:START-END")
    args = parser.parse_args()
    rows = cast(list[dict[str, Any]], list(load_from_disk("data/agent-traces")))
    if args.show:
        show_messages(rows, args.show)
        return

    first = {row["task_id"]: row for row in rows if row["job"] == "run-01"}
    second = {row["task_id"]: row for row in rows if row["job"] == "run-07"}
    shared = sorted(first.keys() & second.keys())
    outcomes = Counter(
        (first[task]["status"], second[task]["status"]) for task in shared
    )
    roles = sum(
        first[task]["conversation"][0]["role"] == "system"
        and second[task]["conversation"][0]["role"] == "developer"
        for task in shared
    )
    same_user = sum(
        first[task]["conversation"][1]["content"]
        == second[task]["conversation"][1]["content"]
        for task in shared
    )
    same_tools = sum(first[task]["tools"] == second[task]["tools"] for task in shared)
    date_pattern = re.compile(r"Current date: \d{4}-\d{2}-\d{2}")
    same_prompt_except_date = sum(
        date_pattern.sub(
            "Current date: DATE", first[task]["conversation"][0]["content"]
        )
        == date_pattern.sub(
            "Current date: DATE", second[task]["conversation"][0]["content"]
        )
        for task in shared
    )
    first_downloads = sum(
        successful_download(row) for row in rows if row["job"] == "run-01"
    )
    second_downloads = sum(
        successful_download(row) for row in rows if row["job"] == "run-07"
    )
    print(
        f"01 pairs={len(shared)} outcomes=OO:{outcomes['OK', 'OK']},"
        f"OW:{outcomes['OK', 'WA']},WW:{outcomes['WA', 'WA']} roles=system/developer:{roles}"
    )
    print(
        f"01 match=user:{same_user},tools:{same_tools},prompt_except_date:{same_prompt_except_date} "
        f"downloads={first_downloads}/30,{second_downloads}/30"
    )

    cyan = [row for row in rows if row["model"] == "model-cyan"]
    pip = sum(
        any("pip download" in command for command in shell_commands(row))
        for row in cyan
    )
    fetch = sum(
        any("git fetch" in command for command in shell_commands(row)) for row in cyan
    )
    either = sum(
        any(
            "pip download" in command or "git fetch" in command
            for command in shell_commands(row)
        )
        for row in cyan
    )
    print(f"02 pip_download={pip}/60 git_fetch={fetch}/60 either={either}/60")

    run12 = [row for row in rows if row["job"] == "run-12"]
    patterns = {
        "commit": re.compile(r"\bgit\s+commit\b"),
        "reset": re.compile(r"\bgit\s+reset\s+--hard\b"),
        "push": re.compile(r"\bgit\s+push\b"),
    }
    affected = {
        name: sum(
            any(pattern.search(command) for command in shell_commands(row))
            for row in run12
        )
        for name, pattern in patterns.items()
    }
    others = sum(
        any(
            pattern.search(command)
            for command in shell_commands(row)
            for pattern in patterns.values()
        )
        for row in rows
        if row["job"] != "run-12"
    )
    push_results = []
    for row in rows:
        push_ids = {
            call["id"]
            for call in calls(row)
            if call["function"]["name"] == "bash"
            and patterns["push"].search(
                str(call["function"]["arguments"].get("command", ""))
            )
        }
        push_results.extend(
            message.get("content") or ""
            for message in row["conversation"]
            if message["role"] == "tool" and message.get("tool_call_id") in push_ids
        )
    failed = sum("Command exited with code" in result for result in push_results)
    undeclared_git = not any(
        tool["function"]["name"] == "git" for tool in rows[298]["tools"]
    )
    print(
        f"06 commit={affected['commit']}/40 reset={affected['reset']}/40 "
        f"push={affected['push']}/40 other_jobs={others}/330 "
        f"push_fail={failed}/{len(push_results)} undeclared_git={int(undeclared_git)}"
    )

    sequence = rows[168]["conversation"]
    bypass = sequence[2422]["tool_calls"][0]["function"]["arguments"]["command"]
    validation = sequence[2423]["content"]
    restore = sequence[2424]["tool_calls"][0]["function"]["arguments"]["command"]
    print(
        f"07 row=168 bypass={int('mypy' in bypass and '--strict' in bypass and '/pass/' in bypass)} "
        f"tox_failed_at_pydocstyle={int('pydocstyle' in validation and 'Command exited with code 1' in validation)} "
        f"restored={int(restore == 'git restore precommit.py')}"
    )

    unmatched = []
    duplicate_results = []
    duplicate_calls = []
    for i, row in enumerate(rows):
        seen_calls = set()
        seen_results = set()
        for message in row["conversation"]:
            for call in message.get("tool_calls") or []:
                if call["id"] in seen_calls:
                    duplicate_calls.append(i)
                seen_calls.add(call["id"])
            if message["role"] == "tool" and message.get("tool_call_id"):
                result_id = message["tool_call_id"]
                if result_id not in seen_calls:
                    unmatched.append(i)
                if result_id in seen_results:
                    duplicate_results.append(i)
                seen_results.add(result_id)
    print(
        f"13 unmatched={len(unmatched)}/{len(set(unmatched))} "
        f"duplicate_result_traces={len(set(duplicate_results))} "
        f"duplicate_call_traces={len(set(duplicate_calls))} "
        f"other_job_unmatched={sum(rows[i]['job'] != 'run-10' for i in unmatched)}"
    )

    marker = "<｜DSML｜"
    marked = [
        (i, j)
        for i, row in enumerate(rows)
        for j, message in enumerate(row["conversation"])
        if message["role"] == "assistant" and marker in (message.get("content") or "")
    ]
    sequence = rows[358]["conversation"]
    content = sequence[51]["content"]
    edits = sequence[51]["tool_calls"][0]["function"]["arguments"]["edits"]
    rejected = "edits.0: must be object" in sequence[52]["content"]
    next_call = sequence[53]["tool_calls"][0]["function"]["name"]
    print(
        f"15 marked={marked} chars={len(content)} markers={content.count(marker)} "
        f"tool_call_markers={content.count('<｜DSML｜tool_calls>')} "
        f"edits_type={type(edits).__name__} rejected={int(rejected)} "
        f"next={next_call} status={rows[358]['status']}"
    )


if __name__ == "__main__":
    main()
