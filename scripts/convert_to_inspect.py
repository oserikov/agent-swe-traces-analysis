"""Convert the agent-traces source table into Inspect AI eval logs.

One source row -> one EvalSample. Samples are grouped into logs by (job, model),
matching the dataset's 15 distinct (job, model) pairs. See inspect-conversion-spec.txt
for the full mapping rules this implements.

Usage:
    uv run scripts/convert_to_inspect.py [--limit N] [--out DIR]
"""

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from fetch_dataset import get_traces
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    ModelOutput,
)
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall

REPO_ID = "mfmVNfpt2q/agent-traces"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "inspect_logs"


def convert_message(
    idx: int, m: dict, model_codename: str, exceptions: list[str]
) -> tuple[ChatMessage, list[str]]:
    """Convert one source conversation message to an Inspect ChatMessage.

    Returns (message, generated_tool_call_ids).
    """
    role = m["role"]
    content = m.get("content")
    metadata: dict = {"source_index": idx, "source_role": role}
    generated_ids: list[str] = []

    if role in ("system", "developer"):
        return ChatMessageSystem(
            content=content or "", metadata=metadata
        ), generated_ids

    if role == "user":
        return ChatMessageUser(content=content or "", metadata=metadata), generated_ids

    if role == "assistant":
        blocks: list = [ContentText(text=content or "")]
        reasoning = m.get("reasoning_content")
        if reasoning:
            blocks.append(ContentReasoning(reasoning=reasoning))

        tool_calls = None
        raw_calls = m.get("tool_calls")
        if raw_calls:
            tool_calls = []
            for k, c in enumerate(raw_calls):
                call_id = c.get("id")
                if not call_id:
                    call_id = f"generated:{idx}:{k}"
                    generated_ids.append(call_id)
                    exceptions.append(
                        f"msg[{idx}] tool_call[{k}]: missing id, generated '{call_id}'"
                    )
                func = c.get("function", {})
                name = func.get("name")
                args = func.get("arguments")
                parse_error = None
                if not isinstance(args, dict):
                    parse_error = f"non-dict arguments of type {type(args).__name__}"
                    exceptions.append(
                        f"msg[{idx}] tool_call[{k}] ({call_id}): {parse_error}"
                    )
                    args = {}
                if not name:
                    exceptions.append(
                        f"msg[{idx}] tool_call[{k}] ({call_id}): missing function name"
                    )
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        function=name or "",
                        arguments=args,
                        parse_error=parse_error,
                    )
                )

        return (
            ChatMessageAssistant(
                content=blocks,
                tool_calls=tool_calls,
                model=model_codename,
                metadata=metadata,
            ),
            generated_ids,
        )

    if role == "tool":
        tool_call_id = m.get("tool_call_id")
        return (
            ChatMessageTool(
                content=content or "", tool_call_id=tool_call_id, metadata=metadata
            ),
            generated_ids,
        )

    # Unrepresentable role: keep it, flag it.
    exceptions.append(
        f"msg[{idx}]: unrepresentable source role '{role}', kept as user message"
    )
    metadata["unrepresentable_role"] = True
    return ChatMessageUser(
        content=str(content) if content is not None else "", metadata=metadata
    ), generated_ids


def build_sample(row: dict) -> EvalSample:
    conv = row["conversation"]
    exceptions: list[str] = []

    messages: list[ChatMessage] = []
    known_call_ids: set[str] = set()
    for idx, m in enumerate(conv):
        msg, _generated = convert_message(idx, m, row["model"], exceptions)
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            for tc in msg.tool_calls:
                known_call_ids.add(tc.id)
        if (
            isinstance(msg, ChatMessageTool)
            and msg.tool_call_id
            and msg.tool_call_id not in known_call_ids
        ):
            exceptions.append(
                f"msg[{idx}]: tool result tool_call_id '{msg.tool_call_id}' "
                "does not match any prior tool_call id in this trace"
            )
        messages.append(msg)

    # input: task prompt from the first user message, if any.
    sample_input: str | list[ChatMessage] = messages
    first_user = next(
        (msg for msg in messages if isinstance(msg, ChatMessageUser)), None
    )
    if first_user is not None:
        sample_input = first_user.text

    # output: final assistant response, if any.
    last_assistant = next(
        (msg for msg in reversed(messages) if isinstance(msg, ChatMessageAssistant)),
        None,
    )
    output = (
        ModelOutput.from_message(last_assistant)
        if last_assistant is not None
        else ModelOutput()
    )

    n_proxy_requests = row["n_proxy_requests"]
    proxy_requests = row["proxy_requests"]
    if n_proxy_requests != len(proxy_requests):
        exceptions.append(
            f"n_proxy_requests ({n_proxy_requests}) != len(proxy_requests) ({len(proxy_requests)})"
        )

    metadata = {
        "source_model": row["model"],
        "source_job": row["job"],
        "source_trial": row["trial"],
        "source_task_id": row["task_id"],
        "n_proxy_requests": n_proxy_requests,
        "proxy_requests": proxy_requests,
        "source_system_prompt": row["system_prompt"],
        "source_tools": row["tools"],
        "source_num_messages": len(conv),
        "target_not_supplied": True,
        "conversion_exceptions": exceptions,
    }

    score = Score(
        value=row["reward"],
        metadata={"status": row["status"]},
        explanation=f"Recorded verdict: {row['status']}",
    )

    return EvalSample(
        id=row["trial"],
        epoch=1,
        input=sample_input,
        target="",
        messages=messages,
        output=output,
        scores={"grade": score},
        metadata=metadata,
    )


def convert(limit: int | None, out_dir: Path) -> list[Path]:
    ds = get_traces()
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in ds:
        groups[(row["job"], row["model"])].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(UTC).isoformat()
    written: list[Path] = []

    for (job, model), rows in sorted(groups.items()):
        samples = [build_sample(row) for row in rows]
        trial_ids = [s.id for s in samples]

        eval_log = EvalLog(
            status="success",
            eval=EvalSpec(
                created=created,
                task="agent-swe-traces-conversion",
                task_id=job,
                run_id=job,
                dataset=EvalDataset(
                    name="agent-traces",
                    location=REPO_ID,
                    samples=len(samples),
                    sample_ids=trial_ids,
                ),
                model=model,
                config=EvalConfig(),
                metadata={
                    "source_job": job,
                    "source_model": model,
                    "source_table": REPO_ID,
                    "conversion_spec": "inspect-conversion-spec.txt",
                },
            ),
            samples=samples,
        )

        out_path = out_dir / f"{job}_{model}.eval"
        write_eval_log(eval_log, str(out_path))
        written.append(out_path)
        print(f"wrote {out_path} ({len(samples)} samples)")

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit source rows (debug)"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory"
    )
    args = parser.parse_args()

    written = convert(args.limit, args.out)
    print(f"\nWrote {len(written)} logs to {args.out}")


if __name__ == "__main__":
    main()
