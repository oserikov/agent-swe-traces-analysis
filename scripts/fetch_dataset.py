"""Download the agent-traces HF dataset and cache it locally for fast reloads.

Usage:
    uv run scripts/fetch_dataset.py

Loads from data/agent-traces/ on subsequent runs instead of hitting the Hub.
"""

from pathlib import Path

from datasets import load_dataset, load_from_disk

REPO_ID = "mfmVNfpt2q/agent-traces"
LOCAL_DIR = Path(__file__).resolve().parent.parent / "data" / "agent-traces"


def get_traces():
    if LOCAL_DIR.exists():
        return load_from_disk(str(LOCAL_DIR))
    traces = load_dataset(REPO_ID, split="train")
    LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    traces.save_to_disk(str(LOCAL_DIR))
    return traces


if __name__ == "__main__":
    ds = get_traces()
    print(f"Loaded {len(ds)} rows, columns: {ds.column_names}")
    print(f"Cached at {LOCAL_DIR}")
