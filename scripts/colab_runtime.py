"""Move files to/from the active headless Colab runtime, or release it — no git, no browser tab.

Uses the colab-mcp OAuth token and the runtime's Jupyter contents API. Paths on the runtime
are absolute (e.g. /content/results/embed). Needs the ats05 colab-mcp package:

    uv run --with ~/src/ats05-colab-mcp scripts/colab_runtime.py push tasks/embed_agnews.py /content/tasks/embed_agnews.py
    uv run --with ~/src/ats05-colab-mcp scripts/colab_runtime.py pull /content/results/embed results/embed
    uv run --with ~/src/ats05-colab-mcp scripts/colab_runtime.py release
"""

import argparse
import asyncio
import base64
from pathlib import Path

import requests
from colab_mcp.auth import get_credentials
from colab_mcp.client import ColabClient, Prod
from colab_mcp.runtime import DirectRuntimeManager

OAUTH_CONFIG = Path.home() / ".config/colab-oauth.json"


def colab_client() -> ColabClient:
    return ColabClient(Prod(), get_credentials(str(OAUTH_CONFIG)))


def contents_api(endpoint: str) -> tuple[str, dict[str, str]]:
    target = asyncio.run(DirectRuntimeManager(colab_client())._select_target(endpoint))
    headers = {
        "X-Colab-Runtime-Proxy-Token": target.token,
        "X-Colab-Client-Agent": "colab-mcp",
    }
    return target.url.rstrip("/") + "/api/contents", headers


def push(local: Path, remote: str, base: str, headers: dict[str, str]) -> None:
    files = (
        sorted(p for p in local.rglob("*") if p.is_file())
        if local.is_dir()
        else [local]
    )
    for f in files:
        dest = (
            remote.strip("/")
            if f == local
            else f"{remote.strip('/')}/{f.relative_to(local).as_posix()}"
        )
        parts = dest.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            requests.put(
                f"{base}/{'/'.join(parts[:i])}",
                headers=headers,
                json={"type": "directory"},
                timeout=60,
            )
        body = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(f.read_bytes()).decode(),
        }
        requests.put(
            f"{base}/{dest}", headers=headers, json=body, timeout=600
        ).raise_for_status()
        print(f"pushed {f} -> /{dest}")


def pull(remote: str, local: Path, base: str, headers: dict[str, str]) -> None:
    r = requests.get(
        f"{base}/{remote.strip('/')}",
        headers=headers,
        params={"content": 1},
        timeout=600,
    )
    r.raise_for_status()
    item = r.json()
    if item["type"] == "directory":
        for child in item["content"]:
            pull(child["path"], local / child["name"], base, headers)
        return
    if item["format"] != "base64":
        r = requests.get(
            f"{base}/{remote.strip('/')}",
            headers=headers,
            params={"format": "base64", "type": "file"},
            timeout=600,
        )
        r.raise_for_status()
        item = r.json()
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(base64.b64decode(item["content"]))
    print(f"pulled /{remote.strip('/')} -> {local}")


def release() -> None:
    client = colab_client()
    for a in client.list_assignments():
        client.unassign(a.endpoint)
        print(f"released {a.endpoint}")
    print(f"active runtimes left: {len(client.list_assignments())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--endpoint", default="", help="runtime endpoint (default: the only active one)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="upload a local file or directory")
    p.add_argument("local", type=Path)
    p.add_argument("remote")
    p = sub.add_parser("pull", help="download a runtime file or directory")
    p.add_argument("remote")
    p.add_argument("local", type=Path)
    sub.add_parser(
        "release", help="unassign all active runtimes (stops compute-unit use)"
    )
    args = parser.parse_args()

    if args.cmd == "release":
        release()
        return
    base, headers = contents_api(args.endpoint)
    if args.cmd == "push":
        push(args.local, args.remote, base, headers)
    else:
        pull(args.remote, args.local, base, headers)


if __name__ == "__main__":
    main()
