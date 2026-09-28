"""Embed AG News test headlines with a sentence-transformers model (GPU when available).

Writes <out>/embeddings.npy (float32, n × dim) and <out>/meta.json (device, shape, timing).

Usage:
    uv run --with sentence-transformers tasks/embed_agnews.py --limit 200   # local debug
    python /content/tasks/embed_agnews.py --out /content/results/embed       # on Colab
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

DATA_URL = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, help="only embed the first N headlines")
    parser.add_argument("--out", type=Path, default=Path("results/embed"))
    args = parser.parse_args()

    texts = pd.read_parquet(DATA_URL)["text"].tolist()[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)

    start = time.time()
    emb = model.encode(
        texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True
    )
    seconds = time.time() - start

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embeddings.npy", emb.astype(np.float32))
    meta = {
        "model": MODEL,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "shape": list(emb.shape),
        "encode_seconds": round(seconds, 2),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
