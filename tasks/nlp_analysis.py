"""NLP contrast analysis of agent-traces text: lexical, style, topic, embedding, classifier stages.

Loads mfmVNfpt2q/agent-traces directly (works on Colab without the repo). See
spec-nlp-analysis.md for the full design.

Usage:
    uv run --with sentence-transformers --with scikit-learn tasks/nlp_analysis.py \
        --limit 40 --device cpu --out results/nlp_analysis_debug
    python /content/tasks/nlp_analysis.py --device cuda --out /content/results/nlp_analysis
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

REPO_ID = "mfmVNfpt2q/agent-traces"

UNIT_MAX_CHARS = 2000
UNIT_MAX_PER_TRACE = 150
RNG_SEED = 0
LIFT_CEILING = 20.0

BASELINE_TERMS = {
    "model-cyan": ["marcus", "the team and i"],
    "model-vega": ["patient", "ailment", "prognosis", "convalesc"],
}
BOILERPLATE_LINE = "The following interfaces are expected by the test suite"

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_traces():
    from datasets import load_dataset

    try:
        from scripts.fetch_dataset import get_traces

        return get_traces()
    except Exception:
        return load_dataset(REPO_ID, split="train")


# ---------------------------------------------------------------------------
# Unit extraction
# ---------------------------------------------------------------------------

BASH_SEGMENT_SPLIT = re.compile(r"\s*&&\s*|\s*\|\s*|\s*;\s*")
LEADING_CD = re.compile(r"^\s*cd\s+\S+\s*&&\s*")


def skeletonize_bash(command: str) -> str:
    command = LEADING_CD.sub("", command)
    segments = BASH_SEGMENT_SPLIT.split(command)
    tokens = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        parts = seg.split()
        tokens.append(" ".join(parts[:2]))
    return " | ".join(tokens)


def split_reasoning_units(text: str) -> list[str]:
    pieces = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged: list[str] = []
    buf = ""
    for p in pieces:
        buf = (buf + "\n\n" + p) if buf else p
        if len(buf) >= 200:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buf
        else:
            merged.append(buf)
    return merged


def extract_units(ds) -> pd.DataFrame:
    rows = []
    for row_idx in range(len(ds)):
        row = ds[row_idx]
        model, job, task_id, trial, status = (
            row["model"],
            row["job"],
            row["task_id"],
            row["trial"],
            row["status"],
        )
        rng = np.random.default_rng(RNG_SEED + row_idx)
        reasoning_units: list[tuple[int, str]] = []
        content_units: list[tuple[int, str]] = []
        bash_units: list[tuple[int, str]] = []
        for msg_idx, msg in enumerate(row["conversation"]):
            if msg.get("role") != "assistant":
                continue
            reasoning = msg.get("reasoning_content") or ""
            if reasoning.strip():
                for piece in split_reasoning_units(reasoning):
                    reasoning_units.append((msg_idx, piece))
            content = msg.get("content") or ""
            if content.strip():
                content_units.append((msg_idx, content))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name") == "bash":
                    args = fn.get("arguments", {})
                    cmd = args.get("command") if isinstance(args, dict) else None
                    if cmd is None and isinstance(args, dict):
                        cmd = args.get("cmd")
                    if isinstance(cmd, dict):
                        cmd = cmd.get("cmd") or cmd.get("command")
                    if isinstance(cmd, list):
                        cmd = " ".join(str(c) for c in cmd)
                    if isinstance(cmd, str) and cmd:
                        bash_units.append((msg_idx, skeletonize_bash(cmd)))

        meta = {
            "row": row_idx,
            "trial": trial,
            "model": model,
            "job": job,
            "task_id": task_id,
            "status": status,
        }
        rows.extend(_capped_rows(reasoning_units, "reasoning", meta, rng))
        rows.extend(_capped_rows(content_units, "content", meta, rng))
        for msg_idx, text in bash_units:
            rows.append({**meta, "msg_idx": msg_idx, "field": "bash", "text": text})
    return pd.DataFrame(rows)


def _capped_rows(
    units: list[tuple[int, str]], field: str, meta: dict, rng: np.random.Generator
) -> list[dict]:
    if not units:
        return []
    kept_idx = list(range(len(units)))
    if len(units) > UNIT_MAX_PER_TRACE:
        idx = rng.choice(len(units), size=UNIT_MAX_PER_TRACE, replace=False)
        idx.sort()
        kept_idx = list(idx)
    return [
        {**meta, "msg_idx": units[pos][0], "field": field, "text": units[pos][1][:UNIT_MAX_CHARS]}
        for pos in kept_idx
    ]


def first_user_message(ds, row_idx: int) -> str:
    for msg in ds[row_idx]["conversation"]:
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

RE_FENCED = re.compile(r"```.*?```", re.DOTALL)
RE_INLINE = re.compile(r"`[^`\n]+`")
RE_URL = re.compile(r"https?://\S+")
RE_PATH_EXT = re.compile(
    r"\b[\w./-]+\.(py|pyi|txt|toml|cfg|ini|md|rst|json|ya?ml|sh|c|h|js|ts)\b"
)
RE_PATH_SLASH = re.compile(r"\b[\w.-]+(/[\w.-]+){2,}")
RE_HEX = re.compile(r"\b[0-9a-f]{7,40}\b")
RE_SNAKE = re.compile(r"\b\w*[a-z0-9]_\w+\b")
RE_CAMEL_LOWER = re.compile(r"\b[a-z]+[A-Z]\w*\b")
RE_CAMEL_UPPER = re.compile(r"\b[A-Z][a-z]+[A-Z]\w*\b")
RE_NUMBER = re.compile(r"\b\d+(\.\d+)?\b")
RE_ALLCAPS = re.compile(r"\b[A-Z]{2,}\b")


def mask_text(text: str, taskwords: set[str]) -> str:
    t = RE_FENCED.sub(" CODE ", text)
    t = RE_INLINE.sub(" CODE ", t)
    t = RE_URL.sub(" URL ", t)
    t = RE_PATH_EXT.sub(" PATH ", t)
    t = RE_PATH_SLASH.sub(" PATH ", t)
    t = RE_HEX.sub(" HEX ", t)

    def ident_sub(m: re.Match) -> str:
        word = m.group(0)
        if RE_ALLCAPS.fullmatch(word):
            return word
        return " IDENT "

    t = RE_SNAKE.sub(ident_sub, t)
    t = RE_CAMEL_LOWER.sub(ident_sub, t)
    t = RE_CAMEL_UPPER.sub(ident_sub, t)
    t = RE_NUMBER.sub(" NUM ", t)

    lowered_tokens = []
    for tok in re.split(r"(\s+)", t):
        stripped = tok.strip()
        if stripped and stripped not in {
            "CODE",
            "URL",
            "PATH",
            "HEX",
            "IDENT",
            "NUM",
        }:
            if RE_ALLCAPS.fullmatch(stripped):
                lowered_tokens.append(tok)
                continue
            tok = tok.lower()
        lowered_tokens.append(tok)
    t = "".join(lowered_tokens)

    if taskwords:
        words = re.split(r"(\W+)", t)
        for i, w in enumerate(words):
            if w.lower() in taskwords:
                words[i] = "TASKWORD"
        t = "".join(words)
    return t


WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def tokenize_for_taskwords(text: str) -> set[str]:
    masked = mask_text(text, set())
    return {w.lower() for w in WORD_RE.findall(masked)}


def build_taskword_masks(ds, row_idx_list: list[int]) -> dict[int, set[str]]:
    task_statements: dict[str, str] = {}
    row_to_task: dict[int, str] = {}
    for row_idx in row_idx_list:
        task_id = ds[row_idx]["task_id"]
        row_to_task[row_idx] = task_id
        if task_id not in task_statements:
            task_statements[task_id] = first_user_message(ds, row_idx)

    boilerplate_words = tokenize_for_taskwords(BOILERPLATE_LINE)
    system_prompt_words: set[str] = set()
    for row_idx in row_idx_list[:1]:
        system_prompt_words |= tokenize_for_taskwords(ds[row_idx]["system_prompt"] or "")

    task_word_sets = {t: tokenize_for_taskwords(s) for t, s in task_statements.items()}
    n_tasks = len(task_statements)
    doc_freq: Counter[str] = Counter()
    for words in task_word_sets.values():
        for w in words:
            doc_freq[w] += 1

    exclude = boilerplate_words | system_prompt_words
    mask_by_task: dict[str, set[str]] = {}
    for task, words in task_word_sets.items():
        mask_by_task[task] = {
            w
            for w in words
            if w not in exclude and (doc_freq[w] / n_tasks) < 0.05
        }
    return {row_idx: mask_by_task[row_to_task[row_idx]] for row_idx in row_idx_list}


# ---------------------------------------------------------------------------
# Tokenizer (shared, lexical/topic stages)
# ---------------------------------------------------------------------------

PLACEHOLDER_TOKENS = {"CODE", "URL", "PATH", "HEX", "IDENT", "NUM", "TASKWORD"}
WHITELIST_PUNCT = ["—", "…", "!", "?!", "##", "**", "->", "✅", "❌", "｜"]


def script_tag(ch: str) -> str | None:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    for script in ("HAN", "CYRILLIC", "ARABIC", "HEBREW", "HIRAGANA", "KATAKANA", "HANGUL"):
        if script in name:
            return script
    return None


def is_emoji(ch: str) -> bool:
    return unicodedata.category(ch) == "So" or 0x1F300 <= ord(ch) <= 0x1FAFF


TOKEN_SCAN = re.compile(
    "|".join(
        [r"\b(?:" + "|".join(PLACEHOLDER_TOKENS) + r")\b"]
        + [r"[^\W\d_]{2,}"]
        + [re.escape(p) for p in WHITELIST_PUNCT]
        + [r"[^\x00-\x7F]"]
    )
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for m in TOKEN_SCAN.finditer(text):
        tok = m.group(0)
        if tok in PLACEHOLDER_TOKENS or tok in WHITELIST_PUNCT:
            tokens.append(tok)
            continue
        if re.fullmatch(r"[^\W\d_]{2,}", tok) and tok.isascii():
            tokens.append(tok)
            continue
        if len(tok) == 1 and not tok.isascii():
            if is_emoji(tok):
                tokens.append(tok)
                continue
            tag = script_tag(tok)
            if tag:
                tokens.append(f"SCRIPT_{tag}")
    return tokens


def ngram_tokenizer(text: str, n_min=1, n_max=3) -> list[str]:
    toks = tokenize(text)
    out = []
    for n in range(n_min, n_max + 1):
        for i in range(len(toks) - n + 1):
            out.append(" ".join(toks[i : i + n]))
    return out


# ---------------------------------------------------------------------------
# Style features
# ---------------------------------------------------------------------------

RE_BOLD = re.compile(r"\*\*[^*]+\*\*")
RE_HEADER = re.compile(r"^#{1,6}\s", re.MULTILINE)
RE_BULLET = re.compile(r"^\s*[-*]\s", re.MULTILINE)
RE_BANG = re.compile(r"!")
RE_ALLCAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")


def style_features_for_text(text: str) -> dict:
    n = max(len(text), 1)
    scripts = Counter()
    non_latin = 0
    emoji_count = 0
    for ch in text:
        if ch.isalpha():
            tag = script_tag(ch)
            if tag:
                scripts[tag] += 1
                non_latin += 1
        if is_emoji(ch):
            emoji_count += 1
    words = re.findall(r"\b[A-Za-z]+\b", text)
    allcaps = sum(1 for w in words if len(w) >= 2 and w.isupper())
    return {
        "non_latin_share": non_latin / n,
        **{f"script_{k}_share": v / n for k, v in scripts.items()},
        "emoji_per_1k": emoji_count / n * 1000,
        "headers_per_1k": len(RE_HEADER.findall(text)) / n * 1000,
        "bullets_per_1k": len(RE_BULLET.findall(text)) / n * 1000,
        "bold_per_1k": len(RE_BOLD.findall(text)) / n * 1000,
        "allcaps_share": allcaps / max(len(words), 1),
        "bang_per_1k": len(RE_BANG.findall(text)) / n * 1000,
    }


def compute_style_features(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (row, model, job, task_id, field), grp in units[
        units.field.isin(["reasoning", "content"])
    ].groupby(["row", "model", "job", "task_id", "field"]):
        text = "\n".join(grp["text"])
        placeholders = Counter()
        total_tok = 0
        for t in grp["text_masked"]:
            for tok in t.split():
                total_tok += 1
                if tok in PLACEHOLDER_TOKENS:
                    placeholders[tok] += 1
        feats = style_features_for_text(text)
        feats.update(
            {
                "row": row,
                "model": model,
                "job": job,
                "task_id": task_id,
                "field": field,
                "masked_share": sum(placeholders.values()) / max(total_tok, 1),
            }
        )
        for ph in PLACEHOLDER_TOKENS:
            feats[f"masked_share_{ph}"] = placeholders.get(ph, 0) / max(total_tok, 1)
        rows.append(feats)
    return pd.DataFrame(rows)


def contrast_style(style_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        c
        for c in style_df.columns
        if c not in ("row", "model", "job", "task_id", "field")
    ]
    candidates = []
    for field, fgrp in style_df.groupby("field"):
        for feat in feature_cols:
            for model, mgrp in fgrp.groupby("model"):
                rest = fgrp[fgrp.model != model]
                if len(mgrp) < 5 or len(rest) < 5:
                    continue
                a, b = mgrp[feat].to_numpy(), rest[feat].to_numpy()
                if np.allclose(a, a[0]) and np.allclose(b, a[0]):
                    continue
                try:
                    _, p = mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    continue
                if p >= 1e-3:
                    continue
                med_a, med_b = np.median(a), np.median(b)
                eps = max(np.median(np.concatenate([a, b])[np.concatenate([a, b]) > 0]), 1e-6) * 0.1 if (a.sum() + b.sum()) > 0 else 1e-6
                ratio = (med_a + eps) / (med_b + eps)
                model_rows = fgrp[fgrp.model == model]
                n_tasks = model_rows["task_id"].nunique()
                other_task_ids = set(rest["task_id"])
                overlap = sum(1 for t in model_rows["task_id"] if t in other_task_ids)
                task_overlap_share = overlap / max(len(model_rows), 1)
                per_job_rates = None
                job_fisher_p = None
                if model in MULTI_JOB_MODELS:
                    per_job_rates = {}
                    for job in MULTI_JOB_MODELS[model]:
                        jrows = model_rows[model_rows.job == job][feat]
                        if len(jrows) > 0:
                            per_job_rates[job] = float(jrows.median())
                    jobs_present = [j for j in MULTI_JOB_MODELS[model] if j in model_rows.job.unique()]
                    if len(jobs_present) >= 2:
                        ja, jb = jobs_present[0], jobs_present[1]
                        va = model_rows[model_rows.job == ja][feat].to_numpy()
                        vb = model_rows[model_rows.job == jb][feat].to_numpy()
                        if len(va) >= 3 and len(vb) >= 3 and not (np.allclose(va, va[0]) and np.allclose(vb, va[0])):
                            try:
                                _, job_fisher_p = mannwhitneyu(va, vb)
                            except ValueError:
                                job_fisher_p = None
                verdict = attribute(n_tasks, task_overlap_share, model, per_job_rates, job_fisher_p)
                candidates.append(
                    {
                        "source": "style",
                        "field": field,
                        "model": model,
                        "label": feat,
                        "p_value": p,
                        "median_ratio": ratio,
                        "median_model": med_a,
                        "median_rest": med_b,
                        "n_tasks": n_tasks,
                        "verdict_rule": verdict,
                    }
                )
    return pd.DataFrame(candidates)


# ---------------------------------------------------------------------------
# Attribution rule
# ---------------------------------------------------------------------------

MULTI_JOB_MODELS = {
    "model-cyan": ["run-01", "run-07"],
    "model-delta": ["run-03", "run-12"],
    "model-flint": ["run-04", "run-06", "run-08", "run-14"],
    "model-garnet": ["run-02", "run-09", "run-11", "run-13"],
}


def attribute(
    n_tasks: int,
    task_overlap_share: float,
    model: str,
    per_job_rates: dict | None,
    job_fisher_p: float | None,
) -> str:
    if n_tasks < 3 or task_overlap_share >= 0.5:
        return "TASK"
    if model in MULTI_JOB_MODELS and job_fisher_p is not None and job_fisher_p < 0.01:
        return "RUN"
    if n_tasks >= 5:
        if model in MULTI_JOB_MODELS:
            if per_job_rates and all(r > 0 for r in per_job_rates.values()):
                return "MODEL"
            return "UNCLEAR"
        return "MODEL-or-RUN (confounded)"
    return "UNCLEAR"


# ---------------------------------------------------------------------------
# Lexical stage
# ---------------------------------------------------------------------------


def trace_documents(units: pd.DataFrame, field: str) -> pd.DataFrame:
    sub = units[units.field == field]
    docs = (
        sub.groupby(["row", "model", "job", "task_id"])["text_masked"]
        .apply(lambda s: " ".join(s))
        .reset_index()
        .rename(columns={"text_masked": "doc"})
    )
    return docs


def build_binary_dtm(docs: list[str], min_traces: int = 5):
    from sklearn.feature_extraction.text import CountVectorizer

    unigram_vec = CountVectorizer(
        tokenizer=lambda t: [tok for tok in tokenize(t) if len(tok) > 1],
        preprocessor=lambda t: t,
        token_pattern=None,
        ngram_range=(1, 1),
        stop_words="english",
        binary=True,
        min_df=min_traces,
    )
    ngram_vec = CountVectorizer(
        tokenizer=lambda t: [tok for tok in tokenize(t) if len(tok) > 1],
        preprocessor=lambda t: t,
        token_pattern=None,
        ngram_range=(2, 3),
        binary=True,
        min_df=min_traces,
    )
    X1 = unigram_vec.fit_transform(docs)
    try:
        X2 = ngram_vec.fit_transform(docs)
        vocab = list(unigram_vec.get_feature_names_out()) + list(
            ngram_vec.get_feature_names_out()
        )
        from scipy.sparse import hstack

        X = hstack([X1, X2]).tocsr()
    except ValueError:
        X, vocab = X1, list(unigram_vec.get_feature_names_out())
    return X, vocab


def lexical_stage(units: pd.DataFrame, ds, field: str, shared_task_ids: set[str]) -> pd.DataFrame:
    docs_df = trace_documents(units, field)
    if len(docs_df) < 5 or docs_df.empty:
        return pd.DataFrame()
    X, vocab = build_binary_dtm(docs_df["doc"].tolist())
    if X.shape[1] == 0:
        return pd.DataFrame()
    models = docs_df["model"].to_numpy()
    task_ids = docs_df["task_id"].to_numpy()
    unique_models = sorted(set(models))

    candidates = []
    for model in unique_models:
        mask = models == model
        n_g = mask.sum()
        if n_g < 5:
            continue
        rest_mask = ~mask
        col_sums_g = np.asarray(X[mask].sum(axis=0)).ravel()
        col_sums_rest = np.asarray(X[rest_mask].sum(axis=0)).ravel()
        rate_g = col_sums_g / n_g
        rate_rest = col_sums_rest / max(rest_mask.sum(), 1)

        # class-based tf-idf across all models for this term
        rate_all_models = []
        for m2 in unique_models:
            mm = models == m2
            rate_all_models.append(
                np.asarray(X[mm].sum(axis=0)).ravel() / max(mm.sum(), 1)
            )
        rate_sum = np.sum(rate_all_models, axis=0)
        W = rate_g * np.log1p(len(unique_models) / np.maximum(rate_sum, 1e-9))

        keep = np.where(rate_g >= 0.2)[0]
        for t_idx in keep:
            a = int(col_sums_g[t_idx])
            b = int(n_g - a)
            c = int(col_sums_rest[t_idx])
            d = int(rest_mask.sum() - c)
            _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            if p >= 1e-3:
                continue
            term = vocab[t_idx]
            hit_mask = np.asarray(X[:, t_idx].todense()).ravel().astype(bool) & mask
            hit_tasks = task_ids[hit_mask]
            n_tasks = len(set(hit_tasks))

            per_job_rates = None
            job_fisher_p = None
            if model in MULTI_JOB_MODELS:
                jobs_df = docs_df[docs_df.model == model]
                job_vals = jobs_df["job"].to_numpy()
                job_hits = np.asarray(X[docs_df.model.to_numpy() == model][:, t_idx].todense()).ravel()
                per_job_rates = {}
                for job in MULTI_JOB_MODELS[model]:
                    jm = job_vals == job
                    if jm.sum() > 0:
                        per_job_rates[job] = float(job_hits[jm].sum() / jm.sum())
                if len(per_job_rates) >= 2:
                    jobs_list = list(per_job_rates.keys())
                    ja, jb = jobs_list[0], jobs_list[1]
                    ja_mask = job_vals == ja
                    jb_mask = job_vals == jb
                    a2 = int(job_hits[ja_mask].sum())
                    b2 = int(ja_mask.sum() - a2)
                    c2 = int(job_hits[jb_mask].sum())
                    d2 = int(jb_mask.sum() - c2)
                    try:
                        _, job_fisher_p = fisher_exact([[a2, b2], [c2, d2]])
                    except ValueError:
                        job_fisher_p = None

            term_hit_all = np.asarray(X[:, t_idx].todense()).ravel().astype(bool)
            other_model_hit_task_ids = set(task_ids[term_hit_all & ~mask])
            overlap_hits = sum(1 for t in hit_tasks if t in other_model_hit_task_ids)
            task_overlap_share = overlap_hits / max(len(hit_tasks), 1)

            survives = "n/a"
            if shared_task_ids:
                shared_row_mask = np.isin(task_ids, list(shared_task_ids))
                if shared_row_mask.sum() >= 5 and mask[shared_row_mask].sum() >= 3:
                    Xs = X[shared_row_mask]
                    models_s = models[shared_row_mask]
                    ms = models_s == model
                    if ms.sum() >= 3:
                        rs = np.asarray(Xs[ms].sum(axis=0)).ravel()[t_idx] / ms.sum()
                        survives = "yes" if rs >= 0.2 else "no"

            verdict = attribute(n_tasks, task_overlap_share, model, per_job_rates, job_fisher_p)

            candidates.append(
                {
                    "field": field,
                    "model": model,
                    "term": term,
                    "W": W[t_idx],
                    "rate_model": rate_g[t_idx],
                    "rate_rest": rate_rest[t_idx],
                    "p_fisher": p,
                    "n_tasks": n_tasks,
                    "task_score": W[t_idx],
                    "per_job_rates": json.dumps(per_job_rates) if per_job_rates else None,
                    "survives_shared_tasks": survives,
                    "verdict_rule": verdict,
                }
            )
    out = pd.DataFrame(candidates)
    if not out.empty:
        out = out.sort_values(["model", "W"], ascending=[True, False])
    return out


# ---------------------------------------------------------------------------
# Topic stage
# ---------------------------------------------------------------------------


def topic_stage(units: pd.DataFrame, field: str) -> tuple[pd.DataFrame, dict]:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_mutual_info_score

    sub = units[units.field == field].reset_index(drop=True)
    if len(sub) < 50:
        return pd.DataFrame(), {}

    vec = TfidfVectorizer(
        tokenizer=lambda t: [tok for tok in tokenize(t) if len(tok) > 1],
        preprocessor=lambda t: t,
        token_pattern=None,
        min_df=5,
    )
    X = vec.fit_transform(sub["text_masked"])
    if X.shape[1] < 25:
        return pd.DataFrame(), {}
    k = min(25, X.shape[1] - 1)
    nmf = NMF(n_components=k, random_state=0, init="nndsvda", max_iter=300)
    W = nmf.fit_transform(X)
    H = nmf.components_
    argmax_topic = W.argmax(axis=1)
    terms = np.array(vec.get_feature_names_out())

    ami = {
        "model": float(adjusted_mutual_info_score(argmax_topic, sub["model"])),
        "job": float(adjusted_mutual_info_score(argmax_topic, sub["job"])),
        "task_id": float(adjusted_mutual_info_score(argmax_topic, sub["task_id"])),
    }

    overall_model_share = sub["model"].value_counts(normalize=True).to_dict()
    rows = []
    for t in range(k):
        top_terms = terms[np.argsort(H[t])[::-1][:12]].tolist()
        topic_mask = argmax_topic == t
        if topic_mask.sum() == 0:
            continue
        topic_units = sub[topic_mask]
        model_dist = topic_units["model"].value_counts(normalize=True)
        dominant_model = model_dist.idxmax()
        lift = model_dist.max() / overall_model_share.get(dominant_model, 1e-9)
        n_tasks = topic_units["task_id"].nunique()
        verdict = "UNCLEAR"
        candidate = False
        if lift >= 3 and n_tasks >= 5:
            candidate = True
            other_model_same_topic = sub[topic_mask & (sub.model != dominant_model)]
            other_task_ids = set(other_model_same_topic["task_id"])
            overlap = sum(1 for t2 in topic_units["task_id"] if t2 in other_task_ids)
            task_overlap_share = overlap / max(len(topic_units), 1)
            per_job_rates = None
            job_fisher_p = None
            if dominant_model in MULTI_JOB_MODELS:
                job_counts = topic_units["job"].value_counts()
                all_job_counts = sub[sub.model == dominant_model]["job"].value_counts()
                per_job_rates = {
                    j: float(job_counts.get(j, 0) / max(all_job_counts.get(j, 1), 1))
                    for j in MULTI_JOB_MODELS[dominant_model]
                }
            verdict = attribute(
                n_tasks, task_overlap_share, dominant_model, per_job_rates, job_fisher_p
            )
        rows.append(
            {
                "field": field,
                "topic": t,
                "top_terms": ", ".join(top_terms),
                "dominant_model": dominant_model,
                "lift": lift,
                "n_tasks": n_tasks,
                "n_units": int(topic_mask.sum()),
                "is_candidate": candidate,
                "verdict_rule": verdict,
            }
        )
    return pd.DataFrame(rows), ami


# ---------------------------------------------------------------------------
# Embedding stage
# ---------------------------------------------------------------------------


def embedding_stage(units: pd.DataFrame, field: str, out_dir: Path, device: str) -> tuple[pd.DataFrame, dict]:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_mutual_info_score

    sub = units[units.field == field].reset_index(drop=True)
    if len(sub) < 60:
        return pd.DataFrame(), {}

    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B", device=device, model_kwargs={"torch_dtype": "float16"} if device == "cuda" else {}
    )
    model.max_seq_length = 512
    emb = model.encode(
        sub["text"].tolist(),
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    np.save(out_dir / f"embeddings_{field}.npy", emb.astype(np.float32))
    sub.to_parquet(out_dir / f"units_{field}.parquet")

    k = min(60, len(sub) - 1)
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3)
    labels = km.fit_predict(emb)

    ami = {
        "model": float(adjusted_mutual_info_score(labels, sub["model"])),
        "job": float(adjusted_mutual_info_score(labels, sub["job"])),
        "task_id": float(adjusted_mutual_info_score(labels, sub["task_id"])),
    }

    overall_model_share = sub["model"].value_counts(normalize=True).to_dict()
    rows = []
    for c in range(k):
        cluster_mask = labels == c
        if cluster_mask.sum() == 0:
            continue
        cluster_units = sub[cluster_mask]
        model_dist = cluster_units["model"].value_counts(normalize=True)
        dominant_model = model_dist.idxmax()
        lift = model_dist.max() / overall_model_share.get(dominant_model, 1e-9)
        n_tasks = cluster_units["task_id"].nunique()
        centroid = km.cluster_centers_[c]
        cluster_emb = emb[cluster_mask]
        dists = np.linalg.norm(cluster_emb - centroid, axis=1)
        closest_local = np.argsort(dists)[:5]
        closest_rows = cluster_units.iloc[closest_local][["row", "msg_idx", "text"]].to_dict("records")

        verdict = "UNCLEAR"
        is_candidate = False
        if lift >= 3 and n_tasks >= 5:
            is_candidate = True
            other_model_same_cluster = sub[cluster_mask & (sub.model != dominant_model)]
            other_task_ids = set(other_model_same_cluster["task_id"])
            overlap = sum(1 for t2 in cluster_units["task_id"] if t2 in other_task_ids)
            task_overlap_share = overlap / max(len(cluster_units), 1)
            per_job_rates = None
            if dominant_model in MULTI_JOB_MODELS:
                job_counts = cluster_units["job"].value_counts()
                all_job_counts = sub[sub.model == dominant_model]["job"].value_counts()
                per_job_rates = {
                    j: float(job_counts.get(j, 0) / max(all_job_counts.get(j, 1), 1))
                    for j in MULTI_JOB_MODELS[dominant_model]
                }
            verdict = attribute(n_tasks, task_overlap_share, dominant_model, per_job_rates, None)

        rows.append(
            {
                "field": field,
                "cluster": c,
                "dominant_model": dominant_model,
                "lift": lift,
                "n_tasks": n_tasks,
                "n_units": int(cluster_mask.sum()),
                "is_candidate": is_candidate,
                "verdict_rule": verdict,
                "closest_units": json.dumps(closest_rows)[:3000],
            }
        )
    return pd.DataFrame(rows), ami


# ---------------------------------------------------------------------------
# Classifier stage
# ---------------------------------------------------------------------------


def _cv_macro_f1(X: np.ndarray, y: np.ndarray, groups: np.ndarray, gkf) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    preds = np.empty(len(y), dtype=object)
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
    return f1_score(y, preds, average="macro")


def classifier_stage(units_reasoning_path: Path, units_content_path: Path) -> dict:
    from sklearn.model_selection import GroupKFold

    results = {}
    for field, path, emb_path in [
        ("reasoning", units_reasoning_path, units_reasoning_path.parent / "embeddings_reasoning.npy"),
        ("content", units_content_path, units_content_path.parent / "embeddings_content.npy"),
    ]:
        if not path.exists() or not emb_path.exists():
            continue
        sub = pd.read_parquet(path)
        emb = np.load(emb_path)
        y = sub["model"].to_numpy()
        groups = sub["task_id"].to_numpy()
        lengths = sub["text"].str.len().to_numpy().reshape(-1, 1)

        n_splits = min(5, pd.Series(groups).nunique())
        if n_splits < 2:
            continue
        gkf = GroupKFold(n_splits=n_splits)

        from sklearn.metrics import f1_score

        counts = pd.Series(y).value_counts()
        f1_majority = f1_score(y, [counts.idxmax()] * len(y), average="macro")
        f1_emb = _cv_macro_f1(emb, y, groups, gkf)
        f1_len = _cv_macro_f1(lengths, y, groups, gkf)
        results[field] = {
            "macro_f1_embeddings": f1_emb,
            "macro_f1_length_only": f1_len,
            "macro_f1_always_majority": float(f1_majority),
            "majority_class_share": float(counts.max() / len(y)),
            "n_units": len(sub),
        }
    return results


# ---------------------------------------------------------------------------
# Examples writer
# ---------------------------------------------------------------------------


def write_examples(candidate_id: str, examples: list[dict], out_dir: Path, source_texts: dict):
    lines = [f"# {candidate_id}\n"]
    for ex in examples:
        excerpt = ex["excerpt"][:300]
        key = (ex["row"], ex["msg_idx"], ex.get("field"))
        sources = source_texts.get(key, [])
        assert any(excerpt in s for s in sources), (
            f"excerpt not substring of source for {candidate_id} {key}"
        )
        lines.append(f"- row={ex['row']} msg_idx={ex['msg_idx']} field={ex.get('field')}\n")
        lines.append(f"  > {excerpt}\n")
    (out_dir / f"{candidate_id}.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/nlp_analysis"))
    parser.add_argument(
        "--stages", default="lexical,topics,embed,classify", help="comma-separated"
    )
    parser.add_argument("--device", default=None, choices=["cuda", "cpu", None])
    args = parser.parse_args()
    stages = set(args.stages.split(","))
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("loading dataset...")
    ds = load_traces()
    n = len(ds)
    if args.limit:
        by_model: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            by_model[ds[i]["model"]].append(i)
        models = sorted(by_model)
        row_idx_list = []
        i = 0
        while len(row_idx_list) < min(args.limit, n):
            for m in models:
                if i < len(by_model[m]):
                    row_idx_list.append(by_model[m][i])
                    if len(row_idx_list) >= args.limit:
                        break
            i += 1
        row_idx_list = sorted(row_idx_list)
        ds = ds.select(row_idx_list)

    print(f"extracting units from {len(ds)} traces...")
    units = extract_units(ds)
    print(units.groupby(["model", "field"]).size().unstack(fill_value=0))

    units_summary = units.groupby(["model", "field"]).size().reset_index(name="n_units")
    units_summary.to_csv(args.out / "units_summary.csv", index=False)

    print("building task-echo masks...")
    taskword_masks = build_taskword_masks(ds, list(range(len(ds))))

    def mask_row_group(g):
        tw = taskword_masks.get(g.name, set())
        return g["text"].apply(lambda t: mask_text(t, tw))

    units["text_masked"] = units.groupby("row", group_keys=False).apply(mask_row_group)

    # masking report
    masking_rows = []
    recall_hits_lost = 0
    for model, mgrp in units.groupby("model"):
        for field, fgrp in mgrp.groupby("field"):
            total_tok = 0
            ph_counts = Counter()
            for t in fgrp["text_masked"]:
                for tok in t.split():
                    total_tok += 1
                    if tok in PLACEHOLDER_TOKENS:
                        ph_counts[tok] += 1
            for ph in PLACEHOLDER_TOKENS:
                masking_rows.append(
                    {
                        "model": model,
                        "field": field,
                        "placeholder": ph,
                        "share": ph_counts.get(ph, 0) / max(total_tok, 1),
                    }
                )
    for terms in BASELINE_TERMS.values():
        for term in terms:
            for _, r in units.iterrows():
                if term in r["text"].lower() and "TASKWORD" in r["text_masked"] and term not in r["text_masked"]:
                    recall_hits_lost += 1
    pd.DataFrame(masking_rows).to_csv(args.out / "masking_report.csv", index=False)
    print(f"recall terms lost to TASKWORD: {recall_hits_lost}")

    print("\n--- masking checkpoint (5 samples per model) ---")
    for model, mgrp in units.groupby("model"):
        sample = mgrp.sample(min(5, len(mgrp)), random_state=0)
        for _, r in sample.iterrows():
            print(f"[{model}] RAW: {r['text'][:120]!r}")
            print(f"[{model}] MSK: {r['text_masked'][:120]!r}")

    task_counts = units.groupby("task_id")["model"].nunique()
    shared_task_ids = set(task_counts[task_counts >= 2].index)
    print(f"shared tasks (>=2 models): {len(shared_task_ids)}")

    all_candidates = []

    if "lexical" in stages:
        print("lexical stage...")
        lexical_frames = {}
        for field in ["reasoning", "content", "bash"]:
            lex = lexical_stage(units, ds, field, shared_task_ids)
            lexical_frames[field] = lex
            lex.to_csv(args.out / f"lexical_terms_{field}.csv", index=False)
            if not lex.empty:
                lc = lex.copy()
                lc["source"] = "term"
                lc["label"] = lc["term"]
                lc["lift"] = (lc["rate_model"] / lc["rate_rest"].clip(lower=0.05)).clip(
                    upper=LIFT_CEILING
                )
                all_candidates.append(
                    lc[
                        [
                            "source",
                            "field",
                            "model",
                            "label",
                            "rate_model",
                            "rate_rest",
                            "n_tasks",
                            "per_job_rates",
                            "survives_shared_tasks",
                            "verdict_rule",
                            "W",
                            "lift",
                        ]
                    ]
                )

        style_df = compute_style_features(units)
        style_df.to_csv(args.out / "style_features.csv", index=False)
        style_contrast = contrast_style(style_df)
        style_contrast.to_csv(args.out / "style_contrast.csv", index=False)
        if not style_contrast.empty:
            sc = style_contrast.copy()
            sc["rate_model"] = sc["median_model"]
            sc["rate_rest"] = sc["median_rest"]
            sc["per_job_rates"] = None
            sc["survives_shared_tasks"] = "n/a"
            sc["W"] = sc["median_ratio"]
            sc["lift"] = sc["median_ratio"].clip(upper=LIFT_CEILING)
            all_candidates.append(
                sc[
                    [
                        "source",
                        "field",
                        "model",
                        "label",
                        "rate_model",
                        "rate_rest",
                        "n_tasks",
                        "per_job_rates",
                        "survives_shared_tasks",
                        "verdict_rule",
                        "W",
                        "lift",
                    ]
                ]
            )

    ami_all = {}
    if "topics" in stages:
        print("topics stage...")
        for field in ["reasoning", "content"]:
            topics_df, ami = topic_stage(units, field)
            topics_df.to_csv(args.out / f"topics_{field}.csv", index=False)
            ami_all[f"topics_{field}"] = ami
            if not topics_df.empty:
                cands = topics_df[topics_df.is_candidate].copy()
                if not cands.empty:
                    cands["source"] = "topic"
                    cands["label"] = cands["top_terms"]
                    cands["rate_model"] = cands["lift"]
                    cands["rate_rest"] = np.nan
                    cands["per_job_rates"] = None
                    cands["survives_shared_tasks"] = "n/a"
                    cands["model"] = cands["dominant_model"]
                    cands["W"] = cands["lift"]
                    cands["lift"] = cands["lift"].clip(upper=LIFT_CEILING)
                    all_candidates.append(
                        cands[
                            [
                                "source",
                                "field",
                                "model",
                                "label",
                                "rate_model",
                                "rate_rest",
                                "n_tasks",
                                "per_job_rates",
                                "survives_shared_tasks",
                                "verdict_rule",
                                "W",
                                "lift",
                            ]
                        ]
                    )

    if "embed" in stages:
        print("embedding stage...")
        for field in ["reasoning", "content"]:
            clusters_df, ami = embedding_stage(units, field, args.out, device)
            clusters_df.to_csv(args.out / f"clusters_{field}.csv", index=False)
            ami_all[f"embed_{field}"] = ami
            if not clusters_df.empty:
                cands = clusters_df[clusters_df.is_candidate].copy()
                if not cands.empty:
                    cands["source"] = "cluster"
                    cands["label"] = "cluster " + cands["cluster"].astype(str)
                    cands["rate_model"] = cands["lift"]
                    cands["rate_rest"] = np.nan
                    cands["per_job_rates"] = None
                    cands["survives_shared_tasks"] = "n/a"
                    cands["model"] = cands["dominant_model"]
                    cands["W"] = cands["lift"]
                    cands["lift"] = cands["lift"].clip(upper=LIFT_CEILING)
                    all_candidates.append(
                        cands[
                            [
                                "source",
                                "field",
                                "model",
                                "label",
                                "rate_model",
                                "rate_rest",
                                "n_tasks",
                                "per_job_rates",
                                "survives_shared_tasks",
                                "verdict_rule",
                                "W",
                                "lift",
                            ]
                        ]
                    )

    (args.out / "ami.json").write_text(json.dumps(ami_all, indent=2))

    classifier_results = {}
    if "classify" in stages and "embed" in stages:
        print("classifier stage...")
        classifier_results = classifier_stage(
            args.out / "units_reasoning.parquet", args.out / "units_content.parquet"
        )
    (args.out / "classifier.json").write_text(json.dumps(classifier_results, indent=2))

    if all_candidates:
        candidates = pd.concat(all_candidates, ignore_index=True)
        candidates.insert(0, "candidate_id", [f"c{i:04d}" for i in range(len(candidates))])
        candidates["verdict_claude"] = ""
        candidates["note"] = ""
        candidates = candidates.sort_values("lift", ascending=False)
        candidates.to_csv(args.out / "candidates.csv", index=False)

        examples_dir = args.out / "examples"
        examples_dir.mkdir(exist_ok=True)
        source_texts: dict[tuple, list[str]] = defaultdict(list)
        for _, r in units.iterrows():
            source_texts[(r["row"], r["msg_idx"], r["field"])].append(r["text"])
        # top 5 per model across all sources combined, capped at 35 overall (spec section 7)
        top_per_model_source = (
            candidates.sort_values("lift", ascending=False).groupby("model").head(5).head(35)
        )
        for _, cand in top_per_model_source.iterrows():
            field = cand["field"]
            model = cand["model"]
            label = str(cand["label"])
            if cand["source"] == "term":
                term_words = label.split(" ")
                mask = (units.model == model) & (units.field == field)
                contains_term = units["text_masked"].apply(
                    lambda t, words=term_words: all(w in t for w in words)
                )
                hits = units[mask & contains_term]
            else:
                mask = (units.model == model) & (units.field == field)
                hits = units[mask]
            sample = hits.head(5)
            examples = [
                {"row": r["row"], "msg_idx": r["msg_idx"], "field": r["field"], "excerpt": r["text"][:300]}
                for _, r in sample.iterrows()
            ]
            if examples:
                write_examples(cand["candidate_id"], examples, examples_dir, source_texts)
    else:
        pd.DataFrame(
            columns=[
                "candidate_id",
                "source",
                "field",
                "model",
                "label",
                "rate_model",
                "rate_rest",
                "n_tasks",
                "per_job_rates",
                "survives_shared_tasks",
                "verdict_rule",
                "verdict_claude",
                "note",
            ]
        ).to_csv(args.out / "candidates.csv", index=False)

    print(f"done. outputs in {args.out}")


if __name__ == "__main__":
    main()
