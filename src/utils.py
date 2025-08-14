"""Common utility functions shared across pipeline stages."""

from __future__ import annotations

import gzip
import json
import os
import re
import unicodedata
from typing import Dict, Iterator, List, Optional

import tempfile
import shutil

# ---------------------------------------------------------------------------
# JSONL and general file I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> Iterator[Dict]:
    """Yield objects from a JSONL file one by one."""
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def save_jsonl(path: str, data: List[Dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "wt", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _append_jsonl_gz(path: str, obj: Dict) -> None:
    """Append ``obj`` to a gzipped JSONL file via decompress → append → recompress."""
    fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        if os.path.exists(path):
            with gzip.open(path, "rt", encoding="utf-8") as gz, open(tmp_path, "wt", encoding="utf-8") as tmp:
                shutil.copyfileobj(gz, tmp)
        with open(tmp_path, "at", encoding="utf-8") as tmp:
            tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        with open(tmp_path, "rt", encoding="utf-8") as tmp, gzip.open(path, "wt", encoding="utf-8") as gz:
            shutil.copyfileobj(tmp, gz)
    finally:
        os.remove(tmp_path)


def append_jsonl(path: str, obj: Dict) -> None:
    """Append a single JSON serialisable object to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".gz"):
        _append_jsonl_gz(path, obj)
    else:
        with open(path, "at", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _next_version_path(path: str) -> str:
    """If ``path`` exists, return ``path`` with an incremented ``.vN`` suffix."""
    base, ext = os.path.splitext(path)
    i = 1
    candidate = f"{base}.v{i}{ext}"
    while os.path.exists(candidate):
        i += 1
        candidate = f"{base}.v{i}{ext}"
    return candidate

def save_jsonl_safely(path: str, data: List[Dict], overwrite: bool = False) -> str:
    """Write JSONL data, versioning the filename if it already exists."""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path or ".", exist_ok=True)
    out_path = path
    if os.path.exists(path) and not overwrite:
        out_path = _next_version_path(path)
    open_fn = gzip.open if out_path.endswith(".gz") else open
    with open_fn(out_path, "wt", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out_path

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise whitespace and remove simple markup for clean text."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"={2,}.*?={2,}", "", text)
    text = unicodedata.normalize("NFKC", text)
    return text

# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------

def pid_plus_title(qid: str, title: str, sent_idx: int) -> str:
    """Create a safe passage identifier using question id and title."""
    if not title:
        safe = "no_title"
    else:
        safe = title.lower()
        safe = re.sub(r"[^a-z0-9]+", "_", safe)
        safe = re.sub(r"_+", "_", safe).strip("_") or "no_title"
    return f"{qid}__{safe}_sent{sent_idx}"

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Maps variant → preferred folder names to search (first match wins).
FOLDERS_BY_VARIANT: Dict[str, List[str]] = {
    "baseline": ["baseline_hoprag"],
    "enhanced": ["enhanced_hoprag"],
}

def resolve_root(model: str, dataset: str, split: str, variant: str) -> Optional[str]:
    """Resolve the root directory containing inputs/outputs for a job.

    Searches under ``data/models/{model}/{dataset}/{split}`` for the first
    directory listed in ``FOLDERS_BY_VARIANT[variant]`` that exists and returns
    its path. If none are found, ``None`` is returned.
    """
    for hoprag_version in FOLDERS_BY_VARIANT[variant]:
        root = f"data/models/{model}/{dataset}/{split}/{hoprag_version}"
        if os.path.isdir(root):
            return root
    return None









from pathlib import Path

def get_result_paths(model, dataset, split, variant):
    base = Path(f"results/{model}/{dataset}/{split}/{variant}")
    return {
        "base": base,
        "answers": base / f"answer_per_query_{variant}_{split}.jsonl.gz",
        "summary": base / f"summary_metrics_{variant}_{split}.json",
    }

def get_traversal_paths(model, dataset, split, variant):
    base = Path(f"data/graphs/{model}/{dataset}/{split}/{variant}/traversal")
    return {
        "base": base,
        "results": base / f"traversal_per_query_{variant}_{split}.jsonl.gz",
        "visited_passages": base / f"all_visited_passages_{variant}_{split}.json.gz",
        "stats": base / f"summary_metrics_{variant}_{split}.json",
    }







__all__ = [
    "load_jsonl",
    "save_jsonl",
    "append_jsonl",
    "save_jsonl_safely",
    "clean_text",
    "pid_plus_title",
    "FOLDERS_BY_VARIANT",
    "resolve_root",
]