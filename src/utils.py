"""Common utility functions shared across pipeline stages."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# JSONL and general file I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path: str, data: List[Dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def append_jsonl(path: str, obj: Dict) -> None:
    """Append a single JSON serialisable object to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out_path = path
    if os.path.exists(path) and not overwrite:
        out_path = _next_version_path(path)
    with open(out_path, "w", encoding="utf-8") as f:
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


def resolve_repr_root(model: str, dataset: str, split: str, variant: str) -> str:
    """Return the root directory for model representations.

    The directory ``data/representations/{model}/{dataset}/{split}/{variant}``
    is created if it does not already exist and the path is returned.
    """
    root = os.path.join(
        "data",
        "representations",
        "models",
        model,
        dataset,
        split,
        variant,
    )
    os.makedirs(root, exist_ok=True)
    return root



__all__ = [
    "load_jsonl",
    "save_jsonl",
    "append_jsonl",
    "save_jsonl_safely",
    "clean_text",
    "pid_plus_title",
    "FOLDERS_BY_VARIANT",
    "resolve_root",
    "resolve_repr_root",
]