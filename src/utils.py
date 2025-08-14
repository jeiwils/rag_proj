
"""Common utility functions shared across pipeline stages."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Dict, Iterator, List, Optional

import logging


# ---------------------------------------------------------------------------
# JSONL and general file I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str, log_skipped: bool = False) -> Iterator[Dict]:
    """Yield objects from a JSONL file one by one.

    Lines that are empty or fail to parse as JSON are skipped. If
    ``log_skipped`` is ``True``, the number of skipped lines is logged at
    debug level.
    """
    skipped = 0
    with open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if log_skipped:
                    logging.debug("Skipping malformed JSON on line %d in %s", line_no, path)
    if log_skipped and skipped:
        logging.debug("Skipped %d empty or malformed lines in %s", skipped, path)

def save_jsonl(path: str, data: List[Dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "wt", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: str, obj: Dict) -> None:
    """Append a single JSON serialisable object to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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
    with open(out_path, "wt", encoding="utf-8") as f:
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

def resolve_root(model: str, dataset: str, split: str, variant: str) -> Optional:
    """
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
        "answers": base / f"answer_per_query_{variant}_{split}.jsonl",
        "summary": base / f"summary_metrics_{variant}_{split}.json",
    }

def get_traversal_paths(model, dataset, split, variant):
    base = Path(f"data/graphs/{model}/{dataset}/{split}/{variant}/traversal")
    return {
        "base": base,
        "results": base / f"traversal_per_query_{variant}_{split}.jsonl",
        "visited_passages": base / f"all_visited_passages_{variant}_{split}.json",
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