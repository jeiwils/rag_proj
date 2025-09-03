from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PLOT_DIR = Path("analysis/plots")

def load_json(path: Path) -> dict:
    """Load a JSON file and return its contents."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSON Lines file into a list of objects."""
    data: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def stylized_subplots(*args, **kwargs):
    """Return ``plt.subplots`` with a consistent style applied."""
    plt.style.use("ggplot")
    return plt.subplots(*args, **kwargs)

def ensure_output_path(path: Path) -> Path:
    """Create parent directories for ``path`` and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

__all__ = [
    "DEFAULT_PLOT_DIR",
    "load_json",
    "load_jsonl",
    "stylized_subplots",
    "ensure_output_path",
]