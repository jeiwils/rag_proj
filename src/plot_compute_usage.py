"""Plot token and time usage metrics across models.

This module loads ``token_usage.json`` files for one or more model result
directories and produces comparative bar charts for key compute statistics:

* ``tokens_total``
* ``t_total_ms``
* ``tps_overall``
* ``trav_tokens_total``
* ``reader_total_tokens``

Usage mirrors :mod:`plot_traversal_metrics`—call :func:`plot_compute_usage`
with a list of directories ``result_dirs`` each pointing to
``data/results/{model}/{dataset}/{split}/{variant}`` and an ``output`` path for
the generated figure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

USAGE_FIELDS = [
    "tokens_total",
    "t_total_ms",
    "tps_overall",
    "trav_tokens_total",
    "reader_total_tokens",
]


def _validate_keys(metrics: Dict[str, float], expected: Sequence[str]) -> None:
    """Log a warning if any ``expected`` keys are absent in ``metrics``."""
    missing = [k for k in expected if k not in metrics]
    if missing:
        logger.warning("Missing metrics: %s", ", ".join(missing))


def _load_usage(result_dir: Path, fields: Sequence[str]) -> Dict[str, float]:
    """Return compute usage metrics from ``token_usage.json`` in ``result_dir``."""
    usage_path = result_dir / "token_usage.json"
    if not usage_path.exists():
        logger.warning("token_usage.json not found in %s", result_dir)
        return {}
    with usage_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _validate_keys(data, fields)
    return data


def _model_label(result_dir: Path) -> str:
    """Infer a short model label from ``result_dir``."""
    parts = result_dir.resolve().parts
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return result_dir.name


def plot_compute_usage(result_dirs: Sequence[Path], output: Path) -> None:
    """Generate bar plots comparing compute usage across models."""
    usage_by_model: Dict[str, Dict[str, float]] = {}

    for rdir in result_dirs:
        metrics = _load_usage(rdir, USAGE_FIELDS)
        label = _model_label(rdir)
        usage_by_model[label] = metrics

    labels = list(usage_by_model.keys())
    fig, axes = plt.subplots(1, len(USAGE_FIELDS), figsize=(5 * len(USAGE_FIELDS), 4))

    for ax, field in zip(axes, USAGE_FIELDS):
        values = [usage_by_model.get(l, {}).get(field, 0.0) for l in labels]
        ax.bar(labels, values)
        ax.set_ylabel(field)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(field.replace("_", " "))

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    RESULT_DIRS = [
        # Path("data/results/model_a/musique/dev/baseline"),
        # Path("data/results/model_b/musique/dev/baseline"),
    ]
    OUTPUT = Path("compute_usage.png")

    if RESULT_DIRS:
        plot_compute_usage(RESULT_DIRS, OUTPUT)