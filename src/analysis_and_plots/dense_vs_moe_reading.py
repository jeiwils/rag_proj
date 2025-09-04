from __future__ import annotations

"""Bar plot comparing Hop-RAG runs with reading-only ablations.

The script scans ``data/results`` for run directories and aggregates the
requested metric across seeds. For each model we expect to find two variants:
``dense_seed*`` (full traversal + reading) and ``baseline_seed*`` (reading
only). Mean scores are plotted with 95% confidence interval error bars. Bars are
colour-coded by model and grouped so that ablations share the same colour with
reduced opacity. Additional reference bars for a baseline and an upper bound can
be supplied and the resulting bars are sorted by the Hop-RAG score.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    ensure_output_path,
    get_result_dirs,
    load_json,
    parse_traversal_run_dir,
    rag_run_paths,
    stylized_subplots,
)

logger = logging.getLogger(__name__)

# Map filesystem model names to prettier plot labels.
MODEL_LABELS: Dict[str, str] = {
    "qwen2.5-7b-instruct": "7B Dense",
    "qwen2.5-14b-instruct": "14B Dense",
    "qwen2.5-2x7b-moe-power-coder-v4": "2x7B MoE",
}

# Colours used for each model and the reference bars.
MODEL_COLOURS: Dict[str, str] = {
    "7B Dense": "#1f77b4",
    "14B Dense": "#ff7f0e",
    "2x7B MoE": "#2ca02c",
    "Baseline": "#7f7f7f",
    "Upper bound": "#000000",
}


def _load_metric(result_dir: Path, metric: str) -> float | None:
    """Return the metric value from ``summary_metrics`` in ``result_dir``."""
    model, dataset, split, seed = parse_traversal_run_dir(result_dir)
    mode, _ = result_dir.name.rsplit("_seed", 1)
    paths = rag_run_paths(model, dataset, split, seed, mode)["answers"]
    if not paths["summary"].exists():
        logger.warning("summary file missing for %s", result_dir)
        return None
    summary = load_json(paths["summary"])
    dense_eval = summary.get("dense_eval", summary)
    return dense_eval.get(metric) or dense_eval.get(metric.lower())

def collect_metrics(metric: str) -> Dict[str, Dict[str, List[float]]]:
    """Gather metrics for Hop-RAG and reading-only runs grouped by model."""
    dirs = get_result_dirs(required="summary_metrics")
    data: Dict[str, Dict[str, List[float]]] = {}
    for rdir in dirs:
        if not (rdir.name.startswith("dense_seed") or rdir.name.startswith("baseline_seed")):
            continue
        model, _, _, _ = parse_traversal_run_dir(rdir)
        label = MODEL_LABELS.get(model)
        if label is None:
            continue
        mode = "hop-rag" if rdir.name.startswith("dense_seed") else "reading"
        val = _load_metric(rdir, metric)
        if val is None:
            continue
        data.setdefault(label, {"hop-rag": [], "reading": []})[mode].append(float(val))
    return data


def compute_stats(data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute mean and 95% CI for each model/mode combination."""
    stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for model, modes in data.items():
        stats[model] = {}
        for mode, vals in modes.items():
            if not vals:
                continue
            mean = float(np.mean(vals))
            if len(vals) > 1:
                err = float(np.std(vals, ddof=1) * 1.96 / np.sqrt(len(vals)))
            else:
                err = 0.0
            stats[model][mode] = (mean, err)
    return stats


def plot(
    stats: Dict[str, Dict[str, Tuple[float, float]]],
    metric: str,
    baseline: float,
    upper: float,
    output: Path,
) -> None:
    """Create the comparison bar plot."""
    if not stats:
        logger.warning("No data available for plotting")
        return

    # Sort models by Hop-RAG mean score.
    order = sorted(
        stats.keys(),
        key=lambda m: stats[m].get("hop-rag", (0.0, 0.0))[0],
        reverse=True,
    )

    fig, ax = stylized_subplots(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(order))

    for idx, model in enumerate(order):
        colour = MODEL_COLOURS.get(model, "#333333")
        hop = stats[model].get("hop-rag")
        read = stats[model].get("reading")
        if hop:
            ax.bar(
                x[idx] - width / 2,
                hop[0],
                width,
                yerr=hop[1],
                color=colour,
                capsize=5,
                label=f"{model} Hop-RAG" if idx == 0 else None,
            )
        if read:
            ax.bar(
                x[idx] + width / 2,
                read[0],
                width,
                yerr=read[1],
                color=colour,
                alpha=0.5,
                capsize=5,
                label=f"{model} Reading" if idx == 0 else None,
            )

    offset = len(order)
    ax.bar(
        offset,
        baseline,
        width,
        color=MODEL_COLOURS["Baseline"],
        capsize=5,
        label="Baseline",
    )
    ax.bar(
        offset + 1,
        upper,
        width,
        color=MODEL_COLOURS["Upper bound"],
        capsize=5,
        label="Upper bound",
    )

    xticks = list(x) + [offset, offset + 1]
    labels = order + ["Baseline", "Upper bound"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_output_path(output))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="F1", help="Metric key to plot (default: F1)")
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.0,
        help="Reference baseline value to plot",
    )
    parser.add_argument(
        "--upper-bound",
        type=float,
        default=1.0,
        help="Reference upper bound value to plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/plots/hop_rag_vs_reading.png"),
        help="Output plot file",
    )
    args = parser.parse_args()

    data = collect_metrics(args.metric)
    stats = compute_stats(data)
    plot(stats, args.metric, args.baseline, args.upper_bound, args.output)


if __name__ == "__main__":
    main()