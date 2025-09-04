from __future__ import annotations

"""Compare dense and MoE model metrics across datasets.

This script aggregates EM/F1 and compute metrics for ``dense`` retrieval runs
produced by dense and MoE models. Metrics are grouped by dataset and visualised
with one subplot per dataset. Individual panels are also saved under
``graphs/<dataset>/`` mirroring other output conventions.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    ensure_output_path,
    get_result_dirs,
    load_json,
    parse_traversal_run_dir,
    rag_run_paths,
    stylized_subplots,
    load_token_usage,
)

logger = logging.getLogger(__name__)

# Map model directory names to short labels used in plots.
MODEL_LABELS = {
    "qwen2.5-14b-instruct": "Dense",
    "qwen2.5-2x7b-moe-power-coder-v4": "MoE",
}

# Accuracy and compute metrics we wish to plot.
ACCURACY_METRICS = ["EM", "F1"]
COMPUTE_METRICS = ["tokens_total", "t_total_ms"]


def _load_run_metrics(result_dir: Path) -> Dict[str, float]:
    """Load EM/F1 and compute metrics for a single run directory."""
    model, dataset, split, seed = parse_traversal_run_dir(result_dir)
    mode, _ = result_dir.name.rsplit("_seed", 1)
    paths = rag_run_paths(model, dataset, split, seed, mode)["answers"]

    summary = load_json(paths["summary"]) if paths["summary"].exists() else {}
    usage = load_token_usage(paths["token_usage"]).get("global", {})

    return {
        "EM": float(summary.get("EM") or summary.get("em") or 0.0),
        "F1": float(summary.get("F1") or summary.get("f1") or 0.0),
        "tokens_total": float(usage.get("tokens_total") or 0.0),
        "t_total_ms": float(usage.get("t_total_ms") or 0.0),
    }


def collect_metrics() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate metrics across seeds for dense and MoE models."""
    result_dirs = get_result_dirs(required="summary_metrics")
    per_dataset: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for rdir in result_dirs:
        # Only look at dense retrieval runs for consistent comparison.
        if not rdir.name.startswith("dense_seed"):
            continue
        model_name = rdir.parts[-4]
        if model_name not in MODEL_LABELS:
            continue
        dataset = rdir.parts[-3]
        label = MODEL_LABELS[model_name]
        metrics = _load_run_metrics(rdir)
        per_dataset.setdefault(dataset, {}).setdefault(label, []).append(metrics)

    averages: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset, model_runs in per_dataset.items():
        averages[dataset] = {}
        for label, runs in model_runs.items():
            avg = {
                key: float(np.mean([r[key] for r in runs]))
                for key in runs[0].keys()
            }
            averages[dataset][label] = avg
    return averages


def _plot_group(
    averages: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str],
    out_name: str,
) -> None:
    """Plot a group of metrics with one subplot per dataset."""
    datasets = sorted(averages.keys())
    fig, axes = stylized_subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        model_vals = averages[dataset]
        x = np.arange(len(metrics))
        width = 0.35
        for i, (model_label, vals) in enumerate(model_vals.items()):
            y = [vals[m] for m in metrics]
            ax.bar(x + (i - len(model_vals) / 2) * width, y, width=width, label=model_label)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_title(dataset)
        if metrics == ACCURACY_METRICS:
            ax.set_ylim(0.0, 1.0)
        ax.legend()

    fig.tight_layout()
    overall_path = ensure_output_path(Path("graphs") / f"{out_name}.png")
    fig.savefig(overall_path)

    # Save individual dataset panels.
    for ax, dataset in zip(axes, datasets):
        panel_path = ensure_output_path(Path("graphs") / dataset / f"{out_name}.png")
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(panel_path, bbox_inches=extent.expanded(1.1, 1.2))

    plt.close(fig)


def main() -> None:
    averages = collect_metrics()
    if not averages:
        logger.warning("No metrics found for plotting")
        return

    _plot_group(averages, ACCURACY_METRICS, "accuracy")
    _plot_group(averages, COMPUTE_METRICS, "compute")


if __name__ == "__main__":
    main()