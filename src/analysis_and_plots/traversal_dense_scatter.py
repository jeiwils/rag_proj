"""Plot accuracy vs. efficiency for traversal and dense retrieval runs.

This script reads traversal and dense retrieval result files and creates a
scatter plot comparing mean F1 to query latency for each approach.  Each seed
run contributes one point per approach.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt

from .utils import (
    ensure_output_path,
    load_json,
    traversal_run_paths,
    answer_run_paths,
    stylized_subplots,
)

logger = logging.getLogger(__name__)


def _gather_traversal_points(
    model: str, dataset: str, split: str, seeds: Iterable[int]
) -> List[Tuple[float, float]]:
    """Collect (latency, F1) pairs for traversal runs."""
    points: List[Tuple[float, float]] = []
    for seed in seeds:
        paths = traversal_run_paths(model, dataset, split, seed)
        stats_path = paths["final_stats"]
        if not stats_path.exists():
            logger.warning("Missing traversal stats: %s", stats_path)
            continue
        stats = load_json(stats_path)
        eval_metrics = stats.get("traversal_eval", {})
        f1 = float(eval_metrics.get("mean_f1", 0.0))
        latency = float(stats.get("query_latency_ms", 0.0))
        points.append((latency, f1))
    return points


def _gather_dense_points(
    model: str, dataset: str, split: str, seeds: Iterable[int]
) -> List[Tuple[float, float]]:
    """Collect (latency, F1) pairs for dense retrieval runs."""
    points: List[Tuple[float, float]] = []
    for seed in seeds:
        paths = answer_run_paths(model, dataset, split, "dense", seed)
        summary_path = paths["summary"]
        if not summary_path.exists():
            logger.warning("Missing dense summary: %s", summary_path)
            continue
        summary = load_json(summary_path)
        dense_eval = summary.get("dense_eval", summary)
        f1 = float(dense_eval.get("F1", dense_eval.get("f1", 0.0)))
        latency = float(dense_eval.get("query_latency_ms", 0.0))
        points.append((latency, f1))
    return points


def plot_scatter(
    traversal_pts: Sequence[Tuple[float, float]],
    dense_pts: Sequence[Tuple[float, float]],
    out_file: Path,
) -> None:
    """Generate and save the scatter plot."""
    fig, ax = stylized_subplots(figsize=(6, 4))
    if traversal_pts:
        x_t, y_t = zip(*traversal_pts)
        ax.scatter(x_t, y_t, marker="o", color="tab:blue", label="Traversal")
    if dense_pts:
        x_d, y_d = zip(*dense_pts)
        ax.scatter(x_d, y_d, marker="x", color="tab:orange", label="Dense + Answer")
    ax.set_xlabel("Query latency (ms)")
    ax.set_ylabel("F1 score")
    ax.set_title("Accuracy vs. efficiency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_output_path(out_file))
    plt.close(fig)


def main(model: str, dataset: str, split: str, seeds: Iterable[int]) -> None:
    traversal_pts = _gather_traversal_points(model, dataset, split, seeds)
    dense_pts = _gather_dense_points(model, dataset, split, seeds)
    if not traversal_pts and not dense_pts:
        logger.warning("No points found for plotting")
        return
    out_dir = Path("analysis/plots")
    out_name = f"{model}_{dataset}_{split}_accuracy_vs_efficiency.png"
    plot_scatter(traversal_pts, dense_pts, out_dir / out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scatter plot of traversal vs. dense retrieval accuracy and efficiency."
    )
    parser.add_argument("model")
    parser.add_argument("dataset")
    parser.add_argument("split")
    parser.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        action="append",
        default=None,
        help="Seed to include (can be used multiple times)",
    )
    args = parser.parse_args()
    seeds = args.seeds if args.seeds else [1]
    main(args.model, args.dataset, args.split, seeds)