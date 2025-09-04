from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from .utils import load_json, ensure_output_path, stylized_subplots
from .seed_stats_plot import _compute_statistics, _plot_metric_distributions, _plot_wilcoxon_heatmap


def plot_mean_with_error(metric: str, stats: dict, out_file: Path) -> None:
    """Plot the per-seed metric values with overall mean and standard deviation.

    Parameters
    ----------
    metric:
        Name of the metric to plot, e.g. ``"EM"`` or ``"F1"``.
    stats:
        Statistics dictionary containing ``per_seed``, ``mean``, and ``std_dev``.
    out_file:
        Path where the figure should be saved.
    """
    per_seed = stats.get("per_seed", {})
    if not per_seed:
        print(f"No per-seed data available for metric {metric}")
        return

    seeds = np.array(sorted(int(s) for s in per_seed.keys()))
    values = [per_seed[str(s)].get(metric) for s in seeds]
    mean_val = stats.get("mean", {}).get(metric)
    std_val = stats.get("std_dev", {}).get(metric)

    fig, ax = stylized_subplots(figsize=(8, 5))
    ax.plot(seeds, values, "o", label="per-seed")
    if mean_val is not None:
        ax.plot(seeds, [mean_val] * len(seeds), label="mean")
        if std_val is not None:
            ax.fill_between(
                seeds,
                mean_val - std_val,
                mean_val + std_val,
                alpha=0.2,
                label="+/-1 std dev",
            )

    ax.set_xlabel("Seed")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_output_path(out_file))
    plt.close(fig)


def main(results_dir: str = "data/results", metrics: list[str] | None = None) -> None:
    """Generate mean-with-error plots and Wilcoxon heatmaps for seed experiments.

    Parameters
    ----------
    results_dir:
        Directory containing seed-specific result subdirectories or summaries.
    metrics:
        List of metric names to plot. Defaults to all metrics with computed means.
    """
    base = Path(results_dir)
    stats_path = base / "seed_variance.json"
    if stats_path.exists():
        stats = load_json(stats_path)
    else:
        stats = _compute_statistics(base)

    if not stats:
        print(f"No seed summaries found in {base}")
        return

    per_seed = stats.get("per_seed", {})
    if not per_seed:
        print(f"No per-seed metrics available in {base}")
        return

    mean = stats.get("mean", {})
    std_dev = stats.get("std_dev", {})
    available_metrics = mean.keys()
    metrics = metrics or list(available_metrics)
    out_dir = base / "graphs"
    _plot_metric_distributions(per_seed, mean, std_dev, out_dir)

    wilcoxon = stats.get("wilcoxon", {})
    if wilcoxon:
        seeds = sorted(int(s) for s in per_seed.keys())
        for key in ("EM_p_value", "F1_p_value"):
            if any(key in v for v in wilcoxon.values()):
                _plot_wilcoxon_heatmap(
                    wilcoxon,
                    seeds,
                    key,
                    out_dir / f"wilcoxon_{key.split('_')[0].lower()}.png",
                )

    for metric in metrics:
        out_file = out_dir / f"mean_{metric}.png"
        plot_mean_with_error(metric, stats, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot seed metrics with standard deviation bands and Wilcoxon heatmaps."
    )
    parser.add_argument("model")
    parser.add_argument("dataset")
    parser.add_argument("split")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=None,
        help="Metric to plot (can be specified multiple times)",
    )
    args = parser.parse_args()
    dir_path = Path("data/results") / args.model / args.dataset / args.split
    main(dir_path, metrics=args.metrics)

__all__ = ["main", "plot_mean_with_error"]