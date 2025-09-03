from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_output_path, load_json, stylized_subplots


def _plot_metric_distributions(per_seed: Dict[str, Dict], std_dev: Dict[str, float], out_dir: Path) -> None:
    """Create bar and box plots for metrics across seeds."""
    metric_names = list(std_dev.keys())
    if not metric_names:
        metric_names = sorted({m for seed in per_seed.values() for m in seed.keys()})

    means: List[float] = []
    stds: List[float] = []
    values_by_metric: List[List[float]] = []
    for metric in metric_names:
        vals = [metrics.get(metric) for metrics in per_seed.values() if metrics.get(metric) is not None]
        if vals:
            arr = np.array(vals, dtype=float)
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
            values_by_metric.append(vals)
        else:
            means.append(float("nan"))
            stds.append(float("nan"))
            values_by_metric.append([])

    fig, ax = stylized_subplots(figsize=(10, 6))
    ax.bar(metric_names, means, yerr=stds, capsize=5)
    ax.set_ylabel("Mean value")
    ax.set_title("Per-seed means with standard deviation error bars")
    fig.tight_layout()
    fig.savefig(ensure_output_path(out_dir / "metrics_errorbars.png"))
    plt.close(fig)

    fig, ax = stylized_subplots(figsize=(10, 6))
    ax.boxplot(values_by_metric, labels=metric_names)
    ax.set_ylabel("Value")
    ax.set_title("Per-seed metric distribution")
    fig.tight_layout()
    fig.savefig(ensure_output_path(out_dir / "metrics_boxplot.png"))
    plt.close(fig)


def _plot_wilcoxon_heatmap(
    wilcoxon: Dict[str, Dict[str, float]],
    seeds: List[int],
    metric_key: str,
    out_file: Path,
) -> None:
    """Plot a heatmap of Wilcoxon p-values for the given metric."""
    n = len(seeds)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        matrix[i, i] = 0.0

    for pair, results in wilcoxon.items():
        if "_vs_" not in pair:
            continue
        a_str, b_str = pair.split("_vs_", 1)
        try:
            a = int(a_str)
            b = int(b_str)
        except ValueError:
            continue
        p_val = results.get(metric_key)
        if p_val is None:
            continue
        i = seeds.index(a)
        j = seeds.index(b)
        matrix[i, j] = p_val
        matrix[j, i] = p_val

    fig, ax = stylized_subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(n), [str(s) for s in seeds])
    ax.set_yticks(range(n), [str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Seed")
    ax.set_title(f"Wilcoxon p-values for {metric_key.split('_')[0]}")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2g}", ha="center", va="center", color=text_color)

    fig.colorbar(im, ax=ax, label="p-value")
    fig.tight_layout()
    fig.savefig(ensure_output_path(out_file))
    plt.close(fig)


def process_seed_variance(path: Path) -> None:
    data = load_json(path)
    per_seed = data.get("per_seed", {})
    std_dev = data.get("std_dev", {})
    wilcoxon = data.get("wilcoxon", {})

    out_dir = path.parent / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if per_seed:
        _plot_metric_distributions(per_seed, std_dev, out_dir)

    if wilcoxon:
        seeds = sorted(int(s) for s in per_seed.keys())
        for key in ("EM_p_value", "F1_p_value"):
            if any(key in v for v in wilcoxon.values()):
                _plot_wilcoxon_heatmap(wilcoxon, seeds, key, out_dir / f"wilcoxon_{key.split('_')[0].lower()}.png")


def main(base_dir: str = "data/results") -> None:
    base = Path(base_dir)
    paths = list(base.rglob("seed_variance.json"))
    if not paths:
        print(f"No seed_variance.json files found under {base}")
        return

    for path in paths:
        process_seed_variance(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from seed_stats variance outputs.",
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="data/results",
        help="Base directory containing results",
    )
    args = parser.parse_args()
    main(args.base_dir)

__all__ = ["main", "process_seed_variance"]