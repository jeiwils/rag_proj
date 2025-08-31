"""Simple evaluation utility with Wilcoxon signed-rank test.

This script expects two files containing numeric metric values, one per line,
for two systems or runs. It loads the paired metrics, computes basic summary
statistics, and applies the Wilcoxon signed-rank test to measure whether the
difference between systems is statistically significant.

Example
-------
python tests/metrics.py run_a.txt run_b.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load_metrics(path: str | Path) -> np.ndarray:
    """Load metric values from a text file into a NumPy array."""
    with open(path, "r", encoding="utf-8") as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return np.array(values, dtype=float)


def compare_runs(run_a: str | Path, run_b: str | Path) -> dict:
    """Compare two runs and return evaluation summary with p-value."""
    metrics_a = load_metrics(run_a)
    metrics_b = load_metrics(run_b)

    if metrics_a.shape != metrics_b.shape:
        raise ValueError("Metric arrays must have the same length for a paired test")

    stat, p_value = wilcoxon(metrics_a, metrics_b)

    summary = {
        "run_a_mean": float(metrics_a.mean()),
        "run_b_mean": float(metrics_b.mean()),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p_value),
    }

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare runs using Wilcoxon signed-rank test")
    parser.add_argument("run_a", help="Path to metrics file for run A")
    parser.add_argument("run_b", help="Path to metrics file for run B")
    args = parser.parse_args()

    compare_runs(args.run_a, args.run_b)