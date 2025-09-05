from __future__ import annotations

"""Compute per-query latency statistics and CDF plots without external libs.

The script traverses ``data/results`` to collect per-query latencies from
``token_usage.json`` files. For each dataset and model combination the
median and 90th percentile wall-times are computed. CDF plots are
rendered as SVG files with one line per model. Outputs are written under
``graphs/latency``.
"""

import argparse
import json
import math
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

def ensure_output_path(path: Path) -> Path:
    """Create parent directories for ``path`` and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data collection helpers


def _gather_token_usage_files(results_dir: Path) -> Iterable[Path]:
    """Yield all ``token_usage.json`` paths under ``results_dir``."""
    return results_dir.glob("*/**/token_usage.json")


def _load_latencies(path: Path) -> Iterable[float]:
    """Load per-query latencies from ``token_usage.json``."""
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    for entry in data.get("per_query_reader", {}).values():
        latency = entry.get("query_latency_ms")
        if latency is not None:
            yield float(latency)


def collect_latencies(results_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Return nested mapping ``dataset -> model -> [latencies]``."""
    latencies: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for tok_path in _gather_token_usage_files(results_dir):
        try:
            model, dataset, split, variant, _ = tok_path.relative_to(results_dir).parts
        except ValueError:
            continue
        for latency in _load_latencies(tok_path):
            latencies[dataset][model].append(latency)
    return latencies


# ---------------------------------------------------------------------------
# Statistics and table generation


def _percentile(sorted_vals: List[float], frac: float) -> float:
    """Return the desired percentile from ``sorted_vals`` (0 < frac <= 1)."""
    if not sorted_vals:
        return float("nan")
    k = max(int(math.ceil(frac * len(sorted_vals))) - 1, 0)
    return sorted_vals[k]


def compute_summary(latencies: Dict[str, Dict[str, List[float]]]) -> List[Tuple[str, str, float, float]]:
    """Compute median and 90th percentile for each dataset×model."""
    rows: List[Tuple[str, str, float, float]] = []
    for dataset, models in latencies.items():
        for model, vals in models.items():
            if not vals:
                continue
            vals_sorted = sorted(vals)
            med = vals_sorted[len(vals_sorted) // 2] if len(vals_sorted) % 2 == 1 else (
                vals_sorted[len(vals_sorted) // 2 - 1] + vals_sorted[len(vals_sorted) // 2]
            ) / 2
            p90 = _percentile(vals_sorted, 0.9)
            rows.append((dataset, model, med, p90))
    rows.sort()
    return rows


def save_table(rows: List[Tuple[str, str, float, float]], out_dir: Path) -> None:
    """Write latency summary to CSV and Markdown files."""
    header = ["dataset", "model", "median_latency_ms", "p90_latency_ms"]
    csv_path = ensure_output_path(out_dir / "latency_summary.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}"])

    md_path = ensure_output_path(out_dir / "latency_summary.md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| dataset | model | median_latency_ms | p90_latency_ms |\n")
        f.write("|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} |\n")


# ---------------------------------------------------------------------------
# SVG plotting


def _color_palette() -> Iterable[str]:
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    while True:
        for c in colors:
            yield c


def plot_cdfs(latencies: Dict[str, Dict[str, List[float]]], out_dir: Path) -> None:
    """Render simple SVG CDF plots for each dataset."""
    color_iter = _color_palette()
    model_colors: Dict[str, str] = {}
    for dataset, models in latencies.items():
        width, height = 900, 600
        margin = 50
        max_latency = max(max(v) for v in models.values())
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
            f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="14">Latency (ms)</text>',
            f'<text x="15" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 15 {height/2})">CDF</text>',
        ]
        legend_y = margin
        for model, vals in models.items():
            color = model_colors.setdefault(model, next(color_iter))
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            points = []
            for i, latency in enumerate(vals_sorted, 1):
                x = margin + (latency / max_latency) * (width - 2 * margin)
                y = (height - margin) - (i / n) * (height - 2 * margin)
                points.append(f"{x},{y}")
            points_str = " ".join(points)
            svg.append(f'<polyline fill="none" stroke="{color}" points="{points_str}"/>')
            svg.append(f'<rect x="{width - margin + 5}" y="{legend_y - 10}" width="20" height="10" fill="{color}"/>')
            svg.append(
                f'<text x="{width - margin + 30}" y="{legend_y}" font-size="12" alignment-baseline="middle">{model}</text>'
            )
            legend_y += 15
        svg.append('</svg>')
        out_file = ensure_output_path(out_dir / f"{dataset}_latency_cdf.svg")
        with out_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(svg))


# ---------------------------------------------------------------------------
# Entry point


def main(results_dir: Path, out_dir: Path) -> None:
    latencies = collect_latencies(results_dir)
    if not latencies:
        raise SystemExit("No latency data found under results directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = compute_summary(latencies)
    save_table(summary, out_dir)
    plot_cdfs(latencies, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("data/results"), help="Results directory"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("graphs/latency"), help="Output directory"
    )
    args = parser.parse_args()
    main(args.results, args.out)