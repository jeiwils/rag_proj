from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

from .utils import (
    ensure_output_path,
    get_result_dirs,
    load_json,
    stylized_subplots,
)


Record = Dict[str, float | str]


def _load_summary(summary_path: Path) -> Dict[str, float | str]:
    data = load_json(summary_path)
    if data.get("split") != "dev":
        return {}
    dense_eval = data.get("dense_eval", data)
    f1 = float(dense_eval.get("F1", dense_eval.get("f1", 0.0)))
    reader_tokens = float(
        dense_eval.get("reader_total_tokens", dense_eval.get("reader_tokens_total", 0.0))
    )
    num_queries = float(dense_eval.get("num_queries", data.get("num_queries", 0.0)))
    apt = reader_tokens / num_queries if num_queries else 0.0
    wall_time = float(dense_eval.get("reader_wall_time_mean_sec", 0.0))
    return {
        "dataset": data.get("dataset"),
        "model": data.get("model"),
        "retriever": data.get("retriever", ""),
        "f1": f1,
        "apt": apt,
        "wall_time": wall_time,
    }


def gather_records(result_dirs: Iterable[Path]) -> List[Record]:
    records: Dict[Tuple[str, str], List[Record]] = {}
    for variant in result_dirs:
        summary_files = list(variant.glob("summary_metrics_*.json"))
        if not summary_files:
            continue
        summary = _load_summary(summary_files[0])
        if not summary:
            continue
        key = (summary["dataset"], summary["model"])
        records.setdefault(key, []).append(summary)
    aggregated: List[Record] = []
    for (dataset, model), recs in records.items():
        n = len(recs)
        f1 = sum(r["f1"] for r in recs) / n
        apt = sum(r["apt"] for r in recs) / n
        wall_time = sum(r["wall_time"] for r in recs) / n
        retriever = recs[0]["retriever"]
        aggregated.append(
            {
                "dataset": dataset,
                "model": model,
                "retriever": retriever,
                "f1": f1,
                "apt": apt,
                "wall_time": wall_time,
            }
        )
    return aggregated


def pareto_frontier(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: List[Tuple[float, float]] = []
    best_y = float("-inf")
    for x, y in sorted_pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def plot(records: List[Record], metric: str, out_file: Path) -> None:
    fig, ax = stylized_subplots(figsize=(6, 4))
    x_vals = [r[metric] for r in records]
    y_vals = [r["f1"] for r in records]
    ax.scatter(x_vals, y_vals, c="tab:blue")
    for r in records:
        ax.annotate(
            f"{r['dataset']}\n{r['model']}",
            (r[metric], r["f1"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )
    frontier = pareto_frontier(zip(x_vals, y_vals))
    if frontier:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, color="black", linestyle="--", label="Pareto frontier")
    dense_f1s = [r["f1"] for r in records if r.get("retriever") == "dense"]
    if dense_f1s:
        best_dense = max(dense_f1s)
        ax.axhspan(best_dense - 1, best_dense + 1, color="gray", alpha=0.2, label="±1 F1 of best Dense")
    ax.set_xlabel("Tokens/query" if metric == "apt" else "Wall-time/query (s)")
    ax.set_ylabel("Answer F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ensure_output_path(out_file))
    plt.close(fig)


def main() -> None:
    result_dirs = get_result_dirs()
    records = gather_records(result_dirs)
    if not records:
        return
    out_dir = Path("graphs/accuracy_efficiency")
    plot(records, "apt", out_dir / "apt_vs_f1.png")
    plot(records, "wall_time", out_dir / "wall_time_vs_f1.png")


if __name__ == "__main__":
    main()