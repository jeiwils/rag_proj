from __future__ import annotations

"""Scatter plot of Path F1 vs Path EM for dense and MoE models."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_output_path


def _parse_mean(value: str) -> float:
    """Extract the mean component from a "mean ± std" string."""
    return float(value.split("±")[0].strip())


def load_records(table_path: Path):
    """Load aggregated metrics and return plotting records."""
    records = []
    with table_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "model_type": "MoE" if "moe" in row["model"].lower() else "Dense",
                    "path_f1": _parse_mean(row["Path F1"]),
                    "path_em": _parse_mean(row["Path EM"]),
                }
            )
    return records


def plot(records, out_path: Path) -> None:
    """Create the scatter plot and write it to ``out_path``."""
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(6, 4))

    markers = {"Dense": "o", "MoE": "s"}
    colors = {"Dense": "tab:blue", "MoE": "tab:orange"}

    for mtype in ("Dense", "MoE"):
        xs = [r["path_f1"] for r in records if r["model_type"] == mtype]
        ys = [r["path_em"] for r in records if r["model_type"] == mtype]
        ax.scatter(xs, ys, label=mtype, marker=markers[mtype], color=colors[mtype])

    for r in records:
        ax.annotate(r["dataset"], (r["path_f1"], r["path_em"]), textcoords="offset points", xytext=(4,4), fontsize=8)

    ax.set_xlabel("Path F1")
    ax.set_ylabel("Path EM")
    ax.legend(title="Model type")
    fig.tight_layout()

    ensure_output_path(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    table_path = Path("analysis_outputs/combined_metrics_table.csv")
    records = load_records(table_path)
    out_dir = Path("graphs/coverage_provenance")
    out_file = out_dir / "path_f1_vs_path_em.png"
    plot(records, out_file)


if __name__ == "__main__":
    main()