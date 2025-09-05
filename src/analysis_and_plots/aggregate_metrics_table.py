"""Aggregate answer and traversal metrics across seeds.

This script scans ``data/results`` and ``data/traversal`` for evaluation
summaries, computes mean and standard deviation across seeds for each
(model, dataset) pair and outputs a combined table with Path F1, Path
EM, Answer F1 and Answer EM.  The table is written to CSV, Markdown and
LaTeX formats under ``analysis_outputs``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, Tuple, List


def _summary_files(root: Path, pattern: str) -> Iterable[Path]:
    """Yield files matching ``pattern`` under ``root`` recursively."""
    return root.glob(pattern)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Return ``(mean, std)`` for a list of numbers.

    If fewer than two values are provided the standard deviation is 0.0.
    """
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def collect_metrics() -> List[Dict[str, str]]:
    """Collect metrics from result and traversal directories."""
    data: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    # Answer metrics
    for fp in _summary_files(Path("data/results"), "**/dev/dense_seed*/summary_metrics_dense_seed*_dev.json"):
        with fp.open("r", encoding="utf-8") as f:
            js = json.load(f)
        model = js.get("model", fp.parts[2])
        dataset = js.get("dataset", fp.parts[3])
        key = (model, dataset)
        d = data.setdefault(key, {"answer_em": [], "answer_f1": [], "path_f1": [], "path_em": []})
        dense_eval = js.get("dense_eval", {})
        d["answer_em"].append(float(dense_eval.get("EM", 0.0)))
        d["answer_f1"].append(float(dense_eval.get("F1", 0.0)))

    # Traversal metrics
    for fp in _summary_files(Path("data/traversal"), "**/dev/baseline_seed*/final_traversal_stats.json"):
        with fp.open("r", encoding="utf-8") as f:
            js = json.load(f)
        model = js.get("model", fp.parts[2])
        dataset = js.get("dataset", fp.parts[3])
        key = (model, dataset)
        d = data.setdefault(key, {"answer_em": [], "answer_f1": [], "path_f1": [], "path_em": []})
        trav_eval = js.get("traversal_eval", {})
        d["path_f1"].append(float(trav_eval.get("mean_f1", 0.0)))
        coverage = float(trav_eval.get("passage_coverage_all_gold_found", 0.0) or 0.0)
        total_calls = float(trav_eval.get("total_traversal_calls", 0.0) or 1.0)
        d["path_em"].append(coverage / total_calls if total_calls else 0.0)

    rows: List[Dict[str, str]] = []
    for (model, dataset), m in sorted(data.items()):
        pf1_mean, pf1_std = _mean_std(m["path_f1"])
        pem_mean, pem_std = _mean_std(m["path_em"])
        af1_mean, af1_std = _mean_std(m["answer_f1"])
        aem_mean, aem_std = _mean_std(m["answer_em"])
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "Path F1": f"{pf1_mean:.3f} ± {pf1_std:.3f}",
                "Path EM": f"{pem_mean:.3f} ± {pem_std:.3f}",
                "Answer F1": f"{af1_mean:.3f} ± {af1_std:.3f}",
                "Answer EM": f"{aem_mean:.3f} ± {aem_std:.3f}",
            }
        )

    return rows


def _rows_to_markdown(rows: List[Dict[str, str]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r[h] for h in headers) + " |")
    return "\n".join(lines) + "\n"


def _rows_to_latex(rows: List[Dict[str, str]], headers: List[str]) -> str:
    """Return a minimal LaTeX tabular representation of ``rows``."""
    lines = ["\\begin{tabular}{" + "l" * len(headers) + "}"]
    lines.append(" & ".join(headers) + " \\")
    lines.append("\\hline")
    for r in rows:
        lines.append(" & ".join(r[h] for h in headers) + " \\")
    lines.append("\\end{tabular}\n")
    return "\n".join(lines)


def main() -> None:
    rows = collect_metrics()
    headers = ["model", "dataset", "Path F1", "Path EM", "Answer F1", "Answer EM"]

    out_dir = Path("analysis_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    with open(out_dir / "combined_metrics_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    md = _rows_to_markdown(rows, headers)
    (out_dir / "combined_metrics_table.md").write_text(md, encoding="utf-8")

    # LaTeX
    tex = _rows_to_latex(rows, headers)
    (out_dir / "combined_metrics_table.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()