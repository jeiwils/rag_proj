from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..metrics import compare_runs
from .utils import ensure_output_path, load_json, stylized_subplots


def _plot_metric_distributions(
    per_seed: Dict[str, Dict],
    mean: Dict[str, float],
    std_dev: Dict[str, float],
    out_dir: Path,
) -> None:
    """Create bar and box plots for metrics across seeds."""
    metric_names = list(mean.keys())
    if not metric_names:
        metric_names = sorted({m for seed in per_seed.values() for m in seed.keys()})

    means: List[float] = [mean.get(metric, float("nan")) for metric in metric_names]
    stds: List[float] = [std_dev.get(metric, float("nan")) for metric in metric_names]
    values_by_metric: List[List[float]] = []
    for metric in metric_names:
        vals = [metrics.get(metric) for metrics in per_seed.values() if metrics.get(metric) is not None]
        values_by_metric.append(vals)

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



def plot_efficiency_scatter(per_seed: Dict[str, Dict], out_dir: Path) -> None:
    """Scatter plots of EM/F1 versus tokens and wall-time.

    Parameters
    ----------
    per_seed:
        Mapping of seed to metric dictionary. Metrics should include ``EM``,
        ``F1``, ``tokens`` and ``wall_time``.
    out_dir:
        Directory where the plot should be written.
    """

    # Gather metrics for seeds that have all required values.
    seeds: list[int] = []
    em_vals: list[float] = []
    f1_vals: list[float] = []
    token_vals: list[float] = []
    wall_vals: list[float] = []
    for seed_key, metrics in per_seed.items():
        em = metrics.get("EM")
        f1 = metrics.get("F1")
        tokens = metrics.get("tokens")
        wall = metrics.get("wall_time")
        if None in (em, f1, tokens, wall):
            continue
        seeds.append(int(seed_key))
        em_vals.append(float(em))
        f1_vals.append(float(f1))
        token_vals.append(float(tokens))
        wall_vals.append(float(wall))

    if not seeds:
        return

    fig, axes = stylized_subplots(2, 2, figsize=(12, 10))
    ax_em_tokens, ax_em_time, ax_f1_tokens, ax_f1_time = axes.flat

    ax_em_tokens.scatter(token_vals, em_vals)
    for x, y, seed in zip(token_vals, em_vals, seeds):
        ax_em_tokens.annotate(str(seed), (x, y))
    ax_em_tokens.set_xlabel("Total tokens")
    ax_em_tokens.set_ylabel("EM")
    ax_em_tokens.set_title("EM vs tokens")

    ax_em_time.scatter(wall_vals, em_vals)
    for x, y, seed in zip(wall_vals, em_vals, seeds):
        ax_em_time.annotate(str(seed), (x, y))
    ax_em_time.set_xlabel("Wall time (s)")
    ax_em_time.set_ylabel("EM")
    ax_em_time.set_title("EM vs wall time")

    ax_f1_tokens.scatter(token_vals, f1_vals)
    for x, y, seed in zip(token_vals, f1_vals, seeds):
        ax_f1_tokens.annotate(str(seed), (x, y))
    ax_f1_tokens.set_xlabel("Total tokens")
    ax_f1_tokens.set_ylabel("F1")
    ax_f1_tokens.set_title("F1 vs tokens")

    ax_f1_time.scatter(wall_vals, f1_vals)
    for x, y, seed in zip(wall_vals, f1_vals, seeds):
        ax_f1_time.annotate(str(seed), (x, y))
    ax_f1_time.set_xlabel("Wall time (s)")
    ax_f1_time.set_ylabel("F1")
    ax_f1_time.set_title("F1 vs wall time")

    fig.tight_layout()
    fig.savefig(ensure_output_path(out_dir / "efficiency_scatter.png"))
    plt.close(fig)


SEED_REGEX = re.compile(r"seed(\d+)")


def _load_summary(path: Path) -> tuple[int | None, dict]:
    """Load a summary JSON file and extract seed and metrics."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seed = None
    match = SEED_REGEX.search(path.stem)
    if match:
        seed = int(match.group(1))
    elif isinstance(data.get("seed"), int):
        seed = data["seed"]

    dense_eval = data.get("dense_eval", data)
    metrics = {
        "EM": dense_eval.get("EM") or dense_eval.get("em"),
        "F1": dense_eval.get("F1") or dense_eval.get("f1"),
        "tokens": dense_eval.get("tokens") or dense_eval.get("tokens_total"),
        "wall_time": dense_eval.get("wall_time")
        or dense_eval.get("wall_time_total_sec")
        or dense_eval.get("reader_wall_time_total_sec"),
        "trav_tokens": dense_eval.get("trav_tokens_total"),
        "reader_tokens": dense_eval.get("reader_total_tokens"),
        "t_total_ms": dense_eval.get("t_total_ms"),
        "tps_overall": dense_eval.get("tps_overall"),
    }

    usage_path = path.parent / "token_usage.json"
    if usage_path.exists():
        with open(usage_path, "r", encoding="utf-8") as f:
            usage_data = json.load(f)
        if isinstance(usage_data, dict):
            if "global" in usage_data:
                usage_data = usage_data.get("global", {})

            trav_tok = usage_data.get("trav_tokens_total")
            metrics.setdefault("trav_tokens", trav_tok)
            reader_tok = usage_data.get("reader_total_tokens")
            metrics.setdefault("reader_tokens", reader_tok)

            t_total_ms = usage_data.get("t_total_ms")
            if t_total_ms is None:
                t_total_ms = usage_data.get("t_traversal_ms", 0) + usage_data.get("t_reader_ms", 0)
            metrics.setdefault("t_total_ms", t_total_ms)

            tokens_total = usage_data.get("tokens_total")
            if tokens_total is None:
                totals = [trav_tok, reader_tok]
                totals = [t for t in totals if t is not None]
                tokens_total = sum(totals) if totals else None
            metrics.setdefault("tokens", tokens_total)

            tps_overall = usage_data.get("tps_overall")
            if tps_overall is None and tokens_total is not None and t_total_ms:
                tps_overall = tokens_total / (t_total_ms / 1000)
            metrics.setdefault("tps_overall", tps_overall)

    per_query_em: List[float] = []
    per_query_f1: List[float] = []
    if "per_query_em" in dense_eval and "per_query_f1" in dense_eval:
        per_query_em = list(map(float, dense_eval["per_query_em"]))
        per_query_f1 = list(map(float, dense_eval["per_query_f1"]))
    elif isinstance(dense_eval.get("per_query"), dict):
        for q in dense_eval["per_query"].values():
            per_query_em.append(float(q.get("EM") or q.get("em") or 0.0))
            per_query_f1.append(float(q.get("F1") or q.get("f1") or 0.0))
    metrics["per_query_em"] = per_query_em
    metrics["per_query_f1"] = per_query_f1

    return seed, metrics


def _compute_statistics(base: Path) -> dict:
    summaries = sorted(base.rglob("*_seed*.json"))

    per_seed: dict[int, dict] = {}
    for summary in summaries:
        seed, metrics = _load_summary(summary)
        if seed is not None:
            per_seed[seed] = metrics

    if not per_seed:
        return {}

    mean: dict[str, float] = {}
    variance: dict[str, float] = {}
    std_dev: dict[str, float] = {}
    for key in (
        "EM",
        "F1",
        "tokens",
        "trav_tokens",
        "reader_tokens",
        "wall_time",
        "tps_overall",
    ):
        values = [m[key] for m in per_seed.values() if m.get(key) is not None]
        if values:
            arr = np.array(values, dtype=float)
            mean[key] = float(np.mean(arr))
            variance[key] = float(np.var(arr))
            std_dev[key] = float(np.std(arr))

    wilcoxon: dict[str, dict[str, float]] = {}
    seeds = sorted(per_seed)
    for a, b in combinations(seeds, 2):
        pair_key = f"{a}_vs_{b}"
        wilcoxon[pair_key] = {}
        for metric_key, out_key in (
            ("per_query_em", "EM_p_value"),
            ("per_query_f1", "F1_p_value"),
        ):
            arr_a = per_seed[a].get(metric_key)
            arr_b = per_seed[b].get(metric_key)
            if arr_a and arr_b and len(arr_a) == len(arr_b):
                with NamedTemporaryFile("w", delete=False) as fa, NamedTemporaryFile(
                    "w", delete=False
                ) as fb:
                    fa.write("\n".join(str(x) for x in arr_a))
                    fb.write("\n".join(str(x) for x in arr_b))
                    fa.flush()
                    fb.flush()
                    summary = compare_runs(fa.name, fb.name)
                wilcoxon[pair_key][out_key] = summary.get("p_value")

    output = {
        "per_seed": {
            str(seed): {k: v for k, v in m.items() if not k.startswith("per_query")}
            for seed, m in per_seed.items()
        },
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "wilcoxon": wilcoxon,
    }

    return output


def process_seed_variance(results_dir: Path) -> None:
    stats_path = results_dir / "seed_variance.json"
    if stats_path.exists():
        data = load_json(stats_path)
    else:
        data = _compute_statistics(results_dir)
        if not data:
            print("No seed summaries found in", results_dir)
            return
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    per_seed = data.get("per_seed", {})
    mean = data.get("mean", {})
    std_dev = data.get("std_dev", {})
    wilcoxon = data.get("wilcoxon", {})

    out_dir = results_dir / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if per_seed:
        _plot_metric_distributions(per_seed, mean, std_dev, out_dir)
        plot_efficiency_scatter(per_seed, out_dir)


    if wilcoxon:
        seeds = sorted(int(s) for s in per_seed.keys())
        for key in ("EM_p_value", "F1_p_value"):
            if any(key in v for v in wilcoxon.values()):
                _plot_wilcoxon_heatmap(
                    wilcoxon, seeds, key, out_dir / f"wilcoxon_{key.split('_')[0].lower()}.png"
                )


def main(base_dir: str = "data/results") -> None:
    base = Path(base_dir)
    seed_files = list(base.rglob("*_seed*.json"))
    variance_files = list(base.rglob("seed_variance.json"))
    dirs = {p.parent for p in seed_files}
    dirs.update(p.parent for p in variance_files)
    if not dirs:
        print(f"No seed summaries found under {base}")
        return

    for d in sorted(dirs):
        process_seed_variance(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute statistics and generate plots for seed variance results.",
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="data/results",
        help="Base directory containing results",
    )
    args = parser.parse_args()
    main(args.base_dir)

__all__ = ["main", "process_seed_variance", "plot_efficiency_scatter"]