
"""Plot traversal statistics and metrics from result directories.

Result directories can be discovered using :func:`src.utils.get_result_dirs`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    DEFAULT_PLOT_DIR,
    ensure_output_path,
    load_json,
    load_jsonl,
    stylized_subplots,
    get_result_dirs
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Traversal statistics helpers
# ---------------------------------------------------------------------------


def find_result_files(root: Path = Path("data/results")) -> list[Path]:
    """Recursively locate ``per_query_traversal_results.jsonl`` under ``root``."""
    return list(root.rglob("per_query_traversal_results.jsonl"))


def load_traversal_stats(paths: Sequence[Path]) -> Dict:
    """Aggregate traversal statistics from a sequence of JSONL files."""
    stats = {
        "n_traversal_calls": [],
        "n_reader_calls": [],
        "hop_candidate_counts": defaultdict(list),
        "hop_edges_chosen": defaultdict(list),
    }

    for path in paths:
        for obj in load_jsonl(path):
            stats["n_traversal_calls"].append(obj.get("n_traversal_calls", 0))
            stats["n_reader_calls"].append(obj.get("n_reader_calls", 0))
            for hop in obj.get("hop_trace", []):
                hop_id = hop.get("hop", 0)
                stats["hop_candidate_counts"][hop_id].append(
                    len(hop.get("candidate_edges", []))
                )
                stats["hop_edges_chosen"][hop_id].append(
                    len(hop.get("edges_chosen", []))
                )
    return stats


def plot_traversal_distributions(
    stats: Dict, outdir: Path = DEFAULT_PLOT_DIR
) -> None:
    """Generate comparative plots for traversal metrics.

    Parameters
    ----------
    result_dirs:
        Sequence of result directories. Use :func:`get_result_dirs` to collect
        them.
    output:
        Destination file path for the generated plot image.
    show_recall_at_k:
        Include the ``mean_recall_at_k`` metric when ``True``.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    for key in ("n_traversal_calls", "n_reader_calls"):
        data = stats[key]
        if not data:
            continue
        fig, ax = stylized_subplots(figsize=(6, 4))
        ax.hist(data, bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"{key} distribution")
        ax.set_xlabel(key)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(ensure_output_path(outdir / f"{key}_hist.png"))
        plt.close(fig)

    hop_ids = sorted(stats["hop_candidate_counts"])
    if hop_ids:
        mean_candidates = [
            np.mean(stats["hop_candidate_counts"][h]) for h in hop_ids
        ]
        mean_chosen = [
            np.mean(stats["hop_edges_chosen"][h]) for h in hop_ids
        ]
        fig, ax = stylized_subplots(figsize=(6, 4))
        ax.plot(hop_ids, mean_candidates, marker="o", label="candidate edges")
        ax.plot(hop_ids, mean_chosen, marker="x", label="edges chosen")
        ax.set_xlabel("Hop")
        ax.set_ylabel("Average count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ensure_output_path(outdir / "hop_counts.png"))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Traversal metrics plotting
# ---------------------------------------------------------------------------

TRAVERSAL_FIELDS = ["mean_precision", "mean_recall", "mean_f1", "mean_hits_at_k"]
RECALL_AT_K_FIELD = "mean_recall_at_k"
ANSWER_FIELDS = ["EM", "F1"]
META_FIELDS = (
    "dataset",
    "split",
    "variant",
    "model",
    "retriever",
    "seed",
    "timestamp",
)

def _validate_keys(metrics: Dict[str, float], expected: Sequence[str]) -> None:
    missing = [k for k in expected if k not in metrics]
    if missing:
        logger.warning("Missing metrics: %s", ", ".join(missing))


def _load_metrics(result_dir: Path, fields: Sequence[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    stats_path = result_dir / "final_traversal_stats.json"
    data = load_json(stats_path)
    meta = {k: data.get(k) for k in META_FIELDS}
    metrics = data.get("traversal_eval", {})
    _validate_keys(metrics, fields)
    return metrics, meta


def _load_answer_metrics(meta: Dict[str, Any]) -> Dict[str, float]:
    variant_for_path = meta.get("variant")
    seed = meta.get("seed")
    if variant_for_path is None:
        return {}
    if seed is not None:
        variant_for_path = f"{variant_for_path}_seed{seed}"
    summary_path = (
        Path("data")
        / "results"
        / meta.get("model", "")
        / meta.get("dataset", "")
        / meta.get("split", "")
        / variant_for_path
        / f"summary_metrics_{variant_for_path}_{meta.get('split', '')}.json"
    )
    if not summary_path.exists():
        return {}
    data = load_json(summary_path)
    metrics = data.get("answer_eval", data)
    _validate_keys(metrics, ANSWER_FIELDS)
    return metrics


def _model_label(result_dir: Path) -> str:
    parts = result_dir.resolve().parts
    if "graphs" in parts:
        idx = parts.index("graphs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return result_dir.name


def plot_traversal_metrics(
    result_dirs: Sequence[Path],
    output: Path,
    show_recall_at_k: bool = False,
) -> None:
    """Generate comparative plots for traversal metrics."""
    traversal_fields = TRAVERSAL_FIELDS + (
        [RECALL_AT_K_FIELD] if show_recall_at_k else []
    )

    traversal_by_model: Dict[str, Dict[str, float]] = {}
    hop_distributions: Dict[str, List[int]] = {}
    answer_by_model: Dict[str, Dict[str, float]] = {}
    metadata_by_model: Dict[str, Dict[str, Any]] = {}

    for rdir in result_dirs:
        metrics, meta = _load_metrics(rdir, traversal_fields)
        label = _model_label(rdir)
        traversal_by_model[label] = metrics
        dist = metrics.get("hop_depth_distribution")
        if isinstance(dist, list):
            hop_distributions[label] = dist
        answer_by_model[label] = _load_answer_metrics(meta)
        metadata_by_model[label] = meta

    labels = list(traversal_by_model.keys())
    num_traversal_axes = len(traversal_fields)
    fig, axes = stylized_subplots(
        1, num_traversal_axes + 2, figsize=(5 * (num_traversal_axes + 2), 4)
    )

    for ax, field in zip(axes[:num_traversal_axes], traversal_fields):
        values = [traversal_by_model[l].get(field, 0.0) for l in labels]
        ax.bar(labels, values)
        ax.set_ylabel(field)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(field.replace("_", " "))

    ax = axes[num_traversal_axes]
    x = list(range(len(labels)))
    width = 0.35
    for i, field in enumerate(ANSWER_FIELDS):
        values = [answer_by_model.get(l, {}).get(field, 0.0) for l in labels]
        offsets = [p + i * width for p in x]
        ax.bar(offsets, values, width=width, label=field)
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("score")
    ax.set_title("final answer metrics")
    ax.legend()

    ax = axes[num_traversal_axes + 1]
    max_len = max((len(d) for d in hop_distributions.values()), default=0)
    x_vals = list(range(max_len))
    for label, dist in hop_distributions.items():
        y = dist + [0] * (max_len - len(dist))
        ax.plot(x_vals, y, marker="o", label=label)
    ax.set_xlabel("hop depth")
    ax.set_ylabel("count")
    if hop_distributions:
        ax.legend()
    ax.set_title("hop_depth_distribution")

    fig.tight_layout()
    fig.savefig(ensure_output_path(output))
    plt.close(fig)

    if metadata_by_model:
        header = ["label"] + list(META_FIELDS)
        print("\t".join(header))
        for label, meta in metadata_by_model.items():
            row = [label] + [str(meta.get(k, "")) for k in META_FIELDS]
            print("\t".join(row))

__all__ = [
    "find_result_files",
    "load_traversal_stats",
    "plot_traversal_distributions",
    "plot_traversal_metrics",
]


if __name__ == "__main__":
    files = find_result_files()
    stats = load_traversal_stats(files)
    plot_traversal_distributions(stats)

    # Example for plotting traversal metrics across result directories.
    dirs = get_result_dirs(required="final_traversal_stats.json")
    plot_traversal_metrics(dirs, DEFAULT_PLOT_DIR / "traversal_metrics.png")