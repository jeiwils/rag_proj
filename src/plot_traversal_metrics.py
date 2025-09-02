"""Plot traversal and answer metrics across models.

For each supplied traversal result directory this module loads the associated
``final_traversal_stats.json`` and the reader's
``summary_metrics_{variant}_{split}.json`` to produce a single figure with:

* Bar charts for traversal ``mean_precision``, ``mean_recall`` and ``mean_f1``.
* A grouped bar chart for final answer ``EM`` and ``F1`` scores indicating the
  traversal model used by the reader.
* A line plot showing the ``hop_depth_distribution`` for each traversal model.

The module follows the project's convention of being executable without
argument parsing—modify the ``RESULT_DIRS`` and ``OUTPUT`` constants in the
``__main__`` block to suit a particular run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


TRAVERSAL_FIELDS = ["mean_precision", "mean_recall", "mean_f1"]
ANSWER_FIELDS = ["EM", "F1"]


def _load_metrics(result_dir: Path) -> Dict[str, float]:
    """Return traversal metrics from ``final_traversal_stats.json``."""

    stats_path = result_dir / "final_traversal_stats.json"
    with stats_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("traversal_eval", {})


def _infer_components(result_dir: Path) -> Tuple[str, str, str, str]:
    """Infer ``(model, dataset, split, variant)`` from ``result_dir``."""

    parts = result_dir.resolve().parts
    if "traversal" in parts:
        idx = parts.index("traversal")
        variant = parts[idx - 1]
        split = parts[idx - 2]
        dataset = parts[idx - 3]
        model = parts[idx - 4]
        return model, dataset, split, variant
    # Fallback: assume result_dir itself encodes variant
    variant = parts[-1]
    split = parts[-2]
    dataset = parts[-3]
    model = parts[-4]
    return model, dataset, split, variant


def _load_answer_metrics(result_dir: Path) -> Dict[str, float]:
    """Load reader evaluation metrics for ``result_dir``.

    Metrics are expected at::

        data/results/{model}/{dataset}/{split}/{variant}/summary_metrics_{variant}_{split}.json
    """

    model, dataset, split, variant = _infer_components(result_dir)
    summary_path = (
        Path("data")
        / "results"
        / model
        / dataset
        / split
        / variant
        / f"summary_metrics_{variant}_{split}.json"
    )
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _model_label(result_dir: Path) -> str:
    """Infer a short label for a result directory.

    The function tries to extract the ``model`` component from paths of the
    form ``data/graphs/{model}/{dataset}/{split}/{variant}/traversal``.  If the
    pattern cannot be matched the directory name is used as the label.
    """

    parts = result_dir.resolve().parts
    if "graphs" in parts:
        idx = parts.index("graphs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return result_dir.name


def plot_traversal_metrics(result_dirs: Sequence[Path], output: Path) -> None:
    """Generate comparative plots for traversal metrics.

    Parameters
    ----------
    result_dirs:
        Iterable of directories that contain ``final_traversal_stats.json``.
    output:
        Path to the image file where the plot will be saved.
    """

    traversal_by_model: Dict[str, Dict[str, float]] = {}
    hop_distributions: Dict[str, List[int]] = {}
    answer_by_model: Dict[str, Dict[str, float]] = {}

    for rdir in result_dirs:
        metrics = _load_metrics(rdir)
        label = _model_label(rdir)
        traversal_by_model[label] = metrics
        dist = metrics.get("hop_depth_distribution")
        if isinstance(dist, list):
            hop_distributions[label] = dist
        answer_by_model[label] = _load_answer_metrics(rdir)

    labels = list(traversal_by_model.keys())
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, field in zip(axes[:3], TRAVERSAL_FIELDS):
        values = [traversal_by_model[l].get(field, 0.0) for l in labels]
        ax.bar(labels, values)
        ax.set_ylabel(field)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(field.replace("_", " "))

    # Final answer metrics grouped bar chart
    ax = axes[3]
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

    # Hop depth distribution line plot
    ax = axes[4]
    max_len = max((len(d) for d in hop_distributions.values()), default=0)
    x = list(range(max_len))
    for label, dist in hop_distributions.items():
        y = dist + [0] * (max_len - len(dist))
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel("hop depth")
    ax.set_ylabel("count")
    if hop_distributions:
        ax.legend()
    ax.set_title("hop_depth_distribution")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":

    RESULT_DIRS = [
        # Path("data/traversal/model_a/musique/dev/baseline"),
        # Path("data/traversal/model_b/musique/dev/baseline"),
    ]
    OUTPUT = Path("traversal_metrics.png")

    if RESULT_DIRS:
        plot_traversal_metrics(RESULT_DIRS, OUTPUT)