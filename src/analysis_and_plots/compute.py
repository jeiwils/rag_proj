
"""Plot compute and token usage across result directories.

Use :func:`src.utils.get_result_dirs` to locate directories containing the
``token_usage.json`` file produced by pipeline runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt

from .utils import ensure_output_path, load_json, stylized_subplots, get_result_dirs


logger = logging.getLogger(__name__)

USAGE_FIELDS = [
    "tokens_total",
    "t_total_ms",
    "tps_overall",
    "trav_tokens_total",
    "reader_total_tokens",
    "trav_prompt_tokens",
    "trav_output_tokens",
    "reader_prompt_tokens",
    "reader_output_tokens",
]



def _validate_keys(metrics: Dict[str, float], expected: Sequence[str]) -> None:
    """Log a warning if any ``expected`` keys are absent."""
    missing = [k for k in expected if k not in metrics]
    if missing:
        logger.warning("Missing metrics: %s", ", ".join(missing))


def _load_usage(result_dir: Path, fields: Sequence[str]) -> Dict[str, float]:
    """Return compute usage metrics from ``token_usage.json`` in ``result_dir``."""
    usage_path = result_dir / "token_usage.json"
    if not usage_path.exists():
        logger.warning("token_usage.json not found in %s", result_dir)
        return {f: 0.0 for f in fields}

    data = load_json(usage_path)

    # Backfill missing fields using per-query metrics when available.
    for field in fields:
        if field not in data:
            if field.startswith("trav") and "per_query_traversal" in data:
                data[field] = sum(
                    m.get(field, 0.0) for m in data["per_query_traversal"].values()
                )
            elif field.startswith("reader") and "per_query_reader" in data:
                data[field] = sum(
                    m.get(field, 0.0) for m in data["per_query_reader"].values()
                )

    _validate_keys(data, fields)
    # Ensure all expected fields are present in the returned mapping.
    return {field: data.get(field, 0.0) for field in fields}


def _model_label(result_dir: Path) -> str:
    """Infer a short model label from ``result_dir``."""
    parts = result_dir.resolve().parts
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return result_dir.name


def plot_compute_usage(result_dirs: Sequence[Path], output: Path) -> None:
    """Generate bar plots comparing compute usage across models.

    Parameters
    ----------
    result_dirs:
        Iterable of result directories. ``get_result_dirs`` can be used to
        discover these locations.
    output:
        File path where the resulting plot image will be written.
    """
    usage_by_model: Dict[str, Dict[str, float]] = {}
    for rdir in result_dirs:
        metrics = _load_usage(rdir, USAGE_FIELDS)
        label = _model_label(rdir)
        usage_by_model[label] = metrics

    labels = list(usage_by_model.keys())
    fig, axes = stylized_subplots(1, len(USAGE_FIELDS), figsize=(5 * len(USAGE_FIELDS), 4))
    for ax, field in zip(axes, USAGE_FIELDS):
        values = [usage_by_model.get(l, {}).get(field, 0.0) for l in labels]
        ax.bar(labels, values)
        ax.set_ylabel(field)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(field.replace("_", " "))

    fig.tight_layout()
    fig.savefig(ensure_output_path(output))
    plt.close(fig)


__all__ = ["plot_compute_usage"]


if __name__ == "__main__":
    # Example usage: gather directories containing token usage information and
    # produce a summary plot.
    dirs = get_result_dirs(required="token_usage.json")
    plot_compute_usage(dirs, Path("analysis/compute_usage.png"))