
"""Plot compute and token usage across result directories.

Use :func:`src.utils.get_result_dirs` to locate directories containing the
``token_usage.json`` file produced by pipeline runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt

from .utils import (
    DEFAULT_PLOT_DIR,
    ensure_output_path,
    load_json,
    stylized_subplots,
    get_result_dirs,
    rag_run_paths,
    parse_traversal_run_dir,
)


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
    parts = result_dir.resolve().parts
    model, dataset, split = parts[-4], parts[-3], parts[-2]
    mode, seed_str = result_dir.name.rsplit("_seed", 1)
    seed = int(seed_str)
    usage_path = rag_run_paths(model, dataset, split, seed, mode)["answers"]["token_usage"]
    if not usage_path.exists():
        logger.warning("token_usage.json not found in %s", result_dir)
        return {f: 0.0 for f in fields}

    data = load_json(usage_path)

    # handle legacy field names - I don't dare touch my loops in the llm calls right now!!!
    if "trav_output_tokens" not in data and "trav_completion_tokens" in data:
        data["trav_output_tokens"] = data["trav_completion_tokens"]
    if "per_query_traversal" in data:
        for m in data["per_query_traversal"].values():
            if (
                "trav_output_tokens" not in m
                and "trav_completion_tokens" in m
            ):
                m["trav_output_tokens"] = m["trav_completion_tokens"]

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
    dirs = get_result_dirs(required="token_usage.json")
    groups: Dict[tuple[str, str, str, int], list[Path]] = {}
    for d in dirs:
        key = parse_traversal_run_dir(d)
        groups.setdefault(key, []).append(d)

    for (model, dataset, split, seed), variant_dirs in groups.items():
        output = ensure_output_path(
            DEFAULT_PLOT_DIR / model / dataset / split / f"compute_seed{seed}.png"
        )
        plot_compute_usage(variant_dirs, output)