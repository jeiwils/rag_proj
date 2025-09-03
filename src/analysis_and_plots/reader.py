from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from .utils import ensure_output_path, load_json, stylized_subplots, answer_run_paths

logger = logging.getLogger(__name__)


def _extract_reader_lengths(token_usage_path: Path) -> tuple[list[int], list[int]]:
    """Return per-query reader prompt and output token counts."""
    try:
        data = load_json(token_usage_path)
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("Failed to load %s: %s", token_usage_path, err)
        return [], []

    per_reader = data.get("per_query_reader", {}) or {}
    prompt_lengths: list[int] = []
    output_lengths: list[int] = []

    for metrics in per_reader.values():
        p = metrics.get("reader_prompt_tokens")
        o = metrics.get("reader_output_tokens")
        if p is not None:
            try:
                prompt_lengths.append(int(p))
            except (TypeError, ValueError):
                logger.debug("Non-numeric reader_prompt_tokens: %r", p)
        if o is not None:
            try:
                output_lengths.append(int(o))
            except (TypeError, ValueError):
                logger.debug("Non-numeric reader_output_tokens: %r", o)

    return prompt_lengths, output_lengths


def _plot_lengths(prompts: Sequence[int], outputs: Sequence[int], output: Path) -> None:
    """Create histograms for prompt and output token lengths."""
    fig, axes = stylized_subplots(1, 2, figsize=(10, 4))

    if prompts:
        axes[0].hist(prompts, bins=30, color="tab:blue", edgecolor="black")
    axes[0].set_title("Reader prompt tokens")
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Count")

    if outputs:
        axes[1].hist(outputs, bins=30, color="tab:orange", edgecolor="black")
    axes[1].set_title("Reader output tokens")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(ensure_output_path(output))
    plt.close(fig)


def process_usage_file(path: Path) -> None:
    """Generate a ``reader_lengths.png`` next to ``path``."""
    prompts, outputs = _extract_reader_lengths(path)
    if not prompts and not outputs:
        logger.warning("No reader token data found in %s", path)
        return
    out_path = path.parent / "reader_lengths.png"
    _plot_lengths(prompts, outputs, out_path)


def main(base_dir: str = "data/results") -> None:
    base = Path(base_dir)
    paths = list(base.rglob("token_usage.json"))
    if not paths:
        logger.warning("No token_usage.json files found under %s", base)
        return
    for p in paths:
        process_usage_file(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate histograms of reader token lengths from token_usage.json files.",
    )
    parser.add_argument(
        "model",
        help="Model name",
    )
    parser.add_argument(
        "dataset",
        help="Dataset name",
    )
    parser.add_argument(
        "split",
        help="Dataset split",
    )
    parser.add_argument(
        "seed",
        type=int,
        help="Random seed used in the runs",
    )
    args = parser.parse_args()
    for mode in ("baseline", "dense"):
        path = answer_run_paths(args.model, args.dataset, args.split, mode, args.seed)[
            "token_usage"
        ]
        if path.exists():
            process_usage_file(path)

__all__ = ["main", "process_usage_file"]