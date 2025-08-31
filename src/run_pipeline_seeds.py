"""Run the RAG pipeline for multiple random seeds and summarise variance.

This script repeatedly runs :func:`src.f_dense_RAG.run_dense_rag` using
different seeds. Per-run metrics are collected and the variance and standard
 deviation for each metric are reported using :mod:`numpy`.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from src.d_traversal import DEFAULT_SEED_TOP_K
from src.f_dense_RAG import run_dense_rag


def run_for_seeds(seeds: List[int], **kwargs) -> Dict[int, Dict[str, float]]:
    """Run ``run_dense_rag`` for each seed and return a mapping of results."""
    results: Dict[int, Dict[str, float]] = {}
    for s in seeds:
        metrics = run_dense_rag(seed=s, **kwargs)
        results[s] = metrics
        print(f"Seed {s}: {metrics}")
    return results


def summarize_metrics(results: Dict[int, Dict[str, float]]) -> None:
    """Print variance and standard deviation for each metric key."""
    if not results:
        return
    keys = list(next(iter(results.values())).keys())
    for key in keys:
        values = [m.get(key, 0.0) for m in results.values()]
        arr = np.array(values, dtype=float)
        print(f"{key} variance={np.var(arr):.4f} std={np.std(arr):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dense RAG across multiple seeds and compute variance/std.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--reader-model", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_SEED_TOP_K)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    per_seed = run_for_seeds(
        args.seeds,
        dataset=args.dataset,
        split=args.split,
        reader_model=args.reader_model,
        server_url=args.server_url,
        top_k=args.top_k,
        resume=args.resume,
    )
    summarize_metrics(per_seed)