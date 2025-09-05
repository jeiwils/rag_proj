from __future__ import annotations

"""Compute per-question F1 deltas between MoE and dense models.

This script loads per-query path and answer F1 scores for a dense model and a
mixture-of-experts (MoE) model across multiple seeds. For each question present
in both runs the difference ``MoE - Dense`` is computed. Distributions of these
deltas are visualised with violin/box plots and saved under ``graphs/deltas``.

The script expects runs to follow the directory layout used throughout the
project, i.e. traversal results under ``data/traversal`` and answer generation
results under ``data/results``. Answer metrics must be saved per-query via the
``--save-per-query`` flag when generating the runs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Tuple, List

try:  # pragma: no cover - seaborn may be absent
    import seaborn as sns
except Exception:  # noqa: BLE001
    sns = None
try:  # pragma: no cover - matplotlib may be absent
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None

from .utils import (
    ensure_output_path,
    traversal_run_paths,
    rag_run_paths,
)

logger = logging.getLogger(__name__)

# Default models compared in plots. These can be overridden via ``main`` args.
DENSE_MODEL_DEFAULT = "qwen2.5-14b-instruct"
MOE_MODEL_DEFAULT = "qwen2.5-2x7b-moe-power-coder-v4"
SPLIT_DEFAULT = "dev"


def _load_path_f1(model: str, dataset: str, seed: int, split: str) -> Dict[str, float]:
    """Return mapping ``{question_id: path_f1}`` for a traversal run."""
    path = traversal_run_paths(model, dataset, split, seed)["per_query"]
    scores: Dict[str, float] = {}
    if not path.exists():
        logger.warning("Missing traversal results: %s", path)
        return scores
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("question_id")
            f1 = rec.get("final_metrics", {}).get("f1")
            if qid is not None and f1 is not None:
                try:
                    scores[qid] = float(f1)
                except (TypeError, ValueError):
                    logger.debug("Non-numeric path f1 for %s: %r", qid, f1)
    return scores


def _load_answer_f1(model: str, dataset: str, seed: int, split: str) -> Dict[str, float]:
    """Return mapping ``{question_id: answer_f1}`` for an answer-gen run."""
    paths = rag_run_paths(model, dataset, split, seed, "dense")["answers"]
    metrics_path = paths["metrics"]
    scores: Dict[str, float] = {}
    if not metrics_path.exists():
        logger.warning(
            "Missing per-query answer metrics for %s seed %s. "
            "Run evaluation with --save-per-query to generate them.",
            model,
            seed,
        )
        return scores
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("question_id")
            f1 = rec.get("f1") or rec.get("F1")
            if qid is not None and f1 is not None:
                try:
                    scores[qid] = float(f1)
                except (TypeError, ValueError):
                    logger.debug("Non-numeric answer f1 for %s: %r", qid, f1)
    return scores


def _available_seeds(model: str, dataset: str, split: str) -> List[int]:
    """Return sorted list of seed numbers available for ``model`` and ``dataset``."""
    res_dir = Path(f"data/results/{model}/{dataset}/{split}")
    seeds: List[int] = []
    if not res_dir.exists():
        return seeds
    for d in res_dir.glob("dense_seed*"):
        m = re.search(r"dense_seed(\d+)", d.name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def _compute_deltas(
    dense_model: str, moe_model: str, dataset: str, split: str
) -> Tuple[List[float], List[float]]:
    """Compute per-question F1 deltas for ``dataset``.

    Returns two lists containing ``MoE - Dense`` deltas for path F1 and answer
    F1 respectively.
    """
    dense_seeds = _available_seeds(dense_model, dataset, split)
    moe_seeds = _available_seeds(moe_model, dataset, split)
    seeds = sorted(set(dense_seeds) & set(moe_seeds))
    path_deltas: List[float] = []
    answer_deltas: List[float] = []

    for seed in seeds:
        dense_path = _load_path_f1(dense_model, dataset, seed, split)
        moe_path = _load_path_f1(moe_model, dataset, seed, split)
        common_q = set(dense_path) & set(moe_path)
        for qid in common_q:
            path_deltas.append(moe_path[qid] - dense_path[qid])

        dense_ans = _load_answer_f1(dense_model, dataset, seed, split)
        moe_ans = _load_answer_f1(moe_model, dataset, seed, split)
        common_q_a = set(dense_ans) & set(moe_ans)
        for qid in common_q_a:
            answer_deltas.append(moe_ans[qid] - dense_ans[qid])

    return path_deltas, answer_deltas


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name.strip())


def _plot_violin(deltas: List[float], title: str, out_path: Path) -> None:
    if not deltas:
        logger.warning("No data available for %s", title)
        return
    if sns is None or plt is None:
        logger.error("seaborn and matplotlib are required for plotting")
        return
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.violinplot(y=deltas, inner="box", color="skyblue")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("ΔF1 (MoE - Dense)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(ensure_output_path(out_path))
    plt.close()


def main(
    dense_model: str = DENSE_MODEL_DEFAULT,
    moe_model: str = MOE_MODEL_DEFAULT,
    split: str = SPLIT_DEFAULT,
) -> None:
    dense_base = Path(f"data/results/{dense_model}")
    moe_base = Path(f"data/results/{moe_model}")
    datasets = sorted(set(d.name for d in dense_base.iterdir() if d.is_dir()) & set(d.name for d in moe_base.iterdir() if d.is_dir()))
    if not datasets:
        logger.warning("No overlapping datasets found for %s and %s", dense_model, moe_model)
        return

    for dataset in datasets:
        path_deltas, ans_deltas = _compute_deltas(dense_model, moe_model, dataset, split)
        dense_short = _sanitize(dense_model)
        moe_short = _sanitize(moe_model)
        base_name = f"{dataset}_{moe_short}_minus_{dense_short}".lower()
        out_dir = Path("graphs/deltas")
        _plot_violin(path_deltas, f"ΔPath F1 on {dataset}", out_dir / f"{base_name}_path_f1.png")
        _plot_violin(ans_deltas, f"ΔAnswer F1 on {dataset}", out_dir / f"{base_name}_answer_f1.png")


if __name__ == "__main__":  # pragma: no cover
    main()