"""Simple orchestration script for running the HopRAG pipeline.

This entry point iterates over combinations of datasets, models and
pipeline variants, invoking :func:`run_pipeline` for each run.  When
``RESUME`` is enabled, previously completed ``query_id`` entries in the
output file are detected via :func:`existing_ids` and skipped.  A short
summary is printed for every run indicating how many queries were
processed versus how many were skipped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from .a2_text_prep import SERVER_CONFIGS, compute_resume_sets, existing_ids

from .e_reranking_answer_gen import run_pipeline
from .utils import load_jsonl


def load_queries(dataset: str, split: str) -> List[Dict]:
    """Load a dataset split from ``data/`` if it exists.

    The function searches two common locations:

    ``data/{dataset}/{split}.jsonl`` and ``data/{dataset}_{split}.jsonl``.
    ``load_jsonl`` is used so the returned items are dictionaries.
    """

    candidates = [
        Path(f"data/{dataset}/{split}.jsonl"),
        Path(f"data/{dataset}_{split}.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            return load_jsonl(str(path))
    return []


# ---------------------------------------------------------------------------
# Configuration – modify these lists to control the runs
# ---------------------------------------------------------------------------
DATASETS = ["demo_dataset"]
MODELS = ["demo_model"]
VARIANTS = ["dense", "hoprag", "enhanced"]
SPLIT = ["dev"]
RESUME = False


if __name__ == "__main__":
    for dataset in DATASETS:
        for model in MODELS:
            for variant in VARIANTS:
                for split in SPLIT:
                    queries = load_queries(dataset, split)

                    # Ensure an output directory exists per combination
                    out_dir = Path("results") / dataset / split / model
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"{variant}.jsonl"

                    # Determine which query_ids have already been processed
                    done = (
                        existing_ids(str(out_file), id_field="query_id")
                        if RESUME
                        else set()
                    )

                    # compute_resume_sets prints a helpful resume message and
                    # returns the intersection of existing IDs with this shard
                    done_ids, shard_ids = compute_resume_sets(
                        resume=RESUME,
                        out_path=str(out_file),
                        items=queries,
                        get_id=lambda x, i: x.get("query_id"),
                        phase_label=f"{dataset}/{split}/{model}/{variant}",
                    )

                    # Skip queries already handled in a previous run
                    remaining = [q for q in queries if q.get("query_id") not in done]

                    if remaining:
                        run_pipeline(
                            mode=variant,
                            query_data=remaining,
                            graph=None,
                            passage_metadata=[],
                            passage_emb=None,
                            passage_index=None,
                            emb_model=None,
                            server_configs=SERVER_CONFIGS,
                            output_path=str(out_file),
                        )

                    total = len(shard_ids)
                    processed = len(remaining)
                    skipped = total - processed
                    print(
                        f"[summary] {dataset}/{split}/{model}/{variant}: "
                        f"processed {processed}, skipped {skipped} of {total}"
                    )