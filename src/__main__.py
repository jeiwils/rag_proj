"""
project-root/
│
├── data/
│   ├── prompts/
│   │   ├── hoprag_iq_prompt.txt
│   │   ├── hoprag_oq_prompt.txt
│   │   ├── 7b_CS_prompt_ultra_updated.txt
│   │   ├── 7b_IQ_prompt_updated.txt
│   │   ├── 7b_OQ_prompt_updated.txt
│   │   └── traversal_prompt.txt
│   │
│   ├── raw_datasets/
│   │   ├── hotpotqa/
│   │   │   ├── hotpot_train_v1.1.json
│   │   │   └── hotpot_dev_fullwiki_v1.json
│   │   ├── 2wikimultihopqa/
│   │   │   ├── train.json
│   │   │   └── dev.json
│   │   └── musique/
│   │       ├── musique_ans_v1.0_train.jsonl
│   │       └── musique_ans_v1.0_dev.jsonl
│   │
│   ├── processed_datasets/
│   │   ├── hotpotqa/
│   │   │   ├── train/
│   │   │   │   ├── questions.jsonl
│   │   │   │   └── passages.jsonl
│   │   │   └── dev/
│   │   │       ├── questions.jsonl
│   │   │       └── passages.jsonl
│   │   ├── 2wikimultihopqa/
│   │   │   ├── train/
│   │   │   │   ├── questions.jsonl
│   │   │   │   └── passages.jsonl
│   │   │   └── dev/
│   │   │       ├── questions.jsonl
│   │   │       └── passages.jsonl
│   │   └── musique/
│   │       ├── train/
│   │       │   ├── questions.jsonl
│   │       │   └── passages.jsonl
│   │       └── dev/
│   │           ├── questions.jsonl
│   │           └── passages.jsonl
│   │
│   ├── models/
│   │   └── {model}/{dataset}/{split}/
│   │       ├── shards/
│   │       │   ├── {split}_passages_shard{N}_{size}.jsonl
│   │       │   └── {hoprag_version}/
│   │       │       ├── {split}_passages_shard{N}_{size}_cs.jsonl
│   │       │       ├── {split}_passages_shard{N}_{size}_iqoq_baseline.jsonl
│   │       │       ├── {split}_passages_shard{N}_{size}_iqoq_enhanced.jsonl
│   │       │       ├── *_cs_debug.txt
│   │       │       ├── *_iqoq_baseline_debug.txt
│   │       │       └── *_iqoq_enhanced_debug.txt
│   │       └── {hoprag_version}/
│   │           ├── cleaned/
│   │           │   ├── iqoq.cleaned.jsonl
│   │           │   └── scored.cleaned.jsonl
│   │           ├── exploded/
│   │           │   ├── iqoq.exploded.jsonl
│   │           │   └── passages.exploded.jsonl
│   │           └── debug/
│   │               └── cleaning_debug.txt
│   │
│   ├── representations/
│   │   ├── datasets/{dataset}/{split}/
│   │   │   ├── {dataset}_passages.jsonl
│   │   │   ├── {dataset}_passages_emb.npy
│   │   │   └── {dataset}_faiss_passages.faiss
│   │   └── models/{model}/{dataset}/{split}/{variant}/
│   │       ├── iqoq.cleaned.jsonl
│   │       ├── {dataset}_iqoq_emb.npy
│   │       └── {dataset}_faiss_iqoq.faiss
│   │
│   └── graphs/
│       └── {model}/{dataset}/{split}/{variant}/
│           ├── {dataset}_{split}_graph.gpickle
│           ├── {dataset}_{split}_edges.jsonl
│           ├── {dataset}_{split}_graph_log.jsonl
│           ├── {dataset}_{split}_graph_results.jsonl
│           └── traversal/
│               ├── per_query_traversal_results.jsonl
│               ├── final_selected_passages.json
│               └── final_traversal_stats.json
│
├── results/
│   └── {model}/{dataset}/{split}/{variant}/
│       ├── answer_per_query_{variant}_{split}.jsonl
│       ├── traversal_per_query_{variant}_{split}.jsonl
│       ├── all_visited_passages_{variant}_{split}.json
│       └── summary_metrics_{variant}_{split}.json
│
├── src/
│   ├── a1_dataset_processing.py
│   ├── a2_text_prep.py
│   ├── a3_file_prep.py
│   ├── b_sparse_dense_representations.py
│   ├── c_graphing.py
│   ├── d_traversal.py
│   ├── e_reranking_answer_gen.py
│   ├── COMPRESS_FILES.py
│   ├── utils.py
│   └── __main__.py
│
├── tests/
│   ├── inspect_datasets.py
│   ├── inspect_outputs.py
│   ├── inspect_passage_ids.py
│   ├── inspect_rows.py
│   └── test.py
│
├── requirements.txt
└── .gitignore







Simple orchestration script for running the HopRAG pipeline.

This entry point iterates over combinations of datasets, models and
pipeline variants, invoking :func:`run_pipeline` for each run.  When
``RESUME`` is enabled, :func:`compute_resume_sets` detects previously
completed ``query_id`` entries in the output file so they can be skipped.
A short summary is printed for every run indicating how many queries were
processed versus how many were skipped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

from .utils import SERVER_CONFIGS, compute_resume_sets

from .e_reranking_answer_gen import run_pipeline
from .utils import load_jsonl


def load_queries(dataset: str, split: str) -> Iterable[Dict]:
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
                    queries = list(load_queries(dataset, split))

                    # Ensure an output directory exists per combination
                    out_dir = Path("results") / dataset / split / model
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / f"{variant}.jsonl"


                    # compute_resume_sets prints a helpful resume message and
                    # returns the intersection of existing IDs with this shard
                    done_ids, shard_ids = compute_resume_sets(
                        resume=RESUME,
                        out_path=str(out_file),
                        items=queries,
                        get_id=lambda x, i: x.get("query_id"),
                        phase_label=f"{dataset}/{split}/{model}/{variant}",
                        id_field="query_id",
                    )

                    # Skip queries already handled in a previous run
                    remaining = [
                        q for q in queries if q.get("query_id") not in done_ids
                    ]

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