"""Downsample processed QA datasets.

This script selects a subset of questions and their associated passages from
preprocessed datasets. It operates on all datasets found in
``data/processed_datasets`` and expects standard datasets such as ``hotpotqa``,
``musique`` and ``2wikimultihopqa`` to be present.

Run:

    python -m src.DAAA

The script validates that the expected datasets exist and raises an informative
error when any are missing.
"""

import json
import random
from pathlib import Path
import logging

from src.utils import load_jsonl, processed_dataset_paths

SEED = 42
SPLITS = ["dev"]
NUM_QUESTIONS = 100
EXPECTED_DATASETS = ["hotpotqa", "musique", "2wikimultihopqa"]


def downsample(
    dataset: str,
    split: str,
    num_questions: int,
    seed: int = 0,
    *,
    output_dir: str | None = None,
) -> None:
    paths = processed_dataset_paths(dataset, split)
    questions_path = paths["questions"]
    passages_path = paths["passages"]

    rng = random.Random(seed)
    questions = list(load_jsonl(str(questions_path)))

    passage_ids_all = {p["passage_id"] for p in load_jsonl(str(passages_path))}
    questions = [
        q
        for q in questions
        if all(pid in passage_ids_all for pid in q["gold_passages"])
    ]
    if len(questions) < num_questions:
        logging.warning(
            "Only %d questions available after filtering; requested %d",
            len(questions),
            num_questions,
        )

    rng.shuffle(questions)
    selected_questions = questions[:num_questions]
    selected_ids = {q["question_id"] for q in selected_questions}
    prefixes = tuple(selected_ids)

    passages = [
        p for p in load_jsonl(str(passages_path))
        if p["passage_id"].startswith(prefixes)
    ]

    passage_ids = {p["passage_id"] for p in passages}
    missing: dict[str, list[str]] = {}
    for q in selected_questions:
        missing_ids = [pid for pid in q["gold_passages"] if pid not in passage_ids]
        if missing_ids:
            logging.warning("Missing passages: %s", missing)
    if missing:
        raise ValueError(f"Missing passages: {missing}")

    if output_dir is None:
        output_dir = f"data/processed_datasets/{dataset}/{split}_subset"
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    q_out = out_base / "questions.jsonl"
    p_out = out_base / "passages.jsonl"
    with open(q_out, "w", encoding="utf-8") as fq:
        for obj in selected_questions:
            fq.write(json.dumps(obj) + "\n")
    with open(p_out, "w", encoding="utf-8") as fp:
        for obj in passages:
            fp.write(json.dumps(obj) + "\n")


def main() -> None:
    """Run the downsampling routine for all datasets.

    The function processes every dataset found in ``data/processed_datasets``
    and verifies that the expected datasets listed in
    :data:`EXPECTED_DATASETS` are present.
    """

    data_dir = Path("data/processed_datasets")
    datasets = [d.name for d in data_dir.iterdir() if d.is_dir()]
    missing = [d for d in EXPECTED_DATASETS if d not in datasets]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Expected datasets not found: {missing_str}"
        )

    for dataset in datasets:
        for split in SPLITS:
            output_dir = f"data/processed_datasets/{dataset}/{split}_subset"
            downsample(
                dataset,
                split,
                num_questions=NUM_QUESTIONS,
                seed=SEED,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()