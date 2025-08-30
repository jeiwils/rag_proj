import json
import random
from pathlib import Path

from src.utils import load_jsonl, processed_dataset_paths

SEED = 42
DATASETS = ["musique"]
SPLITS = ["dev"]
NUM_QUESTIONS = 250


def downsample(
    dataset: str,
    split: str,
    num_questions: int,
    seed: int = 0,
    *,
    output_dir: str | None = None,
) -> None:
    """Downsample ``num_questions`` from the given dataset and split.

    Parameters
    ----------
    dataset:
        Name of the dataset to downsample.
    split:
        Dataset split (e.g., "train", "dev").
    num_questions:
        Number of questions to retain.
    seed:
        Random seed used when shuffling questions.
    output_dir:
        Directory to which the downsampled dataset will be written.  If ``None``,
        a folder ``data/processed_datasets/{dataset}/{split}_subset`` is used.
    """
    paths = processed_dataset_paths(dataset, split)
    questions_path = paths["questions"]
    passages_path = paths["passages"]

    rng = random.Random(seed)
    questions = list(load_jsonl(str(questions_path)))
    rng.shuffle(questions)
    selected_questions = questions[:num_questions]
    selected_ids = {q["question_id"] for q in selected_questions}

    passages = [
        p
        for p in load_jsonl(str(passages_path))
        if p["passage_id"].split("__")[0] in selected_ids
    ]

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


if __name__ == "__main__":
    for dataset in DATASETS:
        for split in SPLITS:
            output_dir = f"data/processed_datasets/{dataset}/{split}_subset"
            downsample(
                dataset,
                split,
                num_questions=NUM_QUESTIONS,
                seed=SEED,
                output_dir=output_dir,
            )