import json
import random
from pathlib import Path

from src.utils import load_jsonl, processed_dataset_paths

SEED = 42
SPLITS = ["dev"]
NUM_QUESTIONS = 100
DATASETS = [d.name for d in Path("data/processed_datasets").iterdir() if d.is_dir()]


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
    rng.shuffle(questions)
    selected_questions = questions[:num_questions]
    selected_ids = {q["question_id"] for q in selected_questions}

    passages = [
        p
        for p in load_jsonl(str(passages_path))
        if p["passage_id"].rsplit("_sent", 1)[0] in selected_ids
    ]

    passage_ids = {p["passage_id"] for p in passages}
    missing: dict[str, list[str]] = {}
    for q in selected_questions:
        missing_ids = [pid for pid in q["gold_passages"] if pid not in passage_ids]
        if missing_ids:
            missing[q["question_id"]] = missing_ids
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