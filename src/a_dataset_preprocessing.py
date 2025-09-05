"""Utilities for processing raw QA datasets into a unified format.

This module exposes a single :func:`process_dataset` helper which converts a
dataset's raw examples into the project wide question and passage JSONL files.
Callers must supply a ``field_map`` describing how to extract the necessary
fields from each example.  Previous wrappers like ``process_hotpotqa`` and the
``PROCESSORS`` registry have been removed so that new datasets can be processed
without modifying this module.

Example
-------

To process a HotpotQA style JSON file::

    from src.a1_dataset_processing import process_dataset
    from src.utils import pid_plus_title

    field_map = {
        "get_id": lambda ex: ex["_id"],
        "get_question": lambda ex: ex["question"],
        "get_answer": lambda ex: ex.get("answer", ""),
        "iter_passages": lambda ex: [
            (pid_plus_title(ex["_id"], title, i), title, sent)
            for title, sents in ex["context"]
            for i, sent in enumerate(sents)
        ],
        "gold_passage_ids": lambda ex: [
            pid_plus_title(ex["_id"], title, idx)
            for title, idx in ex.get("supporting_facts", [])
        ],
    }

    process_dataset(
        dataset="hotpotqa",
        split="dev",
        file_path="data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json",
        field_map=field_map,
    )

The ``field_map`` supplies callables to extract the question id, question
text, gold answer, passage iterator and gold passage IDs.  The processing logic
is otherwise dataset agnostic.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Iterable, List, Tuple

from src.utils import (
    append_jsonl,
    clean_text,
    compute_resume_sets,
    processed_dataset_paths,
    pid_plus_title
)

# ---------------------------------------------------------------------------
# Generic dataset processing

FieldMap = Dict[str, Callable[[Dict], Iterable]]


def process_dataset(
    *,
    dataset: str,
    split: str,
    file_path: str,
    field_map: Dict[str, Callable[[Dict], Iterable]],
    max_examples: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    """Process ``file_path`` using ``field_map``.

    Parameters
    ----------
    dataset:
        Name of the dataset being processed.
    split:
        Dataset split (``train``, ``dev`` ...).
    file_path:
        Path to the raw dataset file.  JSON or JSONL files are supported.
    field_map:
        Mapping of callables that extract fields from each example.  Required
        keys are ``get_id``, ``get_question``, ``get_answer``,
        ``iter_passages`` and ``gold_passage_ids``.  The callables operate on a
        single example and either return a value or an iterable of values.
    max_examples:
        Optional limit for the number of examples processed.
    overwrite:
        Unused but kept for backward compatibility.
    resume:
        Whether to resume from existing processed files.
    """

    # ---- Load raw examples -------------------------------------------------
    examples: List[Dict]
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            examples = []
            for i, line in enumerate(f):
                if isinstance(max_examples, int) and i >= max_examples:
                    break
                examples.append(json.loads(line))
        else:
            examples = json.load(f)
            if isinstance(max_examples, int):
                examples = examples[:max_examples]

    paths = processed_dataset_paths(dataset, split)
    qa_path = str(paths["questions"])
    passages_path = str(paths["passages"])

    get_id = field_map["get_id"]
    get_question = field_map["get_question"]
    get_answer = field_map.get("get_answer", lambda ex: "")
    iter_passages_fn = field_map["iter_passages"]
    gold_ids_fn = field_map.get("gold_passage_ids", lambda ex: [])

    # ---- Determine resume state --------------------------------------------
    done_qids, _ = compute_resume_sets(
        resume=resume,
        out_path=qa_path,
        items=examples,
        get_id=lambda ex, i: get_id(ex),
        phase_label=f"{dataset} {split} questions",
        id_field="question_id",
    )

    def iter_pids() -> Iterable[str]:
        for ex in examples:
            for pid, _title, _text in iter_passages_fn(ex):
                yield pid

    done_pids, _ = compute_resume_sets(
        resume=resume,
        out_path=passages_path,
        items=iter_pids(),
        get_id=lambda pid, i: pid,
        phase_label=f"{dataset} {split} passages",
        id_field="passage_id",
    )

    # ---- Write processed files ---------------------------------------------
    for ex in examples:
        qid = get_id(ex)
        if qid not in done_qids:
            gold_ids, seen = [], set()
            for pid in gold_ids_fn(ex):
                if pid not in seen:
                    gold_ids.append(pid)
                    seen.add(pid)
            append_jsonl(
                qa_path,
                {
                    "question_id": qid,
                    "dataset": dataset,
                    "split": split,
                    "question": clean_text(get_question(ex)),
                    "gold_answer": clean_text(get_answer(ex)),
                    "gold_passages": gold_ids,
                },
            )

        for pid, title, text in iter_passages_fn(ex):
            if pid in done_pids:
                continue
            append_jsonl(
                passages_path,
                {
                    "passage_id": pid,
                    "title": title,
                    "text": clean_text(text),
                },
            )







if __name__ == "__main__":
    RESUME = True
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["dev"]
    MAX_EXAMPLES = 100

    for dataset in DATASETS:
        for split in SPLITS:
            if dataset == "hotpotqa":
                file_path = (
                    "data/raw_datasets/hotpotqa/hotpot_train_v1.1.json"
                    if split == "train"
                    else "data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
                )
                field_map = {
                    "get_id": lambda ex: ex["_id"],
                    "get_question": lambda ex: ex["question"],
                    "get_answer": lambda ex: ex.get("answer", ""),
                    "iter_passages": lambda ex: [
                        (pid_plus_title(ex["_id"], title, i), title, sent)
                        for title, sents in ex["context"]
                        for i, sent in enumerate(sents)
                    ],
                    "gold_passage_ids": lambda ex: [
                        pid_plus_title(ex["_id"], title, idx)
                        for title, idx in ex.get("supporting_facts", [])
                    ],
                }
            elif dataset == "2wikimultihopqa":
                file_path = f"data/raw_datasets/2wikimultihopqa/{split}.json"
                field_map = {
                    "get_id": lambda ex: ex["_id"],
                    "get_question": lambda ex: ex["question"],
                    "get_answer": lambda ex: ex.get("answer", ""),
                    "iter_passages": lambda ex: [
                        (pid_plus_title(ex["_id"], title, i), title, sent)
                        for title, sents in ex["context"]
                        for i, sent in enumerate(sents)
                    ],
                    "gold_passage_ids": lambda ex: [
                        pid_plus_title(ex["_id"], title, idx)
                        for title, idx in ex.get("supporting_facts", [])
                    ],
                }
            elif dataset == "musique":
                file_path = f"data/raw_datasets/musique/musique_ans_v1.0_{split}.jsonl"
                field_map = {
                    "get_id": lambda ex: ex["id"],
                    "get_question": lambda ex: ex.get("question", ""),
                    "get_answer": lambda ex: ex.get("answer", ""),
                    "iter_passages": lambda ex: [
                        (
                            f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}",
                            p.get("title", ""),
                            p.get("paragraph_text", ""),
                        )
                        for i, p in enumerate(ex.get("paragraphs", []))
                    ],
                    "gold_passage_ids": lambda ex: [
                        f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}"
                        for i, p in enumerate(ex.get("paragraphs", []))
                        if p.get("is_supporting")
                    ],
                }

            process_dataset(
                dataset=dataset,
                split=split,
                file_path=file_path,
                field_map=field_map,
                max_examples=MAX_EXAMPLES,
                resume=RESUME,
            )
