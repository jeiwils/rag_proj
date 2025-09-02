"""Utilities for processing raw QA datasets into a unified format.

This module contains light-weight dataset processors that read each dataset's
raw files and emit question and passage JSONL files under
``data/processed_datasets/{dataset}/{split}/``.  The original project stored
these processors in a small registry spread across multiple modules.  For this
exercise the logic is consolidated here so the processors can be created with a
simple mapping.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Type

from src.utils import (
    append_jsonl,
    clean_text,
    compute_resume_sets,
    pid_plus_title,
    processed_dataset_paths,
)


class DatasetProcessor:
    """Base class for dataset processors.

    Sub-classes implement :meth:`process` which performs the end to end
    transformation from raw data to the processed JSONL outputs.  The base class
    simply stores common configuration options.
    """

    DATASET: str = ""

    def __init__(
        self,
        *,
        split: str,
        file_path: str,
        max_examples: int | None = None,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        self.split = split
        self.file_path = file_path
        self.max_examples = max_examples
        self.overwrite = overwrite
        self.resume = resume

    def process(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class HotpotQAProcessor(DatasetProcessor):
    """Processor for the HotpotQA dataset."""

    DATASET = "hotpotqa"

    def process(self) -> None:  # noqa: D401 - short override
        with open(self.file_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        if isinstance(self.max_examples, int):
            examples = examples[: self.max_examples]

        paths = processed_dataset_paths(self.DATASET, self.split)
        qa_path = str(paths["questions"])
        passages_path = str(paths["passages"])

        done_qids, _ = compute_resume_sets(
            resume=self.resume,
            out_path=qa_path,
            items=examples,
            get_id=lambda ex, i: ex["_id"],
            phase_label=f"{self.DATASET} {self.split} questions",
            id_field="question_id",
        )

        def iter_pids() -> Iterable[str]:
            for ex in examples:
                qid = ex["_id"]
                for title, sents in ex["context"]:
                    for i, _ in enumerate(sents):
                        yield pid_plus_title(qid, title, i)

        done_pids, _ = compute_resume_sets(
            resume=self.resume,
            out_path=passages_path,
            items=iter_pids(),
            get_id=lambda pid, i: pid,
            phase_label=f"{self.DATASET} {self.split} passages",
            id_field="passage_id",
        )

        for ex in examples:
            qid = ex["_id"]
            if qid not in done_qids:
                gold_ids, seen = [], set()
                for title, idx in ex.get("supporting_facts", []):
                    pid = pid_plus_title(qid, title, idx)
                    if pid not in seen:
                        gold_ids.append(pid)
                        seen.add(pid)

                append_jsonl(
                    qa_path,
                    {
                        "question_id": qid,
                        "dataset": self.DATASET,
                        "split": self.split,
                        "question": clean_text(ex["question"]),
                        "gold_answer": clean_text(ex.get("answer", "")),
                        "gold_passages": gold_ids,
                    },
                )

            for title, sents in ex["context"]:
                for i, sent in enumerate(sents):
                    pid = pid_plus_title(qid, title, i)
                    if pid in done_pids:
                        continue
                    append_jsonl(
                        passages_path,
                        {
                            "passage_id": pid,
                            "title": title,
                            "text": clean_text(sent),
                        },
                    )


class TwoWikiProcessor(DatasetProcessor):
    """Processor for the 2WikiMultiHopQA dataset."""

    DATASET = "2wikimultihopqa"

    def process(self) -> None:  # noqa: D401 - short override
        with open(self.file_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        if isinstance(self.max_examples, int):
            examples = examples[: self.max_examples]

        paths = processed_dataset_paths(self.DATASET, self.split)
        qa_path = str(paths["questions"])
        passages_path = str(paths["passages"])

        done_qids, _ = compute_resume_sets(
            resume=self.resume,
            out_path=qa_path,
            items=examples,
            get_id=lambda ex, i: ex["_id"],
            phase_label=f"{self.DATASET} {self.split} questions",
            id_field="question_id",
        )

        def iter_pids() -> Iterable[str]:
            for ex in examples:
                qid = ex["_id"]
                for title, sents in ex["context"]:
                    for i, _ in enumerate(sents):
                        yield pid_plus_title(qid, title, i)

        done_pids, _ = compute_resume_sets(
            resume=self.resume,
            out_path=passages_path,
            items=iter_pids(),
            get_id=lambda pid, i: pid,
            phase_label=f"{self.DATASET} {self.split} passages",
            id_field="passage_id",
        )

        for ex in examples:
            qid = ex["_id"]
            if qid not in done_qids:
                gold_ids, seen = [], set()
                for title, idx in ex.get("supporting_facts", []):
                    pid = pid_plus_title(qid, title, idx)
                    if pid not in seen:
                        gold_ids.append(pid)
                        seen.add(pid)


                append_jsonl(
                    qa_path,
                    {
                        "question_id": qid,
                        "dataset": self.DATASET,
                        "split": self.split,
                        "question": clean_text(ex["question"]),
                        "gold_answer": clean_text(ex.get("answer", "")),
                        "gold_passages": gold_ids,
                    },
                )

            for title, sents in ex["context"]:
                for i, sent in enumerate(sents):
                    pid = pid_plus_title(qid, title, i)
                    if pid in done_pids:
                        continue
                    append_jsonl(
                        passages_path,
                        {
                            "passage_id": pid,
                            "title": title,
                            "text": clean_text(sent),
                        },
                    )


class MuSiQueProcessor(DatasetProcessor):
    """Processor for the MuSiQue dataset."""

    DATASET = "musique"

    def process(self) -> None:  # noqa: D401 - short override
        examples: List[Dict] = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for k, line in enumerate(f):
                if isinstance(self.max_examples, int) and k >= self.max_examples:
                    break
                examples.append(json.loads(line))

        paths = processed_dataset_paths(self.DATASET, self.split)
        qa_path = str(paths["questions"])
        passages_path = str(paths["passages"])

        done_qids, _ = compute_resume_sets(
            resume=self.resume,
            out_path=qa_path,
            items=examples,
            get_id=lambda ex, i: ex["id"],
            phase_label=f"{self.DATASET} {self.split} questions",
            id_field="question_id",
        )

        def iter_pids() -> Iterable[str]:
            for ex in examples:
                qid = ex["id"]
                paras = ex.get("paragraphs", [])
                for idx, p in enumerate(paras):
                    j = p.get("idx")
                    pid = f"{qid}_sent{j}" if j is not None else f"{qid}_sent{idx}"
                    yield pid

        done_pids, _ = compute_resume_sets(
            resume=self.resume,
            out_path=passages_path,
            items=iter_pids(),
            get_id=lambda pid, i: pid,
            phase_label=f"{self.DATASET} {self.split} passages",
            id_field="passage_id",
        )

        for ex in examples:
            qid = ex["id"]
            paras = ex.get("paragraphs", [])

            if qid not in done_qids:
                gold_ids, seen = [], set()
                for idx, p in enumerate(paras):
                    if p.get("is_supporting"):
                        j = p.get("idx")
                        pid = f"{qid}_sent{j}" if j is not None else f"{qid}_sent{idx}"
                        if pid not in seen:
                            gold_ids.append(pid)
                            seen.add(pid)
                append_jsonl(
                    qa_path,
                    {
                        "question_id": qid,
                        "dataset": self.DATASET,
                        "split": self.split,
                        "question": clean_text(ex.get("question", "")),
                        "gold_answer": clean_text(ex.get("answer", "")),
                        "gold_passages": gold_ids,
                    },
                )

            for idx, p in enumerate(paras):
                j = p.get("idx")
                pid = f"{qid}_sent{j}" if j is not None else f"{qid}_sent{idx}"
                if pid in done_pids:
                    continue
                append_jsonl(
                    passages_path,
                    {
                        "passage_id": pid,
                        "title": p.get("title", ""),
                        "text": clean_text(p.get("paragraph_text", "")),
                    },
                )


PROCESSORS: Dict[str, Type[DatasetProcessor]] = {
    "hotpotqa": HotpotQAProcessor,
    "2wikimultihopqa": TwoWikiProcessor,
    "musique": MuSiQueProcessor,
}


if __name__ == "__main__":
    RESUME = True
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["dev"]
    MAX_EXAMPLES = 250

    for dataset in DATASETS:
        for split in SPLITS:
            if dataset == "hotpotqa":
                file_path = (
                    "data/raw_datasets/hotpotqa/hotpot_train_v1.1.json"
                    if split == "train"
                    else "data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
                )
            elif dataset == "2wikimultihopqa":
                file_path = f"data/raw_datasets/2wikimultihopqa/{split}.json"
            elif dataset == "musique":
                file_path = f"data/raw_datasets/musique/musique_ans_v1.0_{split}.jsonl"
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unknown dataset: {dataset}")

            processor_cls = PROCESSORS[dataset]
            processor = processor_cls(
                split=split,
                file_path=file_path,
                max_examples=MAX_EXAMPLES,
                resume=RESUME,
            )
            processor.process()