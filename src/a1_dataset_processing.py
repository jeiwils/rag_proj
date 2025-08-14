"""

Module Overview
---------------
Prepare raw multi-hop QA datasets for downstream retrieval and evaluation.

This script reads each dataset's original files, normalizes question text and
passages, and writes paired question and passage JSONL files. Each question
entry includes only the IDs of gold passages, while companion `_passages` files
contain the full text of all available passages.



Inputs
------

### data/raw_datasets/{dataset}/

- hotpot_train_v1.1.json                  - HotpotQA training data.
- hotpot_dev_fullwiki_v1.json                 - HotpotQA dev data.
- {split}.json``                             - 2WikiMultiHopQA data (either 'train' or 'dev')
- musique_ans_v1.0_{split}.jsonl              - MuSiQue data. (either 'train' or 'dev')



Outputs
-------

### data/processed_datasets/{dataset}/

- {split}.jsonl                                     - processed questions with gold passage IDs.
- {split}_passages.jsonl                               - corresponding ID passages with text content.




File Schema
-----------

### {split}.jsonl

{
  "question_id": "{question_id}",
  "dataset": "{dataset}",
  "split": "{split}",
  "question": "{normalized_question}",
  "gold_answer": "{gold_answer}",
  "gold_passages": ["{passage_id_1}", "{passage_id_2}", "..."]
}

Fields
- ``question_id``: unique question identifier.
- ``dataset``: dataset name.
- ``split``: dataset split.
- ``question``: normalized question text.
- ``gold_answer``: reference answer.
- ``gold_passages``: list of gold passage IDs, each pointing to a matching entry in ``{split}_passages.jsonl`` in the same directory




### {split}_passages.jsonl

{
  "passage_id": "{passage_id}",
  "title": "{source_title}",
  "text": "{passage_text}"
}

Fields
- ``passage_id``: unique passage identifier.
- ``title``: source page title.
- ``text``: sentence-level passage text.

"""





import json
import os
from typing import List, Dict
from src.utils import append_jsonl, clean_text, pid_plus_title
from src.utils import existing_ids, compute_resume_sets









os.makedirs("data/processed_datasets", exist_ok=True)






















########## THESE SHOULD ALL ONLY SEND GOLD PASSAGES TO THE {SPLIT} AND ALL PASSAGES TO {SPLIT}_PASSAGES


# ==== HOTPOT: include ALL passages, but ONLY GOLD IDs in {split}.jsonl ====
def process_hotpotqa(
    split: str,
    file_path: str,
    max_examples: int | None = None,
    overwrite: bool = False,
    *,
    resume: bool = False,
) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if isinstance(max_examples, int):
        examples = examples[:max_examples]

    out_dir = "data/processed_datasets/hotpotqa"
    os.makedirs(out_dir, exist_ok=True)
    qa_path = f"{out_dir}/{split}.jsonl"
    passages_path = f"{out_dir}/{split}_passages.jsonl"

    # --- compute resume sets ---
    done_qids, _ = compute_resume_sets(
        resume=resume,
        out_path=qa_path,
        items=examples,
        get_id=lambda ex, i: ex["_id"],
        phase_label=f"hotpotqa {split} questions",
        id_field="passage_id",
    )

    def iter_pids():
        for ex in examples:
            qid = ex["_id"]
            for title, sents in ex["context"]:
                for i, _ in enumerate(sents):
                    yield pid_plus_title(qid, title, i)

    done_pids, _ = compute_resume_sets(
        resume=resume,
        out_path=passages_path,
        items=iter_pids(),
        get_id=lambda pid, i: pid,
        phase_label=f"hotpotqa {split} passages",
        id_field="question_id",
    )

    for ex in examples:
        qid = ex["_id"]

        # Extract GOLD passage IDs from supporting_facts
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
                    "dataset": "hotpotqa",
                    "split": split,
                    "question": clean_text(ex["question"]),
                    "gold_answer": clean_text(ex.get("answer", "")),
                    "gold_passages": gold_ids,
                },
            )

        # Build and append passage list
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







# ==== 2WIKI: include ALL passages, but ONLY GOLD IDs in {split}.jsonl ====
def process_2wikimultihopqa(
    split: str,
    file_path: str,
    max_examples: int | None = None,
    overwrite: bool = False,
    *,
    resume: bool = False,
) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if isinstance(max_examples, int):
        examples = examples[:max_examples]

    out_dir = "data/processed_datasets/2wikimultihopqa"
    os.makedirs(out_dir, exist_ok=True)
    qa_path = f"{out_dir}/{split}.jsonl"
    passages_path = f"{out_dir}/{split}_passages.jsonl"

    # --- compute resume sets ---
    done_qids, _ = compute_resume_sets(
        resume=resume,
        out_path=qa_path,
        items=examples,
        get_id=lambda ex, i: ex["_id"],
        phase_label=f"2wikimultihopqa {split} questions",
        id_field="passage_id",
    )

    def iter_pids():
        for ex in examples:
            qid = ex["_id"]
            for title, sents in ex["context"]:
                for i, _ in enumerate(sents):
                    yield pid_plus_title(qid, title, i)

    done_pids, _ = compute_resume_sets(
        resume=resume,
        out_path=passages_path,
        items=iter_pids(),
        get_id=lambda pid, i: pid,
        phase_label=f"2wikimultihopqa {split} passages",
        id_field="question_id",
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
                    "dataset": "2wikimultihopqa",
                    "split": split,
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







# ==== MUSIQUE: include ALL paragraphs + gold_answer inline ====

def process_musique(
    split: str,
    file_path: str,
    max_examples: int | None = None,
    overwrite: bool = False,
    *,
    resume: bool = False,
) -> None:
    examples: List[Dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for k, line in enumerate(f):
            if isinstance(max_examples, int) and k >= max_examples:
                break
            examples.append(json.loads(line))

    out_dir = "data/processed_datasets/musique"
    os.makedirs(out_dir, exist_ok=True)
    qa_path = f"{out_dir}/{split}.jsonl"
    passages_path = f"{out_dir}/{split}_passages.jsonl"

    done_qids, _ = compute_resume_sets(
        resume=resume,
        out_path=qa_path,
        items=examples,
        get_id=lambda ex, i: ex["id"],
        phase_label=f"musique {split} questions",
    )

    def iter_pids():
        for ex in examples:
            qid = ex["id"]
            paras = ex.get("paragraphs", [])
            for idx, p in enumerate(paras):
                j = p.get("idx")
                pid = f"{qid}_sent{j}" if j is not None else f"{qid}_sent{idx}"
                yield pid

    done_pids, _ = compute_resume_sets(
        resume=resume,
        out_path=passages_path,
        items=iter_pids(),
        get_id=lambda pid, i: pid,
        phase_label=f"musique {split} passages",
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
                    "dataset": "musique",
                    "split": split,
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


























if __name__ == "__main__":

    RESUME = True
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["dev"]
    MAX_EXAMPLES = 200

    for dataset in DATASETS:
        for split in SPLITS:
            if dataset == "hotpotqa":
                file_path = (
                    "data/raw_datasets/hotpotqa/hotpot_train_v1.1.json"
                    if split == "train"
                    else "data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
                )
                process_hotpotqa(split, file_path, max_examples=MAX_EXAMPLES, resume=RESUME)
            elif dataset == "2wikimultihopqa":
                file_path = f"data/raw_datasets/2wikimultihopqa/{split}.json"
                process_2wikimultihopqa(split, file_path, max_examples=MAX_EXAMPLES, resume=RESUME)
            elif dataset == "musique":
                file_path = f"data/raw_datasets/musique/musique_ans_v1.0_{split}.jsonl"
                process_musique(split, file_path, max_examples=MAX_EXAMPLES, resume=RESUME)





