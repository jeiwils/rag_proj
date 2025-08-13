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

- {split}.jsonl                                       - processed questions with gold passage IDs.
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






# 
# with or without updated IQOQ prompts
# with or without CS-guided IQOQ
# with or without updated algorithm 
# timings for iqoq generation - baseline vs enhanced prompt
# 
# 
# 



# ENTIRE PROJECT STRUCTURE 
# - build and tune on train set (200 ROWS)
# - rebuild and test on dev set (500 ROWS)







# TO DO
# - different folders for the 3 pipelines ??????????????????????????????????????
# - neo4j for final graph build + test traversal




















import json
import os
from src.utils import (
    load_jsonl,
    save_jsonl,
    append_jsonl,
    save_jsonl_safely,
    clean_text,
    pid_plus_title,
)

from typing import List, Dict








os.makedirs("data/processed_datasets", exist_ok=True)






















########## THESE SHOULD ALL ONLY SEND GOLD PASSAGES TO THE {SPLIT} AND ALL PASSAGES TO {SPLIT}_PASSAGES


# ==== HOTPOT: include ALL passages, but ONLY GOLD IDs in {split}.jsonl ====
def process_hotpotqa(split: str, file_path: str, max_examples: int | None = None, overwrite: bool = False) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if isinstance(max_examples, int):
        examples = examples[:max_examples]

    qa, passages = [], []

    for ex in examples:
        qid = ex["_id"]
        ex_all_passage_ids = []

        # Build the full passage list
        for title, sents in ex["context"]:
            for i, sent in enumerate(sents):
                pid = pid_plus_title(qid, title, i)
                ex_all_passage_ids.append(pid)
                passages.append({
                    "passage_id": pid,
                    "title": title,
                    "text": clean_text(sent),
                })

        # Extract GOLD passage IDs from supporting_facts
        gold_ids, seen = [], set()
        for title, idx in ex.get("supporting_facts", []):
            pid = pid_plus_title(qid, title, idx)
            if pid not in seen:
                gold_ids.append(pid)
                seen.add(pid)

        qa.append({
            "question_id": qid,
            "dataset": "hotpotqa",
            "split": split,
            "question": clean_text(ex["question"]),
            "gold_answer": clean_text(ex.get("answer", "")),
            "gold_passages": gold_ids,   
        })

    out_dir = "data/processed_datasets/hotpotqa"
    os.makedirs(out_dir, exist_ok=True)
    save_jsonl_safely(f"{out_dir}/{split}.jsonl", qa, overwrite=overwrite)
    save_jsonl_safely(f"{out_dir}/{split}_passages.jsonl", passages, overwrite=overwrite)






# ==== 2WIKI: include ALL passages, but ONLY GOLD IDs in {split}.jsonl ====
def process_2wikimultihopqa(split: str, file_path: str, max_examples: int | None = None, overwrite: bool = False) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if isinstance(max_examples, int):
        examples = examples[:max_examples]

    qa, passages = [], []

    for ex in examples:
        qid = ex["_id"]
        ex_all_passage_ids = []

        # Build the full passage list
        for title, sents in ex["context"]:
            for i, sent in enumerate(sents):
                pid = pid_plus_title(qid, title, i)
                ex_all_passage_ids.append(pid)
                passages.append({
                    "passage_id": pid,
                    "title": title,
                    "text": clean_text(sent),
                })

        # Extract GOLD passage IDs from supporting_facts
        gold_ids, seen = [], set()
        for title, idx in ex.get("supporting_facts", []):
            pid = pid_plus_title(qid, title, idx)
            if pid not in seen:
                gold_ids.append(pid)
                seen.add(pid)

        qa.append({
            "question_id": qid,
            "dataset": "2wikimultihopqa",
            "split": split,
            "question": clean_text(ex["question"]),
            "gold_answer": clean_text(ex.get("answer", "")),
            "gold_passages": gold_ids,   
        })

    out_dir = "data/processed_datasets/2wikimultihopqa"
    os.makedirs(out_dir, exist_ok=True)
    save_jsonl_safely(f"{out_dir}/{split}.jsonl", qa, overwrite=overwrite)
    save_jsonl_safely(f"{out_dir}/{split}_passages.jsonl", passages, overwrite=overwrite)







# ==== MUSIQUE: include ALL paragraphs + gold_answer inline ====

def process_musique(split: str, file_path: str, max_examples: int | None = None, overwrite: bool = False) -> None:
    qa, passages = [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for k, line in enumerate(f):
            if isinstance(max_examples, int) and k >= max_examples:
                break

            ex = json.loads(line)
            qid = ex["id"]
            paras = ex.get("paragraphs", [])

            ex_all_passage_ids = []
            for p in paras:
                j = p.get("idx")
                pid = f"{qid}_sent{j}" if j is not None else f"{qid}_sent{len(ex_all_passage_ids)}"
                ex_all_passage_ids.append(pid)
                passages.append({
                    "passage_id": pid,
                    "title": p.get("title", ""),
                    "text": clean_text(p.get("paragraph_text", "")),
                })

            # GOLD paragraphs are those with is_supporting == True
            gold_ids, seen = [], set()
            for p in paras:
                if p.get("is_supporting"):
                    j = p.get("idx")
                    pid = f"{qid}_sent{j}" if j is not None else f"{qid}_sent{len(gold_ids)}"
                    if pid not in seen:
                        gold_ids.append(pid)
                        seen.add(pid)

            qa.append({
                "question_id": qid,
                "dataset": "musique",
                "split": split,
                "question": clean_text(ex.get("question", "")),
                "gold_answer": clean_text(ex.get("answer", "")),
                "gold_passages": gold_ids,   
            })

    out_dir = "data/processed_datasets/musique"
    os.makedirs(out_dir, exist_ok=True)
    save_jsonl_safely(f"{out_dir}/{split}.jsonl", qa, overwrite=overwrite)
    save_jsonl_safely(f"{out_dir}/{split}_passages.jsonl", passages, overwrite=overwrite)


























if __name__ == "__main__":

    MAX_EXAMPLES = 200


    for split in ["dev"]: # 


        # Hotpot uses different file names for train/dev
        hotpot_path = (
            "data/raw_datasets/hotpotqa/hotpot_train_v1.1.json"
            if split == "train"
            else "data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
        )
        process_hotpotqa(split, hotpot_path, max_examples=MAX_EXAMPLES)

        # 2Wiki: files are {split}.json
        process_2wikimultihopqa(split, f"data/raw_datasets/2wikimultihopqa/{split}.json", max_examples=MAX_EXAMPLES)

        # Musique: files are musique_ans_v1.0_{split}.jsonl
        process_musique(split, f"data/raw_datasets/musique/musique_ans_v1.0_{split}.jsonl", max_examples=MAX_EXAMPLES)







