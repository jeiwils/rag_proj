


from typing import Dict, List
import json

# pull these from wherever you defined them
# (update the module path to match your project)
from ..a1_dataset_processing import load_jsonl, save_jsonl_safely, pid_plus_title






# ==== Utility: rewrite an existing processed {split}.jsonl with GOLD passages (safe save) ====
def update_processed_with_gold_passages(
    dataset: str,
    split: str,
    raw_path: str,
    processed_path: str,
    overwrite: bool = False,
    add_as_field: bool = False,
) -> str:
    """
    Reads RAW dataset to extract gold passage IDs per question, then updates a processed
    {split}.jsonl by either REPLACING the 'passages' field (default) or adding a
    'gold_passages' field (if add_as_field=True). Writes safely (versioned) and returns
    the output path actually written.
    """
    # 1) Build gold map from RAW
    gold_map: Dict[str, List[str]] = {}

    def _add_gold(qid: str, pid_list: List[str]) -> None:
        gold_map[qid] = list(dict.fromkeys(pid_list))  # de-dupe, preserve order

    if dataset == "musique":
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                qid = ex["id"]
                ids = []
                for p in ex.get("paragraphs", []):
                    if p.get("is_supporting"):
                        j = p.get("idx")
                        pid = f"{qid}_sent{j}" if j is not None else f"{qid}_sent{len(ids)}"
                        ids.append(pid)
                _add_gold(qid, ids)



    elif dataset in {"hotpotqa", "2wikimultihopqa"}:
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for ex in raw:
            qid = ex["_id"]
            ids = []
            for title, idx in ex.get("supporting_facts", []):
                ids.append(pid_plus_title(qid, title, idx))
            _add_gold(qid, ids)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")




    # 2) Load existing processed QA and update
    qa = load_jsonl(processed_path)
    updated = []
    for row in qa:
        qid = row.get("question_id") or row.get("id")
        if qid in gold_map:
            if add_as_field:
                row["gold_passages"] = gold_map[qid]
            else:
                row["passages"] = gold_map[qid]
        updated.append(row)




    # 3) Safe save (versioned if target exists and overwrite=False)
    out_path = save_jsonl_safely(processed_path, updated, overwrite=overwrite)
    return out_path






from pathlib import Path

DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
SPLITS = ["train"] #, "dev"]

RAW_BASE = Path("data/raw_datasets")
PROC_BASE = Path("data/processed_datasets")

def raw_path_for(dataset: str, split: str) -> str:
    if dataset == "musique":
        # data/raw_datasets/musique/musique_ans_v1.0_{split}.jsonl
        return str(RAW_BASE / "musique" / f"musique_ans_v1.0_{split}.jsonl")

    if dataset == "hotpotqa":
        # data/raw_datasets/hotpotqa/hotpot_train_v1.1.json or hotpot_dev_fullwiki_v1.json
        fname = "hotpot_train_v1.1.json" if split == "train" else "hotpot_dev_fullwiki_v1.json"
        return str(RAW_BASE / "hotpotqa" / fname)

    if dataset == "2wikimultihopqa":
        # data/raw_datasets/2wikimultihopqa/{split}.json  (fallback to glob if needed)
        p = RAW_BASE / "2wikimultihopqa" / f"{split}.json"
        if p.exists():
            return str(p)
        # be forgiving: accept .jsonl or any single json-like file in the split dir
        for ext in (".jsonl", ".json"):
            q = RAW_BASE / "2wikimultihopqa" / f"{split}{ext}"
            if q.exists():
                return str(q)
        globs = list((RAW_BASE / "2wikimultihopqa").glob(f"*{split}*.*"))
        if globs:
            return str(globs[0])
        raise FileNotFoundError(f"Could not resolve raw path for 2wikimultihopqa/{split}")

    raise ValueError(f"Unknown dataset: {dataset}")

for ds in DATASETS:
    for split in SPLITS:
        raw_path = raw_path_for(ds, split)
        processed_path = str(PROC_BASE / ds / f"{split}.jsonl")
        out_path = update_processed_with_gold_passages(
            dataset=ds,
            split=split,
            raw_path=raw_path,
            processed_path=processed_path,
            overwrite=False,     # set True to replace in-place
            add_as_field=False   # True => keep 'passages' and add 'gold_passages'
        )
        print(f"[{ds} | {split}] -> {out_path}")
