"""





ENTIRE PROJECT FILE STRUCTURE: 
raw_datasets
-	All raw, uncleaned downloaded datasets 
prompts
-	All prompts


processed_datasets – all just cleaned dataset stuff
-	Data set name / text (train or train_passages, dev or dev_passages)



models – all file manipulation/generation done by models
-	Model name / data set name / hoprag mode
metrics 
-	Tests done on independent scripts (checking CS distribution etc) 






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



"""



















import re
import json
import unicodedata
import os

from typing import List, Dict








os.makedirs("data/processed_datasets", exist_ok=True)









"""







PURPOSE:
- 







#   #   #   #   # INPUT: 



data/raw_datasets/{dataset}


- 2wikimultihopqa/{split}


- hotpotqa
-- hotpot_dev_fullwiki_v1.json 
-- hotpot_train_v1.1.json


- musique
-- musique_ans_v1.0_{split}







#   #   #   #   # OUTPUT: 




data/processed_datasets/{dataset}/{split}.jsonl
data/processed_datasets/{dataset}/{split}_passages.jsonl





{ # data/processed_datasets/hotpotqa/{split}.jsonl
  "question_id": "5a7a06935542990198eaf050",
  "dataset": "hotpotqa",
  "split": "{split}",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "gold_answer": "Arthur's Magazine",
  "passages": [
    "5a7a06935542990198eaf050__arthur_s_magazine_sent0",
    "5a7a06935542990198eaf050__first_for_women_sent0"
  ]
}

# data/processed_datasets/hotpotqa/{split}_passages.jsonl
{"passage_id":"5a7a06935542990198eaf050__arthur_s_magazine_sent0","title":"Arthur's Magazine","text":"Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century."}
{"passage_id":"5a7a06935542990198eaf050__first_for_women_sent0","title":"First for Women","text":"First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989."}








{ # data/processed_datasets/2wikimultihopqa/{split}.jsonl
  "question_id": "13f5ad2c088c11ebbd6fac1f6bf848b6",
  "dataset": "2wikimultihopqa",
  "split": "{split}",
  "question": "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?",
  "gold_answer": "no",
  "passages": [
    "13f5ad2c088c11ebbd6fac1f6bf848b6__move_1970_film_sent0",
    "13f5ad2c088c11ebbd6fac1f6bf848b6__m_diterran_e_1963_film_sent0",
    "13f5ad2c088c11ebbd6fac1f6bf848b6__stuart_rosenberg_sent0",
    "13f5ad2c088c11ebbd6fac1f6bf848b6__jean_daniel_pollet_sent0"
  ]
}

# data/processed_datasets/2wikimultihopqa/{split}_passages.jsonl
{"passage_id":"13f5ad2c088c11ebbd6fac1f6bf848b6__move_1970_film_sent0","title":"Move (1970 film)","text":"Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg."}
{"passage_id":"13f5ad2c088c11ebbd6fac1f6bf848b6__m_diterran_e_1963_film_sent0","title":"Méditerranée (1963 film)","text":"Méditerranée is a 1963 French experimental film directed by Jean-Daniel Pollet with assistance from Volker Schlöndorff."}
{"passage_id":"13f5ad2c088c11ebbd6fac1f6bf848b6__stuart_rosenberg_sent0","title":"Stuart Rosenberg","text":"Stuart Rosenberg (1927–2007) was an American film and television director whose motion pictures include Cool Hand Luke (1967) and The Amityville Horror (1979)."}
{"passage_id":"13f5ad2c088c11ebbd6fac1f6bf848b6__jean_daniel_pollet_sent0","title":"Jean-Daniel Pollet","text":"Jean-Daniel Pollet (1936–2004) was a French film director and screenwriter who was most active in the 1960s and 1970s."}







{ # data/processed_datasets/musique/{split}.jsonl
  "question_id": "2hop__482757_12019",
  "dataset": "musique",
  "split": "{split}",
  "question": "When was the institute that owned The Collegian founded?",
  "gold_answer": "1960",
  "passages": [
    "2hop__482757_12019_sent5",
    "2hop__482757_12019_sent9"
  ]
}

# data/processed_datasets/musique/{split}_passages.jsonl
{"passage_id":"2hop__482757_12019_sent5","title":"The Collegian (Houston Baptist University)","text":"The Collegian is the bi-weekly official student publication of Houston Baptist University in Houston, Texas. It was founded in 1963 as a newsletter, and adopted the newspaper format in 1990."}
{"passage_id":"2hop__482757_12019_sent9","title":"Houston","text":"Houston Baptist University, affiliated with the Baptist General Convention of Texas, offers bachelor's and graduate degrees. It was founded in 1960 and is located in the Sharpstown area in Southwest Houston."}






"""












############################
# 1. JSONL and General I/O
############################

def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(path: str, data: List[Dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # create dirs if needed
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")












def _next_version_path(path: str) -> str:
    """If foo.jsonl exists, return foo.v1.jsonl, foo.v2.jsonl, ..."""
    base, ext = os.path.splitext(path)
    i = 1
    candidate = f"{base}.v{i}{ext}"
    while os.path.exists(candidate):
        i += 1
        candidate = f"{base}.v{i}{ext}"
    return candidate


def save_jsonl_safely(path: str, data: List[Dict], overwrite: bool = False) -> str:
    """Write jsonl; if path exists and overwrite=False, write to a versioned filename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out_path = path
    if os.path.exists(path) and not overwrite:
        out_path = _next_version_path(path)
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return out_path
















def clean_text(
        text: str 
        ) -> str:
    """
    for dataset text

    Normalize whitespace, remove HTML/markdown/wiki markup.
    For readable text and dense representation.
    
    """
    # normalise whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # remove leftover HTML stuff 
    text = re.sub(r'\[\[.*?\]\]', '', text)       
    text = re.sub(r'\[.*?\]', '', text)           
    text = re.sub(r'={2,}.*?={2,}', '', text)     

    # remove markdown-style formatting 
    text = unicodedata.normalize('NFKC', text)
    return text
























def pid_plus_title(qid: str, title: str, sent_idx: int) -> str:
    # keep your current style; just consistent slugging
    if not title:
        safe = "no_title"
    else:
        safe = title.lower()
        safe = re.sub(r'[^a-z0-9]+', '_', safe)
        safe = re.sub(r'_+', '_', safe).strip('_') or "no_title"
    return f"{qid}__{safe}_sent{sent_idx}"





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
            "passages": gold_ids,   # ONLY gold passages here
        })

    out_dir = "data/processed_datasets/hotpotqa"
    os.makedirs(out_dir, exist_ok=True)
    save_jsonl_safely(f"{out_dir}/{split}.jsonl", qa, overwrite=overwrite)
    save_jsonl_safely(f"{out_dir}/{split}_passages.jsonl", passages, overwrite=overwrite)







# ==== 2WIKI: include ALL passages + gold_answer inline ====
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
            "passages": gold_ids,   # ONLY gold passages here
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
                "passages": gold_ids,   # ONLY gold passages here
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







