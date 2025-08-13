
# TO DO
# - different folders for the 3 pipelines
# - sort datasets
# - sweep threshold, testing stuff 
# - neo4j for final graph build + test traversal


# - sort out multiprocessing option for smaller models: cs, iqoq, traversal (two model servers running, each processing their own dataset)
# - get llm stuff running on cuda - on servers etc
# I can define the algorithms externally, then send them to the traversal function?



"""

######################################################### pipelines



1) baseline dense retriever
- only retrieve top-k passages from faiss index



2) standard Hop-RAG
- no cs - iqoq generation without cs
- different traveral algorithm - no repeated vertex visits
- I think everything else is the same? 



3) HopRag+ 
- cs - iqoq generation with cs ratio
- different traveral algorithm - repeated vertex visits, but no repeated edge visits ---- could/should I factor cs into the traversal algorithm?
- ????? 



"""




import requests
import os
import re
import json
import unicodedata
import numpy as np
import faiss
import networkx as nx # for graph prototyping/tuning (degree stats, edge filtering, traversal debugging...)
import neo4j # for visualisation and final test queries
import string
import torch

from collections import defaultdict
from paddlenlp import Taskflow
from typing import List, Dict, Optional, Tuple, Callable
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




# Load BGE embedding model
bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")  # 768-dim embeddings

#What's this???
lac = Taskflow("lexical_analysis")



CS_PROMPT = Path("data/prompts/7CS_prompt.txt").read_text()

IQ_prompt = Path("data/prompts/7b_IQ_prompt.txt").read_text()
OQ_prompt = Path("data/prompts/7b_OQ_prompt.txt").read_text()

traversal_prompt = Path("data/prompts/traversal_prompt.txt").read_text()




ACTIVE_MODEL_NAME = "Qwen-7b"
MODEL_SERVERS = [
    "http://localhost:8000",  
    "http://localhost:8001",  
]

MAX_TOKENS = {
    "iq_generation": 256,
    "oq_generation": 256,
    "edge_selection": 64,
    "answer_generation": 128
}

###########################################################################################################
# TUNING PARAMS
###########################################################################################################

# GRAPH CONSTRUCTION TUNING

SIM_THRESHOLD = 0.65
MAX_NEIGHBOURS = 5
ALPHA = 0.5

params = { 
    "sim_threshold": SIM_THRESHOLD,
    "max_neighbors": MAX_NEIGHBOURS,
    "alpha": ALPHA
}

# TRAVERSAL TUNING

TOP_K_SEED_PASSAGES = 50 
NUMBER_HOPS = 2

# ANSWER GENERATION

TOP_K_ANSWER_PASSAGES = 5 


"""
downloading datasets? (train, dev, and test)
- hotpotQA, musique, 2wikimultihop


"""




############################
# 1. JSONL and General I/O
############################

def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')







"""

SORT DATASETS!!!!

"""















################################################################################################################
# TEXT AND FILE PREPROCESSING AND GENERATION 
################################################################################################################


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








"""


## CODE FOR CLEANING RAW DATASET STUFF, LIMITING TO 1000, PUTTING IN THIS ORDER... APPLYING CLEAN_TEXT()???
## I guess I need some function for putting the datasets into this order as well?


#
#
# { # {dataset}_train_source.jsonl
#   "question_id": "hotpot_001",
#   "dataset": "hotpotqa",
#   "split": "train",
#   "question": "Who was the maternal grandfather of Abraham Lincoln?",
#   "gold_answer": "James Hanks", # to see how similar the output of the hyperparam testing is – metric????
#   "supporting_passages": ["hotpot_001_sent1", "hotpot_002_sent4"] # to see how accurately appropriate passage
# }
#
#
#
# { # {dataset}_dev.jsonl
#   "question_id": "hotpot_001",
#   "dataset": "hotpotqa",
#   "split": "dev",
#   "question": "Who was the maternal grandfather of Abraham Lincoln?",
#   "gold_answer": "James Hanks", # to see how similar the output of the hyperparam testing is – metric????
#   "supporting_passages": ["hotpot_001_sent1", "hotpot_002_sent4"] # to see how accurately appropriate passage
# }
#
#
#
# { # {dataset}_test.jsonl
#   "question_id": "hotpot_321",
#   "dataset": "hotpotqa",
#   "split": "test",
#   "question": "What is the capital of the country with the highest GDP in South America?"
# }
#
#



"""



def query_llm(
        prompt: str, 
        server_url: str, 
        max_tokens: int = 5, 
        temperature: float = 0.0 ################# i don't think I need to do anything with temperature? what's the default? 
        ) -> str:
    """
    Send a prompt to the local LLM server and return raw string output.
    """
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(f"{server_url}/completion", json=payload)
    resp.raise_for_status()
    return resp.json().get("content", "").strip()



###



def get_conditioned_score(
        entry: dict, 
        cs_prompt_template: str, 
        server_url: str, 
        max_tokens: int = 30
        ) -> dict:
    """
    Sends a passage to the LLM using the CS prompt and adds a 'conditioned_score' (float) to the entry.
    Returns None if scoring fails or parsing fails.
    """
    passage_text = entry["text"]
    cs_prompt_filled = cs_prompt_template.replace("{{PASSAGE}}", passage_text)

    try:
        response = query_llm(cs_prompt_filled, server_url, max_tokens=max_tokens)

        # Extract score from response 
        match = re.search(r"(?:CS|Conditioned Score):\s*([0-1](?:\.00|\.25|\.50|\.75)?)", response) ############ WILL TUNE THIS ACCORDING TO WHAT THE LLM OUTPUTS
        if match:
            score = float(match.group(1))
        else:
            raise ValueError(f"No valid CS score found in response: {response}")
    
    except Exception as e:
        print(f"[CS ERROR] Failed for {entry.get('passage_id', '?')}: {e}")
        return None

    if not (0.0 <= score <= 1.0):  # sanity check
        print(f"[CS INVALID] Score out of range for {entry.get('passage_id', '?')}: {score}")
        return None

    entry["conditioned_score"] = score
    return entry



"""



entries = load_jsonl("data/hotpotqa_train.jsonl")

output = []

for entry in entries:
    scored = get_conditioned_score(entry, CS_prompt, MODEL_SERVERS[0])
    if scored:
        output.append(scored)

# Save final version with conditioned scores included
save_jsonl(output, "data/hotpotqa_train.jsonl")

#
#
# { # {dataset}_train.jsonl
#   "passage_id": "hotpot_001_sent1",
#   "dataset": "hotpotqa",
#   "split": "train",
#   "text": "Thomas Lincoln was the father of Abraham Lincoln.",
#   "conditioned_score": 0.25 # file is updated to add these
# }
#
#



"""



def iqoq_ratio(
        cs: float,
        qmin: int = 6,
        qmax: int = 9,
        epsilon_iq: int = 2,
        epsilon_oq: int = 2
    ) -> tuple[int, int, int]:
    """
    Compute total number of questions and IQ/OQ split based on conditioned score (CS).

    - Low CS (grounded) → more IQs (incoming), fewer OQs
    - High CS (speculative) → more OQs (outgoing), fewer IQs
    - U-shaped curve controls total questions
    - Enforces minimum 2 IQs, 2 OQs
    """
    # Step 1: U-shaped curve for total questions (CS = 0.5 → qmin)
    q_total = qmin + int(round((qmax - qmin) * (4 * (cs - 0.5)**2)))

    # Step 2: Inverted split: low CS → more IQs; high CS → more OQs
    iq_ratio = 1 - cs
    oq_ratio = cs

    num_iq = int(round(q_total * iq_ratio))
    num_oq = q_total - num_iq

    # Step 3: Enforce minimums
    num_iq = max(num_iq, epsilon_iq)
    num_oq = max(num_oq, epsilon_oq)

    # Step 4: Clamp back to qmax if over
    total = num_iq + num_oq
    if total > qmax:
        excess = total - qmax
        if num_iq > num_oq and num_iq > epsilon_iq:
            reducible = num_iq - epsilon_iq
            reduction = min(excess, reducible)
            num_iq -= reduction
        elif num_oq > epsilon_oq:
            reducible = num_oq - epsilon_oq
            reduction = min(excess, reducible)
            num_oq -= reduction

    return num_iq + num_oq, num_iq, num_oq




def generate_iqoq(
        entry: dict,
        iq_prompt_template: str,
        oq_prompt_template: str,
        server_url: str,
        max_tokens: int = 256,
        conditioned_score: float = None,  # optional CS input
        use_ratio: bool = False           # toggle use of CS-based ratio
    ) -> dict:
    """
    Adds 'IQs' and 'OQs' to an entry by querying the LLM.
    If use_ratio=True and conditioned_score is provided, adjusts number of questions using iqoq_ratio().
    Otherwise defaults to IQ=2, OQ=4.
    """
    passage_text = entry["text"]

    # --- STEP 1: Determine number of IQs and OQs ---
    if use_ratio and conditioned_score is not None:
        _, num_iq, num_oq = iqoq_ratio(conditioned_score)
    else:  # Hop-RAG defaults
        num_iq = 2 
        num_oq = 4  

    # --- STEP 2: Fill prompt templates ---
    iq_prompt_filled = (
        iq_prompt_template
        .replace("{{PASSAGE}}", passage_text)
        .replace("{NUM_QUESTIONS}", str(num_iq))
    )
    oq_prompt_filled = (
        oq_prompt_template
        .replace("{{PASSAGE}}", passage_text)
        .replace("{NUM_QUESTIONS}", str(num_oq))
    )

    # --- STEP 3: Call LLM ---
    try:
        iq_response = query_llm(iq_prompt_filled, server_url, max_tokens=max_tokens)
        oq_response = query_llm(oq_prompt_filled, server_url, max_tokens=max_tokens)
    except Exception as e:
        print(f"[ERROR] LLM failed for {entry.get('passage_id', '?')}: {e}")
        return None

    entry["IQs"] = [q for q in iq_response.split("\n") if q.strip()] # this assumes that all the questions are on separate lines
    entry["OQs"] = [q for q in oq_response.split("\n") if q.strip()]
    entry["generation_model"] = ACTIVE_MODEL_NAME
    entry["split"] = entry.get("split", "train")
    entry["dataset"] = entry.get("dataset", "hotpotqa")
    entry["num_iq"] = num_iq
    entry["num_oq"] = num_oq
    entry["cs_used"] = conditioned_score if use_ratio else None

    return entry




"""



data = load_jsonl("data/hotpotqa_train_source.jsonl")
output = []

output_no_ratio = []
output_with_ratio = []

for entry in data:
    # 1. Without CS ratio (baseline: IQ=2, OQ=4)
    updated_no_ratio = generate_iqoq(
        entry.copy(),  # copy to avoid mutation
        IQ_prompt,
        OQ_prompt,
        MODEL_SERVERS[0],
        use_ratio=False
    )
    if updated_no_ratio:
        output_no_ratio.append(updated_no_ratio)

    # 2. With CS ratio (CS-based IQ/OQ split)
    cs = entry.get("conditioned_score")
    updated_with_ratio = generate_iqoq(
        entry.copy(),
        IQ_prompt,
        OQ_prompt,
        MODEL_SERVERS[0],
        conditioned_score=cs,
        use_ratio=True
    )
    if updated_with_ratio:
        output_with_ratio.append(updated_with_ratio)


save_jsonl(output_no_ratio, "data/hotpotqa_train_iqoq_baseline.jsonl")
save_jsonl(output_with_ratio, "data/hotpotqa_train_iqoq_with_ratio.jsonl")


#
#
# { # data/hotpotqa_train_iqoq_baseline.jsonl
#   "passage_id": "hotpot_001_sent1",
#   "text": "Thomas Lincoln was the father of Abraham Lincoln.",
#   "conditioned_score": 0.25,
#   "IQs": [
#     "1. Who was Abraham Lincoln's father?",
#     "2. What is the relationship between Thomas Lincoln and Abraham Lincoln?"
#   ],
#   "OQs": [
#     "1. What role did Thomas Lincoln play in Abraham Lincoln's early life?",
#     "2. What was Thomas Lincoln's profession?",
#     "3. How did Thomas Lincoln influence Abraham Lincoln's values?",
#     "4. Where did Thomas Lincoln live during his lifetime?"
#   ],
#   "generation_model": "Qwen-7b",
#   "split": "train",
#   "dataset": "hotpotqa",
#   "num_iq": 2,
#   "num_oq": 4,
#   "cs_used": null
# }
#
#
#
# { # data/hotpotqa_train_iqoq_with_ratio.jsonl
#   "passage_id": "hotpot_001_sent1",
#   "text": "Thomas Lincoln was the father of Abraham Lincoln.",
#   "conditioned_score": 0.25,
#   "IQs": [
#     "1. Who was the father of Abraham Lincoln?",
#     "2. Who is Thomas Lincoln?",
#     "3. What is the relationship between Thomas Lincoln and Abraham Lincoln?",
#     "4. Was Thomas Lincoln Abraham Lincoln's biological father?",
#     "5. Does the passage state the birth year of Thomas Lincoln?"
#   ],
#   "OQs": [
#     "1. What else is known about Thomas Lincoln's life?",
#     "2. How might Thomas Lincoln have influenced Abraham Lincoln's upbringing?"
#   ],
#   "generation_model": "Qwen-7b",
#   "split": "train",
#   "dataset": "hotpotqa",
#   "num_iq": 5,
#   "num_oq": 2,
#   "cs_used": 0.25
# }
#
#



"""



def clean_iqoq(questions: list[str]) -> list[str]:
    """
    
    to make sure llm returns proper questions


    """
    cleaned = []

    for q in questions:
        q = q.strip()
        q = re.sub(r"^\d+[\.\)]\s*", "", q)      # remove numbering (e.g., '1. ...')
        q = re.sub(r"^[-*]\s*", "", q)           # remove bullets ('- ', '* ')
        if not q.endswith("?"):
            continue                             # skip if it’s not a question - will need to check this during debugging
        if len(q) < 5:
            continue                             # skip tiny junk 
        if q.lower() in {"n/a", "none", "no question generated"}:
            continue
        cleaned.append(q)

    return cleaned



"""



raw = load_jsonl("data/hotpotqa_train.jsonl")
cleaned = []
debug = []

for entry in raw:
    raw_IQs = entry.get("IQs", [])
    raw_OQs = entry.get("OQs", [])

    entry["IQs"] = clean_iqoq(raw_IQs)
    entry["OQs"] = clean_iqoq(raw_OQs)

    # Re-set dataset/split explicitly just to be safe 
    entry["dataset"] = entry.get("dataset", "hotpotqa")
    entry["split"] = entry.get("split", "train")

    cleaned.append(entry)

    debug.append({
        "passage_id": entry["passage_id"],
        "raw_IQs": raw_IQs,
        "clean_IQs": entry["IQs"],
        "raw_OQs": raw_OQs,
        "clean_OQs": entry["OQs"],
        "generation_model": entry.get("generation_model", "unknown"),
        "dataset": entry.get("dataset", "hotpotqa"),
        "split": entry.get("split", "train")
    })

save_jsonl(cleaned, "data/hotpotqa_train.jsonl")
save_jsonl(debug, "data/hotpotqa_train_iqoq_debug.jsonl") 


#
#
# { # {dataset}_train_iqoq_debug.jsonl 
#   "passage_id": "hotpot_001_sent1",
#   "dataset": "hotpotqa",
#   "raw_IQs": ["1. Who is Abraham Lincoln's dad?", "No question generated"],
#   "clean_IQs": ["Who is Abraham Lincoln's dad?"],
#   "raw_OQs": ["- Who was Abraham Lincoln's mother?", "none"],
#   "clean_OQs": ["Who was Abraham Lincoln's mother?"],
#   "text": "Thomas Lincoln was the father of Abraham Lincoln.",
#   "generation_model": "Qwen-7b"
# }
#
# 
# 
# {  # {dataset}_train.jsonl
#   "passage_id": "hotpot_001_sent1",
#   "dataset": "hotpotqa",
#   "split": "train",
#   "text": "Thomas Lincoln was the father of Abraham Lincoln.",
#   "conditioned_score": 0.25,
#   "IQs": ["Who was Abraham Lincoln's father?", "Who is Thomas Lincoln?"], # file is updated with these - cleaned iq/oq
#   "OQs": ["Who was the maternal grandfather of Abraham Lincoln?"],  # file is updated with these - cleaned iq/oq
#   "generation_model": "Qwen-7b"
# }
#
#



"""




def extract_keywords(text: str) -> list:
    keywords = set()
    results = lac(text)
    
    for result in results:
        for word, tag in zip(result['word'], result['tag']):
            if tag.isupper(): # no filtering for any particular type - as in hopRAG???
                clean_word = word.lower().replace(" ", "_")
                if len(clean_word) > 1:
                    keywords.add(clean_word)
    
    return list(keywords)



def extract_all_keywords(entry: dict) -> dict:
    """
    Adds 'keywords_IQ', 'keywords_OQ', and 'keywords_text' to entry.
    Assumes 'IQs', 'OQs', and 'text' exist.
    """
    entry["keywords_IQ"] = [extract_keywords(q) for q in entry.get("IQs", [])]
    entry["keywords_OQ"] = [extract_keywords(q) for q in entry.get("OQs", [])]
    entry["keywords_text"] = extract_keywords(entry.get("text", ""))
    return entry



"""



# Load cleaned IQ/OQ entries
cleaned_entries = load_jsonl("data/hotpotqa_train.jsonl")

# Extract keywords
keywords = [extract_all_keywords(entry) for entry in cleaned_entries]

save_jsonl(keywords, "data/hotpotqa_train.jsonl")



{ # {dataset}_train.jsonl 
  "passage_id": "hotpot_001_sent1",
  "dataset": "hotpotqa",
  "split": "train",
  "text": "Thomas Lincoln was the father of Abraham Lincoln.",
  "keywords_text": ["thomas_lincoln", "abraham_lincoln", "father"],
  "conditioned_score": 0.25,
  "IQs": ["Who was Abraham Lincoln's father?", "Who is Thomas Lincoln?"],
  "OQs": ["Who was the maternal grandfather of Abraham Lincoln?", "Who were Thomas Lincoln's parents?"],
  "keywords_IQ": [["abraham_lincoln", "father"], ["thomas_lincoln"]],
  "keywords_OQ": [["abraham_lincoln", "maternal_grandfather"], ["thomas_lincoln", "parents"]],
  "generation_model": "Qwen-7b"
}



"""



def explode_passages(
        master_path: str, 
        output_path: str
        ):
    """
    needs CLEANED preprocessed text
    """
    master_data = load_jsonl(master_path)

    passages = []
    for entry in master_data:
        passage = {
            "passage_id": entry["passage_id"],
            "text": entry["text"],
            "conditioned_score": entry.get("conditioned_score", 0.0),
            "keywords": entry.get("keywords_text", [])
        }
        passages.append(passage)

    save_jsonl(passages, output_path)



def explode_iqoq(
        master_path: str, 
        output_path: str
        ):
    """
    needs CLEANED preprocessed text
    """
    master_data = load_jsonl(master_path)
    iqoq = []

    for entry in master_data:
        for i, q in enumerate(entry["IQs"]):
            iqoq.append({
                "parent_passage_id": entry["passage_id"],
                "iqoq_id": f"{entry['passage_id']}_iq{i}",
                "type": "IQ",
                "text": q,
                "keywords": entry["keywords_IQ"][i]
            })

        for i, q in enumerate(entry["OQs"]):
            iqoq.append({
                "parent_passage_id": entry["passage_id"],
                "iqoq_id": f"{entry['passage_id']}_oq{i}",
                "type": "OQ",
                "text": q,
                "keywords": entry["keywords_OQ"][i]
            })

    save_jsonl(iqoq, output_path)



"""



explode_passages(
    master_path="train/hotpot_train_source.jsonl",
    output_path="train/hotpot_passages.jsonl"
)

explode_iqoq(
    master_path="train/hotpot_train_source.jsonl",
    output_path="train/hotpot_iqoq.jsonl"
)


#
# { {dataset}_iqoq.jsonl
#   "parent_passage_id": "hotpot_001_sent1", # to speed up file linking
#   "iqoq_id": "hotpot_001_sent1_iq0",
#   "type": "IQ", # the edge formation thing looks for these rows with type=IQ right? 
#   "text": "Who was Abraham Lincoln's father?", # only kept for debugging I guess?
#   "keywords": ["abraham_lincoln", "father"] 
# }
#
#
#
# { {dataset}_passages.jsonl
#   "passage_id": "hotpot_001_sent1", # why?
#   "text": "Thomas Lincoln was the father of Abraham Lincoln.", 
#   "conditioned_score": 0.25,      	############################################################################################ why?
#   "keywords": ["thomas_lincoln", "abraham_lincoln", "father"] 
# }
#



"""



################################################################################################################
# SPARSE AND DENSE REPRESENTATIONS
################################################################################################################



def embed_and_save(input_jsonl, output_npy, output_jsonl, model, text_key):
    """

    for graph embeddings
    and dev set checks 

    Generate normalized embeddings for FAISS indexing.
    - Adds vec_id to each entry for FAISS alignment.
    - Saves embeddings to .npy and JSONL metadata with vec_id.
    - Raises an error if text_key is not provided.
    """
    if not text_key:
        raise ValueError("You must provide a valid text_key (e.g., 'text' or 'question').")

    data, embeddings = [], []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            entry = json.loads(line)
            text = entry[text_key]  # will raise KeyError if the key is missing
            vec = model.encode(text, normalize_embeddings=True)

            entry["vec_id"] = idx
            embeddings.append(vec)
            data.append(entry)

    embeddings = np.array(embeddings, dtype="float32")
    np.save(output_npy, embeddings)

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for d in data:
            f_out.write(json.dumps(d) + "\n")

    print(f"[Embeddings] Saved {len(data)} vectors to {output_npy} and updated JSONL {output_jsonl}")
    return embeddings



"""



embed_and_save(
    input_jsonl="train/hotpot_passages.jsonl",
    output_npy="train/hotpot_passages_emb.npy",
    output_jsonl="train/hotpot_passages.jsonl",  # updates with vec_id
    model=bge_model,
    text_key="text"
)

embed_and_save(
    input_jsonl="train/hotpot_iqoq.jsonl",
    output_npy="train/hotpot_iqoq_emb.npy",
    output_jsonl="train/hotpot_iqoq.jsonl",
    model=bge_model,
    text_key="text"
)

embed_and_save(
    input_jsonl="dev/hotpot_dev_metadata.jsonl",
    output_npy="dev/hotpot_dev_emb.npy",
    output_jsonl="dev/hotpot_dev_metadata.jsonl",
    model=bge_model,
    text_key="question"
)

passages_path = "train/hotpot_passages.jsonl"
iqoq_path = "train/hotpot_iqoq.jsonl"
dev_path = "train/hotpot_dev.jsonl"

passages_emb_path = "train/hotpot_passages_emb.npy"
iqoq_emb_path = "train/hotpot_iqoq_emb.npy"
dev_emb_path = "train/hotpot_dev_emb.npy"


#
#
# { # {dataset}_passages.jsonl
#   "passage_id": "hotpot_001_sent1",
#   "text": "Thomas Lincoln was the father of Abraham Lincoln.", 
#   "conditioned_score": 0.25,      	
#   "keywords": ["thomas_lincoln", "abraham_lincoln", "father"],
#   "vec_id": 3 # reference for both FAISS and embs.npy 
# }
# 
# [ # {dataset}_passages_emb.npy 
#   [0.031, 0.512, -0.127, ...],  # vec_id 0 # Row i in .npy aligns with the vec_id field in its JSONL file.
#   [0.104, 0.301,  0.215, ...],  # vec_id 1
#   ...
# ]
#
#
#
# { # {dataset}_iqoq.jsonl
#   "parent_passage_id": "hotpot_001_sent1",
#   "iqoq_id": "hotpot_001_sent1_iq0",   
#   "type": "IQ",
#   "text": "Who was Abraham Lincoln's father?",
#   "keywords": ["abraham_lincoln", "father"].
#   "vec_id": 6 # reference for both FAISS and embeddings.npy
# }
#
# [ # {dataset}_iqoq_emb.npy
#   [0.031, 0.512, -0.127, ...],  # vec_id 0 # Row i in .npy aligns with the vec_id field in its JSONL file.
#   [0.104, 0.301,  0.215, ...],  # vec_id 1
#   ...
# ]
#
#
#
# { # {dataset}_dev_metadata.jsonl
#   "question_id": "hotpot_001",
#   "dataset": "hotpotqa",
#   "split": "dev",
#   "question": "Who was the maternal grandfather of Abraham Lincoln?",
#   "gold_answer": "James Hanks", # to see how similar the output of the hyperparam testing is 
#   "supporting_passages": ["hotpot_001_sent1", "hotpot_002_sent4"] # to see how accurately appropriate passage
#   "vec_id": 9 # reference for both FAISS and embeddings.npy
# }
#
# [ {dataset}_dev_emb.npy # i need to compare to gold_answer, for FAISS question-passage retrieval evaluation
#     [0.031, 0.512, -0.127, ...]  # vec_id 0 → first question
#     [0.104, 0.301,  0.215, ...]  # vec_id 1 → second question
# ...
# ]
#
#



"""



###  FAISS dense repesentation + index



def build_and_save_faiss_index(
    embeddings: np.ndarray,
    dataset_name: str,
    index_type: str,
    output_dir: str = "."
):
    """
    Build and save a FAISS cosine-similarity index for a given dataset.

    """
    if not index_type or index_type not in {"passages", "iqoq"}:
        raise ValueError(
            "index_type must be provided and set to either 'passages' or 'iqoq'."
        )

    # Build flat inner-product index
    index = faiss.IndexFlatIP(embeddings.shape[1]) # calculates inner product - vectors already normalised when generated, so no need for L2 norm
    index.add(embeddings)

    # Save with clear naming
    faiss_path = os.path.join(output_dir, f"{dataset_name}_faiss_{index_type}.faiss")
    faiss.write_index(index, faiss_path)

    print(f"[FAISS] Saved {index_type} index to {faiss_path} with {embeddings.shape[0]} vectors.")



def load_faiss_index(dataset_name: str, text_content: str, base_dir: str = "train"): 
    
    """
    Load FAISS index (.faiss) for passages or pseudoquestions.

    """
    faiss_path = f"{base_dir}/{dataset_name}_faiss_{text_content}.faiss" #################################### check directories
    index = faiss.read_index(faiss_path)
    print(f"[FAISS] Loaded {index.ntotal} vectors from {faiss_path}")
    return index



def faiss_search_topk(query_emb: np.ndarray, index, top_k: int = 50):
    """
    retrieves top-k most similar items from .faiss file

    uses vec_id_int
    """
    scores, idx = index.search(query_emb, top_k)
    return idx[0], scores[0]



### sparse representation



def jaccard_similarity(set1: set, set2: set) -> float:
    return len(set1 & set2) / max(1, len(set1 | set2))



### use metrics to find edges


def build_edges(
    oq_metadata: List[Dict],
    iq_metadata: List[Dict],
    oq_emb: np.ndarray,
    iq_index,
    top_k: int = 50, # highest cosine sim iqs considered per oq
    sim_threshold = 0.65,
    output_jsonl: str = None
) -> List[Dict]:
    """
    Build final OQ→IQ edges by:
    1. FAISS top_k retrieval
    2. Compute hybrid similarity
    3. Keep only the single highest-scoring IQ per OQ
    """
    edges = []

    for oq in oq_metadata:
        if oq["type"] != "OQ":
            continue

        oq_vec = oq_emb[oq["vec_id"]]
        idxs, scores = faiss_search_topk(oq_vec.reshape(1, -1), iq_index, top_k=top_k)

        candidate_edges = []
        for rank, iq_idx in enumerate(idxs):
            iq = iq_metadata[iq_idx]
            sim_cos = float(scores[rank])  # FAISS cosine
            sim_jac = jaccard_similarity(set(oq["keywords"]), set(iq["keywords"]))
            sim_hybrid = 0.5 * (sim_cos + sim_jac)

            if sim_hybrid < sim_threshold:
                continue

            candidate_edges.append({
                "oq_id": oq["iqoq_id"],
                "oq_parent": oq["parent_passage_id"],
                "oq_vec_id": oq["vec_id"],
                "oq_text": oq["text"],

                "iq_id": iq["iqoq_id"],
                "iq_parent": iq["parent_passage_id"],
                "iq_vec_id": iq["vec_id"],

                "sim_cos": round(sim_cos, 4),
                "sim_jaccard": round(sim_jac, 4),
                "sim_hybrid": round(sim_hybrid, 4)
            })

        if candidate_edges:
            best_edge = max(candidate_edges, key=lambda e: e["sim_hybrid"])
            edges.append(best_edge)

    # Optionally save edges
    if output_jsonl:
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for e in edges:
                f.write(json.dumps(e) + "\n")
        print(f"[Edges] Saved {len(edges)} edges to {output_jsonl}")

    return edges



"""


# Load metadata
passages_metadata = load_jsonl(passages_path)
iqoq_metadata = load_jsonl(iqoq_path)

# Load embeddings
passages_emb = np.load(passages_emb_path)
iqoq_emb = np.load(iqoq_emb_path)

# Build FAISS index
build_and_save_faiss_index(passages_emb, "hotpot", "passages", output_dir="train")
build_and_save_faiss_index(iqoq_emb, "hotpot", "iqoq", output_dir="train")

# Load FAISS index
iq_index = load_faiss_index("hotpot", "iqoq", base_dir="train")

# Make edges.jsonl
edges = build_edges(
    oq_metadata=iqoq_metadata,
    iq_metadata=iqoq_metadata,
    oq_emb=iqoq_emb,
    iq_emb=iqoq_emb,
    iq_index=iq_index,
    top_k=MAX_NEIGHBOURS,
    sim_threshold=SIM_THRESHOLD,
    output_jsonl="train/hotpot_edges.jsonl"
)


#
# { {dataset}_edges.jsonl
#   "oq_id": "hotpot_001_sent1_oq1",
#   "oq_parent": "hotpot_001_sent1",
#   "oq_vec_id": 12,
#   "oq_text": "What year was the Battle of Hastings?", #include text for LLM to query - saves accessing another file 

#   "iq_id": "hotpot_002_sent4_iq0",
#   "iq_parent": "hotpot_002_sent4",
#   "iq_vec_id": 37,

#   "sim_cos": 0.72,
#   "sim_jaccard": 0.56,
#   "sim_hybrid": 0.64
# }
#



"""




############################################################################### do i need to do any tuning here? or I guess there's nothing to tune here so far really? 




################################################################################################################
# GRAPH CONSTRUCTION
################################################################################################################



def build_networkx_graph(
        passages: List[Dict], 
        edges: List[Dict]
        ) -> nx.DiGraph:
    """
    Build a directed HopRAG-style graph:
    - Nodes: passages
    - Edges: OQ -> IQ with similarity scores
    """
    G = nx.DiGraph()

    # Add passage nodes
    for p in passages:
        G.add_node(
            p["passage_id"],
            text=p.get("text", ""),
            conditioned_score=p.get("conditioned_score", 0.0),
            keywords=p.get("keywords", []),
            vec_id=p.get("vec_id"),
        )

    # Add edges with similarity scores and OQ text
    for e in edges:
        oq_parent = e["oq_parent"]
        iq_parent = e["iq_parent"]

        G.add_edge( 
            oq_parent,
            iq_parent,
            oq_id=e["oq_id"],
            oq_text=e.get("oq_text", ""),  
            iq_id=e["iq_id"],
            sim_cos=e.get("sim_cos", 0.0),
            sim_jaccard=e.get("sim_jaccard", 0.0),
            sim_hybrid=e.get("sim_hybrid", 0.0),
        )

    return G



def basic_graph_eval(
        G: nx.DiGraph, 
        top_k_hubs: int = 5 # how many nodes with the most connections (by total degree) to log 
        ) -> dict:
    """
    Compute general graph-wide statistics: average degree, degree variance, Gini coefficient,
    and top-k hub nodes. Used by both detailed and global graph logging.
    """
    degrees = np.array([deg for _, deg in G.degree()])
    avg_degree = float(np.mean(degrees)) if len(degrees) else 0.0
    degree_var = float(np.var(degrees)) if len(degrees) else 0.0

    def gini(array):
        if len(array) == 0:
            return 0.0
        array = np.sort(array)
        n = len(array)
        cumulative = np.cumsum(array)
        return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

    gini_degree = gini(degrees)

    top_k_hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_k_hubs]
    top_k_hub_nodes = [{"node": nid, "degree": deg} for nid, deg in top_k_hubs]

    return {
        "avg_node_degree": round(avg_degree, 4),
        "node_degree_variance": round(degree_var, 4),
        "gini_degree": round(gini_degree, 4),
        "top_k_hub_nodes": top_k_hub_nodes
    }



def append_global_result( 
    save_path: str,
    total_queries: int,
    graph_eval: Optional[Dict] = None,
    traversal_eval: Optional[Dict] = None,
    answer_eval: Optional[Dict] = None,
    extra_metadata: Optional[Dict] = None
):
    """
    Append a structured global result entry to {dataset}_dev_global_results.jsonl.
    Sections can be partial: graph_eval, traversal_eval, and/or answer_eval.
    """
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "total_queries": total_queries
    }

    if graph_eval:
        result["graph_eval"] = graph_eval

    if traversal_eval:
        result["traversal_eval"] = traversal_eval

    if answer_eval:
        result["answer_eval"] = answer_eval

    if extra_metadata:
        result.update(extra_metadata)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result



"""



# Load passages
with open("data/hotpot_passages.jsonl", "r", encoding="utf-8") as f:
    passages = [json.loads(line) for line in f]

# Load edges
with open("data/hotpot_edges.jsonl", "r", encoding="utf-8") as f:
    edges = [json.loads(line) for line in f]

G = build_networkx_graph(passages, edges)

graph_eval = basic_graph_eval(G, top_k_hubs=5)

append_global_result(
    save_path="outputs/hotpot_dev_global_results.jsonl",
    total_queries=100,
    graph_eval=graph_eval
)



#
#
# { # {dataset}_dev_global_results.jsonl
#   "total_queries": 100,
#
#   "graph_eval": {
#     "avg_node_degree": 3.2,
#     "node_degree_variance": 1.9,
#     "gini_degree": 0.35,
#     "top_k_hub_nodes": [
#       {"node": "hotpot_042_sent1", "degree": 15},
#       {"node": "hotpot_089_sent0", "degree": 12}
#     ]
#   },
# }
#
#
#



"""



def graph_stats(
    G: nx.DiGraph,
    save_path: str = None, # input save path if you want to save
    params: dict = None,
    iteration: int = None,
    dataset: str = None
) -> dict:
    """
    Compute full graph construction stats for tuning and optionally append to JSONL.
    Includes in/out degrees, components, edge sim stats, and high-level summary.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    in_degrees = np.array([d for _, d in G.in_degree()], dtype=int)
    out_degrees = np.array([d for _, d in G.out_degree()], dtype=int)

    comps = list(nx.weakly_connected_components(G))
    largest_comp = max(comps, key=len) if comps else set()

    stats = {
        "dataset": dataset if dataset else "unknown",
        "iteration": iteration if iteration is not None else 0,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "params": params if params else {},

        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_components": len(comps),
        "largest_component_size": len(largest_comp),
        "largest_component_ratio": len(largest_comp) / num_nodes if num_nodes else 0.0,

        "avg_in_degree": float(np.mean(in_degrees)) if num_nodes else 0,
        "min_in_degree": int(np.min(in_degrees)) if num_nodes else 0,
        "max_in_degree": int(np.max(in_degrees)) if num_nodes else 0,
        "var_in_degree": float(np.var(in_degrees)) if num_nodes else 0,

        "avg_out_degree": float(np.mean(out_degrees)) if num_nodes else 0,
        "min_out_degree": int(np.min(out_degrees)) if num_nodes else 0,
        "max_out_degree": int(np.max(out_degrees)) if num_nodes else 0,
        "var_out_degree": float(np.var(out_degrees)) if num_nodes else 0,

        "nodes_with_no_in_edges": int(np.sum(in_degrees == 0)),
        "nodes_with_no_out_edges": int(np.sum(out_degrees == 0)),
    }

    # Edge similarity stats
    if num_edges > 0:
        sim_hybrids = [d.get("sim_hybrid", 0.0) for _, _, d in G.edges(data=True)]
        stats.update({
            "edge_sim_hybrid_mean": float(np.mean(sim_hybrids)),
            "edge_sim_hybrid_max": float(np.max(sim_hybrids)),
            "edge_sim_hybrid_min": float(np.min(sim_hybrids)),
            "edge_sim_hybrid_var": float(np.var(sim_hybrids)),
        })

    # Add high-level graph_eval
    stats.update(basic_graph_eval(G))

    print(json.dumps(stats, indent=2))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")

    return stats



"""


graph_stats(
    G,
    save_path="outputs/hotpot_dev_graph_results.jsonl",
    params=params,
    iteration=12,
    dataset="hotpot_dev"
)


#
#
#
# { # {dataset}_dev_graph_results.jsonl 
#   "dataset": "hotpot_dev",
#   "iteration": 12,
#   "timestamp": "2025-08-06T14:32:11",
#
#   "params": {
#     "sim_threshold": 0.65,
#     "max_neighbors": 5,
#     "beam_width": 3 ######################
#   },
#
#   "num_nodes": 8,
#   "num_edges": 6,
#
#   "num_components": 3, ##################
#   "largest_component_size": 5, ##############
#   "largest_component_ratio": 0.625, ############
#
#   "avg_in_degree": 0.75,
#   "min_in_degree": 0,
#   "max_in_degree": 2,
#   "var_in_degree": 0.4375,
#
#   "avg_out_degree": 0.75,
#   "min_out_degree": 0,
#   "max_out_degree": 2,
#   "var_out_degree": 0.4375,
#
#   "nodes_with_no_in_edges": 4,
#   "nodes_with_no_out_edges": 4,
#
#   "edge_sim_hybrid_mean": 0.63, ################################
#   "edge_sim_hybrid_max": 0.92,
#   "edge_sim_hybrid_min": 0.41,
#   "edge_sim_hybrid_var": 0.033,
# }
#
#
#



"""




######################## do i do some kind of tuning here???? only tuning the graph before actual answer generation? 




################################################################################################################
# 6. RETRIEVAL AND GRAPH TRAVERSAL  
################################################################################################################



### COMPONENT 1 - retrieval 


def select_seed_passages(  # helper for run_dev_set()
    query_text: str,
    query_emb: np.ndarray,
    passage_metadata: List[Dict],
    passage_index,
    seed_top_k: int = 50, # gets faiss dense, applied jaccard dense - this essentially applying hyrid similarity scoring? ########################## this is the main retrieval step???
    alpha: float = 0.5  # weight for hybrid score: alpha * cosine + (1 - alpha) * jaccard
) -> List[str]:
    """
    Select top seed passages by combining FAISS dense similarity and Jaccard keyword overlap.
    Returns a list of passage_ids.
    """
    # Extract keywords from the query
    query_keywords = set(extract_keywords(query_text)) # this is extracting them from the query only, which hasn't been processed yet, right? 

    # Step 1: FAISS search
    idxs, scores = faiss_search_topk(query_emb.reshape(1, -1), passage_index, seed_top_k=seed_top_k)

    # Step 2: Score and filter
    selected = []
    for rank, passage_idx in enumerate(idxs):
        p = passage_metadata[passage_idx]
        sim_cos = float(scores[rank])  # FAISS cosine
        sim_jac = jaccard_similarity(query_keywords, set(p.get("keywords", [])))


        sim_hybrid = alpha * sim_cos + (1 - alpha) * sim_jac

        selected.append({
            "passage_id": p["passage_id"],
            "vec_id": p["vec_id"],
            "sim_cos": round(sim_cos, 4),
            "sim_jaccard": round(sim_jac, 4),
            "sim_hybrid": round(sim_hybrid, 4)
        })

    # Step 3: Sort by hybrid score
    selected.sort(key=lambda x: x["sim_hybrid"], reverse=True)

    # Step 4: Return passage_ids
    return [s["passage_id"] for s in selected]


### COMPONENT 2 - traversal 



def llm_choose_edge( # helper for multi_hop_graph_traverse_llm()
        query_text: str, # from multi_hop_graph_traverse_llm() # the llm reads the query
        passage_text: str, # form build_edges() -> build_networkx_graph() # the llm reads the passages
        candidate_edges: list, # from multi_hop_graph_traverse_llm() # then the llm considers all possible OQs in the outgoing edges
        model_servers: list # from arg. in multi_hop_traverse() 
        ): 
    """
    Ask the local LLM to choose the best outgoing OQ edge to follow.
    Uses the OQ worker (assumed model_servers[1]).
    
    candidate_edges: list of tuples (vk, edge_data)
    Returns the chosen edge tuple or None if no valid choice is made.
    """
    oq_options = [
        f"{i+1}. ({edge_data['oq_id']}) {edge_data['oq_text']}"
        for i, (_, edge_data) in enumerate(candidate_edges)
    ]

    prompt = traversal_prompt.format(
        query_text=query_text,
        passage_text=passage_text,
        candidate_oqs="\n".join(oq_options)
    )
    
    # Send to OQ worker (port 8001)
    answer = query_llm(prompt, server_url=model_servers[1], max_tokens=5, temperature=0.0)
    
    # Extract integer choice
    for token in answer.split():
        if token.isdigit():
            choice_idx = int(token) - 1
            if 0 <= choice_idx < len(candidate_edges):
                return candidate_edges[choice_idx]
    
    # No valid choice -> no traversal
    return None



def hoprag_traversal_algorithm(
    vj, graph, query_text, visited_passages,
    model_servers, ccount, next_Cqueue, hop_log, state
):
    candidates = [
        (vk, graph[vj][vk])
        for vk in graph.successors(vj)
    ]

    if not candidates:
        return set()

    chosen = llm_choose_edge(
        query_text=query_text,
        passage_text=graph.nodes[vj]["text"],
        candidate_edges=candidates,
        model_servers=model_servers
    )

    if chosen is None:
        hop_log["none_count"] += 1
        state["none_count"] += 1
        return set()

    chosen_vk, chosen_edge = chosen
    is_repeat = chosen_vk in visited_passages

    hop_log["edges_chosen"].append({
        "from": vj,
        "to": chosen_vk,
        "oq_id": chosen_edge["oq_id"],
        "iq_id": chosen_edge["iq_id"],
        "repeat_visit": is_repeat
    })

    ccount[chosen_vk] = ccount.get(chosen_vk, 0) + 1

    if is_repeat:
        hop_log["repeat_visit_count"] += 1
        state["repeat_visit_count"] += 1
        return set()

    hop_log["new_passages"].append(chosen_vk)
    next_Cqueue.append(chosen_vk)
    return {chosen_vk}



def enhanced_traversal_algorithm(
    vj, graph, query_text, visited_passages,
    model_servers, ccount, next_Cqueue, hop_log, state
):
    candidates = [
        (vk, graph[vj][vk])
        for vk in graph.successors(vj)
        if (graph[vj][vk]["oq_id"], graph[vj][vk]["iq_id"]) not in state["Evisited"]
    ]

    if not candidates:
        return set()

    chosen = llm_choose_edge(
        query_text=query_text,
        passage_text=graph.nodes[vj]["text"],
        candidate_edges=candidates,
        model_servers=model_servers
    )

    if chosen is None:
        hop_log["none_count"] += 1
        state["none_count"] += 1
        return set()

    chosen_vk, chosen_edge = chosen
    state["Evisited"].add((chosen_edge["oq_id"], chosen_edge["iq_id"]))
    is_repeat = chosen_vk in visited_passages

    hop_log["edges_chosen"].append({
        "from": vj,
        "to": chosen_vk,
        "oq_id": chosen_edge["oq_id"],
        "iq_id": chosen_edge["iq_id"],
        "repeat_visit": is_repeat
    })

    ccount[chosen_vk] = ccount.get(chosen_vk, 0) + 1

    if is_repeat:
        hop_log["repeat_visit_count"] += 1
        state["repeat_visit_count"] += 1
    else:
        hop_log["new_passages"].append(chosen_vk)

    next_Cqueue.append(chosen_vk)
    return {chosen_vk}



def traverse_graph(
    graph: nx.DiGraph,
    query_text: str,
    seed_passages: list,
    n_hops: int,
    model_servers: list,
    traveral_alg: Callable  # custom algorithm step (edge + queueing logic)
):
    Cqueue = seed_passages[:]
    visited_passages = set(seed_passages)
    ccount = {pid: 1 for pid in Cqueue}
    hop_trace = []
    state = {
        "Evisited": set(),
        "none_count": 0,
        "repeat_visit_count": 0
    }

    for hop in range(n_hops):
        next_Cqueue = []
        hop_log = {
            "hop": hop,
            "expanded_from": list(Cqueue),
            "new_passages": [],
            "edges_chosen": [],
            "none_count": 0,
            "repeat_visit_count": 0
        }

        for vj in Cqueue:
            if vj not in graph:
                continue

            new_nodes = traveral_alg(
                vj=vj,
                graph=graph,
                query_text=query_text,
                visited_passages=visited_passages,
                model_servers=model_servers,
                ccount=ccount,
                next_Cqueue=next_Cqueue,
                hop_log=hop_log,
                state=state
            )
            visited_passages.update(new_nodes)

        hop_trace.append(hop_log)
        Cqueue = next_Cqueue

    return list(visited_passages), ccount, hop_trace, {
        "none_count": state["none_count"],
        "repeat_visit_count": state["repeat_visit_count"]
    }





### COMPONENT 3 - metrics 



def compute_hop_metrics(
        hop_trace, 
        gold_passages
        ): # helper for save_dev_results 
    """
    Compute precision, recall, and F1 per hop and final.
    Adds 'metrics' to each hop in-place.
    """
    gold_set = set(gold_passages)
    visited_cumulative = set()
    results = []

    for hop_log in hop_trace:
        visited_cumulative.update(hop_log["new_passages"])
        tp = len(visited_cumulative & gold_set)
        fp = len(visited_cumulative - gold_set)
        fn = len(gold_set - visited_cumulative)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        hop_log["metrics"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

        results.append(hop_log)

    final_metrics = results[-1]["metrics"] if results else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return results, final_metrics



def save_dev_per_query_result( # helper for run_dev_set()
    query_id,
    gold_passages,
    visited_passages,
    ccount,
    hop_trace,
    traveral_alg,
    output_path="dev_results.jsonl"
):
    """
    Save a complete traversal + metric result for a single query.
    """

    hop_trace_with_metrics, final_metrics = compute_hop_metrics(hop_trace, gold_passages)

    result_entry = {
        "query_id": query_id,
        "gold_passages": gold_passages,
        "visited_passages": list(visited_passages),
        "visit_counts": dict(ccount),
        "hop_trace": hop_trace_with_metrics,
        "final_metrics": final_metrics,
        "traversal_algorithm": traveral_alg.__name__
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(result_entry) + "\n")



### all together now!!!



def run_dev_set(
    query_data: List[Dict],
    graph: nx.DiGraph,
    passage_metadata: List[Dict],
    passage_emb: np.ndarray,
    passage_index,
    emb_model, # probs change this to just emb_model? any embeddings model I guess? 
    model_servers: List[str],
    output_path="dev_results.jsonl",
    seed_top_k=50, # top-k seed passages
    alpha=0.5, 
    n_hops=2,
    traveral_alg=None
):
    """
    Run LLM-driven multi-hop traversal on an entire dev set.
    Saves results with precision/recall/F1 to output_path.
    """

    for entry in query_data:
        query_id = entry["query_id"]
        query_text = entry["question"]
        gold_passages = entry["gold_passages"]

        print(f"\n[Query] {query_id} - \"{query_text}\"")

        # --- Embed query ---
        query_emb = emb_model.encode(query_text, normalize_embeddings=True)

        # --- Select seeds with hybrid similarity ---
        seed_passages = select_seed_passages(
            query_text=query_text,
            query_emb=query_emb,
            passage_metadata=passage_metadata,
            passage_emb=passage_emb,
            passage_index=passage_index,
            seed_top_k=seed_top_k,
            alpha=alpha
        )

        print(f"[Seeds] Retrieved {len(seed_passages)} passages.")

        # --- Traverse ---
        visited_passages, ccount, hop_trace, stats = traverse_graph( 
            graph=graph,
            query_text=query_text,
            seed_passages=seed_passages,
            n_hops=n_hops,
            model_servers=model_servers, ######################### this needs to be defined somewhere? isn't it passed as an argument? 
            traveral_alg=traveral_alg
        )

        print(f"[Traversal] Visited {len(visited_passages)} passages (None={stats['none_count']}, Repeat={stats['repeat_visit_count']})")

        # --- Save ---
        save_dev_per_query_result(
            query_id=query_id,
            gold_passages=gold_passages,
            visited_passages=visited_passages,
            ccount=ccount,
            hop_trace=hop_trace,
            output_path=output_path
        )



### overall metrics



def compute_traversal_summary(
        results_path: str
        ) -> dict:
    """
    Summarize traversal-wide metrics across all dev results.
    """
    import json

    total_queries = 0
    sum_precision = 0.0
    sum_recall = 0.0
    hop_depth_counts = defaultdict(int)
    total_none = 0
    total_repeat = 0
    passage_coverage_all_gold_found = 0
    initial_retrieval_coverage = 0

    with open(results_path, "r") as f:
        for line in f:
            total_queries += 1
            entry = json.loads(line)

            final = entry["final_metrics"]
            sum_precision += final["precision"]
            sum_recall += final["recall"]

            if set(entry["gold_passages"]).issubset(set(entry["visited_passages"])):
                passage_coverage_all_gold_found += 1

            # Coverage at hop 0 (seed retrieval)
            hop0_passages = set(entry["hop_trace"][0]["expanded_from"])
            if hop0_passages & set(entry["gold_passages"]):
                initial_retrieval_coverage += 1

            for hop_log in entry["hop_trace"]:
                hop_depth_counts[hop_log["hop"]] += 1
                total_none += hop_log["none_count"]
                total_repeat += hop_log["repeat_visit_count"]

    mean_precision = sum_precision / total_queries if total_queries else 0
    mean_recall = sum_recall / total_queries if total_queries else 0

    return {
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "passage_coverage_all_gold_found": passage_coverage_all_gold_found,
        "initial_retrieval_coverage": initial_retrieval_coverage,
        "avg_hops_before_first_gold": "TODO",  # Optional advanced metric
        "avg_total_hops": round(sum(hop_depth_counts.values()) / total_queries, 2),
        "avg_repeat_visits": round(total_repeat / total_queries, 2),
        "avg_none_count_per_query": round(total_none / total_queries, 2),
        "max_hop_depth_reached": max(hop_depth_counts) if hop_depth_counts else 0,
        "hop_depth_counts": [hop_depth_counts[i] for i in sorted(hop_depth_counts)]
    }





"""



# Load resources for the dev set
dataset = "hotpot"  # or "fever", etc.
passage_metadata = load_jsonl("train/hotpot_passages.jsonl")
passage_emb = np.load("train/hotpot_passages_emb.npy")
passage_index = faiss.read_index("train/hotpot_faiss_passages.faiss")



# Load dev query data
query_data = [json.loads(line) for line in open(f"{dataset}_dev.jsonl")]

# Run pipeline
run_dev_set(
    query_data=query_data,
    graph=graph,
    passage_metadata=passage_metadata,
    passage_emb=passage_emb,
    passage_index=passage_index,
    emb_model=bge_model,
    model_servers=MODEL_SERVERS,
    output_path="results/hotpot_dev_per_query_results.jsonl",
    seed_top_k=TOP_K_SEED_PASSAGES,
    alpha=ALPHA,
    n_hops=NUMBER_HOPS,
    traveral_alg=enhanced_traversal_algorithm
)


# Collect traversal-wide stats from per-query results
traversal_metrics = compute_traversal_summary("results/hotpot_dev_results.jsonl")

# Append to global results log
append_global_result(
    save_path="results/hotpot_dev_global_results.jsonl",
    total_queries=len(query_data),
    traversal_eval=traversal_metrics
)

#
#
# { # {dataset}_dev_results.jsonl ############ I THINK THIS IS THE WRONG LAYOUT # SHOULD BE DEV_PER_QUERY_RESULTS.JSONL
#     query_id = "hotpot_001",
#     gold_passages = ["hotpot_002_sent4", "hotpot_003_sent2"], ############# for precision, recall, f1
#     visited_passages = ["hotpot_001_sent1", "hotpot_002_sent4", "hotpot_003_sent2"], ############# recall coverage - did we see gold passages before pruning?
#     "precision": 0.666,
#     "recall": 1.0,
#     "f1": 1.0,
#     ccount = {"hotpot_002_sent4": 2, "hotpot_003_sent2": 1}, #### for importance (for helpfulness) - identify high-traffic nodes
#     hop_trace = [ ###### prune n-hops (for graph density, pruning thresholds?)
#         {
#             "hop": 0,
#             "expanded_from": ["hotpot_001_sent1"],
#             "new_passages": ["hotpot_002_sent4"],
#             "edges_chosen": [
#                 {
#                     "from": "hotpot_001_sent1",
#                     "to": "hotpot_002_sent4",
#                     "oq_id": "hotpot_001_sent1_oq1",
#                     "iq_id": "hotpot_002_sent4_iq0",
#                     "repeat_visit": False
#                 }
#             ],
#             "none_count": 0,
#             "repeat_visit_count": 0
#         },
#         {
#             "hop": 1,
#             "expanded_from": ["hotpot_002_sent4"],
#             "new_passages": ["hotpot_003_sent2"],
#             "edges_chosen": [
#                 {
#                     "from": "hotpot_002_sent4",
#                     "to": "hotpot_003_sent2",
#                     "oq_id": "hotpot_002_sent4_oq0",
#                     "iq_id": "hotpot_003_sent2_iq0",
#                     "repeat_visit": False
#                 }
#             ],
#             "none_count": 1,
#             "repeat_visit_count": 0
#         }
#     ]
# }
#
#
#
#
#
# { # {dataset}_dev_global_results.jsonl
#   "total_queries": 100,
#   "graph_eval": {
#     "avg_node_degree": 3.2,
#     "node_degree_variance": 1.9,
#     "gini_degree": 0.35,
#     "top_k_hub_nodes": [
#       {"node": "hotpot_042_sent1", "degree": 15},
#       {"node": "hotpot_089_sent0", "degree": 12}
#     ]
#   },
#   "traversal_eval": {
#     "mean_precision": 0.63,
#     "mean_recall": 0.74,
#     "passage_coverage_all_gold_found": 82,
#     "initial_retrieval_coverage": 58,
#     "avg_hops_before_first_gold": 1.8,
#     "avg_total_hops": 2.4,
#     "avg_repeat_visits": 0.3,
#     "avg_none_count_per_query": 0.8,
#     "max_hop_depth_reached": 3,
#     "hop_depth_counts": [100, 95, 72, 20]
#   },
# }
#
#
#



"""



################################################################################################################
# PASSAGE RERANKING AND ANSWER GENERATION 
################################################################################################################



def compute_helpfulness( # helper function for rerank_passages_by_helpfulness()
    vertex_id: str,
    vertex_query_sim: float, # similarity between the passage and the query
    ccount: dict
) -> float:
    """

    what vars does this take? this is for online, llm-interfacing stuff? online, it narrows down the passages processed by the llm in multi_hop_graph_traverse_llm according to their helpfulness? 
    then the new list is sent to the llm? 

    """
    total_visits = sum(ccount.values()) or 1
    importance = ccount.get(vertex_id, 0) / total_visits

    # HopRAG helpfulness formula: (SIM + IMP) / 2
    helpfulness = 0.5 * (vertex_query_sim + importance) # similarity between the passage and the query
    return helpfulness



def rerank_passages_by_helpfulness(
    candidate_passages: List[str],
    query_text: str,
    ccount: dict,
    graph: nx.DiGraph,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Compute helpfulness scores for candidate passages and return top-k ranked list.
    
    Returns:
        List of tuples: [(passage_id, helpfulness_score), ...]
    """
    reranked = []
    for pid in candidate_passages:
        node = graph.nodes.get(pid, {})
        passage_text = node.get("text", "")
        vertex_query_sim = node.get("query_sim", 0.0)  # make sure this is populated when building graph!

        score = compute_helpfulness(pid, vertex_query_sim, ccount)
        reranked.append((pid, score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]



def normalise_answer(s: str) -> str:
    """
    
    Lower text and remove punctuation, articles and extra whitespace.
    to prepare answers for EM/F1
    
    
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def ask_llm_with_passages(
    query_text: str,
    passage_ids: List[str],
    graph: Optional[nx.DiGraph],
    server_url: str,
    max_tokens: int = 100,
    passage_lookup: Optional[Dict[str, str]] = None  # optional for dense mode
) -> Dict[str, str]:
    """
    Format top passages and send them with the query to the LLM.
    Works with or without a graph.
    """
    passage_texts = []

    for pid in passage_ids:
        if graph:
            passage = graph.nodes[pid].get("text", "")
        elif passage_lookup:
            passage = passage_lookup.get(pid, "")
        else:
            passage = f"[{pid}]"

        passage_texts.append(f"[{pid}]: {passage}")

    prompt = (
        f"Answer the question using the following passages:\n\n"
        + "\n\n".join(passage_texts)
        + f"\n\nQuestion: {query_text}\nAnswer:"
    )

    raw = query_llm(prompt, server_url=server_url, max_tokens=max_tokens)
    norm = normalise_answer(raw)
    return {"raw_answer": raw, "normalised_answer": norm}




"""



top_helpful_passages = rerank_passages_by_helpfulness(
    candidate_passages=list(visited_passages),
    query_text=query_text,
    ccount=ccount,
    graph=graph,
    top_k=TOP_K_ANSWER_PASSAGES  
)

top_passage_ids = [pid for pid, _ in top_helpful_passages]

llm_result = ask_llm_with_passages(
    query_text=query_text,
    passage_ids=top_passage_ids,
    graph=graph,
    server_url=model_servers[0],
    max_tokens=100 ################ I think I have this set up globally somewhere? 
)

normalised_answer = llm_result["normalised_answer"]
# do I need to do something with the raw answer here as well??? I guess I need to save it somewhere??


"""



def compute_exact_match(
        pred: str, 
        gold: str
        ) -> int:
    """EM is 1 if normalized prediction == normalized gold, else 0."""
    return int(normalise_answer(pred) == normalise_answer(gold))



def compute_f1(
        pred: str, 
        gold: str
        ) -> float:
    """
    Compute token-level F1 between prediction and gold after normalization.
    """
    pred_tokens = normalise_answer(pred).split()
    gold_tokens = normalise_answer(gold).split()

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)



def evaluate_answers( 
        predictions: dict, 
        gold_answers: dict
        ) -> dict: 
    """
    Evaluate EM and F1 over final answers only.

    predictions: dict of {id: predicted_answer}
    gold_answers: dict of {id: list of gold answers (to allow multiple paraphrases)}

    Returns: dict with overall EM and F1 (%).
    """
    total = len(gold_answers)
    em_total = 0
    f1_total = 0.0

    for qid, gold_list in gold_answers.items():
        pred = predictions.get(qid, "")
        em = max(compute_exact_match(pred, g) for g in gold_list)  
        f1 = max(compute_f1(pred, g) for g in gold_list) 
        em_total += em
        f1_total += f1

    return {
        "EM": 100.0 * em_total / total,
        "F1": 100.0 * f1_total / total
    }




"""


gold_answers = {}

with open("data/dev/hotpot_dev.jsonl", 'r') as f:
    for line in f:
        obj = json.loads(line)
        qid = obj["question_id"]
        ans = obj["gold_answer"]
        gold_answers[qid] = [normalise_answer(ans)] # needs to be a list for evaluate_answers



answers = evaluate_answers(normalised_answer, gold_answers) #### these gold answers need to be normalised somewhere as well 



append_global_result(
    save_path=f"{dataset}_dev_global_results.jsonl",
    total_queries=len(gold_answers),
    answer_eval=answers
)



#
#
# { # {dataset}_dev_global_results.jsonl
#   "total_queries": 100,
#   "graph_eval": {
#     "avg_node_degree": 3.2,
#     "node_degree_variance": 1.9,
#     "gini_degree": 0.35,
#     "top_k_hub_nodes": [
#       {"node": "hotpot_042_sent1", "degree": 15},
#       {"node": "hotpot_089_sent0", "degree": 12}
#     ]
#   },
#   "traversal_eval": {
#     "mean_precision": 0.63,
#     "mean_recall": 0.74,
#     "passage_coverage_all_gold_found": 82,
#     "initial_retrieval_coverage": 58,
#     "avg_hops_before_first_gold": 1.8,
#     "avg_total_hops": 2.4,
#     "avg_repeat_visits": 0.3,
#     "avg_none_count_per_query": 0.8,
#     "max_hop_depth_reached": 3,
#     "hop_depth_counts": [100, 95, 72, 20]
#   },
#   "answer_eval": {
#     "average_em": 72.5,
#     "average_f1": 78.9
#   }
# }
#
#




"""









def sweep_thresholds(edges: List[Dict], thresholds: List[float]):
    for t in thresholds:
        filtered = [e for e in edges if e['sim_hybrid'] >= t]
        print(f"Threshold {t:.2f}: {len(filtered)} edges")














def run_dense_rag_baseline(
    query_data: List[Dict],
    passage_metadata: List[Dict],
    passage_emb: np.ndarray,
    passage_index,
    emb_model,
    model_servers: List[str],
    output_path="results/dense_rag_results.jsonl",
    seed_top_k=50,
    alpha=0.5
):
    """
    Dense RAG baseline: 
    - retrieve top-k by cosine/Jaccard hybrid
    - no traversal or graph
    - rerank by cosine + jaccard
    - generate answer
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        pass  # Clear file before writing

    for entry in query_data:
        query_id = entry["question_id"]
        query_text = entry["question"]
        gold_answer = entry["gold_answer"]
        gold_normalised = normalise_answer(gold_answer)

        print(f"\n[Dense RAG] {query_id} - \"{query_text}\"")

        # --- Embed query ---
        query_emb = emb_model.encode(query_text, normalize_embeddings=True)

        # --- Retrieve top-k seed passages ---
        seed_passages = select_seed_passages(
            query_text=query_text,
            query_emb=query_emb,
            passage_metadata=passage_metadata,
            passage_index=passage_index,
            seed_top_k=seed_top_k,
            alpha=alpha
        )

        print(f"[Retrieved] {len(seed_passages)} passages")

        # --- Generate answer from top passages ---
        llm_output = ask_llm_with_passages(
            query_text=query_text,
            passage_ids=seed_passages,
            graph=None,  # don't use graph — will look up text manually below
            server_url=model_servers[0],
            max_tokens=128
        )

        pred_answer = llm_output["normalised_answer"]
        raw_answer = llm_output["raw_answer"]

        # --- Evaluate ---
        em = compute_exact_match(pred_answer, gold_normalised)
        f1 = compute_f1(pred_answer, gold_normalised)

        # --- Save ---
        result = {
            "query_id": query_id,
            "question": query_text,
            "gold_answer": gold_answer,
            "predicted_answer": raw_answer,
            "normalised_pred": pred_answer,
            "normalised_gold": gold_normalised,
            "retrieved_passages": seed_passages,
            "EM": em,
            "F1": round(f1, 4),
            "method": "dense_rag"
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n[Done] Saved dense RAG results to {output_path}")



def run_pipeline(
    mode: str,
    query_data: List[Dict],
    graph: Optional[nx.DiGraph],
    passage_metadata: List[Dict],
    passage_emb: np.ndarray,
    passage_index,
    emb_model,
    model_servers,
    output_path="results/dev_results.jsonl",
    seed_top_k=50,
    alpha=0.5,
    n_hops=2
):
    """
    Dispatcher to run any of the 3 pipelines:
    - 'dense': Dense-only baseline (no graph)
    - 'hoprag': Standard HopRAG
    - 'enhanced': Enhanced HopRAG
    """
    if mode == "dense":
        run_dense_rag_baseline(
            query_data=query_data,
            passage_metadata=passage_metadata,
            passage_emb=passage_emb,
            passage_index=passage_index,
            emb_model=emb_model,
            model_servers=model_servers,
            output_path=output_path,
            seed_top_k=seed_top_k,
            alpha=alpha
        )
    elif mode == "hoprag":
        run_dev_set(
            query_data=query_data,
            graph=graph,
            passage_metadata=passage_metadata,
            passage_emb=passage_emb,
            passage_index=passage_index,
            emb_model=emb_model,
            model_servers=model_servers,
            output_path=output_path,
            seed_top_k=seed_top_k,
            alpha=alpha,
            n_hops=n_hops,
            traveral_alg=hoprag_traversal_algorithm
        )
    elif mode == "enhanced":
        run_dev_set(
            query_data=query_data,
            graph=graph,
            passage_metadata=passage_metadata,
            passage_emb=passage_emb,
            passage_index=passage_index,
            emb_model=emb_model,
            model_servers=model_servers,
            output_path=output_path,
            seed_top_k=seed_top_k,
            alpha=alpha,
            n_hops=n_hops,
            traveral_alg=enhanced_traversal_algorithm
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

