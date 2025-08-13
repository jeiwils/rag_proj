



# 
# 
# 
# 
# 
# 
# 
# need to put timer on embeddings
# I ALREADY HAVE SOME VEC_IDS - i think it's only qwen7b - musique - train - enhacned  - cs 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 






"""

WRITE UP
- paddleNLP not used - don't need multilingual testing with CHinese




"""

"""







#   #   #   #   # INPUT: 











#   #   #   #   # OUTPUT: 
















"""











import json
import numpy as np
import faiss
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.a1_dataset_processing import clean_text
from src.a2_text_prep import load_jsonl
from src.a3_file_prep import resolve_root

import re
import torch
import os, json, re, spacy
import spacy
from typing import Iterable, Set



SIM_THRESHOLD = 0.65
MAX_NEIGHBOURS = 5
ALPHA = 0.5

params = { 
    "sim_threshold": SIM_THRESHOLD,
    "max_neighbors": MAX_NEIGHBOURS,
    "alpha": ALPHA
}





################################################################################################################
# SPARSE AND DENSE REPRESENTATIONS
################################################################################################################






# def embed_and_save(input_jsonl, output_npy, output_jsonl, model, text_key):
#     """

#     for graph embeddings
#     and dev set checks 

#     Generate normalized embeddings for FAISS indexing.
#     - Adds vec_id to each entry for FAISS alignment.
#     - Saves embeddings to .npy and JSONL metadata with vec_id.
#     - Raises an error if text_key is not provided.
#     """
#     if not text_key:
#         raise ValueError("You must provide a valid text_key (e.g., 'text' or 'question').")

#     data, embeddings = [], []

#     with open(input_jsonl, "r", encoding="utf-8") as f:
#         for idx, line in enumerate(f):
#             entry = json.loads(line)
#             text = entry[text_key]  # will raise KeyError if the key is missing
#             vec = model.encode(text, normalize_embeddings=True)

#             entry["vec_id"] = idx
#             embeddings.append(vec)
#             data.append(entry)

#     embeddings = np.array(embeddings, dtype="float32")
#     os.makedirs(os.path.dirname(output_npy), exist_ok=True)
#     np.save(output_npy, embeddings)

#     with open(output_jsonl, "w", encoding="utf-8") as f_out:
#         for d in data:
#             f_out.write(json.dumps(d) + "\n")

#     print(f"[Embeddings] Saved {len(data)} vectors to {output_npy} and updated JSONL {output_jsonl}")
#     return embeddings





def embed_and_save(input_jsonl, output_npy, output_jsonl, model, text_key):
    if not text_key:
        raise ValueError("You must provide a valid text_key (e.g., 'text' or 'question').")

    # 1) Read all entries + build texts list
    data, texts = [], []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry[text_key])   # raises KeyError if missing
            data.append(entry)

    # 2) ⬇️ Put your batched encode with tqdm RIGHT HERE ⬇️
    embs = model.encode(
        texts,                      # list of strings
        normalize_embeddings=True,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    # 3) Attach vec_id, save npy + jsonl
    for i, entry in enumerate(data):
        entry["vec_id"] = i

    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    np.save(output_npy, embs)

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for d in data:
            f_out.write(json.dumps(d) + "\n")

    print(f"[Embeddings] Saved {len(data)} vectors to {output_npy} and updated JSONL {output_jsonl}")
    return embs



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



# def load_faiss_index(dataset_name: str, text_content: str, base_dir: str = "train"): 
    
#     """
#     Load FAISS index (.faiss) for passages or pseudoquestions.

#     """
#     faiss_path = f"{base_dir}/{dataset_name}_faiss_{text_content}.faiss" #################################### check directories
#     index = faiss.read_index(faiss_path)
#     print(f"[FAISS] Loaded {index.ntotal} vectors from {faiss_path}")
#     return index



def load_faiss_index(path: str):
    index = faiss.read_index(path)
    print(f"[FAISS] Loaded {index.ntotal} vectors from {path}")
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



SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")

nlp = spacy.load(SPACY_MODEL, disable=["parser", "textcat"])

# Keep only these named-entity types
KEEP_ENTS = {
    "PERSON","ORG","GPE","LOC","NORP","FAC","PRODUCT","EVENT",
    "WORK_OF_ART","LAW","LANGUAGE","DATE","TIME"
}

def normalise_text(s: str) -> str:
    """
    Normalise for keyword matching:
    - apply clean_text (assumed defined elsewhere)
    - lowercase
    - collapse any whitespace to a single underscore
    - collapse repeated underscores
    """
    if not s:
        return ""
    t = clean_text(s).lower()
    t = re.sub(r"\s+", "_", t.strip())
    t = re.sub(r"_+", "_", t)
    return t

def extract_keywords(text: str) -> list[str]:
    if not text:
        return []
    doc = nlp(clean_text(text))
    kws = {
        normalise_text(ent.text)
        for ent in doc.ents
        if ent.label_ in KEEP_ENTS and ent.text.strip()
    }
    return sorted(kws)









def extract_all_keywords(entry: dict) -> dict:
    """
    Adds 'keywords_IQ', 'keywords_OQ', and 'keywords_text' to entry.
    Assumes 'IQs', 'OQs', and 'text' exist.
    """
    entry["keywords_IQ"] = [extract_keywords(q) for q in entry.get("IQs", [])]
    entry["keywords_OQ"] = [extract_keywords(q) for q in entry.get("OQs", [])]
    entry["keywords_text"] = extract_keywords(entry.get("text", ""))
    return entry




# def add_keywords_to_passages_jsonl(passages_jsonl: str, merged_with_iqoq: bool = False):
#     """Writes keywords_text, and (optionally) keywords_IQ/keywords_OQ + all_keywords."""
#     rows = [json.loads(l) for l in open(passages_jsonl, "r", encoding="utf-8")]
#     out = []
#     for r in rows:
#         r["keywords_text"] = extract_keywords(r.get("text", ""))
#         if merged_with_iqoq:
#             kws_iq = [extract_keywords(q) for q in (r.get("IQs") or [])]
#             kws_oq = [extract_keywords(q) for q in (r.get("OQs") or [])]
#             r["keywords_IQ"] = kws_iq
#             r["keywords_OQ"] = kws_oq
#             union = set(r["keywords_text"])
#             for lst in kws_iq: union.update(lst)
#             for lst in kws_oq: union.update(lst)
#             r["all_keywords"] = sorted(union)
#         out.append(r)
#     with open(passages_jsonl, "w", encoding="utf-8") as f:
#         for r in out:
#             f.write(json.dumps(r) + "\n")
#     print(f"[sparse] passages keywords → {passages_jsonl}")

def add_keywords_to_passages_jsonl(passages_jsonl: str, merged_with_iqoq: bool = False):
    rows  = [json.loads(l) for l in open(passages_jsonl, "r", encoding="utf-8")]
    texts = [r.get("text", "") for r in rows]  # <-- raw text

    for r, doc in zip(rows, nlp.pipe(texts, batch_size=128, n_process=1)):
        kws = {
            normalise_text(ent.text)  # normalise_text will handle cleaning
            for ent in doc.ents
            if ent.label_ in KEEP_ENTS and ent.text.strip()
        }
        r["keywords_text"] = sorted(kws)
        if merged_with_iqoq:
            kws_iq = [extract_keywords(q) for q in (r.get("IQs") or [])]
            kws_oq = [extract_keywords(q) for q in (r.get("OQs") or [])]
            r["keywords_IQ"] = kws_iq
            r["keywords_OQ"] = kws_oq
            union = set(r["keywords_text"])
            for lst in kws_iq: union.update(lst)
            for lst in kws_oq: union.update(lst)
            r["all_keywords"] = sorted(union)
    with open(passages_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def add_keywords_to_iqoq_jsonl(iqoq_jsonl: str, out_field: str = "keywords"):
    rows  = [json.loads(l) for l in open(iqoq_jsonl, "r", encoding="utf-8")]
    texts = [r.get("text", "") for r in rows]  # <-- raw text

    for r, doc in zip(rows, nlp.pipe(texts, batch_size=128, n_process=1)):
        kws = {
            normalise_text(ent.text)
            for ent in doc.ents
            if ent.label_ in KEEP_ENTS and ent.text.strip()
        }
        r[out_field] = sorted(kws)

    with open(iqoq_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")





############## passages_emb_path, iqoq_emb_path





if __name__ == "__main__":
    print(f"[spaCy] Using: {SPACY_MODEL}")

    BGE_MODEL = os.environ.get("BGE_MODEL", "BAAI/bge-base-en-v1.5")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    bge_model = SentenceTransformer(BGE_MODEL, device=DEVICE)
    print(f"[BGE] Loaded {BGE_MODEL} on {DEVICE}")

    # knobs
    MODELS   = ["qwen-7b", "deepseek-distill-qwen-7b"]
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    VARIANTS = ["baseline", "enhanced"]     # folder name
    SPLIT    = "train"

    CURRENT_VARIANT = "enhanced"



    # -------------------------------
    # Phase A: dataset-only (passages + questions)
    # -------------------------------
    for dataset in DATASETS:
        print(f"\n=== DATASET: {dataset} ({SPLIT}) ===")

        passages_jsonl  = f"data/processed_datasets/{dataset}/{SPLIT}_passages.jsonl"
        questions_jsonl = f"data/processed_datasets/{dataset}/{SPLIT}.jsonl"

        passages_npy  = passages_jsonl.replace(".jsonl", ".emb.npy")
        questions_npy = questions_jsonl.replace(".jsonl", ".emb.npy")





        # Passages embeddings
        if os.path.exists(passages_npy):
            passages_emb = np.load(passages_npy).astype("float32")
            print(f"[skip] {passages_npy} exists; loaded.")
        else:
            passages_emb = embed_and_save(
                input_jsonl=passages_jsonl,
                output_npy=passages_npy,
                output_jsonl=passages_jsonl,
                model=bge_model,
                text_key="text",
            )

        # Questions embeddings
        if os.path.exists(questions_npy):
            print(f"[skip] {questions_npy} exists; loaded.")
        else:
            _ = embed_and_save(
                input_jsonl=questions_jsonl,
                output_npy=questions_npy,
                output_jsonl=questions_jsonl,
                model=bge_model,
                text_key="question",
            )





        # Passage FAISS
        index_tag_passages = f"{dataset}_{SPLIT}"
        build_and_save_faiss_index(
            embeddings=passages_emb,
            dataset_name=index_tag_passages,
            index_type="passages",
            output_dir=os.path.dirname(passages_jsonl),
        )

        # Sparse keywords on passages
        add_keywords_to_passages_jsonl(passages_jsonl, merged_with_iqoq=False)

    # -------------------------------
    # Phase B: model-specific IQ/OQ
    # -------------------------------
    for model in MODELS:
        print(f"\n=== MODEL: {model} ===")
        for dataset in DATASETS:
                variant = CURRENT_VARIANT
                root = resolve_root(model=model, dataset=dataset, split=SPLIT, variant=CURRENT_VARIANT)
                if root is None:
                    print(f"[warn] no folder for {dataset} | {model} | {variant}; skipping.")
                    continue

                iqoq_jsonl = os.path.join(root, "exploded", "iqoq.exploded.jsonl")
                if not os.path.exists(iqoq_jsonl):
                    print(f"[warn] missing IQ/OQ file: {iqoq_jsonl}; skipping.")
                    continue

                iqoq_npy = iqoq_jsonl.replace(".jsonl", ".emb.npy")

                if os.path.exists(iqoq_npy):
                    iqoq_emb = np.load(iqoq_npy).astype("float32")
                    print(f"[skip] {iqoq_npy} exists; loaded.")
                else:
                    iqoq_emb = embed_and_save(
                        input_jsonl=iqoq_jsonl,
                        output_npy=iqoq_npy,
                        output_jsonl=iqoq_jsonl,
                        model=bge_model,
                        text_key="text",
                    )

                add_keywords_to_iqoq_jsonl(iqoq_jsonl)

                index_tag_iqoq = f"{dataset}_{SPLIT}_{model}_{variant}"
                build_and_save_faiss_index(
                    embeddings=iqoq_emb,
                    dataset_name=index_tag_iqoq,
                    index_type="iqoq",
                    output_dir=root,
                )
                print(f"[done] {model} | {dataset} | {variant}")















"""









{ # data/processed_datasets/{dataset}/{split}_passages.jsonl
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b",
  "passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "text": "Arthur's Magazine (1844–1846)...",
  "conditioned_score": 0.25,
  "vec_id": 123,                      // row index in passages.emb.npy
  "keywords_text": [
    "arthur_s_magazine", "american", "literary", "periodical", "philadelphia", "1844_1846"
  ]
}



data/processed_datasets/{dataset}/{split}_passages.emb.npy
# shape: (num_passages, 768), dtype float32
# row i corresponds to the JSONL line with "vec_id": i


data/processed_datasets/{dataset}/{dataset}_{split}_faiss_passages.faiss







{ # data/processed_datasets/{dataset}/{split}.jsonl
  "question_id": "5a7a06935542990198eaf050",
  "dataset": "hotpotqa",
  "split": "train",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "gold_answer": "Arthur's Magazine",
  "passages": [
    "5a7a06935542990198eaf050__arthur_s_magazine_sent0",
    "5a7a06935542990198eaf050__first_for_women_sent0"
  ],
  "vec_id": 42                        // row index in train.emb.npy
}




data/processed_datasets/{dataset}/{split}.emb.npy
# shape: (num_questions, 768), dtype float32


















{root}/{split}_iqoq_items.cleaned.{variant}.jsonl
# root = data/models/{model}/{dataset}/{variant_folder}






{
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b",
  "parent_passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "iqoq_id": "5a7a0693__arthur_s_magazine_sent0_iq0",
  "type": "IQ",
  "index": 0,
  "text": "Who founded Arthur's Magazine?",
  "vec_id": 987,
  "keywords": ["arthur_s_magazine", "founded", "who"]
}



{root}/{split}_iqoq_items.cleaned.{variant}.emb.npy
# shape: (num_items, 768), dtype float32



{root}/{dataset}_{split}_{model}_{variant}_faiss_iqoq.faiss





"""