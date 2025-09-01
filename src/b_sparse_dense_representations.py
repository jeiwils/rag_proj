### TO DO 
# - FIX RAW QUESTION COUNT - IT'S ONLY COUNTING ROWS ATM 


"""

Module Overview
---------------
Build sparse and dense vector representations for passages and IQ/OQ (incoming/outgoing questions),
and index them for efficient retrieval using FAISS. Also extracts sparse keyword features using spaCy NER.



Inputs
------

### data/processed_datasets/{dataset}/{split}/

- passages.jsonl
    → Raw passage entries with passage ID and text.


### data/models/{model}/{dataset}/{split}/{variant}/

- exploded/iqoq.exploded.jsonl 
    → Incoming/outgoing questions (one per row), used for embedding questions.

- cleaned/iqoq.cleaned.jsonl
    → Cleaned IQ/OQ entries, used for indexing and keyword extraction.



Outputs
-------

### data/representations/datasets/{dataset}/{split}/

- {dataset}_passages.jsonl
    → Passage metadata with added vec_id.

- {dataset}_passages_emb.npy
    → Dense passage embeddings (NumPy array).

- {dataset}_faiss_passages.faiss
    → FAISS index over passage vectors.



### data/representations/models/{model}/{dataset}/{split}/{variant}/

- iqoq.cleaned.jsonl
    → Updated input file with vec_id added for each IQ/OQ item.

- {dataset}_iqoq_emb.npy
    → Dense IQ/OQ embeddings (NumPy array).

- {dataset}_faiss_iqoq.faiss
    → FAISS index over IQ/OQ vectors.



File Schema
-----------

### passages.jsonl

{
  "passage_id": "{passage_id}",
  "text": "{passage_text}",
  "vec_id": 0,
  "keywords_passage": ["{kw1}", "{kw2}", "..."],
  "keywords_iq": [["{kw1}"], ["{kw2}"], "..."],
  "keywords_oq": [["{kw1}"], ["{kw2}"], "..."]
}

Fields:
- ``passage_id``: unique identifier for the passage.
- ``text``: raw text of the passage.
- ``vec_id``: embedding index position.
- ``keywords_passage``: extracted named entities via spaCy for the passage text.
- ``keywords_iq``: extracted entities for each incoming question (if present).
- ``keywords_oq``: extracted entities for each outgoing question (if present).


### iqoq.cleaned.jsonl

{
  "iqoq_id": "{iqoq_id}",
  "text": "{question_text}",
  "vec_id": 0,
  "keywords": ["{kw1}", "{kw2}", "..."]
}

Fields:
- ``iqoq_id``: unique identifier for the question (incoming or outgoing).
- ``text``: question text.
- ``vec_id``: embedding index position.
- ``keywords``: extracted named entities via spaCy.



Notes
-----

- All `.jsonl` files are UTF-8 encoded with `ensure_ascii=False`.
- FAISS indexes are built using inner-product over normalized vectors (cosine similarity).
- Embeddings are generated using the BAAI bge-base-en-v1.5 SentenceTransformer.
- Only entities of types like PERSON, ORG, GPE, etc. are retained for keyword features.
- `vec_id` aligns each row in the `.jsonl` file with the corresponding row in the `.npy` file.

"""





import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Callable
import unicodedata
import logging

import faiss
import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer

from src.utils import (
    clean_text,
    compute_resume_sets,
    existing_ids,
    load_jsonl,
    pid_plus_title,
    processed_dataset_paths,
    save_jsonl_safely,
)



RESUME = True ########## WHY SET HERE?

_bge_model = None ########## WHY SET HERE?

logger = logging.getLogger(__name__)


def get_embedding_model():
    """Load and cache the BGE embedding model."""
    global _bge_model

    if _bge_model is None:
        model_name = os.environ.get("BGE_MODEL", "BAAI/bge-base-en-v1.5")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _bge_model = SentenceTransformer(model_name, device=device)
        print(f"[BGE] Loaded {model_name} on {device}")

    return _bge_model

def dataset_rep_paths(dataset: str, split: str) -> Dict[str, str]:
    """Return paths for model-agnostic dataset-level passage representations."""
    base = os.path.join("data", "representations", "datasets", dataset, split)
    return {
        "passages_jsonl": os.path.join(base, f"{dataset}_passages.jsonl"),
        "passages_emb": os.path.join(base, f"{dataset}_passages_emb.npy"),
        "passages_index": os.path.join(base, f"{dataset}_faiss_passages.faiss"),
    }

def model_rep_paths(model: str, dataset: str, split: str, variant: str) -> Dict[str, str]:
    """Return paths for model-specific IQ/OQ representations."""
    base = os.path.join(
        "data", "representations", "models", model, dataset, split, variant
    )
    return {
        "iqoq_jsonl": os.path.join(base, "iqoq.cleaned.jsonl"),
        "iqoq_emb": os.path.join(base, f"{dataset}_iqoq_emb.npy"),
        "iqoq_index": os.path.join(base, f"{dataset}_faiss_iqoq.faiss"),
        "iq_emb": os.path.join(base, f"{dataset}_iq_emb.npy"),
        "iq_index": os.path.join(base, f"{dataset}_faiss_iq.faiss"),
    }

__all__ = [
    "dataset_rep_paths",
    "model_rep_paths",
    "get_embedding_model",
    "build_and_save_faiss_index",
    "load_faiss_index",
    "faiss_search_topk",
    "jaccard_similarity",
    
]




################################################################################################################
# SPARSE AND DENSE REPRESENTATIONS
################################################################################################################







def embed_and_save(
    input_jsonl,
    output_npy,
    output_jsonl,
    model,
    text_key,
    *,
    id_field="passage_id",
    done_ids: Set[str] | None = None,
    output_jsonl_input: str | None = None,
):
    """Embed texts from ``input_jsonl`` and save results.

    Parameters
    ----------
    input_jsonl: str
        Path to the JSONL file containing the text used for embedding (e.g. the
        *exploded* IQ/OQ file).
    output_npy: str
        Destination path for the NumPy embedding array.
    output_jsonl: str
        Destination JSONL path where entries with ``vec_id`` are written.
    model: SentenceTransformer
        The embedding model.
    text_key: str
        Key in each JSON record that contains the text to be embedded.
    id_field: str, optional
        Field holding the unique identifier for each entry.
    done_ids: set[str], optional
        Set of IDs that already have embeddings and should be skipped.
    output_jsonl_input: str, optional
        Path to the JSONL file providing the metadata to write to ``output_jsonl``.
        If ``None``, ``input_jsonl`` is used.
    """

    if not text_key:
        raise ValueError("You must provide a valid text_key (e.g., 'text' or 'question').")

    # If a separate source for the output JSONL is provided (e.g., a cleaned
    # version of the data), build a lookup by ID so we can pair the embedding
    # text from ``input_jsonl`` with the cleaned metadata.
    if output_jsonl_input is None:
        output_jsonl_input = input_jsonl

    by_id = {}
    if output_jsonl_input != input_jsonl:
        with open(output_jsonl_input, "rt", encoding="utf-8") as f_clean:
            for line in f_clean:
                entry = json.loads(line)
                by_id[entry[id_field]] = entry

    data, texts = [], []
    with open(input_jsonl, "rt", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            entry_id = entry.get(id_field)
            if done_ids and entry_id in done_ids:
                continue
            texts.append(entry[text_key])
            if by_id:
                if entry_id not in by_id:
                    raise KeyError(
                        f"{id_field} {entry_id} from {input_jsonl} not found in {output_jsonl_input}"
                    )
                data.append(by_id[entry_id])
            else:
                data.append(entry)

    existing_embs = None
    vec_offset = 0
    if os.path.exists(output_npy):
        existing_embs = np.load(output_npy).astype("float32") #existing_embs = np.load(output_npy)["embs_all"].astype("float32")
        vec_offset = existing_embs.shape[0]
        if os.path.exists(output_jsonl):
            with open(output_jsonl, "rt", encoding="utf-8") as f_old:
                idx = -1
                for idx, line in enumerate(f_old):
                    if json.loads(line).get("vec_id") != idx:
                        raise AssertionError(
                            f"vec_id mismatch at line {idx} in {output_jsonl}"
                        )
                if vec_offset != idx + 1:
                    raise AssertionError(
                        f"Embedding count {vec_offset} does not match JSONL entries {idx + 1}"
                    )

    if not data:
        if existing_embs is not None:
            embs_all = existing_embs
        else:
            embs_all = np.empty(
                (0, model.get_sentence_embedding_dimension()), dtype="float32"
            )
        print(f"[Embeddings] No new items for {input_jsonl}; skipping.")
        return embs_all, np.empty(
            (0, embs_all.shape[1] if embs_all.size else 0), dtype="float32"
        )

    new_embs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    for i, entry in enumerate(data):
        entry["vec_id"] = i + vec_offset

    dir_path = os.path.dirname(output_npy)
    os.makedirs(dir_path or ".", exist_ok=True)
    if existing_embs is not None:
        embs_all = np.vstack([existing_embs, new_embs])
    else:
        embs_all = new_embs
    np.save(output_npy, embs_all)

    mode = "a" if vec_offset > 0 else "w"
    dir_path = os.path.dirname(output_jsonl)
    os.makedirs(dir_path or ".", exist_ok=True)
    with open(output_jsonl, mode + "t", encoding="utf-8") as f_out:
        for d in data:
            f_out.write(json.dumps(d) + "\n")

    print(
        f"[Embeddings] Saved {len(data)} new vectors to {output_npy} and updated JSONL {output_jsonl}"
    )
    return embs_all, new_embs



###  FAISS dense repesentation + index



def build_and_save_faiss_index(
    embeddings: np.ndarray,
    dataset_name: str,
    index_type: str,
    output_dir: str = ".",
    new_vectors: np.ndarray | None = None,
):
    """Build or update a FAISS cosine-similarity index.

    If ``new_vectors`` is provided and an existing index file is found, the new
    vectors are appended to that index. Otherwise, a fresh index is built from
    ``embeddings``. 
    """
    if not index_type or index_type not in {"passages", "iqoq", "iq"}:
        raise ValueError(
            "index_type must be provided and set to 'passages', 'iqoq', or 'iq'."
        )

    faiss_path = os.path.join(output_dir, f"{dataset_name}_faiss_{index_type}.faiss")

    if new_vectors is not None and os.path.exists(faiss_path):
        index = faiss.read_index(faiss_path)
        index.add(new_vectors)
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

    faiss.write_index(index, faiss_path)

    print(f"[FAISS] Saved {index_type} index to {faiss_path} with {index.ntotal} vectors.")





def load_faiss_index(path: str):
    """Load a FAISS index from ``path``."""
    index = faiss.read_index(path)
    print(f"[FAISS] Loaded {index.ntotal} vectors from {path}")
    return index


def faiss_search_topk(query_emb: np.ndarray, index, top_k: int = 50):
    """Retrieve ``top_k`` most similar items from a FAISS index."""
    scores, idx = index.search(query_emb, top_k)
    return idx[0], scores[0]







### sparse representation



def jaccard_similarity(set1: set, set2: set) -> float:
    return len(set1 & set2) / max(1, len(set1 | set2))


def retrieve_hybrid_candidates(
    query_vec: np.ndarray,
    query_keywords: Set[str],
    metadata: List[Dict],
    index,
    top_k: int,
    alpha: float,
    *,
    keyword_field: str | None = None,
    filter_fn: Callable[[int], bool] | None = None,
) -> List[Dict[str, float]]:
    """Hybrid retrieval over dense and sparse signals.

    Parameters
    ----------
    query_vec:
        Dense embedding for the query item.
    query_keywords:
        Set of sparse keywords associated with the query.
    metadata:
        Sequence of metadata dicts aligned with ``index``. Each item must have
        a keyword field (``keywords`` or ``keywords_passage``).
    index:
        FAISS index storing the dense vectors whose ordering corresponds to
        ``metadata``.
    top_k:
        Number of candidates to retrieve from each modality and number of
        results returned.
    alpha:
        Weighting factor for the hybrid score
        ``alpha * sim_cos + (1 - alpha) * sim_jac``.
    keyword_field:
        Optional override for the keyword field name in ``metadata``.
    filter_fn:
        Optional callable ``filter_fn(idx) -> bool`` applied to candidate
        indices. Candidates for which the function returns ``False`` are
        discarded.  This enables caller-specific filtering such as removing
        self-loops.

    Returns
    -------
    List[Dict[str, float]]
        Sorted list (descending ``sim_hybrid``) of dictionaries containing the
        candidate index and similarity scores: ``{"idx", "sim_cos",
        "sim_jac", "sim_hybrid"}``.  The list is truncated to ``top_k``
        entries.
    """

    if keyword_field is None and metadata:
        # Try to infer the keyword field name from the first entry
        if "keywords" in metadata[0]:
            keyword_field = "keywords"
        else:
            keyword_field = "keywords_passage"

    query_vec = query_vec.reshape(1, -1)

    # Dense retrieval via FAISS
    faiss_idxs, faiss_scores = faiss_search_topk(query_vec, index, top_k=top_k)
    n_meta = len(metadata)
    # Filter out invalid indices and associated scores
    valid_pairs = [
        (int(idx), float(score))
        for idx, score in zip(faiss_idxs, faiss_scores)
        if idx != -1 and 0 <= int(idx) < n_meta
    ]
    faiss_dict = {idx: score for idx, score in valid_pairs}

    # Sparse retrieval via Jaccard
    if query_keywords:
        jaccard_scores = [
            jaccard_similarity(query_keywords, set(m.get(keyword_field, []))) for m in metadata
        ]
        jaccard_top = set(
            int(i) for i in np.argsort(jaccard_scores)[::-1][:top_k]
        )
    else:
        logger.warning("No query keywords provided; skipping Jaccard retrieval")
        jaccard_scores = [0.0] * n_meta
        jaccard_top = set()
    candidate_idxs = set(faiss_dict.keys()).union(jaccard_top)
    candidate_idxs = {i for i in candidate_idxs if 0 <= i < n_meta}
    if filter_fn is not None:
        candidate_idxs = {i for i in candidate_idxs if filter_fn(i)}

    # Ensure query vector normalised for cosine calculation
    q_norm = query_vec[0] / max(1e-12, np.linalg.norm(query_vec))

    results: List[Dict[str, float]] = []
    for idx in candidate_idxs:
        meta = metadata[idx]
        if idx in faiss_dict:
            sim_cos = faiss_dict[idx]
        else:
            vec = index.reconstruct(int(idx))
            vec = vec / max(1e-12, np.linalg.norm(vec))
            sim_cos = float(np.dot(q_norm, vec))

        sim_jac = jaccard_scores[idx]
        sim_hybrid = alpha * sim_cos + (1.0 - alpha) * sim_jac
        results.append(
            {
                "idx": idx,
                "sim_cos": round(sim_cos, 4),
                "sim_jac": round(sim_jac, 4),
                "sim_hybrid": round(sim_hybrid, 4),
            }
        )

    results.sort(key=lambda x: x["sim_hybrid"], reverse=True)
    return results[:top_k]

SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")

nlp = spacy.load(SPACY_MODEL, disable=["parser", "textcat"])

# Keep only these named-entity types
KEEP_ENTS = {
    "PERSON","ORG","GPE","LOC","FAC","PRODUCT",
    "WORK_OF_ART","EVENT","LAW"  # drop: DATE, TIME, CARDINAL, ORDINAL, LANGUAGE, NORP
}



ALIAS = {
  # United States variants
  "u": "united_states", "us": "united_states", "u_s": "united_states",
  "u_s_a": "united_states", "united_states_of_america": "united_states",
  "the_united_states": "united_states",
  "u_s_navy": "united_states_navy",
  "u_s_air_force": "united_states_air_force",
  "u_s_army": "united_states_army",
  "u_s_marine_corps": "united_states_marine_corps",

  # United Kingdom / UN
  "uk": "united_kingdom", "u_k": "united_kingdom",
  "great_britain": "united_kingdom",
  "un": "united_nations", "u_n": "united_nations",

  # Map junk to None (drop)
  "what_year": None,
  "the_years": None,
  "year": None,
}


_NOISE_PATTERNS = [
  re.compile(r"^\d{4}$"),            # bare years
  re.compile(r"^\d+$"),              # numbers
  re.compile(r"^(one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)$"),
]


def canonicalize_keyword(kw: str) -> str | None:
    # alias (single step)
    kw = ALIAS.get(kw, kw)
    if kw is None:
        return None
    # noise drop
    for pat in _NOISE_PATTERNS:
        if pat.match(kw):
            return None
    return kw



def strip_accents(t: str) -> str:
    """Transliterate ``t`` to ASCII by stripping accents and special letters."""

    t = unicodedata.normalize("NFKD", t)
    replacements = {"ß": "ss", "æ": "ae", "œ": "oe"}
    for src, tgt in replacements.items():
        t = t.replace(src, tgt)
    return t.encode("ascii", "ignore").decode("ascii")


def normalise_text(s: str) -> str:
    """Return a normalised keyword string suitable for comparisons."""

    if not s:
        return ""
    t = clean_text(s).lower()
    t = t.replace("’", "'")  # unify curly quote
    t = strip_accents(t)
    t = t.replace("&", " and ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\W+", "_", t.strip())
    t = re.sub(r"_s_", "_", t)  # drop possessive
    t = re.sub(r"_s$", "", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def extract_keywords(text: str) -> list[str]:
    if not text:
        return []
    doc = nlp(text)
    out = set()
    for ent in doc.ents:
        if ent.label_ in KEEP_ENTS and ent.text.strip():
            norm = normalise_text(ent.text)
            if norm:
                canon = canonicalize_keyword(norm)
                if canon:
                    out.add(canon)
    return sorted(out)









def extract_all_keywords(entry: dict) -> dict:
    """
    Adds 'keywords_iq', 'keywords_oq', and 'keywords_passage' to entry.
    Assumes 'IQs', 'OQs', and 'text' exist.
    """
    entry["keywords_iq"] = [extract_keywords(q) for q in entry.get("IQs", [])]
    entry["keywords_oq"] = [extract_keywords(q) for q in entry.get("OQs", [])]
    entry["keywords_passage"] = extract_keywords(entry.get("text", ""))

    # to remove old formatting 
    entry.pop("keywords_IQ", None)
    entry.pop("keywords_OQ", None)

    return entry



def add_keywords_to_passages_jsonl(
    passages_jsonl: str,
    merged_with_iqoq: bool = False,
    only_ids: Set[str] | None = None,
):
    rows = [json.loads(l) for l in open(passages_jsonl, "rt", encoding="utf-8")]
    if only_ids:
        targets = [r for r in rows if r.get("passage_id") in only_ids]
    else:
        targets = rows
    texts = [r.get("text", "") for r in targets]

    for r, doc in zip(targets, nlp.pipe(texts, batch_size=128, n_process=1)):
        kws = set()
        for ent in doc.ents:
            if ent.label_ in KEEP_ENTS and ent.text.strip():
                norm = normalise_text(ent.text)
                canon = canonicalize_keyword(norm)
                if canon:
                    kws.add(canon)
        r["keywords_passage"] = sorted(kws)
        if merged_with_iqoq:
            kws_iq = [extract_keywords(q) for q in (r.get("IQs") or [])]
            kws_oq = [extract_keywords(q) for q in (r.get("OQs") or [])]
            r["keywords_iq"] = kws_iq
            r["keywords_oq"] = kws_oq
            union = set(r["keywords_passage"])
            for lst in kws_iq:
                union.update(lst)
            for lst in kws_oq:
                union.update(lst)
            r["all_keywords"] = sorted(union)

    with open(passages_jsonl, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")




def add_keywords_to_iqoq_jsonl(
    iqoq_jsonl: str,
    out_field: str = "keywords",
    only_ids: Set[str] | None = None,
):
    rows = [json.loads(l) for l in open(iqoq_jsonl, "rt", encoding="utf-8")]
    if only_ids:
        targets = [r for r in rows if r.get("iqoq_id") in only_ids]
    else:
        targets = rows
    texts = [r.get("text", "") for r in targets]

    for r, doc in zip(targets, nlp.pipe(texts, batch_size=128, n_process=1)):
        kws = set()
        for ent in doc.ents:
            if ent.label_ in KEEP_ENTS and ent.text.strip():
                norm = normalise_text(ent.text)
                canon = canonicalize_keyword(norm)
                if canon:
                    kws.add(canon)
        r[out_field] = sorted(kws)

    with open(iqoq_jsonl, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")











if __name__ == "__main__":
    print(f"[spaCy] Using: {SPACY_MODEL}")

    BGE_MODEL = os.environ.get("BGE_MODEL", "BAAI/bge-base-en-v1.5")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    bge_model = SentenceTransformer(BGE_MODEL, device=DEVICE)
    print(f"[BGE] Loaded {BGE_MODEL} on {DEVICE}")

    # Config
    MODELS   = ["qwen2.5-7b-instruct"]  # ["deepseek-dis
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    VARIANTS = ["baseline"]
    SPLIT = "dev"




    # -------------------------------
    # Phase A: Passages (Dataset-only)
    # -------------------------------
    for dataset in DATASETS:
        print(f"\n=== DATASET: {dataset} ({SPLIT}) ===")

        # File paths
        pass_paths = dataset_rep_paths(dataset, SPLIT)
        dataset_dir = Path(os.path.dirname(pass_paths["passages_jsonl"]))
        os.makedirs(dataset_dir, exist_ok=True)

        passages_jsonl_src  = str(processed_dataset_paths(dataset, SPLIT)["passages"])
        passages_jsonl      = pass_paths["passages_jsonl"]
        passages_npy        = pass_paths["passages_emb"]

        # === PASSAGE EMBEDDINGS ===
        if os.path.exists(passages_npy) and not RESUME:
            passages_emb = np.load(passages_npy).astype("float32")
            print(f"[skip] {passages_npy} exists; loaded.")
            if not os.path.exists(pass_paths["passages_index"]):
                build_and_save_faiss_index(
                    embeddings=passages_emb,
                    dataset_name=dataset,
                    index_type="passages",
                    output_dir=str(dataset_dir),
                )
        else:
            pass_items = load_jsonl(passages_jsonl_src)
            done_ids, shard_ids = compute_resume_sets(
                resume=RESUME,
                out_path=str(passages_jsonl),
                items=pass_items,
                get_id=lambda x, i: x["passage_id"],
                phase_label="passage embeddings",
                required_field="vec_id",
            )
            new_ids = shard_ids - done_ids
            passages_emb, new_pass_embs = embed_and_save(
                input_jsonl=passages_jsonl_src,
                output_npy=str(passages_npy),
                output_jsonl=str(passages_jsonl),
                model=bge_model,
                text_key="text",
                id_field="passage_id",
                done_ids=done_ids,
            )
            if new_pass_embs.size > 0:
                add_keywords_to_passages_jsonl(
                    str(passages_jsonl),
                    merged_with_iqoq=False,
                    only_ids=new_ids,
                )
                build_and_save_faiss_index(
                    embeddings=passages_emb,
                    dataset_name=dataset,
                    index_type="passages",
                    output_dir=str(dataset_dir),
                    new_vectors=new_pass_embs,
                )
            elif not os.path.exists(pass_paths["passages_index"]):
                build_and_save_faiss_index(
                    embeddings=passages_emb,
                    dataset_name=dataset,
                    index_type="passages",
                    output_dir=str(dataset_dir),
                )

        # Dataset-level phase only handles passage embeddings and indexing





    # -------------------------------
    # Phase B: IQ/OQ (Model-specific)
    # -------------------------------
    for dataset in DATASETS:
        for model in MODELS:
            for variant in VARIANTS:
                hoprag_version = f"{variant}_hoprag"
                print(
                    f"[Run] dataset={dataset} model={model} variant={variant} split={SPLIT}"
                )
                # Input paths for IQ/OQ data
                iqoq_exploded_src = (
                    f"data/models/{model}/{dataset}/{SPLIT}/{hoprag_version}/exploded/iqoq.exploded.jsonl"
                )


                # Output paths for representations
                repr_paths = model_rep_paths(model, dataset, SPLIT, variant)
                iqoq_jsonl = repr_paths["iqoq_jsonl"]
                iqoq_npy = repr_paths["iqoq_emb"]
                iqoq_index = repr_paths["iqoq_index"]
                iq_emb_path = repr_paths["iq_emb"]
                iq_index_path = repr_paths["iq_index"]
                os.makedirs(os.path.dirname(iqoq_jsonl), exist_ok=True)



                if not os.path.exists(iqoq_exploded_src):
                    print(
                        f"[warn] Missing IQ/OQ input file: exploded={iqoq_exploded_src}; skipping."
                    )

                if os.path.exists(iqoq_npy) and not RESUME:
                    iqoq_emb = np.load(iqoq_npy).astype("float32")
                    print(f"[skip] {iqoq_npy} exists; loaded.")
                    if not os.path.exists(iqoq_index):
                        build_and_save_faiss_index(
                            embeddings=iqoq_emb,
                            dataset_name=dataset,
                            index_type="iqoq",
                            output_dir=os.path.dirname(iqoq_index),
                        )
                else:
                    iqoq_items = load_jsonl(iqoq_exploded_src)
                    done_ids, shard_ids = compute_resume_sets(
                        resume=RESUME,
                        out_path=iqoq_jsonl,
                        items=iqoq_items,
                        get_id=lambda x, i: x["iqoq_id"],
                        phase_label="iqoq embeddings",
                        id_field="iqoq_id",
                        required_field="vec_id",
                    )
                    new_ids = shard_ids - done_ids
                    iqoq_emb, new_iqoq_embs = embed_and_save(
                        input_jsonl=iqoq_exploded_src,
                        output_npy=iqoq_npy,
                        output_jsonl=iqoq_jsonl,
                        model=bge_model,
                        text_key="text",
                        id_field="iqoq_id",
                        done_ids=done_ids,
                        output_jsonl_input=iqoq_exploded_src,
                    )
                    if new_iqoq_embs.size > 0:
                        add_keywords_to_iqoq_jsonl(iqoq_jsonl, only_ids=new_ids)
                        build_and_save_faiss_index(
                            embeddings=iqoq_emb,
                            dataset_name=dataset,
                            index_type="iqoq",
                            output_dir=os.path.dirname(iqoq_index),
                            new_vectors=new_iqoq_embs,
                        )
                    elif not os.path.exists(iqoq_index):
                        build_and_save_faiss_index(
                            embeddings=iqoq_emb,
                            dataset_name=dataset,
                            index_type="iqoq",
                            output_dir=os.path.dirname(iqoq_index),
                        )

                # Build IQ-only embeddings and index
                iq_md_all = [m for m in load_jsonl(iqoq_jsonl) if m.get("type") == "IQ"]
                if iq_md_all:
                    iq_vec_ids = [m["vec_id"] for m in iq_md_all]
                    iq_emb = iqoq_emb[iq_vec_ids]
                    np.save(iq_emb_path, iq_emb)
                    build_and_save_faiss_index(
                        embeddings=iq_emb,
                        dataset_name=dataset,
                        index_type="iq",
                        output_dir=os.path.dirname(iq_index_path),
                    )

                print(
                    f"[Done] dataset={dataset} model={model} variant={variant} split={SPLIT}"
                )