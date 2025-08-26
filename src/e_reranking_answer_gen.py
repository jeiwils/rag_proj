"""
Module Overview
---------------
Run multi-hop traversal over a QA dataset using LLM-guided graph expansion.

This script performs seeded retrieval for each question, then expands through
a directed graph of OQ→IQ edges using a local LLM to guide traversal. Results 
are saved per query and summarized globally.

It supports both baseline traversal (no node revisits) and enhanced traversal 
(allowing node revisits but not edge revisits). Outputs are stored *inside the 
graph variant directory* alongside the graph structure and edge logs.


Inputs
------

### data/graphs/{model}/{dataset}/{split}/{variant}/

- `{dataset}_{split}_graph.gpickle`
    Directed NetworkX graph. Nodes = passages, edges = OQ→IQ links.

### WILL THIS ALSO NEED INPUT FROM THE ACTUAL EDGES AND NODES.jsonl? ################################## 

### data/representations/datasets/{dataset}/{split}/

- `{dataset}_passages_emb.npy`, `{dataset}_faiss_passages.faiss`, `{dataset}_passages.jsonl`
    Dense passage embeddings, FAISS index, and passage metadata (including keywords and IDs).

### data/processed_datasets/{dataset}/{split}/questions.jsonl

- Query set (e.g., dev split) with:
    - `question_id`: unique identifier
    - `question`: question text
    - `gold_passages`: list of passage IDs with relevant information

Outputs
-------

### data/graphs/{model}/{dataset}/{split}/{variant}/traversal/

- `per_query_traversal_results.jsonl`  
    One entry per query with hop trace, visited nodes, and precision/recall/F1 metrics.

- `visited_passages.json`  
    Deduplicated union of all passages visited during traversal (used for answer reranking).

- `final_traversal_stats.json`  
    Aggregate metrics across the full query set (e.g., mean precision, recall, hop stats).


File Schemas
------------

### per_query_traversal_results.jsonl

Each line contains a dict with the full traversal trace and evaluation:

{
  "query_id": str,
  "gold_passages": List[str],
  "visited_passages": List[str],
  "visit_counts": Dict[str, int],
  "hop_trace": List[Dict],
  "final_metrics": {
    "precision": float,
    "recall": float,
    "f1": float
  },
  "traversal_algorithm": str
}


### visited_passages.json

A list of unique passage IDs visited across all queries:

[
  "passage_id_1",
  "passage_id_2",
  ...
]


### final_traversal_stats.json

Aggregated summary across all queries:

{
  "mean_precision": float,
  "mean_recall": float,
  "passage_coverage_all_gold_found": int,
  "initial_retrieval_coverage": int,
  "avg_hops_before_first_gold": "TODO",
  "avg_total_hops": float,
  "avg_repeat_visits": float,
  "avg_none_count_per_query": float,
  "max_hop_depth_reached": int,
  "hop_depth_counts": List[int]
}


Notes
-----

- Traversals are run on the `dev` split, not `train`.
- `baseline` = no revisits to previously visited passages.
- `enhanced` = allows passage revisits but avoids revisiting the same edge.
- All outputs are saved in:
  `data/graphs/{model}/{dataset}/{split}/{variant}/traversal/`
"""


from src.utils import (
    get_result_paths,
    get_traversal_paths,
    append_jsonl,
    processed_dataset_paths,
)

from typing import List, Dict, Optional, Tuple

import json
import os
import re
import string

import numpy as np
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer


from src.utils import load_jsonl
from src.b_sparse_dense_representations import dataset_rep_paths, load_faiss_index
from src.a2_text_prep import query_llm, strip_think, is_r1_like
from src.utils import SERVER_CONFIGS
from src.d_traversal import (
    select_seed_passages,
    run_dev_set,
    hoprag_traversal_algorithm,
    enhanced_traversal_algorithm,
)




# ANSWER GENERATION

TOP_K_ANSWER_PASSAGES = 5 

################################################################################################################
# PASSAGE RERANKING AND ANSWER GENERATION 
################################################################################################################



def compute_helpfulness( # helper function for rerank_passages_by_helpfulness()
    vertex_id: str,
    vertex_query_sim: float, # similarity between the passage and the query
    ccount: dict
) -> float:

    """
    Compute a numeric helpfulness score for a passage.

    Args:
        vertex_id: Identifier of the passage vertex.
        vertex_query_sim: Similarity between the passage and the query in ``[0, 1]``.
        ccount: Mapping of passage IDs to visitation counts during traversal.

    Returns:
        float: Helpfulness score in ``[0, 1]`` where higher values indicate
        passages that are both similar to the query and frequently visited.
        The score is the average of ``vertex_query_sim`` and the normalised
        visit count (importance).
    """
    total_visits = sum(ccount.values()) or 1
    importance = ccount.get(vertex_id, 0) / total_visits

    # HopRAG helpfulness formula: (SIM + IMP) / 2
    helpfulness = 0.5 * (
        vertex_query_sim + importance
        ) # similarity between the passage and the query
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
    passage_lookup: Optional[Dict[str, str]] = None,  # optional for dense mode
    model_name: str = ""
) -> Dict[str, str]:
    """Generate an answer from top passages using an LLM server.

    Inputs
    ------
    query_text : str
        User question to be answered.
    passage_ids : List[str]
        Visited passage identifiers ranked by helpfulness.
    graph : Optional[nx.DiGraph]
        Graph containing passage texts. ``None`` if passages are looked up
        elsewhere.
    server_url : str
        URL of the LLM completion endpoint.
    max_tokens : int, optional
        Maximum number of tokens to generate, by default ``100``.
    passage_lookup : Optional[Dict[str, str]]
        Mapping from ``passage_id`` to passage text when ``graph`` is ``None``.
    model_name : str, optional
        Identifier of the active model, passed to ``query_llm``.

    Outputs
    -------
    Dict[str, str]
        ``{"raw_answer": str, "normalised_answer": str}``.
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

    raw = query_llm(
        prompt,
        server_url=server_url,
        max_tokens=max_tokens,
        model_name=model_name,
    )

    if is_r1_like(model_name):
        raw = strip_think(raw)

    norm = normalise_answer(raw)
    return {"raw_answer": raw, "normalised_answer": norm}






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
    server_configs: List[Dict] = SERVER_CONFIGS,
    output_path=None,
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
    if output_path is None:
        raise ValueError("output_path must be provided to run_dense_rag_baseline")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wt", encoding="utf-8") as f:
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
            server_url=server_configs[0]["server_url"],
            max_tokens=128,
            model_name=server_configs[0]["model"]
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

        append_jsonl(output_path, result)

    print(f"\n[Done] Saved dense RAG results to {output_path}")



def run_pipeline(
    mode: str,
    query_data: List[Dict],
    graph: Optional[nx.DiGraph],
    passage_metadata: List[Dict],
    passage_emb: np.ndarray,
    passage_index,
    emb_model,
    server_configs: List[Dict] = SERVER_CONFIGS,
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
            server_configs=server_configs,
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
            model_servers=server_configs,
            output_path=output_path,
            seed_top_k=seed_top_k,
            alpha=alpha,
            n_hops=n_hops,
            traveral_alg=hoprag_traversal_algorithm,
        )
    elif mode == "enhanced":
        run_dev_set(
            query_data=query_data,
            graph=graph,
            passage_metadata=passage_metadata,
            passage_emb=passage_emb,
            passage_index=passage_index,
            emb_model=emb_model,
            model_servers=server_configs,
            output_path=output_path,
            seed_top_k=seed_top_k,
            alpha=alpha,
            n_hops=n_hops,
            traveral_alg=enhanced_traversal_algorithm,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")







if __name__ == "__main__":
    # === Configuration ===
    DATASETS = ["hotpotqa"]
    SPLITS = ["dev"]
    MODELS = ["qwen-7b"]
    VARIANTS = ["baseline", "enhanced"]  # matches the traversal variants

    emb_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    server_configs = SERVER_CONFIGS
    seed_top_k = 50
    alpha = 0.5
    n_hops = 2

    for dataset in DATASETS:
        for split in SPLITS:
            # --- Load data for this dataset + split ---
            rep_paths = dataset_rep_paths(dataset, split)
            passage_metadata = list(load_jsonl(rep_paths["passages_jsonl"]))
            passage_emb = np.load(rep_paths["passages_emb"])["embs_all"]
            passage_index = load_faiss_index(rep_paths["passages_index"])

            query_path = str(processed_dataset_paths(dataset, split)["questions"])
            query_data = list(load_jsonl(query_path))

            for model in MODELS:
                # --- Run DENSE baseline ---
                result_paths = get_result_paths(model, dataset, split, variant="baseline")  # Dense doesn't depend on variant but keep structure unified

                print("\n========== Running DENSE RAG ==========")
                run_pipeline(
                    mode="dense",
                    query_data=query_data,
                    graph=None,
                    passage_metadata=passage_metadata,
                    passage_emb=passage_emb,
                    passage_index=passage_index,
                    emb_model=emb_model,
                    server_configs=server_configs,
                    output_path=result_paths["answers"],
                    seed_top_k=seed_top_k,
                    alpha=alpha
                )

                # --- Run HopRAG and Enhanced ---
                for variant in VARIANTS:
                    print(f"\n========== Running {variant.upper()} HopRAG ==========")

                    graph_path = os.path.join(
                        "data", "graphs", model, dataset, split, variant,
                        f"{dataset}_{split}_graph.gpickle"
                    )
                    
                    graph = nx.read_gpickle(graph_path)

                    result_paths = get_result_paths(model, dataset, split, variant)

                    run_pipeline(
                        mode=variant,
                        query_data=query_data,
                        graph=graph,
                        passage_metadata=passage_metadata,
                        passage_emb=passage_emb,
                        passage_index=passage_index,
                        emb_model=emb_model,
                        server_configs=server_configs,
                        output_path=result_paths["answers"],
                        seed_top_k=seed_top_k,
                        alpha=alpha,
                        n_hops=n_hops
                    )

    print("\n✅ All pipelines complete.")
