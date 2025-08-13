






"""Utilities for passage reranking and answer generation.

Example
-------
>>> visited_passages = {"p1", "p2", "p3"}
>>> graph = nx.DiGraph()
>>> graph.add_node("p1", text="The capital is Paris", query_sim=0.9)
>>> graph.add_node("p2", text="France is in Europe", query_sim=0.5)
>>> graph.add_node("p3", text="Paris has the Louvre", query_sim=0.7)
>>> ccount = {"p1": 2, "p3": 1}  # visitation counts from traversal
>>> reranked = rerank_passages_by_helpfulness(
...     candidate_passages=list(visited_passages),
...     query_text="What is the capital of France?",
...     ccount=ccount,
...     graph=graph,
...     top_k=2,
... )
>>> reranked
[('p1', 0.95), ('p3', 0.60)]
>>> answer = ask_llm_with_passages(
...     query_text="What is the capital of France?",
...     passage_ids=[pid for pid, _ in reranked],
...     graph=graph,
...     server_url="http://localhost:8000",
... )
>>> answer
{'raw_answer': 'The capital of France is Paris',
 'normalised_answer': 'paris'}

Helpfulness scores are represented as floats (0.0–1.0) in the second
element of each tuple returned by :func:`rerank_passages_by_helpfulness`.
They are computed as the average of query similarity and the normalised
visit count for each passage.
"""




### end of: 3 sets of answers+metrics made with the dev query+gold set (is that right? not the train set?)
# - 1) dense retrieval only RAG
# - 2) standard hop-RAG with preceding steps (I don't think there's any actual difference at this point)
# - 3) enhanced hop-RAG with preceding steps




from typing import List, Dict, Optional, Tuple, Callable
import networkx as nx 
import re
import os
import json
import string
import numpy as np


from src.a2_text_prep import query_llm
from src.d_traversal import select_seed_passages, run_dev_set, hoprag_traversal_algorithm, enhanced_traversal_algorithm





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

