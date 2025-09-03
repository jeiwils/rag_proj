"""
Module Overview
---------------
Run multi-hop traversal over a QA dataset using LLM-guided graph expansion.

This script performs seeded retrieval for each question, then expands through
a directed graph of OQ→IQ edges using a local LLM to guide traversal. Results
are saved per query and summarized globally.

It supports both baseline traversal (no node revisits) and an enhanced variant
that biases edge ordering by each destination node's ``conditioned_score``
while still avoiding node revisits. Both variants forward an optional
``seed`` to llama.cpp so edge choices are reproducible. Outputs are stored in
``data/traversal/{model}/{dataset}/{split}/{variant}/``.


Inputs
------

### data/traversal/{model}/{dataset}/{split}/{variant}/

- `{dataset}_{split}_graph.gpickle`  
    Directed NetworkX graph. Nodes = passages, edges = OQ→IQ links.

- `{dataset}_{split}.jsonl`  
    Preprocessed query set, with `question_id`, `question`, and `gold_passages`.

- `passages_emb.npy`, `passages_index.faiss`, `passages.jsonl`
    Passage embeddings, FAISS index, and metadata from dense/sparse encoder setup.


Outputs
-------

### data/graphs/{model}/{dataset}/{split}/{variant}/traversal/

  - `per_query_traversal_results.jsonl'
      One entry per query with hop trace, visited nodes, precision/recall/F1, and
      retrieval metrics (hits@k, recall@k).

- `final_traversal_stats.json`
    Aggregate metrics across the full query set (e.g., mean precision, recall,
    hits@k, recall@k, hop stats).



File Schema
-----------

### per_query_traversal_results.jsonl

{
  "question_id": "{question_id}",
  "gold_passages": ["{passage_id_1}", "..."],
  "visited_passages": ["{passage_id_1}", "..."],
  "visit_counts": {"{passage_id}": count, ...},
  "hop_trace": [<hop dicts>],
  "final_metrics": {
    "precision": float,
    "recall": float,
    "f1": float
  },
  "hits_at_k": float,
  "recall_at_k": float,
  "traversal_algorithm": "{algorithm_name}",
  "wall_time_sec": float
}

### final_traversal_stats.json

{
  "timestamp": "2025-08-13T14:22:31",
  "traversal_eval": {
    "mean_precision": float,
    "mean_recall": float,
    "mean_f1": float,
    "mean_hits_at_k": float,
    "mean_recall_at_k": float,
    "passage_coverage_all_gold_found": int,
    "initial_retrieval_coverage": int,
    "avg_hops_before_first_gold": float | null,
    "avg_total_hops": float,
    "avg_repeat_visits": float,
    "avg_none_count_per_query": float,
    "max_hop_depth_reached": int,
    "hop_depth_distribution": [int, int, ...],
    "wall_time_total_sec": float,
    "wall_time_mean_sec": float,
    "wall_time_median_sec": float
  }
}
"""


import json
import pickle
import re
import inspect
import time 
import random
import string 

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import faiss
import networkx as nx
import numpy as np
from tqdm import tqdm
from datetime import datetime


from src.a2_text_prep import is_r1_like, query_llm, strip_think
from src.b_sparse_dense_representations import (
    dataset_rep_paths,
    extract_keywords,
    retrieve_hybrid_candidates,
    get_embedding_model,
    jaccard_similarity,
    faiss_search_topk,
    build_and_save_faiss_index,
)

from src.c_graphing import DEFAULT_EDGE_BUDGET_ALPHA
from src.utils import (
    SERVER_CONFIGS,
    append_jsonl,
    compute_resume_sets,
    get_server_configs,
    get_server_urls,
    get_traversal_paths,
    load_jsonl,
    processed_dataset_paths,
    run_multiprocess,
    save_jsonl,
    split_jsonl_for_models,
    model_size,
    pool_map,
    compute_hits_at_k,
    compute_recall_at_k,

    log_wall_time,
    validate_vec_ids,
    merge_token_usage,

)
from src.metrics_summary import append_traversal_percentiles
import os

class TraversalOutputError(Exception):
    pass




# # TRAVERSAL TUNING

DEFAULT_SEED_TOP_K = 20
DEFAULT_NUMBER_HOPS = 2
DEFAULT_RETRIEVER_NAME = "hybrid"






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
    Compute helpfulness scores for candidate passages using query similarity and visit frequency,
    returning the top-k ranked list.

    Returns:
        List of tuples: [(passage_id, helpfulness_score), ...]
    """
    reranked = []
    for pid in candidate_passages:
        node = graph.nodes.get(pid, {})
        vertex_query_sim = node.get("query_sim", 0.0)  # precomputed similarity stored on the node

        score = compute_helpfulness(pid, vertex_query_sim, ccount)
        reranked.append((pid, score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]

################################################################################################################
# 6. RETRIEVAL AND GRAPH TRAVERSAL  
################################################################################################################



### COMPONENT 1 - retrieval 



def select_seed_passages(  # helper for run_dev_set()
    query_text: str,
    query_emb: np.ndarray,
    passage_metadata: List[Dict],
    passage_index,
    seed_top_k: int = 50,
    alpha: float = 0.5,
    question_id: str | None = None,
) -> List[str]:
    """Select top seed passages using dense (FAISS) and sparse (Jaccard) signals.

    The hybrid score is ``alpha * sim_cos + (1 - alpha) * sim_jac`` where
    ``alpha`` weights dense vs. sparse similarity. Logs the top ``seed_top_k``
    FAISS and Jaccard results for ``question_id`` to help verify that
    retrieval varies across queries.
    """

    if passage_index.ntotal != len(passage_metadata):
        raise ValueError(
            "FAISS index size mismatch: index has "
            f"{passage_index.ntotal} vectors but metadata lists "
            f"{len(passage_metadata)} passages"
        )

    query_keywords = set(extract_keywords(query_text))

    # Log FAISS dense results
    faiss_idxs, faiss_scores = faiss_search_topk(
        query_emb.reshape(1, -1), passage_index, top_k=seed_top_k
    )
    faiss_pairs = [
        (passage_metadata[int(i)]["passage_id"], float(s))
        for i, s in zip(faiss_idxs, faiss_scores)
    ]

    # Log Jaccard keyword overlap results
    jac_pairs = [
        (
            m["passage_id"],
            jaccard_similarity(query_keywords, set(m.get("keywords_passage", []))),
        )
        for m in passage_metadata
    ]
    jac_pairs.sort(key=lambda x: x[1], reverse=True)
    jac_pairs = jac_pairs[:seed_top_k]

    if question_id is not None:
        print(f"[select_seed_passages][{question_id}] FAISS top: {faiss_pairs[:5]}")
        print(f"[select_seed_passages][{question_id}] Jaccard top: {jac_pairs[:5]}")

    candidates = retrieve_hybrid_candidates(
        query_emb,
        query_keywords,
        passage_metadata,
        passage_index,
        top_k=seed_top_k,
        alpha=alpha,
        keyword_field="keywords_passage",
    )

    return [passage_metadata[c["idx"]]["passage_id"] for c in candidates]


### COMPONENT 2 - traversal 




def llm_choose_edge(
    query_text: str,
    candidate_edges: list,
    graph: nx.DiGraph,
    server_configs: list,
    traversal_prompt: str,
    token_totals: Optional[Dict[str, int]] = None,
    reason: bool = True,
    seed: int | None = None,

):
    """Ask the local LLM to choose among multiple outgoing edges.

    ``candidate_edges`` **must** already be sorted by the caller in whatever
    deterministic order is desired. ``graph`` is included for interface
    compatibility but is not used.

    The ``traversal_prompt`` is a template containing ``{MAIN_QUESTION}`` and
    ``{CANDIDATE_LIST}`` placeholders.  The LLM receives the formatted prompt
    enumerating all candidate auxiliary questions and must respond with JSON of
    the form::

        {"edge_index": int | null}

    The LLM receives a single prompt enumerating all candidate auxiliary
    questions and must respond **only** with an integer ``0..k-1`` or the
    literal string ``null`` if none apply. The returned integer refers to
    the zero-based position of the chosen edge in ``candidate_edges``. The
    function returns the selected edge tuple or ``None`` if ``null`` is
    returned.
    """


    oq_server = server_configs[1] if len(server_configs) > 1 else server_configs[0]
    k = len(candidate_edges)

    # Format prompt template with main question and candidate list
    option_lines = [
        f"{i}. {edge_data['oq_text']}"
        for i, (_, edge_data) in enumerate(candidate_edges)
    ]
    options = "\n".join(option_lines)
    prompt = traversal_prompt.format(
        MAIN_QUESTION=query_text,
        CANDIDATE_LIST=options,
    )

    grammar_choices = " | ".join(
        [f'"{i}"' for i in range(k)] + ['"null"']
    )
    grammar = f"root ::= {grammar_choices}\n"

    def _record_usage(usage: Optional[dict]):
        if token_totals is not None and usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            token_totals["trav_prompt_tokens"] += prompt_tokens
            token_totals["trav_output_tokens"] += completion_tokens
            token_totals["trav_tokens_total"] += usage.get(
                "total_tokens", prompt_tokens + completion_tokens
            )


    answer, usage = query_llm(
        prompt,
        server_url=oq_server["server_url"],
        max_tokens=1,
        temperature=0.7,
        model_name=oq_server["model"],
        phase="edge_selection",
        stop=None,
        reason=reason,
        grammar=grammar,
        top_p=0.95,
        top_k=0,
        mirostat=0,
        repeat_penalty=1.0,
        seed=seed,

    )

    if token_totals is not None:
        token_totals["n_traversal_calls"] += 1

    _record_usage(usage)


    if token_totals is not None and usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        token_totals["trav_prompt_tokens"] += prompt_tokens
        token_totals["trav_output_tokens"] += completion_tokens
        token_totals["trav_tokens_total"] += usage.get(
            "total_tokens", prompt_tokens + completion_tokens
        )

    if is_r1_like(oq_server["model"]):
        answer = strip_think(answer)

    answer = answer.strip()
    retry_count = 0
    while True:
        if answer == "null":
            mode = "null"
            break
        if re.fullmatch(r"[0-9]+", answer):
            mode = "int"
            break
        if retry_count == 0:
            retry_count = 1
            answer, usage = query_llm(
                prompt,
                server_url=oq_server["server_url"],
                max_tokens=1,
                temperature=0.7,
                model_name=oq_server["model"],
                phase="edge_selection",
                stop=None,
                reason=reason,
                grammar=grammar,
                top_p=0.95,
                top_k=0,
                mirostat=0,
                repeat_penalty=1.0,
                seed=seed,

            )
            if token_totals is not None:
                token_totals["n_traversal_calls"] += 1
            _record_usage(usage)
            if is_r1_like(oq_server["model"]):
                answer = strip_think(answer)
            answer = answer.strip()
            continue
        raise TraversalOutputError(answer)

    print(f"[Traversal] grammar=dynamic retry={retry_count} mode={mode}")

    if mode == "null":
        return None

    idx = int(answer)
    if not (0 <= idx < len(candidate_edges)):
        raise TraversalOutputError(answer)

    print(f"[Traversal] selected idx={idx}")
    return candidate_edges[idx]








def hoprag_traversal_algorithm(
    vj,
    graph,
    query_text,
    visited_passages,
    server_configs,
    ccount,
    next_Cqueue,
    hop_log,
    state,
    traversal_prompt: str,
    hop: int = 0,
    token_totals: Optional[Dict[str, int]] = None,
    seed: int | None = None,

    **kwargs,
):
    """Single HopRAG traversal step as described in the HopRAG article.

    Follows the article's step terminology:

    1. **Gather edges** – collect outgoing edges from ``vj`` that are not in
       ``state['Evisited']`` (HopRAG Step 1: eligible edge expansion).
    2. **Sort deterministically** – order candidates for reproducible traversal
       (HopRAG Step 2: canonical ordering).
    3. **Log candidates** – record each option in ``hop_log['candidate_edges']``
       (HopRAG Step 3: trace logging).
    4. **LLM selection** – invoke ``llm_choose_edge`` to pick an edge
       (HopRAG Step 4: LLM-guided choice).
    5. **Handle none** – increment counters and return if no edge is chosen
       (HopRAG Step 5: none case).
    6. **Record choice** – add the selected edge to ``state['Evisited']`` and
       note whether the destination was already visited
       (HopRAG Step 6: edge commit).
    7. **Update counts/queue** – update visit counts, enqueue unseen nodes, and
       log repeat visits (HopRAG Step 7: visit accounting).

    ``hop`` exists for API compatibility but is otherwise unused. Providing a
    ``seed`` enables deterministic llama.cpp edge selection via
    :func:`llm_choose_edge`.
    """
    # Consider all outgoing edges from ``vj``.  We rely solely on
    # ``state['Evisited']`` to avoid traversing the exact same edge more
    # than once; even if a target node has been visited, its edge is still a
    # candidate (the node simply won't be re-queued below).
    candidates = [
        (vk, graph[vj][vk])
        for vk in graph.successors(vj)
        if (vj, vk, graph[vj][vk]["oq_id"], graph[vj][vk]["iq_id"]) not in state["Evisited"]
    ]

    if not candidates:
        return set()

    # Ensure deterministic ordering for edge options
    candidates.sort(key=lambda item: (item[1].get("oq_id", ""), item[0]))

    hop_log["candidate_edges"].extend(
        [
            (
                vj,
                vk,
                edge_data.get("oq_id"),
                edge_data.get("iq_id"),
            )
            for vk, edge_data in candidates
        ]
    )

    # LLM selects among the candidates once and returns the chosen edge
    edge_model = (
        server_configs[1]["model"] if len(server_configs) > 1 else server_configs[0]["model"]
    )
    chosen = llm_choose_edge(
        query_text=query_text,
        candidate_edges=candidates,
        graph=graph,
        server_configs=server_configs,
        traversal_prompt=traversal_prompt,
        token_totals=token_totals,
        reason=is_r1_like(edge_model),
        seed=seed,

    )

    if chosen is None:
        hop_log["none_count"] += 1
        state["none_count"] += 1
        return set()

    chosen_vk, chosen_edge = chosen
    state["Evisited"].add((vj, chosen_vk, chosen_edge["oq_id"], chosen_edge["iq_id"]))
    is_repeat = chosen_vk in visited_passages

    hop_log["edges_chosen"].append({
        "from": vj,
        "to": chosen_vk,
        "oq_id": chosen_edge["oq_id"],
        "iq_id": chosen_edge["iq_id"],
        "repeat_visit": is_repeat,
    })

    ccount[chosen_vk] = ccount.get(chosen_vk, 0) + 1

    if is_repeat:
        hop_log["repeat_visit_count"] += 1
        state["repeat_visit_count"] += 1
        return set()

    hop_log["new_passages"].append(chosen_vk)
    # only enqueue if it's actually new (or if you explicitly want limited revisits)
    if not is_repeat:
        next_Cqueue.append(chosen_vk)
        return {chosen_vk}
    else:
        return set()

















def enhanced_traversal_algorithm(
    vj,
    graph,
    query_text,
    visited_passages,
    server_configs,
    ccount,
    next_Cqueue,
    hop_log,
    state,
    traversal_prompt: str,
    hop: int,
    token_totals: Optional[Dict[str, int]] = None,
    seed: int | None = None,
    **kwargs,
):
    """Enhanced traversal with hop-aware conditioned score bias.

    Mirrors :func:`hoprag_traversal_algorithm`'s no-revisit policy but orders
    untraversed edges by each destination node's ``conditioned_score``. At
    ``hop == 0`` edges are ranked in descending order (high-score first); for
    later hops the order is ascending to explore lower scored options sooner.
    All eligible edges are shown to the LLM for selection. Supplying ``seed``
    yields deterministic llama.cpp decisions via :func:`llm_choose_edge`.
    """

    # 1) Gather outgoing edges not yet traversed
    candidates = [
        (vk, graph[vj][vk])
        for vk in graph.successors(vj)
        if (vj, vk, graph[vj][vk]["oq_id"], graph[vj][vk]["iq_id"]) not in state["Evisited"]
    ]

    if not candidates:
        return set()

    # 2) Sort for deterministic conditioned_score prioritisation.
    #    First ensure a stable base order using (oq_id, destination id), then
    #    order by conditioned_score depending on hop depth.  This keeps ties
    #    deterministic; ``llm_choose_edge`` preserves the ordering provided.
    candidates.sort(key=lambda it: (it[1].get("oq_id", ""), it[0]))
    reverse = hop == 0  # descending when hop==0, else ascending
    candidates.sort(
        key=lambda it: graph.nodes[it[0]].get("conditioned_score", 0.0),
        reverse=reverse,
    )


    hop_log["candidate_edges"].extend(
        [
            (
                vj,
                vk,
                edge_data.get("oq_id"),
                edge_data.get("iq_id"),
            )
            for vk, edge_data in candidates
        ]
    )
    # 3) Ask LLM to pick the next edge among the candidates
    edge_model = (
        server_configs[1]["model"] if len(server_configs) > 1 else server_configs[0]["model"]
    )
    chosen = llm_choose_edge(
        query_text=query_text,
        candidate_edges=candidates,
        graph=graph,
        server_configs=server_configs,
        traversal_prompt=traversal_prompt,
        token_totals=token_totals,
        reason=is_r1_like(edge_model),
        seed=seed,
    )

    if chosen is None:
        hop_log["none_count"] += 1
        state["none_count"] += 1
        return set()

    chosen_vk, chosen_edge = chosen

    # 4) Mark edge as traversed
    state["Evisited"].add((vj, chosen_vk, chosen_edge["oq_id"], chosen_edge["iq_id"]))

    # 5) Queueing mirrors baseline: only enqueue new nodes
    is_repeat = chosen_vk in visited_passages
    hop_log["edges_chosen"].append({
        "from": vj,
        "to": chosen_vk,
        "oq_id": chosen_edge["oq_id"],
        "iq_id": chosen_edge["iq_id"],
        "repeat_visit": is_repeat,
    })

    ccount[chosen_vk] = ccount.get(chosen_vk, 0) + 1

    if is_repeat:
        hop_log["repeat_visit_count"] += 1
        state["repeat_visit_count"] += 1
        return set()

    hop_log["new_passages"].append(chosen_vk)
    next_Cqueue.append(chosen_vk)
    return {chosen_vk}

















def traverse_graph(
    graph: nx.DiGraph,
    query_text: str,
    query_emb: np.ndarray,
    passage_emb: np.ndarray,
    seed_passages: list,
    n_hops: int,
    server_configs: list,
    traversal_alg: Callable,  # custom algorithm step (edge + queueing logic)
    alpha: float = DEFAULT_EDGE_BUDGET_ALPHA,
    traversal_prompt: str = "",
    token_totals: Optional[Dict[str, int]] = None,
    seed: int | None = None,

):
    """Traverse the graph while recording query similarity for visited passages.

    ``alpha`` controls the hybrid weighting between cosine and Jaccard
    similarity when computing ``sim_hybrid`` for each visited node. The optional
    ``seed`` propagates to ``traversal_alg`` and ultimately
    :func:`llm_choose_edge` so llama.cpp edge decisions are reproducible.
    """

    query_keywords = set(extract_keywords(query_text))

    def _update_query_sim(pid: str) -> None:
        node = graph.nodes.get(pid)
        if not node:
            return
        vec_id = node.get("vec_id")
        sim_cos = 0.0
        if vec_id is not None and 0 <= vec_id < len(passage_emb):
            sim_cos = float(np.dot(query_emb, passage_emb[vec_id]))
        sim_jac = jaccard_similarity(
            query_keywords, set(node.get("keywords_passage", []))
        )
        sim_hybrid = alpha * sim_cos + (1 - alpha) * sim_jac
        node["query_sim"] = round(sim_hybrid, 4)

    Cqueue = seed_passages[:]
    for pid in Cqueue:
        _update_query_sim(pid)
    visited_passages = set()
    ccount = {pid: 1 for pid in Cqueue}
    hop_trace = []
    state = {
        "Evisited": set(),
        "none_count": 0,
        "repeat_visit_count": 0,
    }


    for hop in range(n_hops):
        next_Cqueue = []
        hop_log = {
            "hop": hop,
            "expanded_from": list(Cqueue),
            "new_passages": [],
            "edges_chosen": [],
            "candidate_edges": [],
            "none_count": 0,
            "repeat_visit_count": 0,
        }

        for vj in Cqueue:
            if vj not in graph:
                continue

            visited_passages.add(vj)

            new_nodes = traversal_alg(
                vj=vj,
                graph=graph,
                query_text=query_text,
                visited_passages=visited_passages,
                server_configs=server_configs,
                ccount=ccount,
                next_Cqueue=next_Cqueue,
                hop_log=hop_log,
                state=state,
                traversal_prompt=traversal_prompt,
                hop=hop,  # expose hop depth to traversal algorithm
                token_totals=token_totals,
                seed=seed,



            )
            for new_pid in new_nodes:
                _update_query_sim(new_pid)
            visited_passages.update(new_nodes)

        hop_trace.append(hop_log)
        Cqueue = next_Cqueue

    visited_passages.update(seed_passages)

    return list(visited_passages), ccount, hop_trace, {
        "none_count": state["none_count"],
        "repeat_visit_count": state["repeat_visit_count"],
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

    for idx, hop_log in enumerate(hop_trace):
        # include initial seed passages from the first hop so metrics
        # reflect already visited nodes even if no new edges are chosen
        if idx == 0:
            visited_cumulative.update(hop_log.get("expanded_from", []))

        visited_cumulative.update(hop_log.get("new_passages", []))
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








def compute_gold_attention(ccount: Dict[str, int], gold_passages: List[str]) -> Tuple[Dict[str, int], float]:
    """Compute visitation stats for gold passages.

    Args:
        ccount: Mapping of passage IDs to visitation counts during traversal.
        gold_passages: List of gold passage identifiers for the query.

    Returns:
        Tuple ``(gold_counts, attention_ratio)`` where ``gold_counts`` maps each
        gold passage ID to its visit count and ``attention_ratio`` is the fraction
        of total visits accounted for by gold passages.
    """

    gold_counts = {pid: ccount.get(pid, 0) for pid in gold_passages}
    total_visits = sum(ccount.values()) or 1
    attention_ratio = sum(gold_counts.values()) / total_visits
    return gold_counts, attention_ratio






def save_traversal_result(  # helper for run_dev_set()
    question_id,
    gold_passages,
    visited_passages,
    ccount,
    hop_trace,
    traversal_alg,
    helpful_passages,
    hits_at_k,
    recall_at_k,

    dataset: str,
    split: str,
    variant: str,
    retriever_name: str,
    traverser_model: str,
    reader_model: str | None = None,
    wall_time_sec: float | None = None,
    output_path="dev_results.jsonl",
    token_usage: Optional[Dict[str, int]] = None,
    seed: int | None = None,
):
    """
    Save a complete traversal + metric result for a single query.

    Records per-query traversal traces along with seed retrieval metrics such
    as ``hits_at_k`` and ``recall_at_k``.
    """

    hop_trace_with_metrics, final_metrics = compute_hop_metrics(hop_trace, gold_passages)
    gold_counts, gold_attention_ratio = compute_gold_attention(ccount, gold_passages)

    result_entry = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "retriever_name": retriever_name,
        "traverser_model": traverser_model,
        "reader_model": reader_model,
        "question_id": question_id,
        "gold_passages": gold_passages,
        "visited_passages": list(visited_passages),
        "visit_counts": dict(ccount),
        "gold_visit_counts": gold_counts,
        "gold_attention_ratio": round(gold_attention_ratio, 4),
        "hop_trace": hop_trace_with_metrics,
        "final_metrics": final_metrics,
        "traversal_algorithm": traversal_alg.__name__,
        "helpful_passages": [
            {"passage_id": pid, "score": round(score, 4)}
            for pid, score in helpful_passages
        ],
        "hits_at_k": hits_at_k,
        "recall_at_k": recall_at_k,

        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),

    }

    if seed is not None and "seed" not in result_entry:
        result_entry["seed"] = seed

    if token_usage is not None:
        result_entry["trav_prompt_tokens"] = token_usage.get("trav_prompt_tokens", 0)
        result_entry["trav_output_tokens"] = token_usage.get("trav_output_tokens", 0)
        result_entry["trav_tokens_total"] = token_usage.get(
            "trav_tokens_total",
            token_usage.get("trav_prompt_tokens", 0)
            + token_usage.get("trav_output_tokens", 0),
        )
        result_entry["n_traversal_calls"] = token_usage.get("n_traversal_calls", 0)



    if wall_time_sec is not None:
        result_entry["wall_time_sec"] = round(wall_time_sec, 4)



    append_jsonl(str(output_path), result_entry)













### all together now!!!



def run_traversal(
    query_data: List[Dict],
    graph: nx.DiGraph,
    passage_metadata: List[Dict],
    passage_emb: np.ndarray,
    passage_index,
    emb_model,
    server_configs: List[Dict],
    output_paths: Dict[str, Path],  # use traversal_output_paths()
    dataset: str,
    split: str,
    variant: str,
    traverser_model: str,
    retriever_name: str,
    seed_top_k=50,
    alpha=0.5,
    n_hops=2,
    traversal_alg=None,
    traversal_prompt: Optional[str] = None,
    seed: int | None = None,
):
    """Run LLM-guided multi-hop traversal over a QA query set (e.g., train, dev).

    The parameter ``alpha`` controls hybrid weighting between dense cosine and
    Jaccard similarity during seed selection and traversal scoring.

    Outputs
    -------
    - per_query_traversal_results.jsonl: 🔍 Full per-query trace and metrics
    - final_traversal_stats.json: 📈 Aggregate traversal metrics across the query set

    Parameters
    ----------
    seed: int, optional
        Seed used to initialize :mod:`random`, :mod:`numpy`, and the llama.cpp
        calls made during traversal for deterministic behaviour.
    """

    output_paths["base"].mkdir(parents=True, exist_ok=True)
        
    token_totals = {
        "trav_prompt_tokens": 0,
        "trav_output_tokens": 0,
        "trav_tokens_total": 0,
        "n_traversal_calls": 0,

    }
    per_query_usage: Dict[str, Dict[str, int]] = {}
    total_time_ms = 0



    if traversal_prompt is None:
        # Load default template with {MAIN_QUESTION} and {CANDIDATE_LIST}
        # placeholders used by ``llm_choose_edge``
        traversal_prompt = Path("data/prompts/traversal_prompt.txt").read_text()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for entry in tqdm(query_data, desc="queries"):
        start = time.perf_counter()
        question_id = entry["question_id"]
        query_text = entry["question"]
        gold_passages = entry["gold_passages"]
        query_token_totals = {
            "trav_prompt_tokens": 0,
            "trav_output_tokens": 0,
            "trav_tokens_total": 0,
            "n_traversal_calls": 0,

        }
        print(f"\n[Query] {question_id} - \"{query_text}\"")

        # --- Embed query ---
        query_emb = emb_model.encode(query_text, normalize_embeddings=False)
        norm = np.linalg.norm(query_emb)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(
                f"Query embedding norm invalid ({norm}); check emb_model.encode output."
            )
        
        # --- Select seed passages ---
        seed_passages = select_seed_passages(
            query_text=query_text,
            query_emb=query_emb,
            passage_metadata=passage_metadata,
            passage_index=passage_index,
            seed_top_k=seed_top_k,
            alpha=alpha,
            question_id=question_id,
        )

        print(f"[Seeds] Retrieved {len(seed_passages)} passages.")

        hits_val = compute_hits_at_k(seed_passages, gold_passages, seed_top_k)
        recall_val = compute_recall_at_k(seed_passages, gold_passages, seed_top_k)


        # --- Traverse ---
        visited_passages, ccount, hop_trace, stats = traverse_graph(
            graph=graph,
            query_text=query_text,
            query_emb=query_emb,
            passage_emb=passage_emb,
            seed_passages=seed_passages,
            n_hops=n_hops,
            server_configs=server_configs,
            traversal_alg=traversal_alg,
            alpha=alpha,
            traversal_prompt=traversal_prompt,
            token_totals=query_token_totals,
            seed=seed,


        )

        print(f"[Traversal] Visited {len(visited_passages)} passages (None={stats['none_count']}, Repeat={stats['repeat_visit_count']})")

        helpful_passages = rerank_passages_by_helpfulness(
            candidate_passages=visited_passages,
            query_text=query_text,
            ccount=ccount,
            graph=graph,
        )

        elapsed = time.perf_counter() - start
        elapsed_ms = int(elapsed * 1000)


        # --- Save per-query JSONL ---
        save_traversal_result(
            question_id=question_id,
            gold_passages=gold_passages,
            visited_passages=visited_passages,
            ccount=ccount,
            hop_trace=hop_trace,
            traversal_alg=traversal_alg,
            helpful_passages=helpful_passages,
            hits_at_k=hits_val,
            recall_at_k=recall_val,

            dataset=dataset,
            split=split,
            variant=variant,
            retriever_name=retriever_name,
            traverser_model=traverser_model,

            wall_time_sec=elapsed,
            output_path=output_paths["results"],
            token_usage=query_token_totals,
            seed=seed,

        )

        for k in token_totals:
            token_totals[k] += query_token_totals.get(k, 0)
        per_query_usage[question_id] = dict(query_token_totals)
        per_query_usage[question_id]["t_traversal_ms"] = elapsed_ms
        total_time_ms += elapsed_ms


    base_usage_path = output_paths.get(
        "token_usage", output_paths["base"] / "token_usage.json"
    )
    unique = f"{os.getpid()}_{int(time.time())}"
    token_usage_path = base_usage_path.with_name(
        f"{base_usage_path.stem}_{unique}{base_usage_path.suffix}"
    )
    global_usage = {k: v for k, v in token_totals.items()}
    global_usage["t_traversal_ms"] = total_time_ms

    tokens_total = global_usage.get("trav_tokens_total", 0)
    t_total_ms = global_usage.get("t_traversal_ms", 0)
    tps_overall = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0
    global_usage.update(
        {
            "tokens_total": tokens_total,
            "t_total_ms": t_total_ms,
            "tps_overall": tps_overall,
        }
    )

    usage = {"per_query_traversal": per_query_usage, **global_usage}
    with open(token_usage_path, "wt", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)

    print(f"[summary] total traversal tokens: {tokens_total}")
    print(f"[summary] traversal wall time: {t_total_ms} ms")
    print(f"[summary] overall throughput: {tps_overall:.2f} tokens/s")















### overall metrics


















# Backwards compatibility for older imports
run_dev_set = run_traversal



def compute_traversal_summary(
        results_path: str,
        include_ids: Optional[Set[str]] = None
        ) -> dict:
    """Summarize traversal-wide metrics across all dev results or a filtered subset.

    Computes aggregate precision, recall, F1, hits@k, recall@k, mean attention
    to gold passages, and various hop statistics over the provided per-query
    traversal results.
    """

    total_queries = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0

    sum_hits = 0.0
    sum_recall_at_k = 0.0

    sum_gold_attention = 0.0
    sum_traversal_calls = 0


    total_none = 0
    total_repeat = 0
    passage_coverage_all_gold_found = 0
    initial_retrieval_coverage = 0
    first_gold_hops = []
    query_hop_depths: List[int] = []
    wall_times: List[float] = []


    with open(results_path, "rt", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if include_ids is not None and entry["question_id"] not in include_ids:
                continue

            total_queries += 1

            final = entry["final_metrics"]
            sum_precision += final["precision"]
            sum_recall += final["recall"]
            sum_f1 += final.get("f1", 0.0)
            sum_hits += entry.get("hits_at_k", 0.0)
            sum_recall_at_k += entry.get("recall_at_k", 0.0)

            sum_gold_attention += entry.get("gold_attention_ratio", 0.0)
            sum_traversal_calls += entry.get("n_traversal_calls", 0)


            if "wall_time_sec" in entry:
                wall_times.append(entry["wall_time_sec"])


            if set(entry["gold_passages"]).issubset(set(entry["visited_passages"])):
                passage_coverage_all_gold_found += 1

            # Coverage at hop 0 (seed retrieval)
            gold_set = set(entry["gold_passages"])


            # what does this do?????
            hop_trace = entry.get("hop_trace", [])
            if hop_trace:
                hop0_passages = set(hop_trace[0].get("expanded_from", []))
                if hop0_passages & gold_set:
                    initial_retrieval_coverage += 1
                    first_gold_hop = 0
                else:
                    first_gold_hop = None
            else:
                # No hops were taken; treat coverage as zero
                first_gold_hop = None


            deepest_non_empty_hop = -1
            for hop_log in entry["hop_trace"]:
                total_none += hop_log["none_count"]
                total_repeat += hop_log["repeat_visit_count"]

                if hop_log.get("expanded_from") or hop_log.get("new_passages"):
                    deepest_non_empty_hop = hop_log["hop"]

                if first_gold_hop is None and set(hop_log.get("new_passages", [])) & gold_set:
                    first_gold_hop = hop_log["hop"]

            query_hop_depths.append(deepest_non_empty_hop + 1)

            if first_gold_hop is not None:
                first_gold_hops.append(first_gold_hop)

    mean_precision = sum_precision / total_queries if total_queries else 0
    mean_recall = sum_recall / total_queries if total_queries else 0
    mean_f1 = sum_f1 / total_queries if total_queries else 0
    mean_hits = sum_hits / total_queries if total_queries else 0
    mean_recall_at_k = sum_recall_at_k / total_queries if total_queries else 0

    mean_gold_attention = sum_gold_attention / total_queries if total_queries else 0
    mean_traversal_calls = sum_traversal_calls / total_queries if total_queries else 0



    avg_first_gold = (
        round(sum(first_gold_hops) / len(first_gold_hops), 2)
        if first_gold_hops
        else None
    )

    hop_depth_counter = Counter(query_hop_depths)
    max_depth = max(query_hop_depths) if query_hop_depths else 0
    hop_depth_distribution = [hop_depth_counter.get(i, 0) for i in range(max_depth + 1)]

    wall_time_total = sum(wall_times)
    wall_time_mean = wall_time_total / len(wall_times) if wall_times else 0.0
    wall_time_median = float(np.median(wall_times)) if wall_times else 0.0

    summary = {
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "mean_f1": round(mean_f1, 4),
        "mean_hits_at_k": round(mean_hits, 4),
        "mean_recall_at_k": round(mean_recall_at_k, 4),

        "mean_gold_attention_ratio": round(mean_gold_attention, 4),
        "avg_traversal_calls": round(mean_traversal_calls, 2),
        "total_traversal_calls": sum_traversal_calls,
        "passage_coverage_all_gold_found": passage_coverage_all_gold_found,
        "initial_retrieval_coverage": initial_retrieval_coverage,
        "avg_hops_before_first_gold": avg_first_gold,
        "avg_total_hops": round(sum(query_hop_depths) / total_queries, 2) if total_queries else 0,
        "avg_repeat_visits": round(total_repeat / total_queries, 2) if total_queries else 0,
        "avg_none_count_per_query": round(total_none / total_queries, 2) if total_queries else 0,
        "max_hop_depth_reached": max_depth,
        "hop_depth_distribution": hop_depth_distribution,
        "wall_time_total_sec": round(wall_time_total, 4),
        "wall_time_mean_sec": round(wall_time_mean, 4),
        "wall_time_median_sec": round(wall_time_median, 4),
    }
    return summary

























def process_query_batch(cfg: Dict) -> None:
    """Run traversal on a shard of queries and write partial outputs."""

    server_url = cfg["server_url"]
    model = cfg["model"]
    input_path = cfg["input_path"]
    resume = cfg.get("resume", False)
    resume_path = cfg["resume_path"]

    server_config = next(
        (s for s in get_server_configs(model) if s["server_url"] == server_url),
        None,
    )
    if server_config is None:
        raise ValueError(f"Server URL {server_url} not found for model {model}")
    
    emb_model = get_embedding_model()


    queries = [{**q, "query_id": q["question_id"]} for q in load_jsonl(input_path)]

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=str(resume_path),
        items=queries,
        get_id=lambda q, i: q["question_id"],
        phase_label="Traversal",
        id_field="question_id",
    )
    if resume:
        queries = [q for q in queries if q["question_id"] not in done_ids]
    if not queries:
        return

    run_traversal(
        query_data=queries,
        graph=cfg["graph"],
        passage_metadata=cfg["passage_metadata"],
        passage_emb=cfg["passage_emb"],
        passage_index=cfg["passage_index"],
        emb_model=emb_model,
        server_configs=[server_config],
        output_paths=cfg["output_paths"],
        dataset=cfg["dataset"],
        split=cfg["split"],
        variant=cfg["variant"],
        traverser_model=cfg["traverser_model"],
        retriever_name=cfg["retriever_name"],
        seed_top_k=DEFAULT_SEED_TOP_K,
        alpha=DEFAULT_EDGE_BUDGET_ALPHA,
        n_hops=DEFAULT_NUMBER_HOPS,
        traversal_alg=cfg["traversal_alg"],
        seed=cfg.get("seed"),
    )










def process_traversal(cfg: Dict) -> None:
    """Load resources and run traversal for a single configuration."""

    dataset = cfg["dataset"]
    graph_model = cfg["graph_model"]
    model = cfg["model"]  # traversal model for LLM endpoint
    variant = cfg["variant"]
    split = cfg["split"]
    resume = cfg["resume"]
    seed = cfg.get("seed")
    retriever_name = cfg.get("retriever_name", DEFAULT_RETRIEVER_NAME)


    variant_cfg = {
        "baseline": hoprag_traversal_algorithm,
        "enhanced": enhanced_traversal_algorithm,
    }

    variant_for_path = f"{variant}_seed{seed}" if seed is not None else variant

    print(
        f"[Run] dataset={dataset} graph_model={graph_model} traversal_model={model} variant={variant_for_path} split={split}"
    )

    paths = dataset_rep_paths(dataset, split)
    passage_metadata = list(load_jsonl(paths["passages_jsonl"]))
    passage_emb = np.load(paths["passages_emb"])
    validate_vec_ids(passage_metadata, passage_emb)
    passage_index = faiss.read_index(paths["passages_index"])
    if passage_index.ntotal != len(passage_metadata):
        print(
            f"[process_traversal] FAISS index has {passage_index.ntotal} vectors "
            f"but metadata lists {len(passage_metadata)} passages. Rebuilding index."
        )
        output_dir = str(Path(paths["passages_index"]).parent)
        build_and_save_faiss_index(
            passage_emb,
            dataset,
            "passages",
            output_dir=output_dir,
        )
        passage_index = faiss.read_index(paths["passages_index"])
        assert passage_index.ntotal == len(passage_metadata), (
            "FAISS index rebuild failed to match metadata length"
        )
    query_path = processed_dataset_paths(dataset, split)["questions"]

    graph_path = Path(
        f"data/graphs/{graph_model}/{dataset}/{split}/{variant}/{dataset}_{split}_graph.gpickle"
    )
    with open(graph_path, "rb") as f:
        graph_obj = pickle.load(f)
    validate_vec_ids(
        [
            {"passage_id": pid, "vec_id": data.get("vec_id")}
            for pid, data in graph_obj.nodes(data=True)
        ],
        passage_emb,
    )
    trav_alg = variant_cfg[variant]

    output_paths = get_traversal_paths(model, dataset, split, variant_for_path)

    urls = get_server_urls(model)
    shards = split_jsonl_for_models(str(query_path), model, resume=resume)

    batch_configs = []
    for i, (url, shard) in enumerate(zip(urls, shards)):
        batch_paths = {
            "base": output_paths["base"],
            "results": output_paths["base"] / f"results_part{i}.jsonl",
            "visited_passages": output_paths["base"]
            / f"visited_passages_part{i}.json",
            "token_usage": output_paths["base"] / f"token_usage_part{i}.json",
        }
        batch_configs.append(
            {
                "input_path": shard,
                "graph": graph_obj,
                "passage_metadata": passage_metadata,
                "passage_emb": passage_emb,
                "passage_index": passage_index,
                "server_url": url,
                "model": model,
                "output_paths": batch_paths,
                "traversal_alg": trav_alg,
                "resume": resume,
                "resume_path": output_paths["results"],
                "seed": seed,
                "dataset": dataset,
                "split": split,
                "variant": variant_for_path,
                "traverser_model": model,
                "retriever_name": retriever_name,
            }
        )

    run_multiprocess(process_query_batch, batch_configs)

    merge_token_usage(output_paths["base"], cleanup=True)


    new_ids: Set[str] = set()
    with open(output_paths["results"], "at", encoding="utf-8") as fout:
        for i in range(len(urls)):
            part_path = output_paths["base"] / f"results_part{i}.jsonl"
            if part_path.exists():
                with open(part_path, "rt", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                        obj = json.loads(line)
                        new_ids.add(obj.get("question_id"))
                part_path.unlink()

    merged_passages: Set[str] = set()
    for i in range(len(urls)):
        part_path = output_paths["base"] / f"visited_passages_part{i}.json"
        if part_path.exists():
            with open(part_path, "rt", encoding="utf-8") as fin:
                merged_passages.update(json.load(fin))
            part_path.unlink()
    visited_path = output_paths["visited_passages"]
    with open(visited_path, "wt", encoding="utf-8") as fout:
        json.dump(sorted(merged_passages), fout, indent=2)

    traversal_metrics = compute_traversal_summary(
        output_paths["results"], include_ids=new_ids
    )
    stats_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "traversal_eval": traversal_metrics,
    }
    with open(output_paths["stats"], "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    append_traversal_percentiles(
        output_paths["results"], output_paths["stats"]
    )

    print(
        f"[Done] dataset={dataset} graph_model={graph_model} traversal_model={model} variant={variant_for_path} split={split}"
    )






if __name__ == "__main__":
    start_time = time.time()

    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    GRAPH_MODELS = ["llama-3.1-8b-instruct"]


    TRAVERSAL_MODELS = ["qwen2.5-moe-14b"] 

# [
#         "qwen2.5-7b-instruct",
#         "qwen2.5-14b-instruct",
#         "deepseek-r1-distill-qwen-7b",
#         "deepseek-r1-distill-qwen-14b",
#         "qwen2.5-moe-14b",
        # "state-of-the-moe-rp-2x7b",
        # "qwen2.5-2x7b-power-coder-v4"
#     ]

    VARIANTS = ["baseline"]

    RESUME = True
    SPLIT = "dev"
    SEEDS = [0, 1, 3, 4, 5] # for??????????

    configs = [
        {
            "dataset": d,
            "graph_model": gm,
            "model": tm,
            "variant": v,
            "split": SPLIT,
            "resume": RESUME,
            "seed": seed,
        }
        for d in DATASETS
        for gm in GRAPH_MODELS
        for tm in TRAVERSAL_MODELS
        for v in VARIANTS
        for seed in SEEDS
    ]

    result_paths = set()
    for cfg in configs:

        seed = cfg.get("seed")
        variant_for_path = f"{cfg['variant']}_seed{seed}" if seed is not None else cfg["variant"]
        out_path = (
            Path(
                f"data/traversal/{cfg['model']}/{cfg['dataset']}/{cfg['split']}/{variant_for_path}"
            )
            / "per_query_traversal_results.jsonl"
        )
        if out_path in result_paths:
            raise ValueError(f"Duplicate output path detected: {out_path}")
        result_paths.add(out_path)

    configs_by_model: Dict[str, List[Dict]] = {}
    for cfg in configs:
        configs_by_model.setdefault(cfg["model"], []).append(cfg)

    for _, model_configs in configs_by_model.items():
        for cfg in model_configs:
            process_traversal(cfg)
    log_wall_time(__file__, start_time)