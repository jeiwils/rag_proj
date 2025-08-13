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

- `{dataset}_{split}.jsonl`  
    Preprocessed query set, with `question_id`, `question`, and `gold_passages`.

- `passages_emb.npy`, `passages_index.faiss`, `passages.jsonl`  
    Passage embeddings, FAISS index, and metadata from dense/sparse encoder setup.


Outputs
-------

### data/graphs/{model}/{dataset}/{split}/{variant}/traversal/

- `per_query_traversal_results.jsonl`  
    One entry per query with hop trace, visited nodes, and precision/recall/F1.

- `final_selected_passages.json`  
    Deduplicated set of all passages visited during traversal (used for answering).

- `final_traversal_stats.json`  
    Aggregate metrics across the full query set (e.g., mean precision, recall, hop stats).



File Schema
-----------

### per_query_traversal_results.jsonl

{
  "query_id": "{question_id}",
  "gold_passages": ["{passage_id_1}", "..."],
  "visited_passages": ["{passage_id_1}", "..."],
  "visit_counts": {"{passage_id}": count, ...},
  "hop_trace": [<hop dicts>],
  "final_metrics": {
    "precision": float,
    "recall": float,
    "f1": float
  },
  "traversal_algorithm": "{algorithm_name}"
}


### final_selected_passages.json

[
  "{passage_id_1}",
  "{passage_id_2}",
  ...
]


### traversal_stats.json

{
  "total_queries": int,
  "traversal_eval": {
    "mean_precision": float,
    "mean_recall": float,
    "passage_coverage_all_gold_found": int,
    "initial_retrieval_coverage": int,
    "avg_hops_before_first_gold": "TODO",
    "avg_total_hops": float,
    "avg_repeat_visits": float,
    "avg_none_count_per_query": float,
    "max_hop_depth_reached": int,
    "hop_depth_counts": [int, int, ...]
  }
}
"""





### end of: 2 sets of traversals+metrics made with the dev query+gold set (is that right? not the train set?)
# - 1) normal hoprag traversal algorithm (no node revisits)
# - 2) enhanced hoprag traversal algorithm (allows node revists - no edge revisits)




import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from src.a2_text_prep import query_llm, SERVER_CONFIGS, load_jsonl
from src.b_sparse_dense_representations import (
    extract_keywords,
    faiss_search_topk,
    jaccard_similarity,
    get_embedding_model,
    ALPHA,
    dataset_rep_paths,
)
from src.c_graphing import hoprag_graph, enhanced_graph, append_global_result
import faiss
import networkx as nx
import os
from collections import defaultdict
import json
from src.c_graphing import basic_graph_eval
from pathlib import Path

traversal_prompt = Path("data/prompts/traversal_prompt.txt").read_text()


# TRAVERSAL TUNING

TOP_K_SEED_PASSAGES = 50 
NUMBER_HOPS = 2


def traversal_output_paths(model, dataset, split, variant):
    base_dir = Path(f"data/graphs/{model}/{dataset}/{split}/{variant}/traversal")
    base_dir.mkdir(parents=True, exist_ok=True)

    return {
        "dir": base_dir,
        "results": base_dir / "per_query_traversal_results.jsonl",
        "stats": base_dir / "final_traversal_stats.json",  # <-- updated here
        "final_selected_passages": base_dir / "final_selected_passages.json"
    }



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
        sim_jac = jaccard_similarity(query_keywords, set(p["keywords_passage"]))


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



def save_per_query_result( # helper for run_dev_set()
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


def run_traversal(
    query_data: List[Dict],
    graph: nx.DiGraph,
    passage_metadata: List[Dict],
    passage_emb: np.ndarray,
    passage_index,
    emb_model,
    model_servers: List[str],
    output_paths: Dict[str, Path],  # use traversal_output_paths()
    seed_top_k=50,
    alpha=0.5,
    n_hops=2,
    traveral_alg=None
):
    """
    Run LLM-guided multi-hop traversal over a QA query set (e.g., train, dev).

    Outputs:
    - final_selected_passages.json: ✅ Used downstream (answer generation, reranking)
    - per_query_traversal_results.jsonl: 🔍 Full per-query trace and metrics
    - traversal_stats.json: 📈 Aggregate traversal metrics across the query set
    """
        
    all_traversals = []
    all_selected_passages = set()

    for entry in query_data:
        query_id = entry["query_id"]
        query_text = entry["question"]
        gold_passages = entry["gold_passages"]

        print(f"\n[Query] {query_id} - \"{query_text}\"")

        # --- Embed query ---
        query_emb = emb_model.encode(query_text, normalize_embeddings=True)

        # --- Select seed passages ---
        seed_passages = select_seed_passages(
            query_text=query_text,
            query_emb=query_emb,
            passage_metadata=passage_metadata,
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
            model_servers=model_servers,
            traveral_alg=traveral_alg
        )

        print(f"[Traversal] Visited {len(visited_passages)} passages (None={stats['none_count']}, Repeat={stats['repeat_visit_count']})")

        # --- Save per-query JSONL ---
        save_per_query_result(
            query_id=query_id,
            gold_passages=gold_passages,
            visited_passages=visited_passages,
            ccount=ccount,
            hop_trace=hop_trace,
            output_path=output_paths["results"]
        )

        # --- Accumulate traversal + selected passages ---
        all_traversals.append({
            "query_id": query_id,
            "hop_trace": hop_trace
        })
        all_selected_passages.update(visited_passages)

    # --- Save selected_passages.json ---
    with open(output_paths["final_selected_passages"], "w") as f:
        json.dump(sorted(all_selected_passages), f, indent=2)




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









if __name__ == "__main__":
    # === Hardcoded config ===
    model = "qwen-7b"
    dataset = "hotpot"
    split = "dev"
    variant_baseline = "baseline"
    variant_enhanced = "enhanced"

    # === Load dataset representations ===
    paths = dataset_rep_paths(dataset, split)
    passage_metadata = load_jsonl(paths["passages_jsonl"])
    passage_emb = np.load(paths["passages_emb"])
    passage_index = faiss.read_index(paths["passages_index"])
    query_data = [json.loads(line) for line in open(f"{dataset}_{split}.jsonl")]

    bge_model = get_embedding_model()

    ######################################
    # 1. Run baseline (no revisit)
    ######################################
    print(f"\n=== Running BASELINE traversal ===")
    output_paths = traversal_output_paths(model, dataset, split, variant_baseline)

    run_traversal(
        query_data=query_data,
        graph=hoprag_graph,
        passage_metadata=passage_metadata,
        passage_emb=passage_emb,
        passage_index=passage_index,
        emb_model=bge_model,
        model_servers=SERVER_CONFIGS,
        output_paths=output_paths,
        seed_top_k=TOP_K_SEED_PASSAGES,
        alpha=ALPHA,
        n_hops=NUMBER_HOPS,
        traveral_alg=hoprag_traversal_algorithm
    )

    traversal_metrics = compute_traversal_summary(output_paths["results"])
    append_global_result(
        save_path=output_paths["stats"],
        total_queries=len(query_data),
        traversal_eval=traversal_metrics
    )

    ######################################
    # 2. Run enhanced (allows revisit)
    ######################################
    print(f"\n=== Running ENHANCED traversal ===")
    output_paths = traversal_output_paths(model, dataset, split, variant_enhanced)

    run_traversal(
        query_data=query_data,
        graph=enhanced_graph,
        passage_metadata=passage_metadata,
        passage_emb=passage_emb,
        passage_index=passage_index,
        emb_model=bge_model,
        model_servers=SERVER_CONFIGS,
        output_paths=output_paths,
        seed_top_k=TOP_K_SEED_PASSAGES,
        alpha=ALPHA,
        n_hops=NUMBER_HOPS,
        traveral_alg=enhanced_traversal_algorithm
    )

    traversal_metrics = compute_traversal_summary(output_paths["results"])
    append_global_result(
        save_path=output_paths["stats"],
        total_queries=len(query_data),
        traversal_eval=traversal_metrics
    )

    print("\nAll traversals completed.")
