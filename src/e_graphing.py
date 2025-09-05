"""
Module Overview
---------------
Construct a passage-level graph by linking Original Questions (OQs) to Inferred Questions (IQs).

Each OQ retrieves candidate IQs from both a FAISS index and a keyword-based
Jaccard search, computes hybrid similarity (cosine + Jaccard), and retains the
top-scoring IQ. A HopRAG-style global :math:`O(n \log n)` budget then keeps only
the highest-scoring edges across all queries. The resulting OQ→IQ edges form a
directed graph over passages for downstream reasoning and traversal.

Graphs are currently built for the datasets `hotpotqa`, `musique`, and
`2wikimultihopqa`.

This module builds the graph, saves edge lists and NetworkX `.gpickle` files,
and logs diagnostic summaries for structural and similarity-based evaluation.



Inputs
------

### `data/representations/datasets/{dataset}/{split}/`

- `{dataset}_passages.jsonl`  
  – Passage metadata including passage ID, text, vector index, and keywords.

- `{dataset}_passages_emb.npy`
  – Dense passage embeddings (`NumPy` array aligned with `vec_id` field).


### `data/representations/models/{model}/{dataset}/{split}/{variant}/`

- `iqoq.cleaned.jsonl`  
  – Cleaned IQ/OQ items with `vec_id`, `type` (OQ/IQ), `parent_passage_id`, and `keywords`.

- `{dataset}_iqoq_emb.npy`
  – Dense IQ/OQ embeddings (`NumPy` array aligned with `vec_id` field).


Outputs
-------

### `data/graphs/{model}/{dataset}/{split}/{variant}/`

- `{dataset}_{split}_edges.jsonl`
  – Top hybrid-scoring OQ→IQ edge per query with similarity scores and
    retrieval source.

- `{dataset}_{split}_graph.gpickle`  
  – Directed `NetworkX` graph: passages as nodes, OQ→IQ as edges.

- `{dataset}_{split}_graph_log.jsonl`  
  – Global summary with average degree, Gini coefficient, and top-k hubs.

- `{dataset}_{split}_graph_results.jsonl`  
  – Detailed diagnostics: edge similarity stats, degree distributions, components, etc.



File Schema
-----------

### `{dataset}_{split}_edges.jsonl`

{
  "oq_id": "hotpotqa_001_sent1_oq1",
  "oq_parent": "hotpotqa_001_sent1",
  "oq_vec_id": 12,
  "oq_text": "What year was the Battle of Hastings?",

  "iq_id": "hotpotqa_002_sent4_iq0",
  "iq_parent": "hotpotqa_002_sent4",
  "iq_vec_id": 37,

  "sim_cos": 0.7215,
  "sim_jaccard": 0.5632,
  "sim_hybrid": 0.6424,
  "retrieval": "faiss+jaccard"
}





#### `{dataset}_{split}_graph.gpickle`

Node:
{
  "passage_id": "hotpotqa_001_sent1",
  "text": "The Battle of Hastings took place in 1066...",
  "vec_id": 12,
  "keywords_passage": ["battle_of_hastings", "1066"]
}

Edge:
{
  "oq_id": "hotpotqa_001_sent1_oq1",
  "iq_id": "hotpotqa_002_sent4_iq0",
  "oq_text": "What year was the Battle of Hastings?",
  "sim_cos": 0.7215,
  "sim_jaccard": 0.5632,
  "sim_hybrid": 0.6424
}


### {dataset}_{split}_graph_log.jsonl

{
  "timestamp": "2025-08-13T14:22:31",
  "graph_eval": {
    "avg_node_degree": 1.43,
    "node_degree_variance": 0.97,
    "gini_degree": 0.27,
    "top_k_hub_nodes": [
      {"node": "hotpotqa_002_sent4", "degree": 7},
      {"node": "hotpotqa_001_sent3", "degree": 6}
    ]
  },
  "mode": "standard_pipeline",
  "dataset": "hotpotqa",
  "split": "train",
  "edges_file": "data/graphs/llama-3.1-8b-instruct/hotpotqa/train/baseline/hotpotqa_train_edges.jsonl",
  "params": {
    "top_k": 50,
    "edge_budget_alpha": 1.0,
    "edge_budget": 1478
  }
}



### {dataset}_{split}_graph_results.jsonl

{
  "dataset": "hotpotqa_train",
  "timestamp": "2025-08-13T14:22:31",
  "params": {
    "top_k": 50,
    "edge_budget_alpha": 1.0,
    "edge_budget": 1478
  },
  "num_nodes": 1478,
  "num_edges": 1321,
  "num_components": 45,
  "largest_component_size": 732,
  "largest_component_ratio": 0.4953,

  "avg_in_degree": 0.89,
  "min_in_degree": 0,
  "max_in_degree": 9,
  "var_in_degree": 1.12,

  "avg_out_degree": 0.89,
  "min_out_degree": 0,
  "max_out_degree": 1,
  "var_out_degree": 0.27,

  "nodes_with_no_in_edges": 654,
  "nodes_with_no_out_edges": 845,

  "edge_sim_hybrid_mean": 0.6483,
  "edge_sim_hybrid_max": 0.9212,
  "edge_sim_hybrid_min": 0.6512,
  "edge_sim_hybrid_var": 0.0089,

  "avg_node_degree": 1.43,
  "node_degree_variance": 0.97,
  "gini_degree": 0.27,

  "top_k_hub_nodes": [
    {"node": "hotpotqa_002_sent4", "degree": 7},
    {"node": "hotpotqa_001_sent3", "degree": 6}
  ]
}
"""



import json
import math
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

from src.d_sparse_dense_representations import (
    dataset_rep_paths,
    retrieve_hybrid_candidates,
    load_faiss_index,
    model_rep_paths,
)
from src.utils import (
    append_jsonl,
    compute_resume_sets,
    load_jsonl,
    save_jsonl,
    validate_vec_ids,

)






######### POTENTIALLY MOVE THESE TO ANOTHER, SEPARATE, CONFIGS???


DEFAULT_TOP_K = 50

DEFAULT_EDGE_BUDGET_ALPHA = 1.0  # default edge-budget scaling

DEFAULT_PARAMS = {
    "top_k": DEFAULT_TOP_K,
    "edge_budget_alpha": DEFAULT_EDGE_BUDGET_ALPHA,
}










def graph_output_paths(
    model: str, dataset: str, split: str, variant: str
) -> dict:
    base = Path("data") / "graphs" / model / dataset / split / variant
    return {
        "edges": base / f"{dataset}_{split}_edges.jsonl",
        "graph_gpickle": base / f"{dataset}_{split}_graph.gpickle",
        "graph_log": base / f"{dataset}_{split}_graph_log.jsonl",
        "graph_results": base / f"{dataset}_{split}_graph_results.jsonl",
    }




################################################################################################################
# GRAPH CONSTRUCTION
################################################################################################################





def build_edges(
    oq_metadata: List[Dict],
    iq_metadata: List[Dict],
    emb: np.ndarray,
    iq_index,
    top_k: int = DEFAULT_TOP_K,
    output_jsonl: str = None,
    alpha: float = 0.5,
    exclude_self_loops: bool = True,
) -> List[Dict]:
    """Build final OQ->IQ edges.

    Hybrid retrieval is performed for each outgoing question using
    :func:`retrieve_hybrid_candidates`, combining dense FAISS search and
    sparse Jaccard keyword overlap. ``alpha`` is the hybrid weighting factor
    applied to ``alpha * sim_cos + (1 - alpha) * sim_jac``. The highest-scoring
    candidate per OQ is kept as an edge regardless of similarity.
    ``iq_metadata`` must be aligned with the FAISS index and embedding matrix
    ``emb``.
    """
    edges: List[Dict] = []

    for oq in tqdm(
        oq_metadata, desc="Building edges", unit="OQ", miniters=100, mininterval=1
    ):

        oq_vec = emb[oq["vec_id"]]
        oq_keywords = set(oq.get("keywords", []))
        oq_parent = oq["parent_passage_id"]

        candidates = retrieve_hybrid_candidates(
            oq_vec,
            oq_keywords,
            iq_metadata,
            iq_index,
            top_k=top_k,
            alpha=alpha,
            keyword_field="keywords",
            filter_fn=(
                (lambda idx: iq_metadata[idx]["parent_passage_id"] != oq_parent)
                if exclude_self_loops
                else None
            ),
        )

        candidate_edges: List[Dict] = []
        for cand in candidates:
            iq = iq_metadata[cand["idx"]]

            candidate_edges.append(
                {
                    "oq_id": oq["iqoq_id"],
                    "oq_parent": oq_parent,
                    "oq_vec_id": oq["vec_id"],
                    "oq_text": oq.get("text", ""),
                    "iq_id": iq["iqoq_id"],
                    "iq_parent": iq["parent_passage_id"],
                    "iq_vec_id": iq["vec_id"],
                    "sim_cos": cand["sim_cos"],
                    "sim_jaccard": cand["sim_jac"],
                    "sim_hybrid": cand["sim_hybrid"],
                    "retrieval": "hybrid",
                }
            )

        if candidate_edges:
            best_edge = max(candidate_edges, key=lambda e: e["sim_hybrid"])
            edges.append(best_edge)

    # Optionally save edges
    if output_jsonl:
        save_jsonl(output_jsonl, edges)
        print(f"[Edges] Saved {len(edges)} edges to {output_jsonl}")

    if edges:
        src_counts = defaultdict(int)
        for e in edges:
            src_counts[e.get("retrieval", "faiss")] += 1
        print(f"[Edges] Retrieval source counts: {dict(src_counts)}")

    return edges





def enforce_global_edge_budget(edges: List[Dict], budget: int) -> List[Dict]:
    """Keep only the top ``budget`` edges by ``sim_hybrid``."""
    if budget is None or len(edges) <= budget:
        return edges
    return sorted(edges, key=lambda x: x["sim_hybrid"], reverse=True)[:budget]







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
        vec_id = p.get("vec_id")
        if vec_id is None:
            raise ValueError(
                f"passage {p.get('passage_id')} missing vec_id"
            )
        G.add_node(
            p["passage_id"],
            text=p.get("text", ""),
            conditioned_score=p.get("conditioned_score", 0.0),
            vec_id=vec_id,
            keywords=p.get("keywords_passage", []),
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
            retrieval=e.get("retrieval", ""),
        )

    return G






def basic_graph_eval(
        G: nx.DiGraph, 
        top_k_hubs: int = 5 # how many nodes with the most connections (by total degree) to log 
        ) -> dict:
    """
    Compute general graph-wide statistics: average degree, degree variance, Gini coefficient,
    and top-k hub nodes. Used by both detailed and global graph reporting.
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
    graph_eval: Optional[Dict] = None,
    traversal_eval: Optional[Dict] = None,
    answer_eval: Optional[Dict] = None,
    extra_metadata: Optional[Dict] = None
):
    """
    Append a structured global result entry to {dataset}_dev_global_results.jsonl
    Sections can be partial: graph_eval, traversal_eval, and/or answer_eval.
    """
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if graph_eval:
        result["graph_eval"] = graph_eval

    if traversal_eval:
        result["traversal_eval"] = traversal_eval

    if answer_eval:
        result["answer_eval"] = answer_eval

    if extra_metadata:
        result.update(extra_metadata)

    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path or ".", exist_ok=True)
    append_jsonl(save_path, result)

    return result






def graph_stats(
    G: nx.DiGraph,
    save_path: str = None, # input save path if you want to save
    params: dict = None,
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
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path or ".", exist_ok=True)

        with open(save_path, "at", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")

    return stats














def run_graph_pipeline(
    dataset: str,
    split: str,
    model: str,
    variant: str,
    passages_file: str = None,
    iqoq_file: str = None,
    top_k: int = DEFAULT_TOP_K,
    save_graph: bool = True,
    save_graphml: bool = False,
    resume: bool = True,
    edge_budget_alpha: float = DEFAULT_EDGE_BUDGET_ALPHA,
):
    """
    Full pipeline:
        1) Load passage metadata/embeddings from the dataset folder and IQ/OQ
         metadata/embeddings from the model-specific folder
        2) Load FAISS index for all IQ/OQ vectors
        3) build_edges(...) -> save edges jsonl
        4) build_networkx_graph(passages, edges)
        5) basic_graph_eval + append_global_result
        6) graph_stats -> append detailed stats jsonl


    Args:
        edge_budget_alpha: Edge-budget scaling factor for the global
            :math:`O(n \log n)` edge budget. The total number of edges
            retained is ``ceil(edge_budget_alpha * n_passages * log n)``.
    """
    # ---------- 1) Load metadata + embeddings ----------
    pass_paths = dataset_rep_paths(dataset, split)
    model_paths = model_rep_paths(model, dataset, split, variant)
    p_path = passages_file if passages_file else pass_paths["passages_jsonl"]
    q_path = iqoq_file if iqoq_file else model_paths["iqoq_jsonl"]

    passages_md = list(load_jsonl(p_path))
    passage_emb = np.load(pass_paths["passages_emb"], mmap_mode="r")
    validate_vec_ids(passages_md, passage_emb)
    iqoq_md = list(load_jsonl(q_path))   ## mmap_mode=r ?
    iq_md = [m for m in iqoq_md if m["type"] == "IQ"]


    n_passages = len(passages_md)
    edge_budget = None
    if edge_budget_alpha is not None:
        edge_budget = math.ceil(
            edge_budget_alpha * n_passages * math.log(max(n_passages, 2))
        )

    iqoq_emb = np.load(model_paths["iqoq_emb"])

    # ---------- 2) Load FAISS index over IQ-only vectors ----------
    iq_index = load_faiss_index(model_paths["iq_index"])
    oq_items = [q for q in iqoq_md if q.get("type") == "OQ"]

    top_k = top_k if top_k is not None else DEFAULT_TOP_K


    # ---------- 3) Build edges with optional resume ----------
    graph_paths = graph_output_paths(model, dataset, split, variant)
    edges_out = str(graph_paths["edges"])

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=edges_out,
        items=oq_items,
        get_id=lambda x, i: x["iqoq_id"],
        phase_label="edges",
        id_field="oq_id",
    )
    existing_edges = list(load_jsonl(edges_out)) if resume and os.path.exists(edges_out) else []
    oq_to_process = [q for q in oq_items if q["iqoq_id"] not in done_ids]

    new_edges = build_edges(
        oq_metadata=oq_to_process,
        iq_metadata=iq_md,
        emb=iqoq_emb,
        iq_index=iq_index,
        top_k=top_k,
        output_jsonl=None,
    )

    edges = existing_edges + new_edges
    if edge_budget is not None:
        edges = enforce_global_edge_budget(edges, edge_budget)
        print(f"[Edges] Applied global edge budget: {len(edges)}/{edge_budget}")

    os.makedirs(os.path.dirname(edges_out), exist_ok=True)

    if new_edges or not existing_edges or edge_budget is not None:
        save_jsonl(edges_out, edges)
        print(f"[Edges] Saved {len(edges)} edges to {edges_out}")
    else:
        print(f"[Edges] Using existing {len(edges)} edges from {edges_out}")



    # ---------- 4) Build graph ----------
    G = build_networkx_graph(passages=passages_md, edges=edges)

    # Confirm all nodes retain vec_ids
    node_md = [
        {"passage_id": nid, "vec_id": data.get("vec_id")}
        for nid, data in G.nodes(data=True)
    ]
    validate_vec_ids(node_md, passage_emb)
    del passage_emb

    # ---------- 5) Global eval ----------
    graph_eval = basic_graph_eval(G)
    global_path = str(graph_paths["graph_log"])
    stats_path = str(graph_paths["graph_results"])

    append_global_result(
        save_path=global_path,
        graph_eval=graph_eval,
        extra_metadata={
            "mode": "standard_pipeline",
            "dataset": dataset,
            "split": split,
            "edges_file": edges_out,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "params": {
                "top_k": top_k,
                "edge_budget_alpha": edge_budget_alpha,
                "edge_budget": edge_budget,
            },
        },
    )

    # ---------- 6) Detailed stats ----------

    params_used = DEFAULT_PARAMS.copy()
    params_used.update({"top_k": top_k, "edge_budget": edge_budget})
    params_used["edge_budget_alpha"] = (
        edge_budget_alpha if edge_budget_alpha is not None else DEFAULT_EDGE_BUDGET_ALPHA
    )

    stats = graph_stats(
        G,
        save_path=stats_path,
        params=params_used,
        dataset=f"{dataset}_{split}",
    )

    # ---------- 7) Optional graph saves ----------

    graph_gpickle = str(graph_paths["graph_gpickle"])
    graph_graphml = graph_gpickle.replace(".gpickle", ".graphml")

    if save_graph:
        # Save as gpickle-compatible using stdlib pickle
        with open(graph_gpickle, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Graph] Saved -> {graph_gpickle}")

    if save_graphml:
        # GraphML writer is still available
        if hasattr(nx, "write_graphml"):
            nx.write_graphml(G, graph_graphml)
            print(f"[Graph] Saved -> {graph_graphml}")
        else:
            print("[Graph] GraphML writer not available; skipping GraphML.")


    return {
        "graph": G,
        "edges_path": edges_out,
        "global_log_path": global_path,
        "graph_results_path": stats_path,
    }







if __name__ == "__main__":
    
    MODELS   = ["llama-3.1-8b-instruct"]  # ["deepseek-dis
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    VARIANTS = ["baseline"]
    SPLIT    = "dev"

    for dataset in DATASETS: ############## I NEED TO MAKE THESE FOR LOOPS LOGICAL AND COHESIVE ACROSS MODULES 
        for model in MODELS:
            for variant in VARIANTS:
                print(
                    f"[Run] dataset={dataset} model={model} variant={variant} split={SPLIT}"
                )
                result_paths = run_graph_pipeline(
                    dataset=dataset,
                    split=SPLIT,
                    model=model,
                    variant=variant,
                    passages_file=None,
                    iqoq_file=None,
                    save_graph=True,
                    save_graphml=False,
                    edge_budget_alpha=DEFAULT_EDGE_BUDGET_ALPHA,
                )
                print(
                    f"[Done] dataset={dataset} model={model} variant={variant} split={SPLIT}"
                )
                print(result_paths)



