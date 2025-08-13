"""
Module Overview
---------------
Construct a graph that links original questions (OQ) to inferred questions (IQ).
Each OQ retrieves candidate IQs from a FAISS index, computes hybrid similarity
(cosine + Jaccard), and retains the highest-scoring IQ edge.

Inputs
------

################# PATHS / DIRECTORIES

- ``passages.jsonl`` – Passage metadata (IDs, text, keywords, etc.)
- ``iqoq.jsonl`` – IQ/OQ metadata
- ``passages_emb.npy`` – Passage embeddings
- ``iqoq_emb.npy`` – IQ/OQ embeddings
- ``hotpot_faiss_iqoq.faiss`` – FAISS index for IQ/OQ embeddings

Outputs
-------

################# WRITTEN FILES

- ``{dataset}_edges.jsonl`` – JSONL edge list linking each OQ to its top IQ  
  Each line is a top-ranked edge, including cosine, Jaccard, and hybrid similarity.

- ``{dataset}_{split}_graph.gpickle`` – NetworkX directed graph (passage-level, with OQ→IQ edges)

- ``graph_stats/{dataset}_{split}_graph_eval.jsonl`` – Summary graph statistics  
  Includes: avg degree, degree variance, Gini coefficient, top-k hubs.

- ``graph_stats/{dataset}_{split}_graph_results.jsonl`` – Full graph diagnostics (1 line per run)  
  Includes: component stats, degree distributions, edge sim stats, top-k hubs, etc.

Example Edge Record (`{dataset}_edges.jsonl`)
---------------------------------------------
{
    "oq_id": "hotpot_001_sent1_oq1",
    "oq_parent": "hotpot_001_sent1",
    "oq_vec_id": 12,
    "oq_text": "What year was the Battle of Hastings?",

    "iq_id": "hotpot_002_sent4_iq0",
    "iq_parent": "hotpot_002_sent4",
    "iq_vec_id": 37,

    "sim_cos": 0.72,
    "sim_jaccard": 0.56,
    "sim_hybrid": 0.64
}

Example Full Graph Diagnostics Entry (`graph_results.jsonl`)
------------------------------------------------------------
{
  "dataset": "hotpot_train",
  "iteration": 12,
  "timestamp": "2025-08-13T14:22:31",
  "params": {
    "top_k": 50,
    "sim_threshold": 0.65
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
    {"node": "hotpot_002_sent4", "degree": 7},
    {"node": "hotpot_001_sent3", "degree": 6}
  ]
}
"""


### end of: 2 sets of graphs made with the train passage+iqoq set (is that right? not the dev set?)
# - 1) standard hoprag
# - 2) enhanced hoprag (with CS as part of IQOQ generation)


import os
from typing import List, Dict, Optional
import numpy as np
import networkx as nx
from datetime import datetime
import json
from src.b_sparse_dense_representations import (
    params,
    faiss_search_topk,
    jaccard_similarity,
    build_and_save_faiss_index,
    MAX_NEIGHBOURS,
    SIM_THRESHOLD,
    load_faiss_index,
    dataset_rep_paths,
model_rep_paths,
)


from src.utils import load_jsonl

DATASET = "hotpot"
SPLIT = "train"
MODEL = "qwen-7b"
VARIANT = "baseline"

pass_paths = dataset_rep_paths(DATASET, SPLIT)
PASSAGES_PATH = pass_paths["passages_jsonl"]
passages_emb_path = pass_paths["passages_emb"]

iqoq_paths = model_rep_paths(MODEL, DATASET, SPLIT, VARIANT)
IQOQ_PATH = iqoq_paths["iqoq_jsonl"]
iqoq_emb_path = iqoq_paths["iqoq_emb"]

DATASET = "hotpot"
SPLIT = "train"
paths = dataset_rep_paths(DATASET, SPLIT)
PASSAGES_PATH = paths["passages_jsonl"]
IQOQ_PATH = paths["iqoq_jsonl"]
passages_emb_path = paths["passages_emb"]
iqoq_emb_path = paths["iqoq_emb"]

################################################################################################################
# GRAPH CONSTRUCTION
################################################################################################################




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



####





# # Make edges.jsonl
# edges = build_edges(
#     oq_metadata=iqoq_metadata,
#     iq_metadata=iqoq_metadata,
#     oq_emb=iqoq_emb,
#     iq_emb=iqoq_emb,
#     iq_index=iq_index,
#     top_k=MAX_NEIGHBOURS,
#     sim_threshold=SIM_THRESHOLD,
#     output_jsonl="train/hotpot_edges.jsonl"
# )


#
# { {dataset}_edges.jsonl
#   "oq_id": "hotpot_001_sent1_oq1",
#   "oq_parent": "hotpot_001_sent1",
#   "oq_vec_id": 12,
#   "oq_text": "What year was the Battle of Hastings?", #include text for LLM to query - saves accessing another file 
#
#   "iq_id": "hotpot_002_sent4_iq0",
#   "iq_parent": "hotpot_002_sent4",
#   "iq_vec_id": 37,
#
#   "sim_cos": 0.72,
#   "sim_jaccard": 0.56,
#   "sim_hybrid": 0.64
# }
#



####



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















def run_graph_pipeline(
    dataset: str = DATASET,
    split: str = SPLIT,
    model: str = MODEL,
    variant: str = VARIANT,
    passages_file: str = None,
    iqoq_file: str = None,
    top_k: int = None,
    sim_threshold: float = None,
    total_queries: int = 0,
    iteration: int = 0,
    save_graph: bool = True,
    save_graphml: bool = False,
):
    """
    Full pipeline:
        1) Load passage metadata/embeddings from the dataset folder and IQ/OQ
         metadata/embeddings from the model-specific folder
      2) Build/load FAISS index for IQs
      3) build_edges(...) -> save edges jsonl
      4) build_networkx_graph(passages, edges)
      5) basic_graph_eval + append_global_result
      6) graph_stats -> append detailed stats jsonl
    """
    # ---------- 1) Load metadata + embeddings ----------
    pass_paths = dataset_rep_paths(dataset, split)
    model_paths = model_rep_paths(model, dataset, split, variant)
    p_path = passages_file if passages_file else pass_paths["passages_jsonl"]
    q_path = iqoq_file if iqoq_file else model_paths["iqoq_jsonl"]

    passages_md = load_jsonl(p_path)
    iqoq_md = load_jsonl(q_path)

    passages_emb = np.load(pass_paths["passages_emb"])
    iqoq_emb = np.load(model_paths["iqoq_emb"])

    # ---------- 2) Build / load FAISS index ----------
    base_dir = os.path.dirname(model_paths["iqoq_index"])
    os.makedirs(base_dir, exist_ok=True)
    build_and_save_faiss_index(iqoq_emb, dataset, "iqoq", output_dir=base_dir)
    iq_index = load_faiss_index(model_paths["iqoq_index"])

    # ---------- 3) Build edges ----------
    edges_out = os.path.join(base_dir, f"{dataset}_edges.jsonl")
    edges = build_edges(
        oq_metadata=iqoq_md,
        iq_metadata=iqoq_md,
        oq_emb=iqoq_emb,
        iq_index=iq_index,
        top_k=top_k if top_k is not None else MAX_NEIGHBOURS,
        sim_threshold=sim_threshold if sim_threshold is not None else SIM_THRESHOLD,
        output_jsonl=edges_out,
    )

    # ---------- 4) Build graph ----------
    G = build_networkx_graph(passages=passages_md, edges=edges)

    # ---------- 5) Global eval ----------
    graph_eval = basic_graph_eval(G)
    global_path = f"outputs/graphs/graph_stats/{dataset}_{split}_graph_eval.jsonl"
    os.makedirs("outputs", exist_ok=True)
    append_global_result(
        save_path=global_path,
        total_queries=total_queries,
        graph_eval=graph_eval,
        extra_metadata={
            "mode": "standard_pipeline",
            "dataset": dataset,
            "split": split,
            "edges_file": edges_out,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "params": {
                "top_k": top_k if top_k is not None else MAX_NEIGHBOURS,
                "sim_threshold": sim_threshold if sim_threshold is not None else SIM_THRESHOLD,
            },
        },
    )

    # ---------- 6) Detailed stats ----------
    stats_path = f"outputs/graphs/graph_stats/{dataset}_{split}_graph_results.jsonl"

    stats = graph_stats(
        G,
        save_path=stats_path,
        params=params,
        iteration=iteration,
        dataset=f"{dataset}_{split}",
    )

    # ---------- 7) Optional graph saves ----------
    graph_gpickle = f"outputs/{dataset}_{split}_graph.gpickle"
    graph_graphml = f"outputs/{dataset}_{split}_graph.graphml"
    if save_graph:
        nx.write_gpickle(G, graph_gpickle)
        print(f"[Graph] Saved -> {graph_gpickle}")
    if save_graphml:
        nx.write_graphml(G, graph_graphml)
        print(f"[Graph] Saved -> {graph_graphml}")

    return {
        "graph": G,
        "edges_path": edges_out,
        "global_results_path": global_path,
        "graph_results_path": stats_path,
    }



if __name__ == "__main__":
# tweak these as needed
    dataset = "hotpot"
    model = "qwen-7b"
    split = "train"
    variant = "baseline"



    total_queries = 100
    iteration = 12




    result_paths = run_graph_pipeline(
        dataset=dataset,
        split=split,
        model=model,
        variant=variant,
        passages_file=None,   # uses dataset_rep_paths internally
        iqoq_file=None,        # uses model_rep_paths internally
        top_k=None,           # defaults to MAX_NEIGHBOURS
        sim_threshold=None,   # defaults to SIM_THRESHOLD
        total_queries=total_queries,
        iteration=iteration,
        save_graph=True,
        save_graphml=False,
    )

    print("[Done]")
    print(result_paths)




