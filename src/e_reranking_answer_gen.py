"""
Module Overview
---------------
Generate answers from *precomputed traversal results*.

This script never runs traversal. It only:
1. Loads per-query traversal results (helpful_passages).
2. Fetches passage text from the graph.
3. Asks the LLM to generate answers using top-k passages.
4. Saves predictions + EM/F1 evaluation.


Inputs
------

### data/graphs/{model}/{dataset}/{split}/{variant}/

- `{dataset}_{split}_graph.gpickle`
    Directed NetworkX graph (needed only for passage text lookup).

### data/traversal/{model}/{dataset}/{split}/{variant}/

- `per_query_traversal_results.jsonl`
    Saved traversal results with helpful_passages per query.

### data/processed_datasets/{dataset}/{split}/questions.jsonl
    Query set with `question_id`, `question`, and `gold_answer(s)`.

Outputs
-------

### data/traversal/{model}/{dataset}/{split}/{variant}/

- `per_query_traversal_results.jsonl`
    One entry per query with hop trace, visited nodes, and precision/recall/F1 metrics.

- `final_traversal_stats.json`  
    Aggregate metrics across the full query set (e.g., mean precision, recall, hop stats).


File Schemas
------------

### per_query_traversal_results.jsonl

Each line contains a dict with the full traversal trace and evaluation:

{
  "question_id": str,
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



### final_traversal_stats.json

Aggregated summary across all queries:

{
  "mean_precision": float,
  "mean_recall": float,
  "passage_coverage_all_gold_found": int,
  "initial_retrieval_coverage": int,
  "avg_hops_before_first_gold": float | null, 
  "avg_total_hops": float,
  "avg_repeat_visits": float,
  "avg_none_count_per_query": float,
  "max_hop_depth_reached": int,
  "hop_depth_distribution": List[int]
}


Notes
-----

- Traversals are run on the `dev` split, not `train`.
- `baseline` = no revisits to previously visited passages.
- `enhanced` = prioritises edges by destination `conditioned_score` but also
  avoids revisiting nodes.
- All outputs are saved in:
  `data/traversal/{model}/{dataset}/{split}/{variant}/`
"""


import json
import os
import pickle
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from functools import partial
import networkx as nx

from src.a2_text_prep import is_r1_like, query_llm, strip_think, model_size
from src.utils import (
    append_jsonl,
    compute_resume_sets,
    get_result_paths,
    get_server_configs,
    get_traversal_paths,
    load_jsonl,
    processed_dataset_paths,
    save_jsonl,
    pool_map,
    compute_hits_at_k,

)






################################################################################################################
# PASSAGE RERANKING AND ANSWER GENERATION 
################################################################################################################





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
    model_name: str = "",
    top_k_answer_passages: int = 5,
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
    top_k_answer_passages : int, optional
        Number of passages from ``passage_ids`` to include, by default ``5``.

    Outputs
    -------
    Dict[str, str]
        ``{"raw_answer": str, "normalised_answer": str, "prompt_len": int, "output_tokens": int, "total_tokens": int}``
        where the token counts are provided by :func:`query_llm`.
    """
    passage_texts = []

    for pid in passage_ids[:top_k_answer_passages]:
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

    raw, usage = query_llm(
        prompt,
        server_url=server_url,
        max_tokens=max_tokens,
        model_name=model_name,
        phase="answer_generation",
    )

    if is_r1_like(model_name):
        raw = strip_think(raw)

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    norm = normalise_answer(raw)
    return {
        "raw_answer": raw,
        "normalised_answer": norm,
        "prompt_len": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }















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



def generate_answers_from_traversal(
    graph_model: str,
    traversal_model: str,
    dataset: str,
    split: str,
    variant: str,
    top_k_answer_passages: int = 5,
    reader_model: str = "llama-3.1-8b-instruct",
    server_url: str | None = None,
    model_name: str | None = None,
    num_workers: int | None = None,
    resume: bool = False,
) -> Dict[str, float]:
    """Generate answers from pre-computed traversal outputs.

    Parameters
    ----------
    graph_model, traversal_model, dataset, split, variant:
        ``graph_model`` identifies the graph, while ``traversal_model``
        determines traversal outputs, server configuration, and result
        directory.
    top_k_answer_passages:
        Number of passages to supply to the LLM per query.
    reader_model:
        Model used to generate final answers ("reader").
    server_url, model_name:
        LLM server configuration. Defaults to the first server returned by
        :func:`get_server_configs` for ``reader_model`` when not provided.
    num_workers:
        Number of worker processes. When ``None``, uses :func:`model_size` to
        choose ``1`` worker for ``14b``/``19b`` models, ``2`` for ``7b`` models,
        and ``4`` otherwise.
    resume:
        When ``True``, skip questions already present in the output file and
        append newly generated answers instead of overwriting.

    Returns
    -------
    Dict[str, float]
        Evaluation metrics (EM/F1) over the generated answers.

    Notes
    -----
    Shared data for worker processes is passed via ``functools.partial`` and
    tasks are distributed using :func:`src.utils.pool_map` to avoid global
    state and maintain consistent multiprocessing patterns.
    """

    if server_url is None or model_name is None:
        server = get_server_configs(reader_model)[0]
        server_url = server_url or server["server_url"]
        model_name = model_name or server["model"]

    traversal_paths = get_traversal_paths(traversal_model, dataset, split, variant)
    result_paths = get_result_paths(traversal_model, dataset, split, variant)

    traversal_file = traversal_paths["results"]
    graph_file = (
        Path("data")
        / "graphs"
        / graph_model
        / dataset
        / split
        / variant
        / f"{dataset}_{split}_graph.gpickle"
    )
    query_file = processed_dataset_paths(dataset, split)["questions"]

    traversal_records = {r["question_id"]: r for r in load_jsonl(traversal_file)}
    graph = nx.read_gpickle(graph_file)
    passage_lookup = {pid: data.get("text", "") for pid, data in graph.nodes(data=True)}
    queries = {q["question_id"]: q for q in load_jsonl(query_file)}

    answers: List[Dict] = []
    predictions: Dict[str, str] = {}
    gold: Dict[str, List[str]] = {}
    hits: Dict[str, float] = {}
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}




    def _init_worker(q_dict, p_lookup, s_url, m_name, top_k):
        """Store shared data in globals for worker processes."""
        global _QUERIES, _PASSAGE_LOOKUP, _SERVER_URL, _MODEL_NAME, _TOP_K
        _QUERIES = q_dict
        _PASSAGE_LOOKUP = p_lookup
        _SERVER_URL = s_url
        _MODEL_NAME = m_name
        _TOP_K = top_k

    def _generate_answer(
        item: Tuple[str, Dict],
        q_dict: Dict[str, Dict],
        p_lookup: Dict[str, str],
        s_url: str,
        m_name: str,
        top_k: int,
    ) -> Tuple[str, Dict, str]:
        qid, t_entry = item
        q = q_dict[qid]
        question = q["question"]
        gold_answer = q.get("gold_answer", "")  # resolve gold for consistency
        _ = normalise_answer(gold_answer)

        helpful = sorted(
            t_entry.get("helpful_passages", []),
            key=lambda x: x["score"],
            reverse=True,
        )
        passage_ids_sorted = [h["passage_id"] for h in helpful]
        top_passages = passage_ids_sorted[:top_k]
        hits_val = compute_hits_at_k(passage_ids_sorted, q.get("gold_passages", []), top_k)

        llm_out = ask_llm_with_passages(
            query_text=question,
            passage_ids=passage_ids_sorted,
            graph=None,
            server_url=s_url,
            passage_lookup=p_lookup,
            model_name=m_name,
            top_k_answer_passages=top_k,
        )

        answer_dict = {
            "question_id": qid,
            "question": question,
            "raw_answer": llm_out["raw_answer"],
            "normalised_answer": llm_out["normalised_answer"],
            "used_passages": top_passages,
            "hits_at_k": hits_val,
            "prompt_len": llm_out.get("prompt_len", 0),
            "output_tokens": llm_out.get("output_tokens", 0),
            "total_tokens": llm_out.get("total_tokens", 0),
        }

        return qid, answer_dict, llm_out["normalised_answer"]

    if num_workers is None:
        size = model_size(traversal_model if model_name is None else model_name)
        num_workers = {"14b": 1, "19b": 1, "7b": 2}.get(size, 4)

    worker = partial(
        _generate_answer,
        q_dict=queries,
        p_lookup=passage_lookup,
        s_url=server_url,
        m_name=model_name,
        top_k=top_k_answer_passages,
    )

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=str(result_paths["answers"]),
        items=traversal_records.values(),
        get_id=lambda r, i: r["question_id"],
        phase_label="Answer generation",
        id_field="question_id",
    )
    if resume:
        traversal_records = {
            qid: rec for qid, rec in traversal_records.items() if qid not in done_ids
        }
    if not traversal_records:
        print("No new queries to process.")
        return {}

    results = pool_map(worker, traversal_records.items(), processes=num_workers)
    for qid, answer, norm_ans in results:
        answers.append(answer)
        predictions[qid] = norm_ans
        gold[qid] = [queries[qid].get("gold_answer", "")]
        hits[qid] = answer.get("hits_at_k", 0.0)
        token_totals["prompt_tokens"] += answer.get("prompt_len", 0)
        token_totals["completion_tokens"] += answer.get("output_tokens", 0)
        token_totals["total_tokens"] += answer.get(
            "total_tokens", answer.get("prompt_len", 0) + answer.get("output_tokens", 0)
        )

    result_paths["base"].mkdir(parents=True, exist_ok=True)
    if resume:
        append_jsonl(result_paths["answers"], answers)
    else:
        save_jsonl(result_paths["answers"], answers)

    metrics = evaluate_answers(predictions, gold)
    if hits:
        metrics["hits_at_k"] = round(sum(hits.values()) / len(hits), 4)
    with open(result_paths["summary"], "wt", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(result_paths["base"] / "token_usage.json", "w", encoding="utf-8") as f:
        json.dump(token_totals, f, indent=2)

    return metrics




def sweep_thresholds(edges: List[Dict], thresholds: List[float]):
    for t in thresholds:
        filtered = [e for e in edges if e['sim_hybrid'] >= t]
        print(f"Threshold {t:.2f}: {len(filtered)} edges")

















if __name__ == "__main__":
    # === Configuration ===
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["dev"]
    READER_MODEL = "llama-3.1-8b-instruct"
    GRAPH_MODELS = ["llama-3.1-8b-instruct"]
    TRAVERSAL_MODELS = [
        "qwen2.5-7b-instruct"]
    #     "qwen2.5-14b-instruct",
    #     "deepseek-r1-distill-qwen-7b",
    #     "deepseek-r1-distill-qwen-14b",
    #     "qwen2.5-moe-19b",
    # ]
    VARIANTS = ["baseline"]  

    TOP_K_ANSWER_PASSAGES = 5

    for dataset in DATASETS:
        for split in SPLITS:
            for graph_model in GRAPH_MODELS:
                for traversal_model in TRAVERSAL_MODELS:
                    for variant in VARIANTS:
                        print(
                            "[Answers-only] dataset={dataset} graph_model={graph_model} traversal_model={traversal_model} variant={variant} split={split}".format(
                                dataset=dataset,
                                graph_model=graph_model,
                                traversal_model=traversal_model,
                                variant=variant,
                                split=split,
                            )
                        )
                        metrics = generate_answers_from_traversal(
                            graph_model,
                            traversal_model,
                            dataset,
                            split,
                            variant,
                            top_k_answer_passages=TOP_K_ANSWER_PASSAGES,
                            reader_model=READER_MODEL,
                            num_workers=None,
                        )
    print("\n✅ Answers-only complete.")





