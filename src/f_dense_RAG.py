"""Module Overview
---------------
Run a simple dense Retrieval-Augmented Generation (RAG) pipeline.

This module reuses the existing FAISS indexes and embedding model to
answer questions directly without graph traversal. It retrieves the
most similar passages for each query and asks a reader model to
produce an answer from those passages. Per-query retrieval metrics such
as ``hits_at_k`` and ``recall_at_k`` are logged alongside the generated
answers.
"""

from typing import Dict, List

import random
import numpy as np
import time
import json
from datetime import datetime
import os

from tqdm import tqdm

from src.b_sparse_dense_representations import (
    dataset_rep_paths,
    faiss_search_topk,
    get_embedding_model,
    load_faiss_index,
)
from src.d_traversal import DEFAULT_SEED_TOP_K
from src.e_reranking_answer_gen import (
    ask_llm_with_passages,
    evaluate_answers,
    aggregate_answer_scores,

)
from src.utils import (
    append_jsonl,
    compute_resume_sets,
    get_result_paths,
    get_server_configs,
    load_jsonl,
    processed_dataset_paths,
    compute_hits_at_k,
    compute_recall_at_k,

    log_wall_time,
    save_jsonl,
    merge_token_usage,

)
from src.metrics_summary import append_percentiles


def run_dense_rag(
    dataset: str,
    split: str,
    reader_model: str,
    retriever_name: str = "dense",

    server_url: str | None = None,
    top_k: int = DEFAULT_SEED_TOP_K,
    seed: int | None = None,
    resume: bool = False,
) -> Dict[str, float]:
    """Answer queries using dense retrieval over passages and evaluate EM/F1.

    The function retrieves top-``k`` passages for each query and asks a reader
    model to generate an answer. Retrieval metrics ``hits_at_k`` and
    ``recall_at_k`` are computed per query and included in both the per-query
    JSONL output and the summary metrics file.

    Parameters
    ----------
    dataset: str
        Name of the dataset (e.g. ``"hotpotqa"``).
    split: str
        Dataset split (e.g. ``"dev"``).
    reader_model: str
        Name of the reader model used to generate answers. ``server_url``
        defaults to the first entry returned by
        :func:`src.utils.get_server_configs` for this model when ``None``.
    retriever_name: str, optional
        Identifier for the passage retriever used (e.g. ``"dense"``).
    server_url: str, optional
        URL of the completion endpoint for ``reader_model``. When ``None``,
        the first matching server from :func:`get_server_configs` is used.
    top_k: int, optional
        Number of passages to retrieve for each query. Defaults to
        ``DEFAULT_SEED_TOP_K`` from :mod:`src.d_traversal`.
    seed: int, optional
        Seed used to initialize :mod:`random` and :mod:`numpy` for
        deterministic behaviour.
    resume: bool, optional
        Resume a previously interrupted run by reusing existing answers and
        skipping already processed questions. When ``True``,
        :func:`src.utils.compute_resume_sets` determines which question IDs
        have been completed.

    Returns
    -------
    Dict[str, float]
        Exact Match, F1, ``hits_at_k`` and ``recall_at_k`` scores across the
        query set.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if server_url is None:
        server_url = get_server_configs(reader_model)[0]["server_url"]

    rep_paths = dataset_rep_paths(dataset, split)
    passages = list(load_jsonl(rep_paths["passages_jsonl"]))
    passage_lookup = {p["passage_id"]: p["text"] for p in passages}
    index = load_faiss_index(rep_paths["passages_index"])
    encoder = get_embedding_model()

    query_path = processed_dataset_paths(dataset, split)["questions"]
    queries = list(load_jsonl(query_path))

    variant = "dense" if seed is None else f"dense_seed{seed}"
    paths = get_result_paths(reader_model, dataset, split, variant)

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=str(paths["answers"]),
        items=queries,
        get_id=lambda q, i: q["question_id"],
        phase_label="Dense RAG",
        id_field="question_id",
    )
    paths["base"].mkdir(parents=True, exist_ok=True)
    if not resume and paths["answers"].exists():
        paths["answers"].unlink()

    predictions: Dict[str, str] = {}
    gold: Dict[str, List[str]] = {}
    hits_at_k_scores: Dict[str, float] = {}
    recall_at_k_scores: Dict[str, float] = {}

    token_totals = {
        "reader_prompt_tokens": 0,
        "reader_output_tokens": 0,
        "reader_total_tokens": 0,
        "n_reader_calls": 0,

    }
    per_query_reader: Dict[str, Dict[str, int]] = {}
    reader_time_total_ms = 0


    for q in tqdm(queries, desc="queries"):
        q_id = q["question_id"]
        if resume and q_id in done_ids:
            continue
        q_text = q["question"]
        gold[q_id] = [q.get("gold_answer", "")]

        print(f"\n[Query] {q_id} - \"{q_text}\"")

        q_emb = encoder.encode([q_text], normalize_embeddings=False)
        idxs, _ = faiss_search_topk(q_emb, index, top_k=top_k)
        passage_ids = [passages[i]["passage_id"] for i in idxs]
        hits_val = compute_hits_at_k(passage_ids, q.get("gold_passages", []), top_k)
        recall_val = compute_recall_at_k(
            passage_ids, q.get("gold_passages", []), top_k
        )
        start_time = time.perf_counter()

        llm_out = ask_llm_with_passages(
            query_text=q_text,
            passage_ids=passage_ids,
            graph=None,
            server_url=server_url,
            passage_lookup=passage_lookup,
            model_name=reader_model,
            top_k_answer_passages=top_k,
            seed=seed,

        )
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)


        append_jsonl(
            str(paths["answers"]),
            {
                "dataset": dataset,
                "split": split,
                "variant": variant,
                "retriever_name": retriever_name,
                "traverser_model": None,
                "reader_model": reader_model,
                "question_id": q_id,
                "question": q_text,
                "raw_answer": llm_out["raw_answer"],
                "normalised_answer": llm_out["normalised_answer"],
                "used_passages": passage_ids,
                "hits_at_k": hits_val,
                "recall_at_k": recall_val,

                "prompt_len": llm_out.get("prompt_len", 0),
                "output_tokens": llm_out.get("output_tokens", 0),
                "total_tokens": llm_out.get("total_tokens", 0),
                "reader_prompt_tokens": llm_out.get("prompt_len", 0),
                "reader_output_tokens": llm_out.get("output_tokens", 0),
                "reader_total_tokens": llm_out.get(
                    "total_tokens",
                    llm_out.get("prompt_len", 0) + llm_out.get("output_tokens", 0),
                ),
                "t_reader_ms": elapsed_ms,

                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "seed": seed,

            },
        )
        predictions[q_id] = llm_out["normalised_answer"]
        hits_at_k_scores[q_id] = hits_val
        recall_at_k_scores[q_id] = recall_val

        token_totals["n_reader_calls"] += 1
        token_totals["reader_prompt_tokens"] += llm_out.get("prompt_len", 0)
        token_totals["reader_output_tokens"] += llm_out.get("output_tokens", 0)
        token_totals["reader_total_tokens"] += llm_out.get(
            "total_tokens", llm_out.get("prompt_len", 0) + llm_out.get("output_tokens", 0)
        )
        per_query_reader[q_id] = {
            "reader_prompt_tokens": llm_out.get("prompt_len", 0),
            "reader_output_tokens": llm_out.get("output_tokens", 0),
            "reader_total_tokens": llm_out.get(
                "total_tokens",
                llm_out.get("prompt_len", 0) + llm_out.get("output_tokens", 0),
            ),
            "n_reader_calls": 1,
            "t_reader_ms": elapsed_ms,
        }
        reader_time_total_ms += elapsed_ms

    if not gold:
        print("No new queries to process.")
        return {}

    per_query = evaluate_answers(predictions, gold)
    agg_scores = aggregate_answer_scores(predictions, gold)

    now_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    metric_records = [
        {
            "dataset": dataset,
            "split": split,
            "variant": variant,
            "retriever_name": retriever_name,
            "traverser_model": None,
            "reader_model": reader_model,
            "question_id": qid,
            **m,
            "timestamp": now_ts,
        }
        for qid, m in per_query.items()
    ]
    if resume:
        for rec in metric_records:
            append_jsonl(str(paths["answer_metrics"]), rec)
    else:
        save_jsonl(str(paths["answer_metrics"]), metric_records)

    metrics = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "retriever_name": retriever_name,
        "traverser_model": None,
        "reader_model": reader_model,
        "EM": agg_scores["EM"],
        "F1": agg_scores["F1"],
        "timestamp": now_ts,
    }
    if seed is not None:
        metrics["seed"] = seed
    if hits_at_k_scores:
        metrics["hits_at_k"] = round(
            sum(hits_at_k_scores.values()) / len(hits_at_k_scores), 4
        )
    if recall_at_k_scores:
        metrics["recall_at_k"] = round(
            sum(recall_at_k_scores.values()) / len(recall_at_k_scores), 4
        )
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    extra = append_percentiles(paths["answer_metrics"], paths["summary"])
    metrics.update(extra)

    t_reader_ms = reader_time_total_ms
    usage = {
        "per_query_traversal": {},
        "per_query_reader": per_query_reader,
        "trav_prompt_tokens": 0,
        "trav_output_tokens": 0,
        "trav_tokens_total": 0,
        **token_totals,
        "t_traversal_ms": 0,
        "t_reader_ms": t_reader_ms,
    }
    run_id = str(int(time.time()))  # Identifier to group token usage shards
    usage_path = paths["base"] / f"token_usage_{run_id}_{os.getpid()}.json"
    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)

    merge_token_usage(paths["base"], run_id=run_id, cleanup=True)

    metrics.update({
        "trav_prompt_tokens": 0,
        "trav_output_tokens": 0,
        "trav_tokens_total": 0,
        **token_totals,
        "t_traversal_ms": 0,
        "t_reader_ms": t_reader_ms,
    })

    tokens_total = (
        metrics.get("trav_tokens_total", 0)
        + metrics.get("reader_total_tokens", 0)
    )
    t_total_ms = metrics.get("t_traversal_ms", 0) + metrics.get("t_reader_ms", 0)
    tps_overall = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0
    metrics.update(
        {
            "tokens_total": tokens_total,
            "t_total_ms": t_total_ms,
            "tps_overall": tps_overall,
        }
    )

    print(f"[summary] overall throughput: {tps_overall:.2f} tokens/s")
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics









if __name__ == "__main__":
    start_time = time.time()
    DATASETS = ["hotpotqa", "2wikimultihopqa", "musique"]
    SPLITS = ["dev"]

    READER_MODELS = ["deepseek-r1-distill-qwen-7b"]

    #     "qwen2.5-7b-instruct",
    #     "qwen2.5-14b-instruct",

    #     "deepseek-r1-distill-qwen-7b",
    #     "deepseek-r1-distill-qwen-14b",

    #     "qwen2.5-moe-14b",

    #     "state-of-the-moe-rp-2x7b",

    #     "qwen2.5-2x7b-moe-power-coder-v4"
    # ]


    SEEDS = [1, 2, 3]

    TOP_K = DEFAULT_SEED_TOP_K

    for seed in SEEDS:
        for dataset in DATASETS:
            for split in SPLITS:
                for reader in READER_MODELS:
                    print(
                        f"[Dense RAG] dataset={dataset} split={split} reader={reader} top_k={TOP_K} seed={seed}"
                    )
                    metrics = run_dense_rag(
                        dataset,
                        split,
                        reader_model=reader,
                        top_k=TOP_K,
                        seed=seed,
                        resume=True,
                    )
                    print(metrics)
    print("\nDense RAG complete.")
    log_wall_time(__file__, start_time)
