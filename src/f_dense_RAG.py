"""Module Overview
---------------
Run a simple dense Retrieval-Augmented Generation (RAG) pipeline.

This module reuses the existing FAISS indexes and embedding model to
answer questions directly without graph traversal. It retrieves the
most similar passages for each query and asks a reader model to
produce an answer from those passages.
"""

from typing import Dict, List

import random
import numpy as np
import time
import json
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
)
from src.utils import (
    append_jsonl,
    compute_resume_sets,
    get_result_paths,
    get_server_configs,
    load_jsonl,
    processed_dataset_paths,
    compute_hits_at_k,
    log_wall_time   
)


def run_dense_rag(
    dataset: str,
    split: str,
    reader_model: str,
    server_url: str | None = None,
    top_k: int = DEFAULT_SEED_TOP_K,
    seed: int | None = None,
    resume: bool = False,
) -> Dict[str, float]:
    """Answer queries using dense retrieval over passages and evaluate EM/F1.

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
        Exact Match and F1 scores across the query set.
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
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}



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
        llm_out = ask_llm_with_passages(
            query_text=q_text,
            passage_ids=passage_ids,
            graph=None,
            server_url=server_url,
            passage_lookup=passage_lookup,
            model_name=reader_model,
            top_k_answer_passages=top_k,
        )

        append_jsonl(
            str(paths["answers"]),
            {
                "question_id": q_id,
                "question": q_text,
                "raw_answer": llm_out["raw_answer"],
                "normalised_answer": llm_out["normalised_answer"],
                "used_passages": passage_ids,
                "hits_at_k": hits_val,
                "prompt_len": llm_out.get("prompt_len", 0),
                "output_tokens": llm_out.get("output_tokens", 0),
                "total_tokens": llm_out.get("total_tokens", 0),
                "seed": seed,

            },
        )
        predictions[q_id] = llm_out["normalised_answer"]
        hits_at_k_scores[q_id] = hits_val
        token_totals["prompt_tokens"] += llm_out.get("prompt_len", 0)
        token_totals["completion_tokens"] += llm_out.get("output_tokens", 0)
        token_totals["total_tokens"] += llm_out.get(
            "total_tokens", llm_out.get("prompt_len", 0) + llm_out.get("output_tokens", 0)
        )

    if not gold:
        print("No new queries to process.")
        return {}

    metrics = evaluate_answers(predictions, gold)
    if seed is not None:
        metrics["seed"] = seed
    if hits_at_k_scores:
        metrics["hits_at_k"] = round(
            sum(hits_at_k_scores.values()) / len(hits_at_k_scores), 4
        )
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(paths["base"] / "token_usage.json", "w", encoding="utf-8") as f:
        json.dump(token_totals, f, indent=2)

    return metrics









if __name__ == "__main__":
    start_time = time.time()
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["dev"]
    READER_MODELS = ["llama-3.1-8b-instruct"]



    READER_MODELS = [

        "qwen2.5-7b-instruct",
        "qwen2.5-14b-instruct",

        "deepseek-r1-distill-qwen-7b",
        "deepseek-r1-distill-qwen-14b",

        "qwen2.5-moe-19b",

        "state-of-the-moe-rp-2x7b",

        "qwen2.5-2x7b-power-coder-v4"
    ]


    SEEDS = [0, 1, 3, 4, 5]

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
