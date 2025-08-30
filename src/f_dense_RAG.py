"""Module Overview
---------------
Run a simple dense Retrieval-Augmented Generation (RAG) pipeline.

This module reuses the existing FAISS indexes and embedding model to
answer questions directly without graph traversal. It retrieves the
most similar passages for each query and asks a reader model to
produce an answer from those passages.
"""

from typing import Dict, List

import json
import numpy as np

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
    get_result_paths,
    get_server_configs,
    load_jsonl,
    processed_dataset_paths,
    save_jsonl,
)


def run_dense_rag(
    dataset: str,
    split: str,
    reader_model: str,
    server_url: str | None = None,
    top_k: int = DEFAULT_SEED_TOP_K,
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

    Returns
    -------
    Dict[str, float]
        Exact Match and F1 scores across the query set.
    """

    if server_url is None:
        server_url = get_server_configs(reader_model)[0]["server_url"]

    rep_paths = dataset_rep_paths(dataset, split)
    passages = list(load_jsonl(rep_paths["passages_jsonl"]))
    passage_lookup = {p["passage_id"]: p["text"] for p in passages}
    index = load_faiss_index(rep_paths["passages_index"])
    encoder = get_embedding_model()

    query_path = processed_dataset_paths(dataset, split)["questions"]

    answers: List[Dict[str, str]] = []
    predictions: Dict[str, str] = {}
    gold: Dict[str, List[str]] = {}

    for q in load_jsonl(query_path):
        q_id = q["question_id"]
        q_text = q["question"]
        gold[q_id] = [q.get("gold_answer", "")]

        q_emb = encoder.encode([q_text], normalize_embeddings=True)
        idxs, _ = faiss_search_topk(q_emb, index, top_k=top_k)
        passage_ids = [passages[i]["passage_id"] for i in idxs]
        llm_out = ask_llm_with_passages(
            query_text=q_text,
            passage_ids=passage_ids,
            graph=None,
            server_url=server_url,
            passage_lookup=passage_lookup,
            model_name=reader_model,
            top_k_answer_passages=top_k,
        )

        answers.append(
            {
                "question_id": q_id,
                "question": q_text,
                "raw_answer": llm_out["raw_answer"],
                "normalised_answer": llm_out["normalised_answer"],
                "used_passages": passage_ids,
            }
        )
        predictions[q_id] = llm_out["normalised_answer"]

    paths = get_result_paths(reader_model, dataset, split, "dense")
    paths["base"].mkdir(parents=True, exist_ok=True)
    save_jsonl(paths["answers"], answers)

    metrics = evaluate_answers(predictions, gold)
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics




if __name__ == "__main__":
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["dev"]
    HOPRAG_VERSIONS = ["baseline", "enhanced"]
    READER_MODELS = ["llama-3.1-8b-instruct"]
    TOP_K = DEFAULT_SEED_TOP_K

    for dataset in DATASETS:
        for split in SPLITS:
            for hop_version in HOPRAG_VERSIONS:
                for reader in READER_MODELS:
                    print(
                        f"[Dense RAG] dataset={dataset} split={split} hoprag_version={hop_version} reader={reader} top_k={TOP_K}"
                    )
                    metrics = run_dense_rag(
                        dataset,
                        split,
                        reader_model=reader,
                        top_k=TOP_K,
                    )
                    print(metrics)
    print("\n✅ Dense RAG complete.")