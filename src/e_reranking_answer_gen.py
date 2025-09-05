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
    When a `seed` is provided, `{variant}` becomes `{variant}_seed{seed}`.


### data/processed_datasets/{dataset}/{split}/questions.jsonl
    Query set with `question_id`, `question`, and `gold_answer(s)`.

Outputs
-------

### data/results/{model}/{dataset}/{split}/{variant}/

- `answer_per_query_{variant}_{split}.jsonl`
    LLM-generated answers per query.

- `answer_metrics_{variant}_{split}.jsonl`
    Exact-match and F1 metrics for each query.

- `summary_metrics_{variant}_{split}.json`
    Aggregate answer metrics in the form `{ "timestamp": ..., "answer_eval": {..} }`.
    Like the traversal path, include `_seed{seed}` after `variant` when a
    seed is specified.


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



### summary_metrics_{variant}_{split}.json

Aggregated answer evaluation with run metadata:

{
  "dataset": str,
  "split": str,
  "variant": str,
  "model": str,
  "retriever": str | null,
  "seed": int | null,
  "timestamp": "2025-08-13T14:22:31",
  "answer_eval": {
    "EM": float,
    "F1": float,
    "hits_at_k": float,
    "recall_at_k": float
  },
  "median_f1": float,
  "p90_f1": float,
  "median_em": float,
  "p90_em": float,
  "median_trav_tokens_total": float,
  "p90_trav_tokens_total": float,
  "median_reader_total_tokens": float,
  "p90_reader_total_tokens": float,
  "median_latency_ms": float,
  "p90_latency_ms": float,
  "median_t_reader_ms": float,
  "p90_t_reader_ms": float,
  "median_n_traversal_calls": float,
  "p90_n_traversal_calls": float,
  "median_n_reader_calls": float,
  "p90_n_reader_calls": float
}



Notes
-----

- Traversals are run on the `dev` split, not `train`.
- `baseline` = no revisits to previously visited passages.
- `enhanced` = prioritises edges by destination `conditioned_score` but also
  avoids revisiting nodes.
- All outputs are saved in:
  `data/results/{model}/{dataset}/{split}/{variant}/`
"""

import os
import time
import json
import pickle
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime

from functools import partial
import networkx as nx
import pickle

from src.llm_utils import is_r1_like, query_llm, strip_think
from src.config import MAX_TOKENS, TEMPERATURE
from src.utils import (
    append_jsonl,
    compute_resume_sets,
    get_result_paths,
    get_server_configs,
    get_traversal_paths,
    load_jsonl,
    processed_dataset_paths,
    compute_hits_at_k,
    compute_recall_at_k,
    merge_token_usage,
    model_size,
)


from src.metrics_summary import append_percentiles



# Public exports from this module
__all__ = [
    "ask_llm_with_passages",
    "evaluate_answers",
    "aggregate_answer_scores",
]

################################################################################################################
# PASSAGE RERANKING AND ANSWER GENERATION 
################################################################################################################


def extract_final_answer(raw: str) -> str:
    """Extract the last stated answer from raw LLM output.

    Searches for patterns like "final answer", "answer is", or "Answer:" and
    returns the text following the last occurrence. If no such pattern is
    found, the original string is returned stripped.
    """
    pattern = re.compile(
        r"(?:final answer|answer(?: is)?)\s*:?\s*(.+)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    matches = pattern.findall(raw)
    if matches:
        return matches[-1].strip()
    return raw.strip()


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


def clean_tokens(tokens: list[str]) -> list[str]:
    """Remove duplicate tokens while keeping order.

    If tokens besides ``"unknown"`` exist, drop all occurrences of
    ``"unknown"``. Otherwise, keep a single ``"unknown"`` token.
    """
    drop_unknown = any(t != "unknown" for t in tokens)
    seen: set[str] = set()
    cleaned: list[str] = []
    for t in tokens:
        if t == "unknown" and drop_unknown:
            continue
        if t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned


def first_fragment(text: str) -> str:
    """Return the first non-empty fragment split by period or newline."""

    for part in re.split(r"[.\n]", text):
        frag = part.strip()
        if frag:
            return frag
    return ""



def post_process_answer(
    text: str,
    max_words: int = 40,
    repeat_threshold: int = 3,
) -> Optional[str]:
    """Validate and possibly reject an LLM answer.

    Returns the original ``text`` if it is within ``max_words`` and no token
    appears more than ``repeat_threshold`` times. Otherwise ``None`` is
    returned to signal that the answer should be retried or replaced with a
    fallback.
    """

    tokens = text.split()
    if len(tokens) > max_words:
        return None

    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
        if counts[t] > repeat_threshold:
            return None

    return text



def ask_llm_with_passages(
    query_text: str,
    passage_ids: List[str],
    graph: Optional[nx.DiGraph],
    server_url: str,
    max_tokens: int = MAX_TOKENS["answer_generation"],
    passage_lookup: Optional[Dict[str, str]] = None,  # optional for dense mode
    model_name: str = "",
    top_k_answer_passages: int = 20,
    seed: int | None = None,

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
        Maximum number of tokens to generate, by default ``MAX_TOKENS["answer_generation"]``.
    passage_lookup : Optional[Dict[str, str]]
        Mapping from ``passage_id`` to passage text when ``graph`` is ``None``.
    model_name : str, optional
        Identifier of the active model, passed to ``query_llm``.
    top_k_answer_passages : int, optional
        Number of passages from ``passage_ids`` to include, by default ``20``.

    Outputs
    -------
    Dict[str, str]
        ``{"raw_answer": str, "normalised_answer": str, "prompt_len": int, "output_tokens": int, "total_tokens": int}``
        where the token counts are provided by :func:`query_llm`.
    """
    passage_texts = []

    for i, pid in enumerate(passage_ids[:top_k_answer_passages], start=1):
        if graph:
            passage = graph.nodes[pid].get("text", "")
        elif passage_lookup:
            passage = passage_lookup.get(pid, "")
        else:
            passage = "[missing passage]"

        passage_texts.append(f"[{i}]: {passage}")

    prompt = (
        "Answer the question using the following passages.\n"
        "Return exactly one concise fact. If the fact is unknown, reply `unknown`.\n\n"
        + "\n\n".join(passage_texts)
        + f"\n\nQuestion: {query_text}\nAnswer:"
    )

    raw = ""
    raw_clean = ""
    usage: Dict[str, int] = {}
    max_attempts = 2

    stop_sequences: list[str] | None = ["\n", ".", "Answer:", "Final answer:"]
    if is_r1_like(model_name):
        stop_sequences = None

    for attempt in range(max_attempts):
        raw, usage = query_llm(
            prompt,
            server_url=server_url,
            max_tokens=max_tokens,
            model_name=model_name,
            stop=stop_sequences,
            temperature=TEMPERATURE.get("answer_generation", 0.0),
            phase="answer_generation",
            seed=seed,
        )

        if is_r1_like(model_name):
            raw = strip_think(raw)

        raw_clean = first_fragment(extract_final_answer(raw))
        if post_process_answer(raw_clean) is not None:
            break
    else:
        raw = raw_clean = "unknown"

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    tokens = clean_tokens(normalise_answer(raw_clean).split())
    norm = " ".join(tokens)

    return {
        "raw_answer": raw,
        "raw_clean": raw_clean,
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
        predictions: dict[str, str],
        gold_answers: dict[str, list[str]],
        ) -> dict[str, dict]:
    """Compute per-query EM and F1 scores."""
    results: dict[str, dict] = {}
    for qid, gold_list in gold_answers.items():
        pred = predictions.get(qid, "")
        em = max((compute_exact_match(pred, g) for g in gold_list), default=0)
        f1 = max((compute_f1(pred, g) for g in gold_list), default=0.0)
        results[qid] = {"prediction": pred, "em": em, "f1": f1}
    return results



def aggregate_answer_scores(
        predictions: dict,
        gold_answers: dict
        ) -> dict:
    """Aggregate EM and F1 over a set of predicted answers.

    Parameters
    ----------
    predictions: dict
        Mapping ``{id: predicted_answer}``.
    gold_answers: dict
        Mapping ``{id: list of gold answers}`` allowing multiple paraphrases.

    Returns
    -------
    dict
        Overall EM and F1 expressed as percentages.
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
        "F1": 100.0 * f1_total / total,
    }



def _generate_answer(
    item: Tuple[str, Dict],
    q_dict: Dict[str, Dict],
    p_lookup: Dict[str, str],
    s_url: str,
    m_name: str,
    top_k: int,
    seed: int | None = None,

) -> Tuple[str, Dict, str]:
    """Worker function to generate an answer for a single query."""
    qid, t_entry = item
    q = q_dict[qid]
    question = q["question"]
    gold_answer = q.get("gold_answer", "")
    _ = normalise_answer(gold_answer)

    helpful = sorted(
        t_entry.get("helpful_passages", []),
        key=lambda x: x["score"],
        reverse=True,
    )
    passage_ids_sorted = [h["passage_id"] for h in helpful]
    top_passages = passage_ids_sorted[:top_k]
    hits_val = compute_hits_at_k(
        passage_ids_sorted, q.get("gold_passages", []), top_k
    )

    recall_val = compute_recall_at_k(
        passage_ids_sorted, q.get("gold_passages", []), top_k
    )

    print(f"\n[Query] {qid} - \"{question}\"")
    start_time = time.perf_counter()
    llm_out = ask_llm_with_passages(
        query_text=question,
        passage_ids=passage_ids_sorted,
        graph=None,
        server_url=s_url,
        passage_lookup=p_lookup,
        model_name=m_name,
        top_k_answer_passages=top_k,
        seed=seed,

    )
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    prompt_tokens = llm_out.get("prompt_len", 0)
    output_tokens = llm_out.get("output_tokens", 0)
    total_tokens = llm_out.get(
        "total_tokens", prompt_tokens + output_tokens
    )

    answer_dict = {
        "question_id": qid,
        "question": question,
        "raw_answer": llm_out["raw_answer"],
        "normalised_answer": llm_out["normalised_answer"],
        "used_passages": top_passages,
        "hits_at_k": hits_val,
        "recall_at_k": recall_val,

        # Legacy token fields
        "prompt_len": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,

        # Reader usage fields
        "reader_prompt_tokens": prompt_tokens,
        "reader_output_tokens": output_tokens,
        "reader_total_tokens": total_tokens,
        "n_reader_calls": 1,
        "t_reader_ms": elapsed_ms,
    }

    return qid, answer_dict, llm_out["normalised_answer"]



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
    seed: int | None = None,

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
        choose ``1`` worker for ``14b`` models, ``2`` for ``7b`` models,
        and ``4`` otherwise.
    seed:
        Random seed forwarded to the LLM for reproducible outputs.
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
    tasks are distributed using :class:`multiprocessing.Pool` to avoid global
    state and maintain consistent multiprocessing patterns.
    """

    if server_url is None or model_name is None:
        server = get_server_configs(reader_model)[0]
        server_url = server_url or server["server_url"]
        model_name = model_name or server["model"]

    variant_for_path = f"{variant}_seed{seed}" if seed is not None else variant
    traversal_paths = get_traversal_paths(
        traversal_model, dataset, split, variant_for_path
    )
    result_paths = get_result_paths(
        traversal_model, dataset, split, variant_for_path
    )

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
    sample_record = next(iter(traversal_records.values()), {})
    retriever_name = sample_record.get("retriever_name")
    with open(graph_file, "rb") as f:
        graph = pickle.load(f)
    passage_lookup = {pid: data.get("text", "") for pid, data in graph.nodes(data=True)}
    queries = {q["question_id"]: q for q in load_jsonl(query_file)}

    predictions: Dict[str, str] = {}
    gold: Dict[str, List[str]] = {}
    hits: Dict[str, float] = {}
    recalls: Dict[str, float] = {}

    token_totals = {
        "reader_prompt_tokens": 0,
        "reader_output_tokens": 0,
        "reader_total_tokens": 0,
        "n_reader_calls": 0,
    }
    per_query_reader: Dict[str, Dict[str, int]] = {}
    reader_time_total_ms = 0



    if num_workers is None:
        size = model_size(traversal_model if model_name is None else model_name)
        num_workers = {"14b": 1, "7b": 2}.get(size, 4)

    worker = partial(
        _generate_answer,
        q_dict=queries,
        p_lookup=passage_lookup,
        s_url=server_url,
        m_name=model_name,
        top_k=top_k_answer_passages,
        seed=seed,

    )

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=str(result_paths["answers"]),
        items=traversal_records.values(),
        get_id=lambda r, i: r["question_id"],
        phase_label="Answer generation",
        id_field="question_id",
    )
    result_paths["base"].mkdir(parents=True, exist_ok=True)
    if not resume:
        result_paths["answers"].unlink(missing_ok=True)
        result_paths["answer_metrics"].unlink(missing_ok=True)
    if resume:
        traversal_records = {
            qid: rec for qid, rec in traversal_records.items() if qid not in done_ids
        }
    if not traversal_records:
        print("No new queries to process.")
        return {}

    items = list(traversal_records.items())
    with Pool(num_workers) as pool:
        for qid, answer, norm_ans in tqdm(
            pool.imap(worker, items),
            total=len(items),
            desc="queries",
        ):
            predictions[qid] = norm_ans
            gold_list = [queries[qid].get("gold_answer", "")]
            gold[qid] = gold_list
            hits[qid] = answer.get("hits_at_k", 0.0)
            recalls[qid] = answer.get("recall_at_k", 0.0)

            rp = answer.get("reader_prompt_tokens", answer.get("prompt_len", 0))
            ro = answer.get("reader_output_tokens", answer.get("output_tokens", 0))
            rt = answer.get(
                "reader_total_tokens",
                answer.get("total_tokens", rp + ro),
            )
            n_calls = answer.get("n_reader_calls", 1)
            t_ms = answer.get("t_reader_ms", 0)

            token_totals["reader_prompt_tokens"] += rp
            token_totals["reader_output_tokens"] += ro
            token_totals["reader_total_tokens"] += rt
            token_totals["n_reader_calls"] += n_calls
            reader_time_total_ms += t_ms

            per_query_reader[qid] = {
                "reader_prompt_tokens": rp,
                "reader_output_tokens": ro,
                "reader_total_tokens": rt,
                "n_reader_calls": n_calls,
                "t_reader_ms": t_ms,
                "query_latency_ms": t_ms,
                "call_latency_ms": t_ms / max(n_calls, 1),
            }

            append_jsonl(result_paths["answers"], answer)
            em = max(
                (compute_exact_match(norm_ans, g) for g in gold_list),
                default=0,
            )
            f1 = max(
                (compute_f1(norm_ans, g) for g in gold_list),
                default=0.0,
            )
            append_jsonl(
                result_paths["answer_metrics"],
                {"question_id": qid, "prediction": norm_ans, "em": em, "f1": f1},
            )

    per_query = evaluate_answers(predictions, gold)

    metrics = {
        "EM": 100.0 * np.mean([m["em"] for m in per_query.values()]),
        "F1": 100.0 * np.mean([m["f1"] for m in per_query.values()]),
    }
    if hits:
        metrics["hits_at_k"] = round(sum(hits.values()) / len(hits), 4)

    if recalls:
        metrics["recall_at_k"] = round(
            sum(recalls.values()) / len(recalls), 4
        )

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    summary_payload = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "model": traversal_model,
        "retriever": retriever_name,
        "seed": seed,
        "timestamp": timestamp,
        "answer_eval": metrics.copy(),
    }
    with open(result_paths["summary"], "wt", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    t_reader_ms = reader_time_total_ms
    num_queries = len(per_query_reader)
    n_reader_calls = token_totals["n_reader_calls"]

    query_latency_ms = t_reader_ms / num_queries if num_queries else 0
    call_latency_ms = t_reader_ms / max(n_reader_calls, 1)
    query_qps_reader = num_queries / (t_reader_ms / 1000) if t_reader_ms else 0.0
    cps_reader = (
        n_reader_calls / (t_reader_ms / 1000) if t_reader_ms else 0.0
    )
    usage = {
        "per_query_traversal": {},
        "per_query_reader": per_query_reader,
        "trav_prompt_tokens": 0,
        "trav_output_tokens": 0,
        "trav_tokens_total": 0,
        **token_totals,
        "t_traversal_ms": 0,
        "t_reader_ms": t_reader_ms,
        "num_queries": num_queries,
        "query_latency_ms": query_latency_ms,
        "call_latency_ms": call_latency_ms,
        "query_qps_reader": query_qps_reader,
        "cps_reader": cps_reader,
    }

    run_id = str(int(time.time()))  # Identifier to group token usage shards
    usage_path = result_paths["base"] / f"token_usage_{run_id}_{os.getpid()}.json"
    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)

    # Consolidate token usage files for this run and remove the temporary parts
    merge_token_usage(result_paths["base"], run_id=run_id, cleanup=True)

    metrics.update({
        "trav_prompt_tokens": 0,
        "trav_output_tokens": 0,
        "trav_tokens_total": 0,
        **token_totals,
        "t_traversal_ms": 0,
        "t_reader_ms": t_reader_ms,
        "num_queries": num_queries,
        "query_latency_ms": query_latency_ms,
        "call_latency_ms": call_latency_ms,

    })

    tokens_total = (
        metrics.get("trav_tokens_total", 0)
        + metrics.get("reader_total_tokens", 0)
    )
    t_total_ms = metrics.get("t_traversal_ms", 0) + metrics.get("t_reader_ms", 0)
    tps_overall = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0

    metrics.update({
        "tokens_total": tokens_total,
        "t_total_ms": t_total_ms,
        "tps_overall": tps_overall,
        "query_qps_reader": query_qps_reader,
        "cps_reader": cps_reader,
    })

    summary_payload["answer_eval"] = metrics
    with open(result_paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    token_usage_file = result_paths["base"] / "token_usage.json"
    try:
        with open(token_usage_file, "r", encoding="utf-8") as f:
            token_usage_data = json.load(f)
    except FileNotFoundError:
        token_usage_data = {}
    token_usage_data["query_qps_reader"] = query_qps_reader
    token_usage_data["cps_reader"] = cps_reader
    token_usage_data["num_queries"] = num_queries
    token_usage_data["query_latency_ms"] = query_latency_ms
    token_usage_data["call_latency_ms"] = call_latency_ms
    with open(token_usage_file, "w", encoding="utf-8") as f:
        json.dump(token_usage_data, f, indent=2)

    extra = append_percentiles(
        result_paths["answer_metrics"], result_paths["summary"]
    )
    metrics.update(extra)
    summary_payload.update(extra)

    print(
        f"[summary] overall throughput: {tps_overall:.2f} tokens/s | "
        f"reader query throughput: {query_qps_reader:.2f} queries/s | "
        f"reader call throughput: {cps_reader:.2f} calls/s | "
        f"reader query latency: {query_latency_ms:.2f} ms | "
        f"reader call latency: {call_latency_ms:.2f} ms"
    )

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
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
    "deepseek-r1-distill-qwen-7b",
    "deepseek-r1-distill-qwen-14b",
    "state-of-the-moe-rp-2x7b",
    "qwen2.5-2x7b-moe-power-coder-v4"]

#["qwen2.5-7b-instruct", "deepseek-r1-distill-qwen-7b"]
    #     "qwen2.5-14b-instruct",
    #     "deepseek-r1-distill-qwen-7b",
    #     "deepseek-r1-distill-qwen-14b",
    #     "qwen2.5-moe-14b",
    # "qwen2.5-2x7b-moe-power-coder-v4" ]
    VARIANTS = ["baseline"]
    TOP_K_ANSWER_PASSAGES = 20
    SEEDS = [1, 2, 3]
    for dataset in DATASETS:
        for split in SPLITS:
            for graph_model in GRAPH_MODELS:
                for traversal_model in TRAVERSAL_MODELS:
                    for variant in VARIANTS:
                        for seed in SEEDS:
                            print(
                                "[Answers-only] dataset={dataset} graph_model={graph_model} traversal_model={traversal_model} variant={variant} split={split} seed={seed}".format(
                                    dataset=dataset,
                                    graph_model=graph_model,
                                    traversal_model=traversal_model,
                                    variant=variant,
                                    split=split,
                                    seed=seed,
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
                                seed=seed,
                            )
    print("\nAnswers-only complete.")





