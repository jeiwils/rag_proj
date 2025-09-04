import json
from pathlib import Path
from typing import Dict

from .analysis_and_plots.utils import load_token_usage


import numpy as np


def append_percentiles(metrics_path: str | Path, summary_path: str | Path) -> Dict[str, float]:
    """Append median and p90 metrics to an existing summary file.

    The function looks for two sources of per-query data within the same
    directory as ``summary_path``:

    * ``metrics_path`` - JSONL with ``em`` and ``f1`` per query.
    * ``token_usage.json`` - JSON containing optional ``per_query_traversal``
      and ``per_query_reader`` mappings with token usage, timings and latency.

    Parameters
    ----------
    metrics_path: str or Path
        JSONL file containing per-query metrics with fields ``em`` and ``f1``.
    summary_path: str or Path
        Path to the summary JSON file to update.

    Returns
    -------
    Dict[str, float]
        The computed statistics added to the summary file.
    """
    metrics_path = Path(metrics_path)
    summary_path = Path(summary_path)

    if not metrics_path.exists():
        return {}

    records = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return {}

    f1s = [r.get("f1", 0.0) for r in records]
    ems = [r.get("em", 0.0) for r in records]

    stats: Dict[str, float] = {}
    if f1s:
        stats["median_f1"] = float(np.median(f1s))
        stats["p90_f1"] = float(np.percentile(f1s, 90))
    if ems:
        stats["median_em"] = float(np.median(ems))
        stats["p90_em"] = float(np.percentile(ems, 90))

    # Attempt to compute token and timing percentiles from token_usage.json
    token_usage_path = summary_path.parent / "token_usage.json"
    if token_usage_path.exists():
        usage = load_token_usage(token_usage_path)
        per_query = usage.get("per_query", {})

        tokens: list[float] = []
        query_latencies_ms: list[float] = []
        call_latencies_ms: list[float] = []
        call_latencies_ms_traversal: list[float] = []
        call_latencies_ms_reader: list[float] = []
        tps: list[float] = []
        query_qps_trav: list[float] = []
        cps_trav: list[float] = []
        query_qps_read: list[float] = []
        cps_read: list[float] = []
        query_qps_overall: list[float] = []
        cps_overall: list[float] = []
        total_t_trav_ms = 0.0
        total_t_reader_ms = 0.0
        total_n_trav_calls = 0.0
        total_n_reader_calls = 0.0

        for metrics in per_query.values():
            tok = float(metrics.get("tokens_total", 0))
            t_ms = float(metrics.get("t_total_ms", 0))
            tokens.append(tok)
            query_latencies_ms.append(t_ms)
            tps.append(
                float(metrics.get("tps_overall", tok / (t_ms / 1000) if t_ms else 0.0))
            )

            n_trav_calls = float(metrics.get("n_traversal_calls", 0))
            n_reader_calls = float(metrics.get("n_reader_calls", 0))
            total_n_trav_calls += n_trav_calls
            total_n_reader_calls += n_reader_calls

            t_trav_ms = float(metrics.get("t_traversal_ms", 0))
            total_t_trav_ms += t_trav_ms
            t_reader_ms = float(metrics.get("t_reader_ms", 0))
            total_t_reader_ms += t_reader_ms

            if "call_latency_ms_traversal" in metrics:
                latency_trav = float(metrics.get("call_latency_ms_traversal", 0))
            else:
                latency_trav = t_trav_ms / max(n_trav_calls, 1)
            if "call_latency_ms_reader" in metrics:
                latency_reader = float(metrics.get("call_latency_ms_reader", 0))
            else:
                latency_reader = t_reader_ms / max(n_reader_calls, 1)
            call_latencies_ms_traversal.append(latency_trav)
            call_latencies_ms_reader.append(latency_reader)

            total_calls = n_trav_calls + n_reader_calls
            if "call_latency_ms" in metrics:
                latency = float(metrics.get("call_latency_ms", 0))
            else:
                latency = (t_trav_ms + t_reader_ms) / max(total_calls, 1)
            call_latencies_ms.append(latency)

            query_qps_trav.append(1 / (t_trav_ms / 1000) if t_trav_ms else 0.0)
            cps_trav.append(n_trav_calls / (t_trav_ms / 1000) if t_trav_ms else 0.0)

            query_qps_read.append(1 / (t_reader_ms / 1000) if t_reader_ms else 0.0)
            cps_read.append(n_reader_calls / (t_reader_ms / 1000) if t_reader_ms else 0.0)

            total_time_ms = t_trav_ms + t_reader_ms
            query_qps_overall.append(
                1 / (total_time_ms / 1000) if total_time_ms else 0.0
            )
            cps_overall.append(
                (n_trav_calls + n_reader_calls) / (total_time_ms / 1000)
                if total_time_ms
                else 0.0
            )

        num_queries = len(per_query)
        total_time_ms = total_t_trav_ms + total_t_reader_ms
        if total_time_ms:
            stats["overall_qps"] = num_queries / (total_time_ms / 1000)
            stats["overall_cps"] = (
                (total_n_trav_calls + total_n_reader_calls)
                / (total_time_ms / 1000)
            )
        else:
            stats["overall_qps"] = 0.0
            stats["overall_cps"] = 0.0

        if tokens:
            stats["median_tokens_total"] = float(np.median(tokens))
            stats["p90_tokens_total"] = float(np.percentile(tokens, 90))
        if query_latencies_ms:
            stats["median_t_total_ms"] = float(np.median(query_latencies_ms))
            stats["p90_t_total_ms"] = float(np.percentile(query_latencies_ms, 90))
            stats["median_latency_ms"] = float(np.median(query_latencies_ms))
            stats["p90_latency_ms"] = float(np.percentile(query_latencies_ms, 90))
        if call_latencies_ms:
            stats["median_call_latency_ms"] = float(np.median(call_latencies_ms))
            stats["p90_call_latency_ms"] = float(np.percentile(call_latencies_ms, 90))
        if call_latencies_ms_traversal:
            stats["median_call_latency_ms_traversal"] = float(
                np.median(call_latencies_ms_traversal)
            )
            stats["p90_call_latency_ms_traversal"] = float(
                np.percentile(call_latencies_ms_traversal, 90)
            )
        if call_latencies_ms_reader:
            stats["median_call_latency_ms_reader"] = float(
                np.median(call_latencies_ms_reader)
            )
            stats["p90_call_latency_ms_reader"] = float(
                np.percentile(call_latencies_ms_reader, 90)
            )
        if tps:
            stats["median_tps_overall"] = float(np.median(tps))
            stats["p90_tps_overall"] = float(np.percentile(tps, 90))
        if query_qps_trav:
            stats["median_query_qps_traversal"] = float(np.median(query_qps_trav))
            stats["p90_query_qps_traversal"] = float(
                np.percentile(query_qps_trav, 90)
            )
        if cps_trav:
            stats["median_cps_traversal"] = float(np.median(cps_trav))
            stats["p90_cps_traversal"] = float(np.percentile(cps_trav, 90))
        if query_qps_read:
            stats["median_query_qps_reader"] = float(np.median(query_qps_read))
            stats["p90_query_qps_reader"] = float(
                np.percentile(query_qps_read, 90)
            )
        if cps_read:
            stats["median_cps_reader"] = float(np.median(cps_read))
            stats["p90_cps_reader"] = float(np.percentile(cps_read, 90))

        if query_qps_overall:
            stats["median_query_qps_overall"] = float(np.median(query_qps_overall))
            stats["p90_query_qps_overall"] = float(
                np.percentile(query_qps_overall, 90)
            )
        if cps_overall:
            stats["median_cps_overall"] = float(np.median(cps_overall))
            stats["p90_cps_overall"] = float(np.percentile(cps_overall, 90))

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    summary.update(stats)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return stats




def append_traversal_percentiles(
    results_path: str | Path, stats_path: str | Path
) -> Dict[str, float]:
    """Append median and p90 traversal metrics to ``final_traversal_stats.json``.

    Parameters
    ----------
    results_path: str or Path
        JSONL file containing per-query traversal results with ``final_metrics``.
    stats_path: str or Path
        Path to ``final_traversal_stats.json`` to update.

    Returns
    -------
    Dict[str, float]
        The computed statistics added to the stats file.
    """

    results_path = Path(results_path)
    stats_path = Path(stats_path)

    if not results_path.exists():
        return {}

    f1s: list[float] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fm = obj.get("final_metrics", {})
            f1 = fm.get("f1")
            if f1 is not None:
                f1s.append(float(f1))

    stats: Dict[str, float] = {}
    if f1s:
        stats["median_final_f1"] = float(np.median(f1s))
        stats["p90_final_f1"] = float(np.percentile(f1s, 90))

    token_usage_path = stats_path.parent / "token_usage.json"
    if token_usage_path.exists():
        try:
            with open(token_usage_path, "r", encoding="utf-8") as f:
                usage = json.load(f)
        except json.JSONDecodeError:
            usage = {}

        per_trav = usage.get("per_query_traversal", {}) or {}

        trav_tokens: list[float] = []
        query_latencies_ms: list[float] = []
        call_latencies_ms: list[float] = []
        tps: list[float] = []
        query_qps: list[float] = []
        cps: list[float] = []
        total_t_ms = 0.0
        total_n_calls = 0.0

        for q in per_trav.values():
            tok = float(q.get("trav_tokens_total", 0))
            t_ms = float(q.get("t_traversal_ms", 0))
            trav_tokens.append(tok)
            query_latencies_ms.append(t_ms)
            tps.append(tok / (t_ms / 1000) if t_ms else 0.0)

            n_calls = float(q.get("n_traversal_calls", 0))
            query_qps.append(1 / (t_ms / 1000) if t_ms else 0.0)
            cps.append(n_calls / (t_ms / 1000) if t_ms else 0.0)
            total_t_ms += t_ms
            total_n_calls += n_calls
            if "call_latency_ms" in q:
                latency = float(q.get("call_latency_ms", 0))
            else:
                latency = t_ms / max(n_calls, 1)
            call_latencies_ms.append(latency)

        num_queries = len(per_trav)
        if total_t_ms:
            stats["overall_qps"] = num_queries / (total_t_ms / 1000)
            stats["overall_cps"] = total_n_calls / (total_t_ms / 1000)
        else:
            stats["overall_qps"] = 0.0
            stats["overall_cps"] = 0.0

        if trav_tokens:
            stats["median_trav_tokens_total"] = float(np.median(trav_tokens))
            stats["p90_trav_tokens_total"] = float(np.percentile(trav_tokens, 90))
        if query_latencies_ms:
            stats["median_t_traversal_ms"] = float(np.median(query_latencies_ms))
            stats["p90_t_traversal_ms"] = float(np.percentile(query_latencies_ms, 90))
            stats["median_latency_ms"] = float(np.median(query_latencies_ms))
            stats["p90_latency_ms"] = float(np.percentile(query_latencies_ms, 90))
        if tps:
            stats["median_tps_overall"] = float(np.median(tps))
            stats["p90_tps_overall"] = float(np.percentile(tps, 90))
        if query_qps:
            stats["median_query_qps_traversal"] = float(np.median(query_qps))
            stats["p90_query_qps_traversal"] = float(np.percentile(query_qps, 90))
        if cps:
            stats["median_cps_traversal"] = float(np.median(cps))
            stats["p90_cps_traversal"] = float(np.percentile(cps, 90))
        if call_latencies_ms:
            stats["median_call_latency_ms"] = float(np.median(call_latencies_ms))
            stats["p90_call_latency_ms"] = float(np.percentile(call_latencies_ms, 90))

    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    summary.update(stats)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return stats