from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any


def _merge_numeric(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Merge numeric values from ``src`` into ``dst``."""
    for k, v in src.items():
        if isinstance(v, (int, float)):
            dst[k] = dst.get(k, 0) + v
        else:
            dst[k] = v
    return dst


def merge_token_usage(output_dir: str | Path) -> Path:
    """Merge ``token_usage_*.json`` files in ``output_dir`` into one.

    The function aggregates global token counts and per-query metrics across
    multiple partial usage files. The merged result is written to
    ``token_usage.json`` inside ``output_dir``.
    """

    out_dir = Path(output_dir)
    usage_files = sorted(out_dir.glob("token_usage_*.json"))
    if not usage_files:
        return out_dir / "token_usage.json"

    per_query_trav: Dict[str, Dict[str, Any]] = {}
    per_query_reader: Dict[str, Dict[str, Any]] = {}
    global_totals: Dict[str, Any] = defaultdict(float)

    for fp in usage_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if pq := data.get("per_query_traversal"):
            for qid, metrics in pq.items():
                per_query_trav[qid] = _merge_numeric(per_query_trav.get(qid, {}), metrics)
        if pq := data.get("per_query_reader"):
            for qid, metrics in pq.items():
                per_query_reader[qid] = _merge_numeric(per_query_reader.get(qid, {}), metrics)

        for k, v in data.items():
            if k.startswith("per_query") or k in {"tokens_total", "t_total_ms", "tps_overall"}:
                continue
            if isinstance(v, (int, float)):
                global_totals[k] += v
            else:
                global_totals[k] = v

    tokens_total = (
        global_totals.get("trav_tokens_total", 0)
        + global_totals.get("trav_total_tokens", 0)
        + global_totals.get("reader_total_tokens", 0)
        + global_totals.get("reader_tokens_total", 0)
    )
    t_total_ms = global_totals.get("t_traversal_ms", 0) + global_totals.get("t_reader_ms", 0)

    merged: Dict[str, Any] = {}
    if per_query_trav:
        merged["per_query_traversal"] = per_query_trav
    if per_query_reader:
        merged["per_query_reader"] = per_query_reader
    merged.update(global_totals)
    merged["tokens_total"] = tokens_total
    merged["t_total_ms"] = t_total_ms
    merged["tps_overall"] = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0

    out_path = out_dir / "token_usage.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return out_path