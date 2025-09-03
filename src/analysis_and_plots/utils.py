"""Helper utilities shared across plotting scripts.

Result-directory discovery is handled by :func:`src.utils.get_result_dirs`.
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Iterable, List, Dict, Any

import logging
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PLOT_DIR = Path("analysis/plots")

def load_json(path: Path) -> dict:
    """Load a JSON file and return its contents."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSON Lines file into a list of objects."""
    data: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def stylized_subplots(*args, **kwargs):
    """Return ``plt.subplots`` with a consistent style applied."""
    plt.style.use("ggplot")
    return plt.subplots(*args, **kwargs)

def ensure_output_path(path: Path) -> Path:
    """Create parent directories for ``path`` and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_result_dirs(
    base: str | Path = "data/results",
    *,
    required: str | Iterable[str] | None = None,
) -> List[Path]:
    """Return variant-level result directories under ``base``.

    Directories follow the layout
    ``{base}/{model}/{dataset}/{split}/{variant}``. When ``required`` is
    provided, only directories containing all specified file name(s) are
    returned.

    Parameters
    ----------
    base:
        Root directory containing result subdirectories.
    required:
        Filename or iterable of filenames that must exist within a directory for
        it to be included.

    Returns
    -------
    List[Path]
        Sorted list of matching directories.
    """

    base_path = Path(base)
    if isinstance(required, (str, Path)):
        required_files = [Path(required)]
    elif required is None:
        required_files = []
    else:
        required_files = [Path(r) for r in required]

    result_dirs: List[Path] = []
    if not base_path.exists():
        return result_dirs

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for split_dir in dataset_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                for variant_dir in split_dir.iterdir():
                    if not variant_dir.is_dir():
                        continue
                    if required_files and not all(
                        (variant_dir / rf).exists() for rf in required_files
                    ):
                        continue
                    result_dirs.append(variant_dir)

    return sorted(result_dirs)

def load_token_usage(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load and normalize token usage metrics.

    Parameters
    ----------
    path:
        Directory containing ``token_usage.json`` or the file itself.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A mapping with ``global`` totals and ``per_query`` metrics. Missing
        totals such as ``tokens_total`` and ``t_total_ms`` are derived when
        possible, and per-query metrics are merged from traversal and reader
        entries with aggregate statistics added (``tokens_total``,
        ``t_total_ms``, ``tps_overall``).
    """
    fp = Path(path)
    if fp.is_dir():
        fp = fp / "token_usage.json"
    if not fp.exists():
        logger.warning("token_usage.json not found at %s", fp)
        return {"global": {}, "per_query": {}}
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("Failed to load %s: %s", fp, err)
        return {"global": {}, "per_query": {}}

    if isinstance(data, dict) and set(data.keys()) == {"global"}:
        data = data["global"]

    global_metrics: Dict[str, Any] = {
        k: v for k, v in data.items() if not k.startswith("per_query")
    }

    trav_tokens = float(global_metrics.get("trav_tokens_total", 0) or 0)
    reader_tokens = float(global_metrics.get("reader_total_tokens", 0) or 0)
    tokens_total = global_metrics.get("tokens_total")
    if tokens_total is None:
        tokens_total = trav_tokens + reader_tokens
    global_metrics["trav_tokens_total"] = trav_tokens
    global_metrics["reader_total_tokens"] = reader_tokens
    global_metrics["tokens_total"] = tokens_total

    t_trav_ms = float(global_metrics.get("t_traversal_ms", 0) or 0)
    t_reader_ms = float(global_metrics.get("t_reader_ms", 0) or 0)
    t_total_ms = global_metrics.get("t_total_ms")
    if t_total_ms is None:
        t_total_ms = t_trav_ms + t_reader_ms
    global_metrics["t_traversal_ms"] = t_trav_ms
    global_metrics["t_reader_ms"] = t_reader_ms
    global_metrics["t_total_ms"] = t_total_ms

    tps_overall = global_metrics.get("tps_overall")
    if tps_overall is None:
        tps_overall = tokens_total / (t_total_ms / 1000) if t_total_ms else 0.0
    global_metrics["tps_overall"] = tps_overall

    per_trav = data.get("per_query_traversal") or {}
    per_read = data.get("per_query_reader") or {}
    per_query: Dict[str, Dict[str, Any]] = {}
    for qid in set(per_trav) | set(per_read):
        q_trav = per_trav.get(qid, {})
        q_read = per_read.get(qid, {})
        merged: Dict[str, Any] = {}
        merged.update(q_trav)
        merged.update(q_read)

        q_trav_tokens = float(q_trav.get("trav_tokens_total", 0) or 0)
        q_reader_tokens = float(q_read.get("reader_total_tokens", 0) or 0)
        q_tokens_total = merged.get("tokens_total")
        if q_tokens_total is None:
            q_tokens_total = q_trav_tokens + q_reader_tokens
        merged["trav_tokens_total"] = q_trav_tokens
        merged["reader_total_tokens"] = q_reader_tokens
        merged["tokens_total"] = q_tokens_total

        q_t_trav_ms = float(q_trav.get("t_traversal_ms", 0) or 0)
        q_t_reader_ms = float(q_read.get("t_reader_ms", 0) or 0)
        q_t_total_ms = merged.get("t_total_ms")
        if q_t_total_ms is None:
            q_t_total_ms = q_t_trav_ms + q_t_reader_ms
        merged["t_traversal_ms"] = q_t_trav_ms
        merged["t_reader_ms"] = q_t_reader_ms
        merged["t_total_ms"] = q_t_total_ms

        q_tps = merged.get("tps_overall")
        if q_tps is None:
            q_tps = q_tokens_total / (q_t_total_ms / 1000) if q_t_total_ms else 0.0
        merged["tps_overall"] = q_tps

        per_query[qid] = merged

    return {"global": global_metrics, "per_query": per_query}


__all__ = [
    "DEFAULT_PLOT_DIR",
    "load_json",
    "load_jsonl",
    "stylized_subplots",
    "ensure_output_path",
    "get_result_dirs",
]