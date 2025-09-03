
"""Common utility functions shared across pipeline stages."""

from __future__ import annotations

import time
import json
import os
import re
import unicodedata
from typing import Any, Callable, Dict, Hashable, Iterable, Iterator, List, Optional, Set, Tuple
from collections import defaultdict
import warnings

import numpy as np

from multiprocessing import Process, Pool 


from pathlib import Path






SERVER_CONFIGS = [
    # --- Graphing & Reading: Meta-Llama-3.1-8B-Instruct (2) ---
    {"server_url": "http://localhost:8000", "model": "llama-3.1-8b-instruct"},
    {"server_url": "http://localhost:8001", "model": "llama-3.1-8b-instruct"},

    # --- Traversal: Qwen2.5-7B-Instruct (2) ---
    {"server_url": "http://localhost:8002", "model": "qwen2.5-7b-instruct"},
    {"server_url": "http://localhost:8003", "model": "qwen2.5-7b-instruct"},

    # --- Traversal: Qwen2.5-14B-Instruct (1) ---
    {"server_url": "http://localhost:8004", "model": "qwen2.5-14b-instruct"},

    # --- Traversal: DeepSeek-R1-Distill-Qwen-7B (2) ---
    {"server_url": "http://localhost:8005", "model": "deepseek-r1-distill-qwen-7b"},
    {"server_url": "http://localhost:8006", "model": "deepseek-r1-distill-qwen-7b"},

    # --- Traversal: DeepSeek-R1-Distill-Qwen-14B (1) ---
    {"server_url": "http://localhost:8007", "model": "deepseek-r1-distill-qwen-14b"},

    # --- Traversal: Qwen2.5-MOE-19B (1) ---
    {"server_url": "http://localhost:8008", "model": "qwen2.5-moe-14b"},

    # --- Traversal: State-of-the-MoE-RP-2x7B (1) ---
    {"server_url": "http://localhost:8052", "model": "state-of-the-moe-rp-2x7b"},

    # --- Traversal: Qwen2.5-2x7B-Power-Coder-V4 (1) ---
    {"server_url": "http://localhost:8051", "model": "qwen2.5-2x7b-power-coder-v4"},
]


def get_server_configs(model: str) -> List[Dict[str, str]]:
    """Return all server configuration dicts for ``model``.

    Parameters
    ----------
    model:
        Model name (e.g. ``"qwen-7b"``).

    Returns
    -------
    List[Dict[str, str]]
        All matching entries from :data:`SERVER_CONFIGS`.

    Raises
    ------
    ValueError
        If ``model`` is unknown.
    """

    configs = [cfg for cfg in SERVER_CONFIGS if cfg["model"] == model]
    if not configs:
        raise ValueError(f"Unknown model: {model}")
    return configs


def get_server_urls(model: str) -> List[str]:
    """Return just the server URLs for ``model``."""

    return [cfg["server_url"] for cfg in get_server_configs(model)]


# def get_server_urls(model):
#     """
#     Return all server URLs for a given model (e.g., 4 for 1.5B, 2 for 7B).


#     goes in the __main__ loop to activate all relevant servers 

#     """
#     urls = [config["server_url"] for config in SERVER_CONFIGS if config["model"] == model]
#     if not urls:
#         raise ValueError(f"Unknown model: {model}")
#     return urls



def existing_ids(path, id_field="passage_id", required_field: str | None = None):
    """Return IDs from ``path`` when available.

    Parameters
    ----------
    path:
        JSONL file to scan.
    id_field:
        Name of the identifier field whose values should be collected.
    required_field:
        Optional field that must also be present in a line for it to
        contribute an ID. This is useful when resuming an embedding step:
        rows that have been written but lack the embedding field (e.g.
        ``vec_id``) will then be ignored, keeping them eligible for
        processing.
    """
    if not Path(path).exists():
        return set()
    done = set()
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get(id_field)
                if pid is not None and (
                    required_field is None or obj.get(required_field) is not None
                ):
                    done.add(pid)
            except Exception:
                # tolerate a possibly truncated last line ################################################ ?
                continue
    return done


def validate_vec_ids(
    metadata: List[Dict], emb: np.ndarray, id_field: str = "passage_id"
) -> None:
    """Raise ``ValueError`` if any ``vec_id`` is missing or out of bounds."""

    n = len(emb)
    for item in metadata:
        vec_id = item.get("vec_id")
        if vec_id is None or not (0 <= vec_id < n):
            raise ValueError(
                f"{id_field} {item.get(id_field)} has invalid vec_id {vec_id}"
            )


def compute_resume_sets(
    *,
    resume: bool,
    out_path: str,
    items: Iterable[Any],
    get_id: Callable[[Any, int], Hashable],
    phase_label: str,
    id_field: str = "passage_id",
    required_field: str | None = None,
) -> Tuple[Set[Hashable], Set[Hashable]]:
    
    """Return ``(done_ids, shard_ids)`` for a single shard.

    When ``resume`` is ``True``, :func:`existing_ids` reads ``out_path`` and the
    function prints a message describing how many items are skipped for *this*
    shard. Pipelines that split work across multiple shards should call this
    function separately for each shard's output file – resumption is per shard
    only. The ``items`` iterable is fully consumed to build ``shard_ids``; pass a
    list or other re-iterable sequence if it will be reused later.

    Parameters
    ----------
    resume:
        Whether to check ``out_path`` and report existing IDs.
    out_path:
        JSONL file produced by the current shard.
    items:
        Input sequence for the shard.
    get_id:
        Callable extracting an identifier from ``items`` with signature
        ``(item, index) -> Hashable``.
    phase_label:
        Human-readable label used in log messages.
    id_field:
        Name of the identifier field inside ``out_path`` JSON objects.
    required_field:
        Optional field that must exist in a JSON object for the corresponding
        ID to be considered "done". This allows partially written records to
        be retried on resume.

    Returns
    -------
    Tuple[Set[Hashable], Set[Hashable]]
        ``done_ids``: IDs already present in ``out_path`` for this shard.
        ``shard_ids``: IDs for all items in the shard.
    """
    shard_ids = {get_id(x, i) for i, x in enumerate(items)}
    if not resume:
        return set(), shard_ids

    done_all = existing_ids(
        out_path, id_field=id_field, required_field=required_field
    )  # only this shard's file; caller handles other shards
    done_ids = done_all & shard_ids  # defensive intersection
    print(
        f"[resume] {phase_label}: {len(done_ids)}/{len(shard_ids)} already present in this shard – skipping those"
    )
    return done_ids, shard_ids













# ---------------------------------------------------------------------------
# JSONL and general file I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str, log_skipped: bool = False) -> Iterator[Dict]:
    """Yield objects from a JSONL file one by one.

    Lines that are empty or fail to parse as JSON are skipped. If
    ``log_skipped`` is ``True``, the number of skipped lines is printed.
    """
    skipped = 0
    with open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if log_skipped:
                    print(f"Skipping malformed JSON on line {line_no} in {path}")
    if log_skipped and skipped:
        print(f"Skipped {skipped} empty or malformed lines in {path}")

def save_jsonl(path: str, data: List[Dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "wt", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: str, obj: Dict) -> None:
    """Append a single JSON serialisable object to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _next_version_path(path: str) -> str:
    """If ``path`` exists, return ``path`` with an incremented ``.vN`` suffix."""
    base, ext = os.path.splitext(path)
    i = 1
    candidate = f"{base}.v{i}{ext}"
    while os.path.exists(candidate):
        i += 1
        candidate = f"{base}.v{i}{ext}"
    return candidate

def save_jsonl_safely(path: str, data: List[Dict], overwrite: bool = False) -> str:
    """Write JSONL data, versioning the filename if it already exists."""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path or ".", exist_ok=True)
    out_path = path
    if os.path.exists(path) and not overwrite:
        out_path = _next_version_path(path)
    with open(out_path, "wt", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out_path

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise whitespace and remove simple markup for clean text."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"={2,}.*?={2,}", "", text)
    text = unicodedata.normalize("NFKC", text)
    return text

# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------



def pid_plus_title(qid: str, title: str, sent_idx: int) -> str:
    """Create a safe passage identifier using question id and title.


    only used with 2wikimulihop and hotpotqa - musique already has a unique identifier for each passage

    The title is normalised by converting to lowercase and replacing any
    non-alphanumeric characters with underscores.  If the provided title is
    empty or sanitisation results in an empty string, ``"no_title"`` is used
    instead.

    Parameters
    ----------
    qid:
        The base identifier, typically the question or passage id.
    title:
        Title text associated with the passage.
    sent_idx:
        Sentence index within the passage.

    Returns
    -------
    str
        A combined identifier ``"{qid}__{safe}_sent{sent_idx}"``.
    """
    if not title:
        safe = "no_title"
    else:
        # Replace any non-word characters with underscores and collapse
        # repeated underscores.  ``\w`` matches alphanumerics and ``_``.
        safe = re.sub(r"[^0-9A-Za-z]+", "_", title.lower()).strip("_")
        if not safe:
            safe = "no_title"
    return f"{qid}__{safe}_sent{sent_idx}"


# Maps public variant names to potential directory names on disk.  The first
# entry for each variant is considered the canonical folder name.
FOLDERS_BY_VARIANT: Dict[str, List[str]] = {
    "baseline": ["baseline_hoprag", "baseline"],
    "enhanced": ["enhanced_hoprag", "enhanced"],
}

def resolve_root(model: str, dataset: str, split: str, variant: str) -> Optional[str]:
    """
    Searches under ``data/models/{model}/{dataset}/{split}`` for the first
    directory listed in ``FOLDERS_BY_VARIANT[variant]`` that exists and returns
    its path. If none are found, ``None`` is returned.
    """
    for hoprag_version in FOLDERS_BY_VARIANT[variant]:
        root = f"data/models/{model}/{dataset}/{split}/{hoprag_version}"
        if os.path.isdir(root):
            return root
    return None









def get_result_paths(model, dataset, split, variant):
    base = Path(f"data/results/{model}/{dataset}/{split}/{variant}")
    return {
        "base": base,
        "answers": base / f"answer_per_query_{variant}_{split}.jsonl",
        "answer_metrics": base / f"answer_metrics_{variant}_{split}.jsonl",
        "summary": base / f"summary_metrics_{variant}_{split}.json",
    }

def get_traversal_paths(model, dataset, split, variant):
    """Return standard paths for traversal artifacts.

    Creates ``data/traversal/{model}/{dataset}/{split}/{variant}/`` if
    necessary and returns paths to key traversal output files.
    """

    base = Path(f"data/traversal/{model}/{dataset}/{split}/{variant}")
    base.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "results": base / "per_query_traversal_results.jsonl",
        "visited_passages": base / "visited_passages.json",
        "stats": base / "final_traversal_stats.json",
    }

def processed_dataset_paths(dataset: str, split: str) -> Dict[str, Path]:
    """Return standard paths for processed dataset files.

    Creates ``data/processed_datasets/{dataset}/{split}/`` if necessary and
    returns paths for ``questions.jsonl`` and ``passages.jsonl``.
    """
    base = Path(f"data/processed_datasets/{dataset}/{split}")
    base.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "questions": base / "questions.jsonl",
        "passages": base / "passages.jsonl",
    }








def model_size(model: str) -> str:
    """Return a normalized size string for ``model``.

    Examples
    --------
    ``qwen2.5-7b-instruct`` → ``7b``
    ``deepseek-r1-distill-qwen-14b`` → ``14b``
    ``qwen2.5-moe-14b`` → ``14b``

    This is used purely for file naming and shard selection.
    """
    m = re.search(r'(\d+(?:\.\d+)?)b', model, re.I)
    if not m:
        raise ValueError(f"Cannot infer size from model name: {model}")
    return m.group(1).lower() + "b"















def split_jsonl(path: str, out1: str, out2: str):
    data = list(load_jsonl(path))  # Convert generator to list
    half = len(data) // 2
    save_jsonl(out1, data[:half])
    save_jsonl(out2, data[half:])



def split_jsonl_into_four(path, out1, out2, out3, out4):
    data = list(load_jsonl(path))
    total_rows = len(data)

    if total_rows == 0:
        save_jsonl(out1, []); save_jsonl(out2, []); save_jsonl(out3, []); save_jsonl(out4, [])
        return

    # Base size per chunk + how many leftovers to spread
    base_size, leftovers = divmod(total_rows, 4)

    # First `leftovers` chunks get one extra row
    chunk_sizes = [(base_size + 1 if i < leftovers else base_size) for i in range(4)]

    chunks, start = [], 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(data[start:end])
        start = end

    save_jsonl(out1, chunks[0]); save_jsonl(out2, chunks[1])
    save_jsonl(out3, chunks[2]); save_jsonl(out4, chunks[3])





def split_jsonl_into_four(path: str, out1: str, out2: str, out3: str, out4: str) -> None:
    """Split ``path`` into four roughly equal JSONL shards."""
    data = list(load_jsonl(path))
    total_rows = len(data)
    if total_rows == 0:
        save_jsonl(out1, []); save_jsonl(out2, []); save_jsonl(out3, []); save_jsonl(out4, [])
        return
    base_size, leftovers = divmod(total_rows, 4)
    chunk_sizes = [(base_size + 1 if i < leftovers else base_size) for i in range(4)]
    chunks, start = [], 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(data[start:end])
        start = end
    save_jsonl(out1, chunks[0]); save_jsonl(out2, chunks[1])
    save_jsonl(out3, chunks[2]); save_jsonl(out4, chunks[3])




def model_shard_dir(model: str, dataset: str, split: str) -> Path:
    """Return directory for model-specific shards.

    The directory follows ``data/models/{model}/{dataset}/{split}/shards`` and
    is created if it does not already exist.
    """
    base = Path(f"data/models/{model}/{dataset}/{split}/shards")
    base.mkdir(parents=True, exist_ok=True)
    return base



def model_shard_paths(model: str, dataset: str, split: str, stem: str, size: str) -> List[Path]:
    """Return shard file paths for a given model size.

    Parameters
    ----------
    model, dataset, split:
        Identify the model dataset split.
    stem:
        Base filename (typically input JSONL stem).
    size:
        Model size string (``"1.5b"``, ``"7b"``, ``"8b"``, ``"14b"``).
    """
    out_dir = model_shard_dir(model, dataset, split)
    counts = {"1.5b": 4, "7b": 2, "8b": 2, "14b": 1}
    n_shards = counts.get(size)
    if n_shards is None:
        raise ValueError(f"Unsupported model size: {size}")
    return [out_dir / f"{stem}_shard{i}_{size}.jsonl" for i in range(1, n_shards + 1)]



def split_jsonl_for_models(path: str, model: str, *, resume: bool = False) -> List[str]:
    """Split ``path`` into shards appropriate for ``model``.

    Supported model sizes and shard counts:
    ``1.5b`` → 4 shards, ``7b``/``8b`` → 2 shards, ``14b`` → 1 shard.

    Parameters
    ----------
    path:
        Input JSONL file to shard.
    model:
        Model name whose size determines the number of shards.
    resume:
        When ``True``, existing shard files are reused and splitting is skipped
        if all expected shards are already present.

    Returns
    -------
    List[str]
        Paths to the shard files produced (or reused).
    """

    size = model_size(model)
    p = Path(path)
    dataset = p.parent.parent.name
    split = p.parent.name
    stem = p.stem
    out_paths = model_shard_paths(model, dataset, split, stem, size)

    if resume and all(op.exists() for op in out_paths):
        return [str(op) for op in out_paths]

    if size == "1.5b":
        split_jsonl_into_four(path, *(str(op) for op in out_paths))
    elif size in {"7b", "8b"}:
        split_jsonl(path, str(out_paths[0]), str(out_paths[1]))
    elif size == "14b":
        data = list(load_jsonl(path))
        save_jsonl(str(out_paths[0]), data)
    else:
        raise ValueError(f"Unsupported model size: {size}")

    return [str(op) for op in out_paths]





def run_multiprocess(
        func: Callable, 
        configs: List[Dict]):
    """
    Run a given function in parallel across multiple processes.
    
    Parameters:
        func (Callable): The function to run in each process.
        configs (List[Dict]): A list of dicts containing keyword arguments for each call.
    """
    processes = []
    for cfg in configs:
        p = Process(target=func, args=(cfg,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def pool_map(
    func: Callable,
    items: Iterable,
    processes: Optional[int] = None,
):
    """Map ``func`` across ``items`` using a :class:`multiprocessing.Pool`.

    This helper mirrors :func:`run_multiprocess` to provide a consistent
    multiprocessing interface across the codebase.

    Parameters
    ----------
    func:
        Callable executed for each item.
    items:
        Iterable of work items passed to ``func``.
    processes:
        Number of worker processes. Defaults to ``None`` which lets
        ``multiprocessing.Pool`` decide.

    Returns
    -------
    List
        Results returned by ``func``.
    """
    with Pool(processes) as pool:
        return pool.map(func, items)


def compute_recall_at_k(
    pred_passages: List[str], gold_passages: List[str], k: int
) -> float:
    """Return recall of top-``k`` predictions against gold passages.

    Parameters
    ----------
    pred_passages:
        Retrieved passages ordered by relevance.
    gold_passages:
        Gold passage identifiers.
    k:
        Evaluation cutoff.

    Returns
    -------
    float
        Fraction of ``gold_passages`` found among the first ``k`` predictions, or
        ``0.0`` when ``gold_passages`` is empty.
    """

    if k <= 0 or not gold_passages:
        return 0.0

    gold_set = set(gold_passages)
    return len(set(pred_passages[:k]) & gold_set) / len(gold_passages)


def compute_hits_at_k(pred_passages: List[str], gold_passages: List[str], k: int) -> float:
    """Return whether any of the top-``k`` predicted passages match a gold passage.

    Parameters
    ----------
    pred_passages:
        Retrieved passages ordered by relevance.
    gold_passages:
        Gold passage identifiers.
    k:
        Evaluation cutoff.

    Returns
    -------
    float
        ``1.0`` if any gold passage is found in the first ``k`` predictions,
        otherwise ``0.0``.
    """

    if k <= 0:
        return 0.0

    gold_set = set(gold_passages)
    return float(any(pid in gold_set for pid in pred_passages[:k]))




def log_wall_time(
    script_name: str,
    start_time: float,
    log_file: str | Path | None = None,
) -> float:
    """Append elapsed wall time for ``script_name`` to ``wall_time.log``.

    Parameters
    ----------
    script_name:
        Name or path of the running script (e.g. ``__file__``).
    start_time:
        Timestamp captured at the start of the script via :func:`time.time`.
    log_file:
        Optional path to the log file. Defaults to ``wall_time.log`` in the
        repository root.

    Returns
    -------
    float
        The elapsed wall time in seconds.
    """

    elapsed = time.time() - start_time
    log_path = (
        Path(log_file)
        if log_file is not None
        else Path(__file__).resolve().parent.parent / "wall_time.log"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{script_name}\t{elapsed:.2f}\n")
    return elapsed


def aggregate_wall_times(log_file: str | Path | None = None) -> Dict[str, float]:
    """Aggregate total wall times per script from ``log_file``.

    Parameters
    ----------
    log_file:
        Optional path to the log file. Defaults to ``wall_time.log`` in the
        repository root.

    Returns
    -------
    Dict[str, float]
        Mapping of script names to total elapsed seconds across runs.
    """

    log_path = (
        Path(log_file)
        if log_file is not None
        else Path(__file__).resolve().parent.parent / "wall_time.log"
    )
    totals: Dict[str, float] = {}
    if not log_path.exists():
        return totals
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            name, sec = parts
            try:
                totals[name] = totals.get(name, 0.0) + float(sec)
            except ValueError:
                continue
    return totals




def _merge_numeric(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Merge numeric values from ``src`` into ``dst``."""
    for k, v in src.items():
        if isinstance(v, (int, float)):
            dst[k] = dst.get(k, 0) + v
        else:
            dst[k] = v
    return dst


def merge_token_usage(
    output_dir: str | Path,
    *,
    run_id: str | None = None,
    cleanup: bool = False,
) -> Path:
    """Merge ``token_usage`` shards in ``output_dir`` into one.

    The function aggregates global token counts and per-query metrics across
    multiple partial usage files. If no usage files are found, an empty
    ``token_usage.json`` file is created. The merged result is written to
    ``token_usage.json`` inside ``output_dir``.
    Parameters
    ----------
    output_dir:
        Directory containing ``token_usage_*.json`` shard files.
    run_id:
        Optional identifier used in shard filenames. When provided, only files
        matching ``token_usage_{run_id}_*.json`` are merged. Any non-matching
        shards are ignored and, if ``cleanup`` is ``True``, removed.
    cleanup:
        If ``True``, the individual shard files are removed after the merged
        ``token_usage.json`` is written. Defaults to ``False`` to preserve
        backward compatibility with callers expecting shards to remain.
    """

    out_dir = Path(output_dir)
    if run_id is not None:
        stale = [fp for fp in out_dir.glob("token_usage_*.json") if run_id not in fp.name]
        for fp in stale:
            try:
                fp.unlink()
            except OSError:
                pass
        pattern = f"token_usage_{run_id}_*.json"
    else:
        pattern = "token_usage_*.json"
    usage_files = sorted(out_dir.glob(pattern))
    if not usage_files:
        out_path = out_dir / "token_usage.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return out_path

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
                existing = global_totals.get(k)
                if existing is None:
                    global_totals[k] = v
                elif existing != v:
                    warnings.warn(
                        f"Conflicting values for '{k}': keeping {existing!r}, ignoring {v!r}",
                        stacklevel=1,
                    )

    tokens_total = (
        global_totals.get("trav_tokens_total", 0)
        + global_totals.get("reader_total_tokens", 0)
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

    if cleanup:
        for fp in usage_files:
            try:
                fp.unlink()
            except OSError:
                pass

    return out_path


__all__ = [
    "SERVER_CONFIGS",
    "get_server_configs",
    "get_server_urls",
    "load_jsonl",
    "save_jsonl",
    "append_jsonl",
    "save_jsonl_safely",
    "clean_text",
    "pid_plus_title",
    "FOLDERS_BY_VARIANT",
    "resolve_root",
    "get_result_paths",
    "get_traversal_paths",
    "processed_dataset_paths",
    "model_shard_dir",
    "model_shard_paths",
    "run_multiprocess",
    "pool_map",
    "model_size",
    "split_jsonl",
    "split_jsonl_into_four",
    "split_jsonl_for_models",
    "compute_recall_at_k",

    "compute_hits_at_k",
    "log_wall_time",
    "aggregate_wall_times",
    "merge_token_usage",

]