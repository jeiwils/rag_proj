
"""Common utility functions shared across pipeline stages."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Callable, Dict, Hashable, Iterable, Iterator, List, Optional, Set, Tuple


from multiprocessing import Process, Pool 


from pathlib import Path






SERVER_CONFIGS = [
    # --- Legacy: Qwen 1.5B (4) ---
    {"server_url": "http://localhost:8000", "model": "qwen-1.5b"},
    {"server_url": "http://localhost:8001", "model": "qwen-1.5b"},
    {"server_url": "http://localhost:8002", "model": "qwen-1.5b"},
    {"server_url": "http://localhost:8003", "model": "qwen-1.5b"},

    # --- Legacy: Qwen 7B (2) ---
    {"server_url": "http://localhost:8004", "model": "qwen-7b"},
    {"server_url": "http://localhost:8005", "model": "qwen-7b"},

    # --- Legacy: Qwen 14B (1) ---
    {"server_url": "http://localhost:8006", "model": "qwen-14b"},

    # --- Legacy: DeepSeek-Distill-Qwen 1.5B (4) ---
    {"server_url": "http://localhost:8007", "model": "deepseek-distill-qwen-1.5b"},
    {"server_url": "http://localhost:8008", "model": "deepseek-distill-qwen-1.5b"},
    {"server_url": "http://localhost:8009", "model": "deepseek-distill-qwen-1.5b"},
    {"server_url": "http://localhost:8010", "model": "deepseek-distill-qwen-1.5b"},

    # --- Legacy: DeepSeek-Distill-Qwen 7B (2) ---
    {"server_url": "http://localhost:8011", "model": "deepseek-distill-qwen-7b"},
    {"server_url": "http://localhost:8012", "model": "deepseek-distill-qwen-7b"},

    # --- Legacy: DeepSeek-Distill-Qwen 14B (1) ---
    {"server_url": "http://localhost:8013", "model": "deepseek-distill-qwen-14b"},

    # --- NEW: Graphing — Qwen2.5-7B (2) ---
    {"server_url": "http://localhost:8014", "model": "qwen2.5-7b"},
    {"server_url": "http://localhost:8015", "model": "qwen2.5-7b"},

    # --- NEW: Traversal — DeepSeek-R1-Distill-Qwen-7B (2) ---
    {"server_url": "http://localhost:8016", "model": "deepseek-r1-distill-qwen-7b"},
    {"server_url": "http://localhost:8017", "model": "deepseek-r1-distill-qwen-7b"},

    # --- NEW: Qwen2.5-7B-Instruct (2) ---
    {"server_url": "http://localhost:8018", "model": "qwen2.5-7b-instruct"},
    {"server_url": "http://localhost:8019", "model": "qwen2.5-7b-instruct"},

    # --- NEW: Reader — Meta-Llama-3.1-8B-Instruct (2) ---
    {"server_url": "http://localhost:8020", "model": "llama-3.1-8b-instruct"},
    {"server_url": "http://localhost:8021", "model": "llama-3.1-8b-instruct"},
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
    base = Path(f"results/{model}/{dataset}/{split}/{variant}")
    return {
        "base": base,
        "answers": base / f"answer_per_query_{variant}_{split}.jsonl",
        "summary": base / f"summary_metrics_{variant}_{split}.json",
    }

def get_traversal_paths(model, dataset, split, variant):
    base = Path(f"data/graphs/{model}/{dataset}/{split}/{variant}/traversal")
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
    """
    Normalizes 'qwen-1.5b' -> '1.5b', 'qwen-7b' -> '7b', 'deepseek-distill-qwen-14b' -> '14b'
        
    just a helper function for file naming         
                
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
        Model size string (``"1.5b"``, ``"7b"`` or ``"14b"``).
    """
    out_dir = model_shard_dir(model, dataset, split)
    counts = {"1.5b": 4, "7b": 2, "14b": 1}
    n_shards = counts.get(size)
    if n_shards is None:
        raise ValueError(f"Unsupported model size: {size}")
    return [out_dir / f"{stem}_shard{i}_{size}.jsonl" for i in range(1, n_shards + 1)]



def split_jsonl_for_models(path: str, model: str, *, resume: bool = False) -> List[str]:
    """Split ``path`` into shards appropriate for ``model``.

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
    elif size == "7b":
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
]