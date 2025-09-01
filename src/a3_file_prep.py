"""
Module Overview
---------------
Clean, merge, and explode question-generation (IQ/OQ) output shards from generative models.

This module processes IQ/OQ and optional conditioned-score outputs for each dataset, model, and variant
(e.g., baseline, enhanced). It prepares standardized files for downstream graph construction, traversal,
and evaluation in HopRAG-style QA pipelines.


Workflow
--------

For each (dataset, model, variant, split):

1. Clean IQ/OQ shards using variant-specific filters.
2. Merge cleaned IQ/OQ and score files into consolidated JSONLs.
3. Explode the merged files into:
   - one row per IQ/OQ (for graph edges)
   - one row per passage (for node representations)


Input Structure
---------------

Sharded files are located at:

### data/models/{model}/{dataset}/{split}/shards/

- {split}_passages_shard{N}_{size}.jsonl
    → Raw input shards, split by model size:
        - 1.5B → 4 shards
        - 7B   → 2 shards
        - 14B  → 1 shard

### data/models/{model}/{dataset}/{split}/shards/{hoprag_version}/

- {split}_passages_shard{N}_{size}_iqoq_baseline.jsonl
    → IQ/OQ generation (baseline variant, fixed IQ/OQ ratio)

- {split}_passages_shard{N}_{size}_iqoq_enhanced.jsonl
    → IQ/OQ generation (enhanced variant, CS-guided ratio)

- {split}_passages_shard{N}_{size}_cs.jsonl
    → Conditioned-score values for passages

- *_iqoq_baseline_debug.txt / *_enhanced_debug.txt / *_cs_debug.txt
    → Per-shard debug logs


Output Structure
----------------

### data/models/{model}/{dataset}/{split}/{hoprag_version}/cleaned/

- iqoq.cleaned.jsonl
    → Merged and cleaned IQ/OQ entries for all passages

- scored.cleaned.jsonl
    → Merged conditioned score entries (if available)


### data/models/{model}/{dataset}/{split}/{hoprag_version}/exploded/

- iqoq.exploded.jsonl
    → One row per IQ/OQ question with minimal metadata

- passages.exploded.jsonl
    → One row per passage with optional conditioned score


### data/models/{model}/{dataset}/{split}/{hoprag_version}/debug/

- cleaning_debug.txt
    → Aggregated stats, counts, drop rates, timing for cleaning stage


Schemas
-------

### cleaned/iqoq.cleaned.jsonl

{
  "passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "title": "Arthur's Magazine",
  "text": "Arthur's Magazine (1844–1846)...",
  "IQs": ["Who founded Arthur's Magazine?", "..."],
  "OQs": ["Which city was it based in?", "..."],
  "num_iq": 3,
  "num_oq": 3,
  "cs_used": 0.25,                     # only in enhanced variant if scores are used
  "hoprag_version": "enhanced_hoprag",
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "llama-3.1-8b-instruct"
}


### cleaned/scored.cleaned.jsonl

{
  "passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "title": "Arthur's Magazine",
  "text": "Arthur's Magazine (1844–1846)...",
  "conditioned_score": 0.25,
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "llama-3.1-8b-instruct"
}


### exploded/iqoq.exploded.jsonl

{
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "llama-3.1-8b-instruct",
  "parent_passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "iqoq_id": "5a7a0693__arthur_s_magazine_sent0_oq2",
  "type": "OQ",
  "index": 2,
  "text": "Who were notable contributors?"
}


### exploded/passages.exploded.jsonl

{
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "llama-3.1-8b-instruct",
  "passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "text": "Arthur's Magazine (1844–1846)...",
  "conditioned_score": 0.25
}

"""



import json
import os
import re
import time
from pathlib import Path

from src.utils import (
    FOLDERS_BY_VARIANT,
    SERVER_CONFIGS,
    append_jsonl,
    compute_resume_sets,
    existing_ids,
    load_jsonl,
    model_shard_dir,
    resolve_root,
    save_jsonl,
)



### *   *   * ###

# def clean_iqoq(questions: list[str]) -> list[str]: 
#     """
#     Normalize and filter generated IQ/OQ strings.

#     """
#     cleaned = []

#     for q in questions:
#         q = q.strip()
#         q = re.sub(r"^\d+[\.\)]\s*", "", q)      # remove numbering (e.g., '1. ...')
#         q = re.sub(r"^[-*]\s*", "", q)           # remove bullets ('- ', '* ')
#         if not q.endswith("?"):
#             continue                             # skip if it’s not a question - will need to check this during debugging
#         if len(q) < 5:
#             continue                             # skip tiny junk 
#         if q.lower() in {"n/a", "none", "no question generated"}:
#             continue
#         cleaned.append(q)

#     return cleaned


def clean_iqoq(questions: list[str]) -> list[str]:
    """
    Normalize and filter generated IQ/OQ strings.

    """
    cleaned = []
    skipped_not_question = skipped_too_short = skipped_banned = 0

    for q in questions:
        q = q.strip()
        q = re.sub(r"^\d+[\.\)]\s*", "", q)      # remove numbering (e.g., '1. ...')
        q = re.sub(r"^[-*]\s*", "", q)           # remove bullets ('- ', '* ')
        q = q.rstrip(" \"'“”‘’`)]}>,.!")
        if not q.endswith("?"):
            skipped_not_question += 1
            continue                             # skip if it’s not a question - will need to check this during debugging
        if len(re.findall(r"\w+", q)) < 1:
            skipped_too_short += 1               # skip if no word tokens
            continue
        if q.lower() in {"n/a", "none", "no question generated"}:
            skipped_banned += 1
            continue
        cleaned.append(q)

    print(
        f"clean_iqoq filtered - not_question: {skipped_not_question}, "
        f"too_short: {skipped_too_short}, banned: {skipped_banned}"
    )

    # Remove duplicates while preserving order
    return list(dict.fromkeys(cleaned))

# def clean_baseline(questions):
#     """Parse baseline 'json {"Question List":[...]}' strings then clean."""
#     if not questions:
#         return []
#     seq = questions if isinstance(questions, list) else [questions]
#     collected = []
#     for s in seq:
#         s = re.sub(r"^\s*json\s*", "", str(s).strip(), flags=re.I)
#         for b in re.findall(r"\{.*?\}", s, flags=re.S):
#             try:
#                 obj = json.loads(b)
#             except Exception:
#                 continue
#             qlist = obj.get("Question List")
#             if qlist is None:
#                 for alt in ("questions","question_list","qlist"):
#                     if alt in obj:
#                         qlist = obj[alt]; break
#             if isinstance(qlist, list):
#                 collected.extend(map(str, qlist))
#     return clean_iqoq(collected if collected else list(map(str, seq)))


def clean_baseline(questions):
    """Parse baseline 'json {"Question List":[...]}' strings then clean."""
    if not questions:
        return []

    seq = questions if isinstance(questions, list) else [questions]
    collected: list[str] = []

    for s in seq:
        s = re.sub(r'^\s*json\s+', '', str(s), flags=re.I).strip()
        try:
            obj = json.loads(s)
        except Exception as e:
            print(f"clean_baseline parse error: {e}")
            matches = re.findall(r'"((?:[^"\\]|\\.)+)"', s)  # quoted strings incl. escapes
            collected.extend([m.strip() for m in matches if m.strip().endswith('?')])
            continue

        while isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except Exception as e:
                print(f"clean_baseline parse error: {e}")
                matches = re.findall(r'"((?:[^"\\]|\\.)+)"', s)
                collected.extend([m.strip() for m in matches if m.strip().endswith('?')])
                obj = None
                break

        if obj is None:
            continue

        # Normalize the parsed object before extracting questions.
        qlist = None
        if isinstance(obj, dict):
            qlist = obj.get("Question List")
            if qlist is None:
                for alt in ("questions", "question_list", "qlist"):
                    if alt in obj:
                        qlist = obj[alt]
                        break
        elif isinstance(obj, list):
            qlist = obj
        elif isinstance(obj, str):
            qlist = [obj]
        else:
            continue

        if isinstance(qlist, list):
            collected.extend(map(str, qlist))
        elif isinstance(qlist, str):
            collected.append(str(qlist))

    # 👇 normalize then deduplicate
    if collected:
        cleaned = clean_iqoq(collected)
    else:
        cleaned = clean_iqoq(list(map(str, seq)))

    # Deduplicate after normalization
    return list(dict.fromkeys(cleaned))



def clean_file(in_path, out_path, cleaner, resume: bool = False):
    """Clean one shard. Returns a compact summary. No debug files here."""
    raw = list(load_jsonl(in_path)) 
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=out_path,
        items=raw,
        get_id=lambda e, i: e.get("passage_id", f"idx:{i}"),
        phase_label="clean",
    )

    cleaned = []
    total_raw_iq = total_raw_oq = 0
    total_clean_iq = total_clean_oq = 0
    skipped = 0
    malformed_iq_types = malformed_oq_types = 0

    for i, e in enumerate(raw):
        pid = e.get("passage_id", f"idx:{i}")
        if resume and pid in done_ids:
            print(f"[resume] clean skipping {pid}")
            skipped += 1
            continue

        raw_IQs = e.get("IQs")
        if isinstance(raw_IQs, str):
            raw_IQs = [raw_IQs]
        else:
            if raw_IQs is None:
                raw_IQs = []
            else:
                if not isinstance(raw_IQs, (list, tuple, set)):
                    malformed_iq_types += 1
                raw_IQs = list(raw_IQs)

        raw_OQs = e.get("OQs")
        if isinstance(raw_OQs, str):
            raw_OQs = [raw_OQs]
        else:
            if raw_OQs is None:
                raw_OQs = []
            else:
                if not isinstance(raw_OQs, (list, tuple, set)):
                    malformed_oq_types += 1
                raw_OQs = list(raw_OQs)

        total_raw_iq += len(raw_IQs)
        total_raw_oq += len(raw_OQs)

        cIQ = cleaner(raw_IQs)
        cOQ = cleaner(raw_OQs)

        e["IQs"], e["OQs"] = cIQ, cOQ
        e["num_iq"], e["num_oq"] = len(cIQ), len(cOQ)

        total_clean_iq += len(cIQ)
        total_clean_oq += len(cOQ)
        cleaned.append(e)

    if resume:
        for e in cleaned:
            append_jsonl(out_path, e)
    else:
        save_jsonl(out_path, cleaned)

    if malformed_iq_types or malformed_oq_types:
        print(
            f"[clean_file] Malformed IQ types: {malformed_iq_types}, "
            f"Malformed OQ types: {malformed_oq_types}"
        )

    return {
        "shard_path": in_path,
        "out_path": out_path,
        "num_entries": len(cleaned),
        "raw_iq": total_raw_iq,
        "raw_oq": total_raw_oq,
        "clean_iq": total_clean_iq,
        "clean_oq": total_clean_oq,
        "skipped": skipped,
        "malformed_iq": malformed_iq_types,
        "malformed_oq": malformed_oq_types,
    }


def write_cleaning_debug(
    *, model: str, dataset: str, variant: str, split: str,
    hoprag_version: str, start_time: float,
    shard_summaries: list, merged_output_path: str | None,
    debug_dir: str,
):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    # NEW: no split in filename
    out_path = debug_dir / "cleaning_debug.txt"

    duration = time.time() - start_time
    total_shards = len(shard_summaries)
    total_passages = sum(s["num_entries"] for s in shard_summaries)
    raw_iq  = sum(s["raw_iq"]  for s in shard_summaries)
    raw_oq  = sum(s["raw_oq"]  for s in shard_summaries)
    clean_iq= sum(s["clean_iq"]for s in shard_summaries)
    clean_oq= sum(s["clean_oq"]for s in shard_summaries)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Total shards: {total_shards}\n")
        f.write(f"Total passages processed: {total_passages}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Variant: {variant}\n")
        f.write(f"HopRAG version: {hoprag_version}\n")
        f.write("Phase: cleaning\n")
        f.write(f"Total time taken (seconds): {duration:.2f}\n")

        f.write("\nCounts (IQ/OQ):\n")
        f.write(f"- Raw IQ total: {raw_iq}\n")
        f.write(f"- Raw OQ total: {raw_oq}\n")
        f.write(f"- Clean IQ total: {clean_iq}\n")
        f.write(f"- Clean OQ total: {clean_oq}\n")
        f.write(f"- Dropped IQ total: {raw_iq - clean_iq}\n")
        f.write(f"- Dropped OQ total: {raw_oq - clean_oq}\n")

        if merged_output_path:
            f.write("\nMerged cleaned file:\n")
            f.write(f"- {merged_output_path}\n")




### *   *   * ###


def merge_jsonl_files(in_paths, out_path, dedup_key=None, resume: bool = False):
    """Merge multiple JSONL files, optionally deduplicating by ``dedup_key``.

    If ``resume`` is ``True``, existing IDs in ``out_path`` are loaded once and
    compared against the union of IDs from all shard inputs. Only the missing
    records are written, allowing interrupted runs to resume without reprocessing
    previously merged entries.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Collect rows from all shards keyed by their identifier. When duplicates
    # are encountered, merge IQ/OQ lists while preserving question order and
    # removing duplicates.  All entries will have ``num_iq``/``num_oq``
    # recomputed after merging.
    rows_by_id: dict[str, dict] = {}
    for p in in_paths:
        for i, row in enumerate(load_jsonl(p)):
            k = row.get(dedup_key, f"idx:{i}") if dedup_key else f"idx:{i}"
            if k in rows_by_id and dedup_key:
                existing = rows_by_id[k]
                for field, count_field in (("IQs", "num_iq"), ("OQs", "num_oq")):
                    combined = existing.get(field, []) + row.get(field, [])
                    seen = set()
                    deduped = []
                    for q in combined:
                        if q not in seen:
                            deduped.append(q)
                            seen.add(q)
                    existing[field] = deduped
                    existing[count_field] = len(deduped)
            else:
                rows_by_id[k] = row

    # Ensure counts are accurate for entries that never encountered duplicates
    for row in rows_by_id.values():
        if "IQs" in row:
            row["num_iq"] = len(row.get("IQs", []))
        if "OQs" in row:
            row["num_oq"] = len(row.get("OQs", []))

    shard_ids = set(rows_by_id.keys())
    existing = existing_ids(out_path, id_field=dedup_key) if resume else set()
    pending_ids = shard_ids - existing if resume else shard_ids

    mode = "a" if resume else "w"
    with open(out_path, mode + "t", encoding="utf-8") as f:
        for k, row in rows_by_id.items():
            if k in pending_ids:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")



### *   *   * ###



def explode_passages(
    master_path: str, output_path: str, resume: bool = False
) -> None:
    """Explode merged IQ/OQ file into one row per passage.

    When ``resume`` is ``True``, existing ``passage_id`` values in
    ``output_path`` are skipped so the operation can safely restart.
    """

    data = list(load_jsonl(master_path))
    items = [e["passage_id"] for e in data]
    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=output_path,
        items=items,
        get_id=lambda x, i: x,
        phase_label="explode_passages",
        id_field="passage_id",
    )

    out = []
    for e in data:
        pid = e["passage_id"]
        if resume and pid in done_ids:
            print(f"[resume] explode_passages skipping {pid}")
            continue
        out.append(
            {
                "dataset": e.get("dataset"),
                "split": e.get("split"),
                "generation_model": e.get("generation_model"),
                "passage_id": pid,
                "text": e.get("text", ""),
                "conditioned_score": e.get("cs_used"),
            }
        )

    if resume:
        for rec in out:
            append_jsonl(output_path, rec)
    else:
        save_jsonl(output_path, out)



def explode_iqoq(master_path: str, output_path: str, resume: bool = False):
    """
    Input: CLEANED, merged file per variant
    Output: one row per IQ/OQ with core metadata (no keywords).
    """
    data = list(load_jsonl(master_path))
    items: list[str] = []
    for e in data:
        pid = e["passage_id"]
        for i, _ in enumerate(e.get("IQs", []) or []):
            items.append(f"{pid}_iq{i}")
        for i, _ in enumerate(e.get("OQs", []) or []):
            items.append(f"{pid}_oq{i}")
    done_ids, _ = compute_resume_sets(
        resume=resume,
        out_path=output_path,
        items=items,
        get_id=lambda x, i: x,
        phase_label="explode_iqoq",
        id_field="iqoq_id",
    )
    seen = set()
    out = []
    for e in data:
        pid = e["passage_id"]

        for i, q in enumerate(e.get("IQs", []) or []):
            iqid = f"{pid}_iq{i}"
            if resume and iqid in done_ids:
                print(f"[resume] explode_iqoq skipping {iqid}")
                continue
            if iqid in seen:
                continue
            seen.add(iqid)
            out.append({
                "dataset": e.get("dataset"),
                "split": e.get("split"),
                "generation_model": e.get("generation_model"),
                "parent_passage_id": pid,
                "iqoq_id": iqid,
                "type": "IQ",
                "index": i,
                "text": q,
            })

        for i, q in enumerate(e.get("OQs", []) or []):
            oqid = f"{pid}_oq{i}"
            if resume and oqid in done_ids:
                print(f"[resume] explode_iqoq skipping {oqid}")
                continue
            if oqid in seen:
                continue
            seen.add(oqid)
            out.append({
                "dataset": e.get("dataset"),
                "split": e.get("split"),
                "generation_model": e.get("generation_model"),
                "parent_passage_id": pid,
                "iqoq_id": oqid,
                "type": "OQ",
                "index": i,
                "text": q,
            })
    if resume:
        for rec in out:
            append_jsonl(output_path, rec)
    else:
        save_jsonl(output_path, out)


















def list_inputs(root: str, split: str, variant: str):
    """
    Discover shard input files under `root`, returning IQ/OQ shards and scored shards separately.

      - IQ/OQ shards:
          * `.jsonl` files that either end with `_{variant}.jsonl`
            (e.g., *_baseline.jsonl) or start with `{split}_iqoq_`.
          * Excludes cleaned/debug files.

      - Scored shards:
          * `.jsonl` files that end with `_cs.jsonl`.
          * Excludes cleaned/debug files.

    Args:
        root: Root directory to scan.
        split: Data split (e.g., "train", "dev", "test") used for the iqoq prefix match.
        variant: Variant string (e.g., "baseline", "enhanced") used for the iqoq suffix match.

    Returns:
        dict with:
            {
              "iqoq":   [<paths to IQ/OQ shard files>],
              "scored": [<paths to *_cs.jsonl files>]
            }
        If root does not exist, both lists are empty.
    """
    iqoq, scored = [], []
    if not os.path.isdir(root):
        return {"iqoq": [], "scored": []}

    for name in os.listdir(root):
        if not name.endswith(".jsonl"):
            continue
        if ".cleaned." in name or name.endswith("_iqoq_debug.jsonl"):
            continue

        full = os.path.join(root, name)

        if name.endswith("_cs.jsonl"):
            scored.append(full)
        elif name.endswith(f"_{variant}.jsonl") or name.startswith(f"{split}_iqoq_"):
            iqoq.append(full)

    return {"iqoq": sorted(iqoq), "scored": sorted(scored)}















# Maps variant → cleaning function to apply to IQ/OQ fields.
CLEANER_BY_VARIANT = {
    "baseline": clean_baseline,
    "enhanced": clean_iqoq,
}
















def process_job(dataset: str, model: str, variant: str, split: str,
                run_clean: bool, run_merge: bool, run_explode: bool,
                keep_shard_outputs: bool = False,
                resume: bool = False) -> None:
    hoprag_version = FOLDERS_BY_VARIANT[variant][0]  # e.g., "baseline_hoprag"
    base = resolve_root(model, dataset, split, variant) or f"data/models/{model}/{dataset}/{split}/{hoprag_version}"
    root = Path(base)

    # prefer .../raw as input if it exists, otherwise the variant folder itself
    search_dir = model_shard_dir(model, dataset, split) / hoprag_version
    if not search_dir.is_dir():
        print(f"[warn] no shard inputs for {dataset} | {model} | {variant} in {search_dir}")
        return
    
    # ensure output dirs exist
    cleaned_dir  = root / "cleaned";  cleaned_dir.mkdir(parents=True, exist_ok=True)
    exploded_dir = root / "exploded"; exploded_dir.mkdir(parents=True, exist_ok=True)
    # Debug lives under the split dir; filename has no split (debug/cleaning_debug.txt)
    debug_dir    = root / "debug";    debug_dir.mkdir(parents=True, exist_ok=True)

    # discover inputs
    files = list_inputs(str(search_dir), split, variant)
    iqoq_inputs, scored_inputs = files["iqoq"], files["scored"]

    if not iqoq_inputs and not scored_inputs:
        print(f"[warn] no shard inputs for {dataset} | {model} | {variant} in {search_dir}")
        return

    print(f"\n=== {dataset} | {model} | {variant} ===")
    cleaner = CLEANER_BY_VARIANT[variant]
    cleaned_paths, summaries = [], []
    phase_start = time.time()





    # CLEAN shards → cleaned/{stem}.cleaned.jsonl (variant suffix dropped for simplicity)
    if run_clean and iqoq_inputs:
        for p in iqoq_inputs:
            stem = Path(p).stem
            out_clean = cleaned_dir / f"{stem}.cleaned.jsonl"
            summary = clean_file(p, str(out_clean), cleaner, resume=resume)
            summaries.append(summary)
            cleaned_paths.append(str(out_clean))

    # MERGE cleaned IQ/OQ → cleaned/iqoq.cleaned.jsonl   (NEW name)
    merged_iqoq = cleaned_dir / "iqoq.cleaned.jsonl"
    if run_merge and cleaned_paths:
        merge_jsonl_files(cleaned_paths, str(merged_iqoq), dedup_key="passage_id", resume=resume)

    # MERGE scored shards (accepts *_cs.jsonl) → cleaned/scored.cleaned.jsonl
    merged_scored = cleaned_dir / "scored.cleaned.jsonl"
    if run_merge and scored_inputs:
        merge_jsonl_files(scored_inputs, str(merged_scored), dedup_key="passage_id", resume=resume)
        print(f"[scores] merged → {merged_scored}")









    # EXPLODE (from merged_iqoq) → exploded/passages.exploded.jsonl and exploded/iqoq.exploded.jsonl
    if run_explode and merged_iqoq.exists():
        passages_out = exploded_dir / "passages.exploded.jsonl"
        iqoq_out    = exploded_dir / "iqoq.exploded.jsonl"
        explode_passages(str(merged_iqoq), str(passages_out), resume=resume)
        explode_iqoq(str(merged_iqoq), str(iqoq_out), resume=resume)

    # DEBUG (one TXT per phase) → debug/cleaning_debug.txt   (NEW name)
    if run_clean and summaries:
        write_cleaning_debug(
            model=model, dataset=dataset, variant=variant, split=split,
            hoprag_version=hoprag_version, start_time=phase_start,
            shard_summaries=summaries, merged_output_path=str(merged_iqoq) if merged_iqoq.exists() else None,
            debug_dir=str(debug_dir),
        )

    # optionally remove per-shard cleaned files
    if cleaned_paths and not keep_shard_outputs:
        for p in cleaned_paths:
            try: Path(p).unlink()
            except FileNotFoundError: pass

    print(f"done: {dataset} | {model} | {variant}")

















def iter_jobs(datasets, models, variants, skip):
    """
    Generate all (dataset, model, variant) triples, honoring a skip list.

    This utility iterates the Cartesian product of `datasets × models × variants`
    and yields each triple unless it appears in `skip`.

    Args:
        datasets: Iterable of dataset names (e.g., ["musique", "hotpotqa"]).
        models: Iterable of model identifiers (e.g., ["Qwen-7b"]).
        variants: Iterable of variant names (e.g., ["baseline", "enhanced"]).
        skip: A set of (dataset, model, variant) tuples to exclude.
              Example: {("musique", "Qwen-7b", "baseline")}

    Yields:
        Tuples of the form (dataset: str, model: str, variant: str).

    Side Effects:
        Prints a "Skipping ..." message for any triple found in `skip`.

    Examples:
        >>> for d, m, v in iter_jobs(["musique"], ["Qwen-7b"], ["baseline"], set()):
        ...     print(d, m, v)
        musique Qwen-7b baseline
    """
    for d in datasets:
        for m in models:
            for v in variants:
                if (d, m, v) in skip:
                    print(f"Skipping {d} | {m} | {v}")
                    continue
                yield d, m, v

















if __name__ == "__main__":
    # --- config ---
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    MODEL_TO_PROCESS   = ["llama-3.1-8b-instruct"]  # ["deepseek-distill-qwen-7b"]
    SPLIT = "dev"

    # All models we know about (from your SERVER_CONFIGS)
    ALL_MODELS = sorted({cfg["model"] for cfg in SERVER_CONFIGS})

    # REQUIRED: list of models to run
    # e.g., MODEL_TO_PROCESS = ["qwen-7b", "deepseek-distill-qwen-7b"]
    if not MODEL_TO_PROCESS:
        raise ValueError("MODEL_TO_PROCESS must be a non-empty list of model names.")

    invalid = [m for m in MODEL_TO_PROCESS if m not in ALL_MODELS]
    if invalid:
        raise ValueError(f"Unknown models in MODEL_TO_PROCESS: {invalid}. "
                         f"Choose from: {ALL_MODELS}")

    # Preserve order, remove duplicates
    MODELS = list(dict.fromkeys(MODEL_TO_PROCESS))

    RUN_BASELINE = True
    RUN_ENHANCED = False
    RUN_CLEAN    = True
    RUN_MERGE    = True
    RUN_EXPLODE  = True

    
    RESUME       = True

    # optional: skip specific combos
    SKIP = set()  # e.g., {("musique", "qwen-7b", "baseline")}

    VARIANTS = []
    if RUN_BASELINE: VARIANTS.append("baseline")
    if RUN_ENHANCED: VARIANTS.append("enhanced")

    for dataset, model, variant in iter_jobs(DATASETS, MODELS, VARIANTS, SKIP):
        process_job(
            dataset=dataset,
            model=model,
            variant=variant,
            split=SPLIT,
            run_clean=RUN_CLEAN,
            run_merge=RUN_MERGE,
            run_explode=RUN_EXPLODE,
            resume=RESUME
        )



