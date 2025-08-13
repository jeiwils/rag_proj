
import re
import os
import json
import time

from src.a1_dataset_processing import load_jsonl, save_jsonl, append_jsonl
from src.a2_text_prep import SERVER_CONFIGS
from pathlib import Path







"""


#   #   #   #   # INPUT: 




NOW I NEED TO UPDATE THIS FOR THE NEW DIRECTORIES!!!






#   #   #   #   # OUTPUT: 


data/models/{model}/{dataset}/{split}/{variant_hoprag}/


cleaned/
  iqoq.cleaned.jsonl
  scored.cleaned.jsonl            # only if *_cs.jsonl shards exist - SO I HAVE TO CHANGE THIS 
  {stem}.cleaned.jsonl            # per-shard temp; deleted unless keep_shard_outputs=True

exploded/
  passages.exploded.jsonl
  iqoq.exploded.jsonl

debug/
  cleaning_debug.txt




    







###################### I DON' THINK I ACTUALLY HAVE TO SAVE THESE? WHAT PURPOSE DO THEY SERVE?
#### MERGED CS

{ # train_scored.jsonl
  "passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "title": "Arthur's Magazine",
  "text": "Arthur's Magazine (1844–1846)...",
  "conditioned_score": 0.25,
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b"
}




#### CLEANED AND MERGED IQOQ

{ # train_iqoq.cleaned.baseline.jsonl
  "passage_id": "5a7a0693__first_for_women_sent0",
  "title": "First for Women",
  "text": "First for Women is a woman's magazine...",
  "IQs": ["Who publishes First for Women?", "When was First for Women launched?"],
  "OQs": ["What is the target audience?", "How often is it published?", "Who is the editor?", "What is its circulation?"],
  "num_iq": 2,
  "num_oq": 4,
  "cs_used": null,
  "hoprag_version": "baseline_hoprag",
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b"
}



{ #train_iqoq.cleaned.enhanced.jsonl
  "passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "title": "Arthur's Magazine",
  "text": "Arthur's Magazine (1844–1846)...",
  "IQs": ["Who founded Arthur's Magazine?", "When was it published?", "What content did it feature?"],
  "OQs": ["Which city was it based in?", "Why did it cease?", "Who were notable contributors?"],
  "num_iq": 3,
  "num_oq": 3,
  "cs_used": 0.25,
  "hoprag_version": "enhanced_hoprag",
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b"
}






###### EXPLODED IQOQ

{ # train_iqoq_items.cleaned.enhanced.jsonl
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b",
  "parent_passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "iqoq_id": "5a7a0693__arthur_s_magazine_sent0_iq0",
  "type": "IQ",
  "index": 0,
  "text": "Who founded Arthur's Magazine?"
}



{ # train_iqoq_items.cleaned.enhanced.jsonl
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b",
  "parent_passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "iqoq_id": "5a7a0693__arthur_s_magazine_sent0_oq2",
  "type": "OQ",
  "index": 2,
  "text": "Who were notable contributors?"
}







### EXPLODED PASSAGES

{ # train_passages.cleaned.enhanced.jsonl
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b",
  "passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "text": "Arthur's Magazine (1844–1846)...",
  "conditioned_score": 0.25,
}



"""







def clean_iqoq(questions: list[str]) -> list[str]: ##################### should I recalculate the number of iqoq after this? - ALREADY DO IT IN CLEAN_FILE BUT i THIK i SHOULD DO IT HERE INSTEAD?
    """
    to make sure llm returns proper questions

    """
    cleaned = []

    for q in questions:
        q = q.strip()
        q = re.sub(r"^\d+[\.\)]\s*", "", q)      # remove numbering (e.g., '1. ...')
        q = re.sub(r"^[-*]\s*", "", q)           # remove bullets ('- ', '* ')
        if not q.endswith("?"):
            continue                             # skip if it’s not a question - will need to check this during debugging
        if len(q) < 5:
            continue                             # skip tiny junk 
        if q.lower() in {"n/a", "none", "no question generated"}:
            continue
        cleaned.append(q)

    return cleaned
















def clean_baseline(questions):
    """Parse baseline 'json {"Question List":[...]}' strings then clean."""
    if not questions:
        return []
    seq = questions if isinstance(questions, list) else [questions]
    collected = []
    for s in seq:
        s = re.sub(r"^\s*json\s*", "", str(s).strip(), flags=re.I)
        for b in re.findall(r"\{.*?\}", s, flags=re.S):
            try:
                obj = json.loads(b)
            except Exception:
                continue
            qlist = obj.get("Question List")
            if qlist is None:
                for alt in ("questions","question_list","qlist"):
                    if alt in obj:
                        qlist = obj[alt]; break
            if isinstance(qlist, list):
                collected.extend(map(str, qlist))
    return clean_iqoq(collected if collected else list(map(str, seq)))


def clean_file(in_path, out_path, cleaner):
    """Clean one shard. Returns a compact summary. No debug files here."""
    raw = load_jsonl(in_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cleaned = []
    total_raw_iq = total_raw_oq = 0
    total_clean_iq = total_clean_oq = 0

    for e in raw:
        raw_IQs = list(e.get("IQs") or [])
        raw_OQs = list(e.get("OQs") or [])
        total_raw_iq += len(raw_IQs)
        total_raw_oq += len(raw_OQs)

        cIQ = cleaner(raw_IQs)
        cOQ = cleaner(raw_OQs)

        e["IQs"], e["OQs"] = cIQ, cOQ
        e["num_iq"], e["num_oq"] = len(cIQ), len(cOQ)

        total_clean_iq += len(cIQ)
        total_clean_oq += len(cOQ)
        cleaned.append(e)

    save_jsonl(out_path, cleaned)
    return {
        "shard_path": in_path,
        "out_path": out_path,
        "num_entries": len(raw),
        "raw_iq": total_raw_iq,
        "raw_oq": total_raw_oq,
        "clean_iq": total_clean_iq,
        "clean_oq": total_clean_oq,
    }


def merge_jsonl_files(in_paths, out_path, dedup_key=None):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    seen = set(); out = []
    for p in in_paths:
        for row in load_jsonl(p):
            if dedup_key:
                k = row.get(dedup_key)
                if k in seen: continue
                seen.add(k)
            out.append(row)
    save_jsonl(out_path, out)





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





















def explode_passages(master_path: str, output_path: str):
    data = load_jsonl(master_path)
    out = []
    for e in data:
        out.append({
            "dataset": e.get("dataset"),
            "split": e.get("split"),
            "generation_model": e.get("generation_model"),
            "passage_id": e["passage_id"],
            "text": e.get("text", ""),
            "conditioned_score": e.get("cs_used"),
        })
    save_jsonl(output_path, out)








def explode_iqoq(master_path: str, output_path: str):
    """
    Input: CLEANED, merged file per variant
    Output: one row per IQ/OQ with core metadata (no keywords).
    """
    data = load_jsonl(master_path)
    out = []
    for e in data:
        pid = e["passage_id"]

        for i, q in enumerate(e.get("IQs", []) or []):
            out.append({
                "dataset": e.get("dataset"),
                "split": e.get("split"),
                "generation_model": e.get("generation_model"),
                "parent_passage_id": pid,
                "iqoq_id": f"{pid}_iq{i}",
                "type": "IQ",
                "index": i,
                "text": q,
            })

        for i, q in enumerate(e.get("OQs", []) or []):
            out.append({
                "dataset": e.get("dataset"),
                "split": e.get("split"),
                "generation_model": e.get("generation_model"),
                "parent_passage_id": pid,
                "iqoq_id": f"{pid}_oq{i}",
                "type": "OQ",
                "index": i,
                "text": q,
            })
    save_jsonl(output_path, out)


















def list_inputs(root: str, split: str, variant: str):
    """
    Discover shard input files under `root`, returning IQ/OQ shards and scored shards separately.

    - IQ/OQ shards:
        * .jsonl files that either end with `_{variant}.jsonl` (e.g., *_baseline.jsonl, *_enhanced.jsonl)
          or start with `{split}_iqoq_`.
        * Excludes cleaned/debug files.

    - Scored shards:
        * .jsonl files that end with `_cs.jsonl`.
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














# Maps variant → preferred folder names to search (first match wins).
FOLDERS_BY_VARIANT = {
    "baseline": ["baseline_hoprag"],
    "enhanced": ["enhanced_hoprag"],
}

# Maps variant → cleaning function to apply to IQ/OQ fields.
CLEANER_BY_VARIANT = {
    "baseline": clean_baseline,
    "enhanced": clean_iqoq,
}












def resolve_root(model: str, ######################################## could i use this earlier? isn't it useful in all my directories?
                 dataset: str,
                 split: str,
                 variant: str) -> str | None:
    """
    Resolve the root directory that contains input/output files for a job.

    Given a `(model, dataset, variant)` triple, this tries each candidate folder
    defined in `FOLDERS_BY_VARIANT[variant]` under the path:
        data/models/{model}/{dataset}/{candidate_folder}
    and returns the first folder that exists on disk.

    Args:
        model: Model identifier (e.g., "Qwen-7b") used in the directory path.
        dataset: Dataset identifier (e.g., "hotpotqa") used in the directory path.
        variant: Variant name (e.g., "baseline", "enhanced") used to choose folders.

    Returns:
        The first existing root directory path as a string, or `None` if no
        candidate folder exists.

    Examples:
        >>> resolve_root("Qwen-7b", "hotpotqa", "baseline")
        'data/models/Qwen-7b/hotpotqa/baseline_hoprag'   # if it exists
    """
    for hoprag_version in FOLDERS_BY_VARIANT[variant]:
        root = f"data/models/{model}/{dataset}/{split}/{hoprag_version}"
        if os.path.isdir(root):
            return root
    return None









def process_job(dataset: str, model: str, variant: str, split: str,
                run_clean: bool, run_merge: bool, run_explode: bool,
                keep_shard_outputs: bool = False) -> None:
    hoprag_version = FOLDERS_BY_VARIANT[variant][0]  # e.g., "baseline_hoprag"
    base = resolve_root(model, dataset, split, variant) or f"data/models/{model}/{dataset}/{split}/{hoprag_version}"
    root = Path(base)

    # prefer .../raw as input if it exists, otherwise the variant folder itself
    search_dir = root / "raw" if (root / "raw").is_dir() else root

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
            summary = clean_file(p, str(out_clean), cleaner)
            summaries.append(summary)
            cleaned_paths.append(str(out_clean))

    # MERGE cleaned IQ/OQ → cleaned/iqoq.cleaned.jsonl   (NEW name)
    merged_iqoq = cleaned_dir / "iqoq.cleaned.jsonl"
    if run_merge and cleaned_paths:
        merge_jsonl_files(cleaned_paths, str(merged_iqoq), dedup_key="passage_id")

    # MERGE scored shards (accepts *_cs.jsonl) → cleaned/scored.cleaned.jsonl  
    merged_scored = cleaned_dir / "scored.cleaned.jsonl"
    if run_merge and scored_inputs:
        merge_jsonl_files(scored_inputs, str(merged_scored), dedup_key="passage_id")
        print(f"[scores] merged → {merged_scored}")









    # EXPLODE (from merged_iqoq) → exploded/passages.exploded.jsonl and exploded/iqoq.exploded.jsonl  
    if run_explode and merged_iqoq.exists():
        passages_out = exploded_dir / "passages.exploded.jsonl"
        iqoq_out    = exploded_dir / "iqoq.exploded.jsonl"
        explode_passages(str(merged_iqoq), str(passages_out))
        explode_iqoq(str(merged_iqoq), str(iqoq_out))

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
    MODEL_TO_PROCESS   = ["qwen-7b"]  # ["deepseek-distill-qwen-7b"]
    SPLIT = "train"

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
    RUN_ENHANCED = True
    RUN_CLEAN    = True
    RUN_MERGE    = True
    RUN_EXPLODE  = True

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
        )



