"""
Module Overview
---------------
Split passages into model-specific shards and generate conditioned scores (CS) 
alongside incoming and outgoing question (IQ/OQ) lists for each passage.

This script supports both baseline and enhanced HopRAG workflows using multiple
LLM servers. Enhanced IQ/OQ generation is guided by conditioned scores.

Key steps include:
- Splitting `{split}_passages.jsonl` into N shards based on model size.
- Generating conditioned scores for each passage.
- Creating IQ/OQ lists either with fixed or score-based ratios.
- Writing debug summaries for each shard and phase.



Inputs
------
data/processed_datasets/{dataset}/{split}_passages.jsonl
    Source passages with the following fields:
    - ``passage_id``: unique identifier.
    - ``title``: source article or document title.
    - ``text``: passage content.

    

Outputs
-------
Sharded output files written to:

data/models/{model}/{dataset}/{split}/shards/
    - {split}_passages_shard{N}_{size}.jsonl
        → Raw input shards split by model size (1.5b → 4 shards, 7b → 2, 14b → 1).

data/models/{model}/{dataset}/{split}/shards/{hoprag_version}/


    - {split}_passages_shard{N}_{size}_cs.jsonl
        → Passages with conditioned scores.

    - {split}_passages_shard{N}_{size}_iqoq_baseline.jsonl
        → IQ/OQ questions from baseline HopRAG prompts (fixed ratio).

    - {split}_passages_shard{N}_{size}_iqoq_enhanced.jsonl
        → IQ/OQ questions generated using conditioned score–based ratios.

    - *_cs_debug.txt
    - *_iqoq_baseline_debug.txt
    - *_iqoq_enhanced_debug.txt
        → Per-shard debug summaries (counts, missing values, timing).

        

Examples
--------

### _cs.jsonl 

{
  "passage_id": "5a7a0693__arthur_s_magazine_sent0",
  "title": "Arthur's Magazine",
  "text": "Arthur's Magazine (1844–1846)...",
  "conditioned_score": 0.25,
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b"
}



### _iqoq_baseline.jsonl

{
  "passage_id": "5a7a0693__first_for_women_sent0",
  "title": "First for Women",
  "text": "First for Women is a woman's magazine...",
  "IQs": ["...", "..."],
  "OQs": ["...", "...", "...", "..."],
  "num_iq": 2,
  "num_oq": 4,
  "cs_used": null,                                             # baseline uses fixed ratio, no CS - enhanced would have a float here
  "hoprag_version": "baseline_hoprag",
  "dataset": "hotpotqa",
  "split": "train",
  "generation_model": "qwen-7b"
}



Debug Logs
----------
Each debug file logs:
- Total passages processed
- Model, dataset, and phase info
- Total time taken (seconds)
- List of missing conditioned scores or missing IQ/OQ generations


"""



#
# WRITE UP
# - conditioned score reverts to 0.5 if missing, the 2-4 default hop-rag iqoq ratio
# - justification for tokens (budget; size of prompt, expected input, expected output - context window)
# - justification for temperature (deterministic - taken from hopRAG)
#








from pathlib import Path
import re
import requests
from multiprocessing import Process
from src.utils import load_jsonl, save_jsonl, append_jsonl
from tqdm import tqdm
from typing import Callable, List, Dict
import time 
import json





SERVER_CONFIGS = [ 
    # 1.5B models (4 servers for all datasets)
    {"server_url": "http://localhost:8000", "model": "qwen-1.5b"},
    {"server_url": "http://localhost:8001", "model": "qwen-1.5b"},
    {"server_url": "http://localhost:8002", "model": "qwen-1.5b"},
    {"server_url": "http://localhost:8003", "model": "qwen-1.5b"},

    # 7B models (2 servers for all datasets)
    {"server_url": "http://localhost:8004", "model": "qwen-7b"},
    {"server_url": "http://localhost:8005", "model": "qwen-7b"},

    # 14B models (1 server for all datasets)
    {"server_url": "http://localhost:8006", "model": "qwen-14b"},
    
    # Deepseek-distill-qwen models (4 servers for all datasets)
    {"server_url": "http://localhost:8007", "model": "deepseek-distill-qwen-1.5b"},
    {"server_url": "http://localhost:8008", "model": "deepseek-distill-qwen-1.5b"},
    {"server_url": "http://localhost:8009", "model": "deepseek-distill-qwen-1.5b"},
    {"server_url": "http://localhost:8010", "model": "deepseek-distill-qwen-1.5b"},

    # Deepseek-distill-qwen 7B models (2 servers for all datasets)
    {"server_url": "http://localhost:8011", "model": "deepseek-distill-qwen-7b"},
    {"server_url": "http://localhost:8012", "model": "deepseek-distill-qwen-7b"},

    # Deepseek-distill-qwen 14B models (1 server for all datasets)
    {"server_url": "http://localhost:8013", "model": "deepseek-distill-qwen-14b"},
]



# at top with other constants
CS_GRAMMAR = r'''
root ::= "CS: 0.00" | "CS: 0.25" | "CS: 0.50" | "CS: 0.75" | "CS: 1.00"
'''



MAX_TOKENS = {
    "cs": 200, # 50, 
    "iqoq_generation": 256,
    "edge_selection": 64, ################################# to tune
    "answer_generation": 256 ########################### to tune 
}

TEMPERATURE = {
    "cs": 0.0, 
    "iqoq_generation": 0.1,
    "edge_selection": 0.1, ################################# to tune
    "answer_generation": 0.1 ################################# to tune
}



CS_PROMPT = Path("data/prompts/7b_CS_prompt_ultra_updated.txt").read_text(encoding="utf-8")

enhanced_IQ_prompt = Path("data/prompts/7b_IQ_prompt_updated.txt").read_text(encoding="utf-8")
enhanced_OQ_prompt = Path("data/prompts/7b_OQ_prompt_updated.txt").read_text(encoding="utf-8") 

hoprag_IQ_prompt = Path("data/prompts/hoprag_iq_prompt.txt").read_text(encoding="utf-8")
hoprag_OQ_prompt = Path("data/prompts/hoprag_oq_prompt.txt").read_text(encoding="utf-8") 








def get_server_urls(model):
    """
    Return all server URLs for a given model (e.g., 4 for 1.5B, 2 for 7B).


    goes in the __main__ loop to activate all relevant servers 

    """
    urls = [config["server_url"] for config in SERVER_CONFIGS if config["model"] == model]
    if not urls:
        raise ValueError(f"Unknown model: {model}")
    return urls



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
    data = load_jsonl(path)
    half = len(data) // 2
    save_jsonl(out1, data[:half])
    save_jsonl(out2, data[half:])



def split_jsonl_into_four(path, out1, out2, out3, out4):
    data = load_jsonl(path)
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




def split_jsonl_for_models(path: str, model: str) -> list[str]:
    """
    Split input JSONL into N shards based on model size and write them to:
      data/models/{model}/{dataset}/{split}/shards/{stem}_shard{N}_{size}.jsonl
    Returns the shard paths. For 14B we still write a single shard file.
    """
    size = model_size(model)  # '1.5b' | '7b' | '14b'
    dataset = Path(path).parent.name
    split_name = Path(path).stem.split("_")[0]  # 'train' or 'dev' 
    out_dir = Path(f"data/models/{model}/{dataset}/{split_name}/shards")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(path).stem

    if size == "1.5b":
        out_paths = [
            str(out_dir / f"{stem}_shard1_{size}.jsonl"),
            str(out_dir / f"{stem}_shard2_{size}.jsonl"),
            str(out_dir / f"{stem}_shard3_{size}.jsonl"),
            str(out_dir / f"{stem}_shard4_{size}.jsonl"),
        ]
        split_jsonl_into_four(path, *out_paths)
        return out_paths

    if size == "7b":
        out_paths = [
            str(out_dir / f"{stem}_shard1_{size}.jsonl"),
            str(out_dir / f"{stem}_shard2_{size}.jsonl"),
        ]
        split_jsonl(path, *out_paths)
        return out_paths

    if size == "14b":
        out_path = out_dir / f"{stem}_shard1_{size}.jsonl"
        if not out_path.exists():
            save_jsonl(str(out_path), load_jsonl(path))
        return [str(out_path)]

    raise ValueError(f"Unsupported model size: {size}")









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













def query_llm(prompt: str, server_url: str, max_tokens: int = 5, temperature: float = 0.0, stop=None, grammar=None) -> str:
    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "n_predict": max_tokens,
    }
    if stop:
        payload["stop"] = stop
    if grammar:
        payload["grammar"] = grammar
    resp = requests.post(f"{server_url}/completion", json=payload)
    resp.raise_for_status()
    return resp.json().get("content", "").strip()


















def write_debug_file(
    task_type: str,
    total_processed: int,
    missing_cs: list,
    missing_iq: list,
    missing_oq: list,
    start_time: float,
    model: str,
    dataset: str,
    hoprag_version: str,
    shard_path: str,
    split_name: str,
    skipped: int = 0,
):
    size = model_size(model)
    shard_stem = Path(shard_path).stem
    debug_dir = Path(f"data/models/{model}/{dataset}/{split_name}/shards/{hoprag_version}")
    debug_dir.mkdir(parents=True, exist_ok=True)

    if task_type == "cs":
        debug_file_path = debug_dir / f"{shard_stem}_cs_debug.txt"
    else:
        debug_file_path = debug_dir / f"{shard_stem}_{task_type}_debug.txt"

    duration = time.time() - start_time
    with open(debug_file_path, "w", encoding="utf-8") as f:
        f.write(f"Total passages processed: {total_processed}\n")
        f.write(f"Skipped due to resume: {skipped}\n")
        f.write(f"Model used: {model}\n")
        f.write(f"Model size: {size}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Split: {split_name}\n")  # optional but handy
        f.write(f"Phase: {task_type}\n")
        f.write(f"Total time taken: {duration:.2f} seconds\n")
        if missing_cs:
            f.write(f"\nPassages missing CS: {len(missing_cs)}\n")
            for passage in missing_cs:
                pid = passage.get("passage_id", "?")
                err = passage.get("error", "")
                txt = passage.get("passage_text", "")
                f.write(f"- Missing CS (ID: {pid}): {err}\n")
                if txt:
                    f.write(f"  Passage text: {txt}\n")
        if missing_iq:
            f.write(f"\nPassages missing IQ: {len(missing_iq)}\n")
            for pid in missing_iq:
                f.write(f"- {pid}\n")
        if missing_oq:
            f.write(f"\nPassages missing OQ: {len(missing_oq)}\n")
            for pid in missing_oq:
                f.write(f"- {pid}\n")







def get_conditioned_score(
    entry: dict,
    cs_prompt_template: str,
    server_url: str,
    cs_tokens: int,
    cs_temperature: float = 0.0
):
    """
    Query CS for a single passage. Returns the UPDATED entry on success; None on failure.
    Note: This function does NOT set dataset/split. Do that where you write results.
    """
    passage_text = entry["text"]
    cs_prompt_filled = cs_prompt_template.replace("{{PASSAGE}}", passage_text)

    try:
        response = query_llm(
            cs_prompt_filled,
            server_url,
            max_tokens=cs_tokens,          # keep this small
            temperature=cs_temperature,
            stop=["\n"],                   # stop at first newline
            grammar=CS_GRAMMAR             # hard-constrain output
        )
        match = (
            re.search(r"(?:CS|Conditioned\s*Score)\s*:?\s*(?P<score>(?:0?\.\d+|0|1(?:\.0+)?))", response, re.I)
            or re.search(r"\b(?P<score>(?:0?\.\d+|0|1(?:\.0+)?))\b", response)
        )
        if not match:
            raise ValueError(f"No valid CS score found in response: {response!r}")

        score = float(match.group("score"))
    except Exception as e:
        print(f"[CS ERROR] Failed for {entry.get('passage_id', '?')}: {e}")
        return None

    if not (0.0 <= score <= 1.0):
        print(f"[CS INVALID] {entry.get('passage_id', '?')}: {score}")
        return None

    entry["conditioned_score"] = score
    return entry





def iqoq_ratio( ####################################### FOCUS ON THIS 
        cs: float,
        qmin: int = 6,
        qmax: int = 9,
        epsilon_iq: int = 2,
        epsilon_oq: int = 2,
        iq_weight: float = 1.0,
        oq_weight: float = 2.0
    ) -> tuple[int, int, int]:
    """
    Weighted split:
      - w_iq = iq_weight * (1 - cs)
      - w_oq = oq_weight * cs
    This makes cs=0.5 => IQ share = 1/(1+2) = 1/3 => for q_total=6 => 2 IQ, 4 OQ.
    """
    # Step 1: U-shaped total questions; cs=0.5 -> qmin
    q_total = qmin + int(round((qmax - qmin) * (4 * (cs - 0.5)**2)))

    # Step 2: Weighted split (bias toward OQ at mid cs)
    w_iq = iq_weight * (1.0 - cs)
    w_oq = oq_weight * cs
    denom = w_iq + w_oq
    iq_share = 1.0 if denom == 0 else (w_iq / denom)

    num_iq = int(round(q_total * iq_share))
    num_oq = q_total - num_iq

    # Step 3: Enforce minimums
    num_iq = max(num_iq, epsilon_iq)
    num_oq = max(num_oq, epsilon_oq)

    # Step 4 (optional but recommended): keep total == q_total after mins
    total = num_iq + num_oq
    if total > q_total:
        excess = total - q_total
        # reduce the larger side first while respecting epsilons
        if num_iq >= num_oq and num_iq - epsilon_iq >= excess:
            num_iq -= excess
        elif num_oq - epsilon_oq >= excess:
            num_oq -= excess
        else:
            take = min(excess, max(0, num_iq - epsilon_iq))
            num_iq -= take
            excess -= take
            if excess > 0:
                num_oq -= min(excess, max(0, num_oq - epsilon_oq))
    elif total < q_total:
        deficit = q_total - total
        # add to the side favored by cs (or OQ when cs==0.5)
        if cs >= 0.5:
            num_oq += deficit
        else:
            num_iq += deficit

    return num_iq + num_oq, num_iq, num_oq




# Modify the generate_iqoq function to include version
def generate_iqoq(
    entry: dict,
    iq_prompt_template: str,
    oq_prompt_template: str,
    server_url: str,
    iq_tokens: int,
    oq_tokens: int,
    iq_temperature: float = 0.1,
    oq_temperature: float = 0.1,
    conditioned_score: float = None,
    use_ratio: bool = False,
    hoprag_version: str = "standard_hoprag",
    debug_dir: Path = None  # kept for signature compatibility; not used
):
    """
    Generates IQ and OQ lists for a passage.
    Returns (entry, missing_iq_ids, missing_oq_ids).
    On failure, entry=None and the lists indicate what failed.

    Note: This function does NOT set dataset/split. Set those where you write results.
    """
    passage_text = entry["text"]

    if use_ratio and conditioned_score is not None:
        _, num_iq, num_oq = iqoq_ratio(conditioned_score)
    else:
        num_iq, num_oq = 2, 4

    iq_prompt_filled = (
        iq_prompt_template
        .replace("{{PASSAGE}}", passage_text)
        .replace("{NUM_QUESTIONS}", str(num_iq))
    )
    oq_prompt_filled = (
        oq_prompt_template
        .replace("{{PASSAGE}}", passage_text)
        .replace("{NUM_QUESTIONS}", str(num_oq))
    )

    try:
        iq_response = query_llm(iq_prompt_filled, server_url, max_tokens=iq_tokens, temperature=iq_temperature)
        oq_response = query_llm(oq_prompt_filled, server_url, max_tokens=oq_tokens, temperature=oq_temperature)
    except Exception as e:
        print(f"[ERROR] LLM failed for {entry.get('passage_id','?')}: {e}")
        pid = entry.get("passage_id", "?")
        return None, [pid], [pid]

    missing_iq_this, missing_oq_this = [], []
    if not iq_response.strip():
        missing_iq_this.append(entry.get("passage_id", "?"))
    if not oq_response.strip():
        missing_oq_this.append(entry.get("passage_id", "?"))

    if missing_iq_this or missing_oq_this:
        return None, missing_iq_this, missing_oq_this

    entry["IQs"] = [q for q in iq_response.split("\n") if q.strip()]
    entry["OQs"] = [q for q in oq_response.split("\n") if q.strip()]
    entry["num_iq"] = num_iq
    entry["num_oq"] = num_oq
    entry["cs_used"] = conditioned_score if use_ratio else None
    entry["hoprag_version"] = hoprag_version

    return entry, [], [] ########################## WHY?????









def existing_ids(path, id_field="passage_id"):
    if not Path(path).exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                obj = json.loads(line)
                pid = obj.get(id_field)
                if pid is not None:
                    done.add(pid)
            except Exception:
                # tolerate a possibly truncated last line ################################################ ?
                continue
    return done






from typing import Iterable, Callable, Hashable, Tuple, Set, Any

def compute_resume_sets(
    *,
    resume: bool,
    out_path: str,
    items: Iterable[Any],
    get_id: Callable[[Any, int], Hashable],
    phase_label: str
) -> Tuple[Set[Hashable], Set[Hashable]]:
    """
    Returns (done_ids, shard_ids) for the current shard/phase and prints an accurate
    per-shard resume message if resume=True.

    - shard_ids: IDs present in this shard's input sequence (items)
    - done_ids:  IDs already written to this shard's *output* file (out_path), intersected with shard_ids
    """
    shard_ids = {get_id(x, i) for i, x in enumerate(items)}
    if not resume:
        return set(), shard_ids

    done_all = existing_ids(out_path)         # only this shard's file
    done_ids = done_all & shard_ids           # defensive intersection
    print(f"[resume] {phase_label}: {len(done_ids)}/{len(shard_ids)} already present in this shard – skipping those")
    return done_ids, shard_ids







def process_server_task(config: dict):
    """
    Expects: server_url, dataset, model, hoprag_version, task_type, input_path,
             cs_tokens, iq_tokens, oq_tokens, tqdm_position, (optional) split, (optional) resume

    Runs one shard on one server for the given phase and writes a single debug file:
      - <shard>_cs_debug.txt
      - <shard>_baseline_debug.txt
      - <shard>_enhanced_debug.txt
    """
    # --- required config values ---
    model          = config["model"]
    dataset        = config["dataset"]
    hoprag_version = config["hoprag_version"]
    server_url     = config["server_url"]
    cs_tokens      = config["cs_tokens"]
    iq_tokens      = config["iq_tokens"]
    oq_tokens      = config["oq_tokens"]
    task_type      = config["task_type"]
    input_path     = config["input_path"]

    # --- optional ---
    pos        = int(config.get("tqdm_position", 0))
    split_name = config.get("split") or Path(input_path).stem.split("_")[0]
    resume     = bool(config.get("resume", False))

    # outputs live under .../{dataset}/{split}/{hoprag_version}
    phase_dir = Path(f"data/models/{model}/{dataset}/{split_name}/shards/{hoprag_version}")
    phase_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    passages = load_jsonl(input_path)

    # Aggregation
    total_processed = 0
    total_success   = 0  # successes for the phase

    # Miss buckets
    missing_cs: list = []
    missing_iq: list = []
    missing_oq: list = []

    shard_stem   = Path(input_path).stem
    scored_tmp   = str(phase_dir / f"{shard_stem}_cs.jsonl")
    out_baseline = str(phase_dir / f"{shard_stem}_iqoq_baseline.jsonl")
    out_enhanced = str(phase_dir / f"{shard_stem}_iqoq_enhanced.jsonl")

    # pick phase output for unlink logic
    if task_type == "cs":
        out_path = scored_tmp
    elif task_type == "iqoq_baseline":
        out_path = out_baseline
    elif task_type == "iqoq_enhanced":
        out_path = out_enhanced
    else:
        raise ValueError(f"Unexpected task_type: {task_type}")

    # unlink only when NOT resuming
    if not resume and Path(out_path).exists():
        Path(out_path).unlink()





    # ---------------- Enhanced HopRAG ----------------
    if hoprag_version == "enhanced_hoprag":

        if task_type == "cs":
            done_ids, shard_ids = compute_resume_sets(
                resume=resume,
                out_path=scored_tmp,
                items=passages,
                get_id=lambda p, i: p.get("passage_id", f"idx:{i}"),
                phase_label="CS"
            )
            skipped = 0
            for i, p in enumerate(tqdm(
                passages,
                desc=f"CS | {dataset} | {model} @{server_url.split(':')[-1]}",
                position=pos, dynamic_ncols=True, leave=False, mininterval=0.2,
            )):
                pid = p.get("passage_id", f"idx:{i}")
                if resume and pid in done_ids:
                    skipped += 1
                    continue

                total_processed += 1
                s = get_conditioned_score(
                    p, CS_PROMPT, server_url,
                    cs_tokens=cs_tokens,
                    cs_temperature=TEMPERATURE["cs"],
                )
                if s:
                    s["dataset"] = dataset
                    s["split"] = split_name
                    s["generation_model"] = model
                    append_jsonl(scored_tmp, s)
                    total_success += 1
                else:
                    missing_cs.append({
                        "passage_id": p.get("passage_id", "?"),
                        "error": "scoring failed or invalid score",
                        "passage_text": p.get("text", "")
                    })

            duration = time.time() - t0
            print(f"[summary] CS: processed={total_processed}, skipped={skipped}, wrote={Path(scored_tmp).name}")
            write_debug_file("cs", total_processed, missing_cs, [], [], t0, model, dataset,
                            hoprag_version, input_path, split_name, skipped=skipped)
            return duration






        # Phase: Enhanced IQ/OQ generation using CS-driven ratios

        elif task_type == "iqoq_enhanced":
            # Require CS file; if missing, skip this shard entirely
            if not Path(scored_tmp).exists():
                print(f"[skip] Enhanced: CS file not found for {shard_stem} – skipping IQ/OQ.")
                write_debug_file(
                    "iqoq_enhanced",
                    total_processed=0,
                    missing_cs=[], missing_iq=[], missing_oq=[],
                    start_time=t0, model=model, dataset=dataset,
                    hoprag_version=hoprag_version, shard_path=input_path,
                    split_name=split_name, skipped=0
                )
                return 0.0

            # Load CS results for this shard
            scored_list = load_jsonl(scored_tmp)

            # (Optional) keep only those passage_ids that are actually in this shard's input
            shard_ids = {p.get("passage_id", f"idx:{i}") for i, p in enumerate(passages)}
            scored_items = [
                (rec.get("passage_id"), rec)
                for rec in scored_list
                if rec.get("passage_id") in shard_ids
            ]

            # Accurate per-shard resume setup
            done_ids, _ = compute_resume_sets(
                resume=resume,
                out_path=out_enhanced,
                items=scored_items,
                get_id=lambda kv, i: kv[0],  # pid
                phase_label="Enhanced"
            )

            skipped = 0
            for i, (pid, p) in enumerate(tqdm(
                scored_items,
                desc=f"Enhanced IQ/OQ | {dataset} | {model} @{server_url.split(':')[-1]}",
                position=pos, dynamic_ncols=True, leave=False, mininterval=0.2,
            )):
                if resume and pid in done_ids:
                    skipped += 1
                    continue

                total_processed += 1

                # Per-entry fallback: if CS is missing/invalid, use 0.5 (→ 2 IQ / 4 OQ)
                cs = p.get("conditioned_score", 0.5)
                if not isinstance(cs, (int, float)) or not (0.0 <= cs <= 1.0):
                    cs = 0.5

                out, mi, mo = generate_iqoq(
                    p, enhanced_IQ_prompt, enhanced_OQ_prompt, server_url,
                    iq_tokens=iq_tokens, oq_tokens=oq_tokens,
                    iq_temperature=TEMPERATURE["iqoq_generation"],
                    oq_temperature=TEMPERATURE["iqoq_generation"],
                    conditioned_score=cs,  # 0.5 default drives 2–4 with your weighted iqoq_ratio
                    use_ratio=True,
                    hoprag_version=hoprag_version,
                    debug_dir=None
                )
                if mi: missing_iq.extend(mi)
                if mo: missing_oq.extend(mo)
                if out:
                    out["dataset"] = dataset
                    out["split"] = split_name
                    out["generation_model"] = model  # generate_iqoq already sets cs_used to the cs above
                    append_jsonl(out_enhanced, out)
                    total_success += 1

            duration = time.time() - t0
            print(f"[summary] Enhanced: processed={total_processed}, skipped={skipped}, wrote={Path(out_enhanced).name}")
            write_debug_file(
                "iqoq_enhanced",
                total_processed, [], missing_iq, missing_oq,
                t0, model, dataset, hoprag_version, input_path, split_name, skipped=skipped
            )
            return duration



        else:
            raise ValueError(f"Unexpected task_type for enhanced_hoprag: {task_type}")









    # ---------------- Baseline HopRAG ----------------
    elif hoprag_version == "baseline_hoprag":
        if task_type != "iqoq_baseline":
            raise ValueError(f"Baseline expects task_type='iqoq_baseline', got {task_type}")

        done_ids, shard_ids = compute_resume_sets(
            resume=resume,
            out_path=out_baseline,
            items=passages,
            get_id=lambda p, i: p.get("passage_id", f"idx:{i}"),
            phase_label="Baseline"
        )
        skipped = 0
        for i, p in enumerate(tqdm(
            passages,
            desc=f"Baseline IQ/OQ | {dataset} | {model} @{server_url.split(':')[-1]}",
            position=pos, dynamic_ncols=True, leave=False, mininterval=0.2,
        )):
            pid = p.get("passage_id", f"idx:{i}")
            if resume and pid in done_ids:
                skipped += 1
                continue

            total_processed += 1
            out, mi, mo = generate_iqoq(
                p, hoprag_IQ_prompt, hoprag_OQ_prompt, server_url,
                iq_tokens=iq_tokens, oq_tokens=oq_tokens,
                iq_temperature=TEMPERATURE["iqoq_generation"],
                oq_temperature=TEMPERATURE["iqoq_generation"],
                use_ratio=False, hoprag_version=hoprag_version,
                debug_dir=None
            )
            if mi: missing_iq.extend(mi)
            if mo: missing_oq.extend(mo)
            if out:
                out["dataset"] = dataset
                out["split"] = split_name
                out["generation_model"] = model
                append_jsonl(out_baseline, out)
                total_success += 1

        duration = time.time() - t0
        print(f"[summary] Baseline: processed={total_processed}, skipped={skipped}, wrote={Path(out_baseline).name}")
        write_debug_file("iqoq_baseline", total_processed, [], missing_iq, missing_oq,
                        t0, model, dataset, hoprag_version, input_path, split_name, skipped=skipped)
        return duration



    else:
        raise ValueError(f"Unknown hoprag_version: {hoprag_version}")



















if __name__ == "__main__": 


    RESUME = True
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    ACTIVE_MODEL_NAMES   = ["qwen-7b"]

    RUN_CS        = True        # enhanced scoring step
    RUN_BASELINE  = True        # hopRAG baseline IQ/OQ
    RUN_ENHANCED  = True        # enhanced IQ/OQ



    SPLIT = "train"             # or "dev"

    # --- per-phase skip rules (dataset, model) ---
    SKIP_CS        = set()
    SKIP_BASELINE  = set()
    SKIP_ENHANCED  = set()




    for dataset in DATASETS:
        input_path = f"data/processed_datasets/{dataset}/{SPLIT}_passages.jsonl" #################################################################### 

        for model in ACTIVE_MODEL_NAMES:
            print(f"\n=== {dataset} | {model} ===")
            urls   = get_server_urls(model)                    # all servers for this model
            shards = split_jsonl_for_models(input_path, model) # one shard per server






            # ---- Phase 1: CS (only if using enhanced + CS_GUIDED) ----
            if RUN_CS:
                if (dataset, model) in SKIP_CS:
                    print(f"Skipping CS for {dataset} | {model}")
                else:
                    t0 = time.time()
                    configs = [{
                        "server_url": url,
                        "dataset": dataset,
                        "model": model,
                        "hoprag_version": "enhanced_hoprag",
                        "task_type": "cs",
                        "input_path": shards[i],
                        "cs_tokens": MAX_TOKENS["cs"],
                        "iq_tokens": MAX_TOKENS["iqoq_generation"],
                        "oq_tokens": MAX_TOKENS["iqoq_generation"],
                        "tqdm_position": i,  # one tqdm row per worker
                        "split": SPLIT,
                        "resume": RESUME
                    } for i, url in enumerate(urls)]
                    run_multiprocess(process_server_task, configs)
                    print(f"[timing] CS: {time.time() - t0:.2f}s")




            # ---- Phase 2: Baseline IQ/OQ ----
            if RUN_BASELINE:
                if (dataset, model) in SKIP_BASELINE:
                    print(f"Skipping baseline IQ/OQ for {dataset} | {model}")
                else:
                    t0 = time.time()
                    configs = [{
                        "server_url": url,
                        "dataset": dataset,
                        "model": model,
                        "hoprag_version": "baseline_hoprag",
                        "task_type": "iqoq_baseline",
                        "input_path": shards[i],
                        "cs_tokens": MAX_TOKENS["cs"],
                        "iq_tokens": MAX_TOKENS["iqoq_generation"],
                        "oq_tokens": MAX_TOKENS["iqoq_generation"],
                        "tqdm_position": i,  # tqdm row
                        "split": SPLIT,
                        "resume": RESUME
                    } for i, url in enumerate(urls)]
                    run_multiprocess(process_server_task, configs)
                    print(f"[timing] baseline IQ/OQ: {time.time() - t0:.2f}s")




            # ---- Phase 3: Enhanced IQ/OQ ----
            if RUN_ENHANCED:
                if (dataset, model) in SKIP_ENHANCED:
                    print(f"Skipping enhanced IQ/OQ for {dataset} | {model}")
                else:
                    t0 = time.time()
                    configs = [{
                        "server_url": url,
                        "dataset": dataset,
                        "model": model,
                        "hoprag_version": "enhanced_hoprag",
                        "task_type": "iqoq_enhanced",
                        "input_path": shards[i],
                        "cs_tokens": MAX_TOKENS["cs"],
                        "iq_tokens": MAX_TOKENS["iqoq_generation"],
                        "oq_tokens": MAX_TOKENS["iqoq_generation"],
                        "tqdm_position": i,  # tqdm row
                        "split": SPLIT,
                        "resume": RESUME
                    } for i, url in enumerate(urls)]
                    run_multiprocess(process_server_task, configs)
                    print(f"[timing] enhanced IQ/OQ: {time.time() - t0:.2f}s")

    print("\nDone.")





__all__ = [
    "SERVER_CONFIGS",
    "extract_keywords",
    "get_server_urls",
    "model_size",
    "split_jsonl",
    "split_jsonl_into_four",
    "split_jsonl_for_models",
    "run_multiprocess",
    "query_llm",
    "write_debug_file",
    "get_conditioned_score",
    "iqoq_ratio",
    "generate_iqoq",
    "existing_ids",
    "compute_resume_sets",
    "process_server_task",
    "load_jsonl",
    "save_jsonl",
    "append_jsonl",
]