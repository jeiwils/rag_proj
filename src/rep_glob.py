#!/usr/bin/env python3
"""
Run strict rep_health across *all* discovered triplets.

Scans:
  - data/representations/datasets/** for *_passages_emb.npy triplets
  - data/representations/models/**   for *_iqoq_emb.npy triplets
Calls: python -m src.rep_health --strict on each
Exit code 0 if all pass, 1 if any fail.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]  # project root (rag_proj)
DATASETS_ROOT = ROOT / "data" / "representations" / "datasets"
MODELS_ROOT   = ROOT / "data" / "representations" / "models"

def find_passage_triplets():
    """Yield (emb, index, meta) for dataset passages."""
    for emb in DATASETS_ROOT.rglob("*_passages_emb.npy"):
        base = Path(str(emb).removesuffix("_passages_emb.npy"))
        index = Path(f"{base}_faiss_passages.faiss")
        meta  = Path(f"{base}_passages.jsonl")
        yield (emb, index, meta)

def find_iqoq_triplets():
    """Yield (emb, index, meta) for model IQ/OQ."""
    for emb in MODELS_ROOT.rglob("*_iqoq_emb.npy"):
        base = Path(str(emb).removesuffix("_iqoq_emb.npy"))
        index = Path(f"{base}_faiss_iqoq.faiss")
        # meta usually has no dataset prefix; try canonical then fallback
        d = emb.parent
        meta = d / "iqoq.cleaned.jsonl"
        if not meta.exists():
            cands = sorted(d.glob("*iqoq*.jsonl"))
            meta = cands[0] if cands else d / "iqoq.cleaned.jsonl"  # may not exist; rep_health will error
        yield (emb, index, meta)

def run_one(emb: Path, index: Path, meta: Path, sample: int, recall_trials: int) -> int:
    cmd = [
        sys.executable, "-m", "src.rep_health",
        "--emb", str(emb),
        "--index", str(index),
        "--meta", str(meta),
        "--strict",
        "--sample", str(sample),
        "--recall_trials", str(recall_trials),
    ]
    print("\n=== RUN ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode

def main():
    ap = argparse.ArgumentParser(description="Run rep_health over all datasets+models")
    ap.add_argument("--which", choices=["all","datasets","models"], default="all",
                    help="What to scan")
    ap.add_argument("--sample", type=int, default=5, help="norm sample size")
    ap.add_argument("--recall_trials", type=int, default=5, help="self-recall trials per triplet")
    args = ap.parse_args()

    triplets = []
    if args.which in ("all","datasets"):
        triplets += list(find_passage_triplets())
    if args.which in ("all","models"):
        triplets += list(find_iqoq_triplets())

    if not triplets:
        print("No triplets found. Check your folder layout and file patterns.")
        sys.exit(1)

    failures = 0
    for emb, index, meta in triplets:
        rc = run_one(emb, index, meta, args.sample, args.recall_trials)
        if rc != 0:
            failures += 1
            print(f"[SUMMARY] FAIL -> {emb.parent}")

    total = len(triplets)
    print("\n====================")
    print(f"Checked: {total} triplet(s)")
    print(f"Failures: {failures}")
    print("====================\n")

    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
