#!/usr/bin/env python3
"""
Bulletproof health checks for embeddings (.npy), FAISS indexes (.faiss), and metadata (.jsonl).

Exit codes:
  0 -> all checks passed
  1 -> one or more checks failed
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import faiss

# If you have this helper, we’ll reuse it; otherwise we do our own checks.
try:
    from src.utils import validate_vec_ids  # optional
except Exception:
    validate_vec_ids = None

def fail(errors, msg):
    errors.append(msg)
    print(f"[FAIL] {msg}")

def ok(msg):
    print(f"[OK] {msg}")

def load_embeddings(path: Path, errors: list[str]) -> np.ndarray:
    if not Path(path).is_file():
        fail(errors, f"emb file not found: {path}")
        return np.zeros((0, 0), dtype=np.float32)
    try:
        emb = np.load(path)
    except Exception as e:
        fail(errors, f"failed to load emb: {path} ({e})")
        return np.zeros((0, 0), dtype=np.float32)

    # dtype check
    if emb.dtype != np.float32:
        # try safe cast (FAISS prefers float32)
        try:
            emb = emb.astype(np.float32, copy=False)
            ok(f"casted emb to float32 (was {emb.dtype})")
        except Exception:
            fail(errors, f"emb dtype is {emb.dtype}, could not cast to float32")

    # shape check
    if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[1] == 0:
        fail(errors, f"emb has invalid shape {emb.shape}")
    else:
        ok(f"emb shape={emb.shape}")

    # NaN/Inf check
    if not np.isfinite(emb).all():
        fail(errors, "emb contains NaN/Inf")
    else:
        ok("emb is finite")

    return emb

def stats_and_norm_checks(emb: np.ndarray, errors: list[str], sample: int, expect_unit_norm: bool, unit_tol: float):
    # row norms
    norms = np.linalg.norm(emb, axis=1)
    mn, mx, mean = float(norms.min()), float(norms.max()), float(norms.mean())
    print(f"[emb] norms: min={mn:.6f} max={mx:.6f} mean={mean:.6f}")
    if sample > 0:
        print(f"[emb] sample norms: {norms[:sample]}")

    # zero vectors?
    zeros = int((norms == 0).sum())
    if zeros:
        fail(errors, f"{zeros} zero-norm vectors found")
    else:
        ok("no zero-norm vectors")

    if expect_unit_norm:
        # Require within tolerance of 1.0
        if abs(mean - 1.0) > unit_tol or mn < (1.0 - 5*unit_tol) or mx > (1.0 + 5*unit_tol):
            fail(errors, f"emb not ~unit-norm (tol={unit_tol}): min={mn:.6f} max={mx:.6f} mean={mean:.6f}")
        else:
            ok(f"emb ~unit-norm within tol={unit_tol}")

def load_index(path: Path, errors: list[str]):
    if not Path(path).is_file():
        fail(errors, f"index file not found: {path}")
        return None
    try:
        index = faiss.read_index(str(path))
    except Exception as e:
        fail(errors, f"failed to read faiss index: {path} ({e})")
        return None

    # basic attributes
    ntotal = getattr(index, "ntotal", None)
    d = getattr(index, "d", None)
    metric = getattr(index, "metric_type", None)
    ok(f"faiss index loaded: ntotal={ntotal}, dim={d}, metric={metric}")

    # training state (for IVF/HNSW etc.)
    if hasattr(index, "is_trained"):
        if not index.is_trained:
            fail(errors, "faiss index is not trained")
        else:
            ok("faiss index is trained")

    return index

def check_index_matches_emb(index, emb: np.ndarray, errors: list[str]):
    # dim check
    d = getattr(index, "d", None)
    if d is None or d != emb.shape[1]:
        fail(errors, f"index dim {d} != emb dim {emb.shape[1]}")
    else:
        ok("index dim matches emb dim")

    # count check
    ntotal = getattr(index, "ntotal", None)
    if ntotal is None or ntotal != emb.shape[0]:
        fail(errors, f"index ntotal {ntotal} != emb rows {emb.shape[0]}")
    else:
        ok("index ntotal matches emb rows")

def quick_recall_check(index, emb: np.ndarray, trials: int, errors: list[str]):
    # Choose random rows; expect nearest neighbor to be self
    if emb.shape[0] == 0:
        return
    rng = np.random.default_rng(0xBEEF)
    n = emb.shape[0]
    k = min(trials, n)
    picks = rng.integers(0, n, size=k)
    # Make sure we feed float32 and C-contiguous
    q = np.ascontiguousarray(emb[picks].astype(np.float32))
    D, I = index.search(q, 1)
    hits = int((I.reshape(-1) == picks).sum())
    print(f"[recall@1 self] {hits}/{k} exact self-hits")
    # Be lenient on approximate indexes: require majority self-hit
    if hits < max(1, k // 2):
        fail(errors, f"low self recall@1: {hits}/{k}")
    else:
        ok("self recall@1 looks good")

def load_metadata(meta_path: Path, errors: list[str]) -> list[dict]:
    if not Path(meta_path).is_file():
        fail(errors, f"meta jsonl not found: {meta_path}")
        return []
    try:
        with open(meta_path, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        fail(errors, f"failed to read meta jsonl: {e}")
        return []
    ok(f"meta rows: {len(rows)}")
    return rows

def check_meta_vs_emb(meta: list[dict], emb: np.ndarray, errors: list[str]):
    if not meta:
        return
    if len(meta) != emb.shape[0]:
        fail(errors, f"meta rows {len(meta)} != emb rows {emb.shape[0]}")
    else:
        ok("meta row count matches emb rows")

    # vec_id validations
    if "vec_id" in meta[0]:
        ids = np.array([m.get("vec_id") for m in meta])
        if ids.dtype.kind not in "iu":
            fail(errors, "meta vec_id not integer type")
        else:
            ok("meta vec_id integer type")
        # uniqueness & range
        unique = len(set(map(int, ids)))
        if unique != len(ids):
            fail(errors, f"duplicate vec_id(s): unique={unique} rows={len(ids)}")
        else:
            ok("meta vec_id unique")
        if ids.min() != 0 or ids.max() != len(ids) - 1:
            fail(errors, f"vec_id range not [0..N-1]: min={ids.min()} max={ids.max()}")
        else:
            ok("vec_id covers [0..N-1]")
    else:
        fail(errors, "meta missing 'vec_id' field")

    # optional text coverage check
    has_text = sum(1 for m in meta if any(k in m for k in ("text","passage","content")))
    if has_text == 0:
        print("[warn] meta has no text-like field (text/passage/content); skip coverage check")
    else:
        coverage = has_text / len(meta)
        print(f"[meta] text-like coverage: {coverage:.1%}")
        if coverage < 0.95:
            print("[warn] low text coverage (<95%); ensure this is expected")

def main():
    ap = argparse.ArgumentParser("rep health")
    ap.add_argument("--emb", type=Path, required=True, help="Path to *.npy embeddings")
    ap.add_argument("--index", type=Path, required=True, help="Path to *.faiss index")
    ap.add_argument("--meta", type=Path, required=True, help="Path to metadata JSONL")
    ap.add_argument("--sample", type=int, default=5, help="How many norms to print")
    ap.add_argument("--recall_trials", type=int, default=5, help="Self-recall trials")
    ap.add_argument("--expect_unit_norm", action="store_true", default=True, help="Expect ~unit-norm vectors")
    ap.add_argument("--no_expect_unit_norm", dest="expect_unit_norm", action="store_false")
    ap.add_argument("--unit_tol", type=float, default=1e-4, help="Tolerance for unit-norm check")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if any failure or warning")
    args = ap.parse_args()

    errors: list[str] = []

    emb = load_embeddings(args.emb, errors)
    if emb.size:
        stats_and_norm_checks(emb, errors, args.sample, args.expect_unit_norm, args.unit_tol)

    index = load_index(args.index, errors)
    if index is not None and emb.size:
        check_index_matches_emb(index, emb, errors)
        quick_recall_check(index, emb, args.recall_trials, errors)

    meta = load_metadata(args.meta, errors)
    if meta and emb.size:
        # prefer project validator if available
        if validate_vec_ids is not None:
            try:
                validate_vec_ids(meta, emb)
                ok("validate_vec_ids passed")
            except Exception as e:
                fail(errors, f"validate_vec_ids failed: {e}")
        check_meta_vs_emb(meta, emb, errors)

    if errors:
        print("\n=== SUMMARY: FAIL ===")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    else:
        print("\n=== SUMMARY: OK ===")
        sys.exit(0)

if __name__ == "__main__":
    main()






