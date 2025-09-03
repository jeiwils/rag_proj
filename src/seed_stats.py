from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from metrics import compare_runs


SEED_REGEX = re.compile(r"seed(\d+)")


def _load_summary(path: Path) -> tuple[int | None, dict]:
    """Load a summary JSON file and extract seed and metrics."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seed = None
    match = SEED_REGEX.search(path.stem)
    if match:
        seed = int(match.group(1))
    elif isinstance(data.get("seed"), int):
        seed = data["seed"]

    metrics = {
        "EM": data.get("EM") or data.get("em"),
        "F1": data.get("F1") or data.get("f1"),
        "tokens": data.get("tokens") or data.get("total_tokens"),
        "wall_time": data.get("wall_time") or data.get("wall_time_total_sec"),
        "trav_tokens": data.get("trav_tokens_total"),
        "reader_tokens": data.get("reader_total_tokens"),
        "t_total_ms": data.get("t_total_ms"),
        "tps_overall": data.get("tps_overall"),
    }

    usage_path = path.parent / "token_usage.json"
    if usage_path.exists():
        with open(usage_path, "r", encoding="utf-8") as f:
            usage_data = json.load(f)
        if isinstance(usage_data, dict):
            if "global" in usage_data:
                usage_data = usage_data.get("global", {})

            trav_tok = usage_data.get("trav_tokens_total")
            metrics.setdefault("trav_tokens", trav_tok)
            reader_tok = usage_data.get("reader_total_tokens")
            metrics.setdefault("reader_tokens", reader_tok)

            t_total_ms = usage_data.get("t_total_ms")
            if t_total_ms is None:
                t_total_ms = usage_data.get("t_traversal_ms", 0) + usage_data.get("t_reader_ms", 0)
            metrics.setdefault("t_total_ms", t_total_ms)

            tokens_total = usage_data.get("tokens_total")
            if tokens_total is None:
                totals = [trav_tok, reader_tok]
                totals = [t for t in totals if t is not None]
                tokens_total = sum(totals) if totals else None
            metrics.setdefault("tokens", tokens_total)

            tps_overall = usage_data.get("tps_overall")
            if tps_overall is None and tokens_total is not None and t_total_ms:
                tps_overall = tokens_total / (t_total_ms / 1000)
            metrics.setdefault("tps_overall", tps_overall)

    # Attempt to collect per-query metrics for statistical tests
    per_query_em: list[float] = []
    per_query_f1: list[float] = []
    if "per_query_em" in data and "per_query_f1" in data:
        per_query_em = list(map(float, data["per_query_em"]))
        per_query_f1 = list(map(float, data["per_query_f1"]))
    elif isinstance(data.get("per_query"), dict):
        for q in data["per_query"].values():
            per_query_em.append(float(q.get("EM") or q.get("em") or 0.0))
            per_query_f1.append(float(q.get("F1") or q.get("f1") or 0.0))
    metrics["per_query_em"] = per_query_em
    metrics["per_query_f1"] = per_query_f1

    return seed, metrics


def main(base_dir: str) -> None:
    base = Path(base_dir)
    summaries = sorted(base.rglob("*_seed*.json"))

    per_seed: dict[int, dict] = {}
    for summary in summaries:
        seed, metrics = _load_summary(summary)
        if seed is not None:
            per_seed[seed] = metrics

    if not per_seed:
        print("No seed summaries found in", base)
        return

    # Compute average, variance, and standard deviation across seeds
    mean: dict[str, float] = {}
    variance: dict[str, float] = {}
    std_dev: dict[str, float] = {}
    for key in (
        "EM",
        "F1",
        "tokens",
        "trav_tokens",
        "reader_tokens",
        "wall_time",
        "tps_overall",
    ):
        values = [m[key] for m in per_seed.values() if m.get(key) is not None]
        if values:
            arr = np.array(values, dtype=float)
            mean[key] = float(np.mean(arr))

            variance[key] = float(np.var(arr))
            std_dev[key] = float(np.std(arr))

    # Pairwise Wilcoxon tests using existing metrics utility
    wilcoxon: dict[str, dict[str, float]] = {}
    seeds = sorted(per_seed)
    for a, b in combinations(seeds, 2):
        pair_key = f"{a}_vs_{b}"
        wilcoxon[pair_key] = {}
        for metric_key, out_key in (
            ("per_query_em", "EM_p_value"),
            ("per_query_f1", "F1_p_value"),
        ):
            arr_a = per_seed[a].get(metric_key)
            arr_b = per_seed[b].get(metric_key)
            if arr_a and arr_b and len(arr_a) == len(arr_b):
                with NamedTemporaryFile("w", delete=False) as fa, NamedTemporaryFile(
                    "w", delete=False
                ) as fb:
                    fa.write("\n".join(str(x) for x in arr_a))
                    fb.write("\n".join(str(x) for x in arr_b))
                    fa.flush()
                    fb.flush()
                    summary = compare_runs(fa.name, fb.name)
                wilcoxon[pair_key][out_key] = summary.get("p_value")

    output = {
        "per_seed": {
            str(seed): {k: v for k, v in m.items() if not k.startswith("per_query")}
            for seed, m in per_seed.items()
        },
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "wilcoxon": wilcoxon,
    }

    out_path = base / "seed_variance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-seed variance and Wilcoxon tests from result summaries",
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="data/results",
        help="Base directory containing *_seed*.json files",
    )
    args = parser.parse_args()
    main(args.base_dir)