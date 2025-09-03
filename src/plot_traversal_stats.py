# plot_traversal_stats.py
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def find_result_files(root: Path = Path("data/results")) -> list[Path]:
    """Recursively locate per_query_traversal_results.jsonl under the results directory."""
    return list(root.rglob("per_query_traversal_results.jsonl"))


def load_traversal_stats(paths: list[Path]) -> dict:
    """
    Load traversal logs and collect:
        - n_traversal_calls
        - n_reader_calls
        - candidate counts per hop
        - chosen edge counts per hop
    """
    stats = {
        "n_traversal_calls": [],
        "n_reader_calls": [],
        "hop_candidate_counts": defaultdict(list),  # hop → [candidate edge counts]
        "hop_edges_chosen": defaultdict(list),      # hop → [chosen edge counts]
    }

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                stats["n_traversal_calls"].append(obj.get("n_traversal_calls", 0))
                stats["n_reader_calls"].append(obj.get("n_reader_calls", 0))
                for hop in obj.get("hop_trace", []):
                    hop_id = hop.get("hop", 0)
                    stats["hop_candidate_counts"][hop_id].append(
                        len(hop.get("candidate_edges", []))
                    )
                    stats["hop_edges_chosen"][hop_id].append(
                        len(hop.get("edges_chosen", []))
                    )
    return stats


def plot_distributions(stats: dict, outdir: Path = Path("analysis/plots")) -> None:
    """Generate histograms for call counts and line charts over hops."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Histogram: traversal & reader call counts
    for key in ("n_traversal_calls", "n_reader_calls"):
        data = stats[key]
        if not data:
            continue
        plt.figure()
        plt.hist(data, bins=20, color="skyblue", edgecolor="black")
        plt.title(f"{key} distribution")
        plt.xlabel(key)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(outdir / f"{key}_hist.png")
        plt.close()

    # Line chart: average candidate edges and edges chosen per hop
    hop_ids = sorted(stats["hop_candidate_counts"])
    if hop_ids:
        mean_candidates = [np.mean(stats["hop_candidate_counts"][h]) for h in hop_ids]
        mean_chosen = [np.mean(stats["hop_edges_chosen"][h]) for h in hop_ids]

        plt.figure()
        plt.plot(hop_ids, mean_candidates, marker="o", label="candidate edges")
        plt.plot(hop_ids, mean_chosen, marker="x", label="edges chosen")
        plt.xlabel("Hop")
        plt.ylabel("Average count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "hop_counts.png")
        plt.close()


if __name__ == "__main__":
    files = find_result_files()
    stats = load_traversal_stats(files)
    plot_distributions(stats)

    # Print basic statistics for quick inspection
    summary = {
        "mean_n_traversal_calls": np.mean(stats["n_traversal_calls"]) if stats["n_traversal_calls"] else 0,
        "mean_n_reader_calls": np.mean(stats["n_reader_calls"]) if stats["n_reader_calls"] else 0,
    }
    print(json.dumps(summary, indent=2))
