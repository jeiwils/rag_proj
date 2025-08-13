import pandas as pd
import json
from pathlib import Path

# === CONFIG ===
model_name = "deepseek-distill-qwen-7b"
dataset_name = "musique"
files = ["train_passages_part1_7b_scored.jsonl", "train_passages_part2_7b_scored.jsonl"]




base_dir = Path(fr"C:\Users\jeiwi\rag_proj\data\models\{model_name}\{dataset_name}\enhanced_hoprag")


metrics_dir = Path(r"C:\Users\jeiwi\rag_proj\data\metrics\CS")
metrics_dir.mkdir(parents=True, exist_ok=True)

output_file = metrics_dir / "combined_cs_metrics.jsonl"


def load_jsonl_safely(path):
    good_rows, bad_rows = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                good_rows.append(json.loads(line))
            except Exception:
                bad_rows.append((i, line.strip()[:200]))
    return pd.DataFrame(good_rows), bad_rows







def main():
    all_counts = pd.Series(dtype=float)
    total_entries = 0
    total_bad = 0

    for fname in files:
        file_path = base_dir / fname
        df, bad = load_jsonl_safely(file_path)
        total_bad += len(bad)

        if "conditioned_score" in df.columns:
            df["conditioned_score"] = pd.to_numeric(df["conditioned_score"], errors="coerce")
            counts = df["conditioned_score"].value_counts(dropna=False)
            all_counts = all_counts.add(counts, fill_value=0)  # combine counts
            total_entries += len(df)
        else:
            print(f"Warning: No 'conditioned_score' column in {fname}")

    # sort by score value
    all_counts = all_counts.sort_index()
    proportions = all_counts / total_entries
    percentages = (proportions * 100).round(2)

    # build record for appending
    summary = {
        "generation_model": model_name,
        "dataset_name": dataset_name,
        "prompt_used": "original (long) CS prompt",
        "total_entries": int(total_entries),
        "total_bad_lines": int(total_bad),
        "counts": {str(k): int(v) for k, v in all_counts.items()},
        "proportions": {str(k): round(v, 6) for k, v in proportions.items()},
        "percentages": {str(k): float(v) for k, v in percentages.items()}
    }





    # append to JSONL (machine-friendly)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # also maintain a pretty JSON snapshot (human-friendly)
    pretty_file = metrics_dir / "combined_cs_metrics_pretty.json"
    if pretty_file.exists() and pretty_file.stat().st_size > 0:
        with open(pretty_file, "r", encoding="utf-8") as f:
            buf = json.load(f)
            if not isinstance(buf, list):
                buf = [buf]
    else:
        buf = []

    buf.append(summary)
    with open(pretty_file, "w", encoding="utf-8") as f:
        json.dump(buf, f, ensure_ascii=False, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
