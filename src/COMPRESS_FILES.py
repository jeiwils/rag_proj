from pathlib import Path

datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
split = "train"

for ds in datasets:
    path = Path(f"data/processed_datasets/{ds}/{split}_passages.jsonl.gz")
    print(f"{ds}: {'✅ Exists' if path.exists() else '❌ MISSING'}")
