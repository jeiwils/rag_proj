# make_dirs.py
from pathlib import Path

# fill these with what you need
models   = ["qwen-7b", "deepseek-distill-qwen-7b"]                      # e.g., ["qwen-1.5b", "qwen-7b", "deepseek-distill-qwen-7b"]
datasets = ["hotpotqa", "2wikimultihopqa", "musique"]  # e.g., ["hotpotqa", "musique", "2wikimultihopqa"]
splits   = ["train", "dev"]

BASE = Path("data/models")

def ensure_dirs(models, datasets, splits, base: Path = BASE):
    created = []
    for model in models:
        for dataset in datasets:
            for split in splits:
                p = base / model / dataset / split / "shards"
                p.mkdir(parents=True, exist_ok=True)
                created.append(p)
    return created

if __name__ == "__main__":
    for path in ensure_dirs(models, datasets, splits):
        print(path)
