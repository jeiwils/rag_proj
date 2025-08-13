
#### needs updating with new datasets







import shutil
from pathlib import Path

cache_root = Path.home() / ".cache" / "huggingface" / "hub"

keep = {
    # Existing
    "models--roberta-large-mnli",
    "models--sentence-transformers--all-MiniLM-L6-v2",

    # DeepSeek Distilled Qwen (q8 GGUF)
    "models--bartowski--DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    "models--bartowski--DeepSeek-R1-Distill-Qwen-7B-GGUF",
    "models--bartowski--DeepSeek-R1-Distill-Qwen-14B-GGUF",

    # Qwen2.5 dense GGUF (q8, non-instruct)
    "models--QuantFactory--Qwen2.5-1.5B-GGUF",
    "models--QuantFactory--Qwen2.5-7B-GGUF",
    "models--QuantFactory--Qwen2.5-14B-GGUF",

    # Datasets
    "datasets--hotpot_qa",
    "datasets--dgslibisey--MuSiQue",
    "datasets--pubmed_qa",
    "datasets--domenicrosati--TruthfulQA",
    "datasets--truthful_qa",
    "datasets--fever",
    "datasets--wikipedia",
}

# Delete folders not in the keep list
for item in cache_root.iterdir():
    if item.is_dir() and item.name not in keep:
        print(f"🗑️ Deleting: {item.name}")
        shutil.rmtree(item)

# Categorize remaining cache entries
models = []
datasets = []
others = []

for item in sorted(cache_root.iterdir()):
    if item.is_dir():
        if item.name.startswith("models--"):
            models.append(item.name)
        elif item.name.startswith("datasets--"):
            datasets.append(item.name)
        else:
            others.append(item.name)

def get_dir_size(path: Path) -> float:
    """Return total size of directory (in GB)."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024**3)  # Convert bytes to GB

# Print remaining cache contents
print("\n✅ Remaining model cache:")
for name in models:
    print(f"  📦 {name}")

print("\n✅ Remaining dataset cache:")
for name in datasets:
    print(f"  📚 {name}")

if others:
    print("\n⚠️ Other entries not classified as model or dataset:")
    for name in others:
        print(f"  ❓ {name}")

cache_size_gb = get_dir_size(cache_root)
print(f"\n💾 Total Hugging Face cache size: {cache_size_gb:.2f} GB")
