import os
import json

def inspect_dataset(path, n=3):
    print(f"\n📄 Inspecting: {path}")
    is_jsonl = path.endswith(".jsonl")
    
    examples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            if is_jsonl:
                for i, line in enumerate(f):
                    if i >= n:
                        break
                    examples.append(json.loads(line))
            else:
                data = json.load(f)
                examples = data[:n]
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return

    for i, ex in enumerate(examples):
        print(f"\n🔹 Example {i + 1}:")
        for k, v in ex.items():
            typename = type(v).__name__
            preview = str(v)[:80].replace("\n", " ")
            print(f"  - {k} ({typename}): {preview}")
        print("  Keys:", list(ex.keys()))


def inspect_all(base_dir="rag_proj/data/datasets"):
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name.endswith(".json") or name.endswith(".jsonl"):
                inspect_dataset(os.path.join(root, name))


if __name__ == "__main__":
    inspect_all("C:/Users/jeiwi/rag_proj/data/datasets")
