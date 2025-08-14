import json
from pathlib import Path

def inspect_json_columns(file_path):
    """
    Inspect column (key) names in a JSON or JSONL file,
    and print one example row.
    """
    path = Path(file_path)
    keys = set()
    example_row = None

    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            keys.update(obj.keys())
                            if example_row is None:
                                example_row = obj
        else:  # .json
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for r in data:
                    if isinstance(r, dict):
                        keys.update(r.keys())
                        if example_row is None:
                            example_row = r
            elif isinstance(data, dict):
                keys.update(data.keys())
                example_row = data
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return

    print(f"\n📄 {path.name} ({len(keys)} keys)")
    print("   Keys:", sorted(keys))
    if example_row is not None:
        print("   Example row:")
        print(json.dumps(example_row, indent=2, ensure_ascii=False))
    else:
        print("No example row found.")



if __name__ == "__main__":





    # inspect_json_columns("data/processed_datasets/hotpotqa/train/questions.jsonl")
    # inspect_json_columns("data/processed_datasets/hotpotqa/train/passages.jsonl")

    # inspect_json_columns("data/processed_datasets/2wikimultihopqa/train/questions.jsonl")
    # inspect_json_columns("data/processed_datasets/2wikimultihopqa/train/passages.jsonl")

    # inspect_json_columns("data/processed_datasets/musique/train/questions.jsonl")
    # inspect_json_columns("data/processed_datasets/musique/train/passages.jsonl")


    inspect_json_columns("data/raw_datasets/hotpotqa/hotpot_train_v1.1.json")
    #inspect_json_columns("data/raw_datasets/hotpotqa/hotpot_dev_distractor_v1.json")

    inspect_json_columns("data/raw_datasets/2wikimultihopqa/train.json")
    #inspect_json_columns("data/raw_datasets/2wikimultihopqa/dev.json")

    inspect_json_columns("data/raw_datasets/musique/musique_ans_v1.0_train.jsonl")
    #inspect_json_columns("data/raw_datasets/musique/musique_ans_v1.0_dev.jsonl")
