import json
from pathlib import Path


CLEAN_MISMATCHED_FILES = True  # 🔁 Set to False to disable cleanup


def extract_passage_ids(files):
    ids = set()
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    pid = obj.get("passage_id")
                    if pid is not None:
                        ids.add(pid)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] {file.name}, line {i}: {e}")
    return ids


def remove_extra_passage_ids(file_path, valid_ids):
    """
    Overwrites the file with only valid passage_id rows.
    """
    file_path = Path(file_path)
    temp_path = file_path.with_suffix(".tmp.jsonl")

    kept = 0
    removed = 0

    with open(file_path, "r", encoding="utf-8") as fin, \
         open(temp_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            try:
                obj = json.loads(line)
                pid = obj.get("passage_id")
                if pid in valid_ids:
                    fout.write(json.dumps(obj) + "\n")
                    kept += 1
                else:
                    removed += 1
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode failed at line {i} in {file_path.name}: {e}")

    temp_path.replace(file_path)
    print(f"🧹 Cleaned {file_path.name}: kept {kept}, removed {removed}")


def compare_passage_ids_across_configs(base_dir):
    base_dir = Path(base_dir)

    folders = {
        "baseline_hoprag": base_dir / "baseline_hoprag",
        "enhanced_hoprag": base_dir / "enhanced_hoprag",
    }

    id_sets = {}
    file_lists = {}

    for label, folder in folders.items():
        if label == "enhanced_hoprag":
            # Only *_cs.jsonl and *_iqoq_enhanced.jsonl
            files = sorted(folder.glob("*_cs.jsonl")) + sorted(folder.glob("*_iqoq_enhanced.jsonl"))
        else:
            files = sorted(folder.glob("*.jsonl"))

        if not files:
            print(f"⚠️ No matching files in {label}")
            continue

        print(f"\n📂 Collecting passage_ids from {label} ({len(files)} files)")
        ids = extract_passage_ids(files)
        print(f"   ➤ Total unique passage_ids: {len(ids)}")
        id_sets[label] = ids
        file_lists[label] = files

    # Compare sets
    keys = list(id_sets.keys())
    base_key = keys[0]
    base_ids = id_sets[base_key]
    consistent = True

    print(f"\n🔍 Comparing all sets against: {base_key}")
    for key in keys[1:]:
        current_ids = id_sets[key]
        if base_ids != current_ids:
            consistent = False
            missing = base_ids - current_ids
            extra = current_ids - base_ids
            print(f"❌ {key} does not match {base_key}:")
            if missing:
                print(f"   - Missing {len(missing)} IDs")
            if extra:
                print(f"   - Has {len(extra)} extra IDs")

                # 🔧 CLEAN if enabled
                if CLEAN_MISMATCHED_FILES:
                    print(f"   🔧 Cleaning extra IDs from {key}...")
                    for file in file_lists[key]:
                        remove_extra_passage_ids(file, base_ids)

        else:
            print(f"✅ {key} matches {base_key}")

    if consistent:
        print("\n✅ All folders contain the same passage_id set.")
    else:
        print("\n❌ Some folders had mismatching passage_id sets.")


if __name__ == "__main__":
    compare_passage_ids_across_configs("data/models/qwen-7b/musique/train")
