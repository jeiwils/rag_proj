import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types
sys.modules.setdefault("requests", types.ModuleType("requests"))
tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = lambda x, **kwargs: x
sys.modules.setdefault("tqdm", tqdm_module)
prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "prompts"))
os.makedirs(prompt_dir, exist_ok=True)
for fname in [
    "7b_CS_prompt_ultra_updated.txt",
    "7b_IQ_prompt_updated.txt",
    "7b_OQ_prompt_updated.txt",
    "hoprag_iq_prompt.txt",
    "hoprag_oq_prompt.txt",
]:
    path = os.path.join(prompt_dir, fname)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
from src.a2_text_prep import existing_ids, compute_resume_sets

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def test_existing_ids(tmp_path):
    path = tmp_path / "shard.jsonl"
    write_jsonl(path, [{"passage_id": 1}, {"passage_id": 2}])
    assert existing_ids(path) == {1, 2}

def test_compute_resume_sets_multiple_shards(tmp_path, capsys):
    shard1 = tmp_path / "s1.jsonl"
    shard2 = tmp_path / "s2.jsonl"
    write_jsonl(shard1, [{"passage_id": 0}, {"passage_id": 1}])
    write_jsonl(shard2, [{"passage_id": 3}, {"passage_id": 4}, {"passage_id": 5}])

    items1 = [{"passage_id": i} for i in range(2)]
    items2 = [{"passage_id": i} for i in range(3, 8)]

    done1, shard_ids1 = compute_resume_sets(
        resume=True,
        out_path=str(shard1),
        items=items1,
        get_id=lambda x, i: x["passage_id"],
        phase_label="s1",
    )
    msg1 = capsys.readouterr().out.strip()

    done2, shard_ids2 = compute_resume_sets(
        resume=True,
        out_path=str(shard2),
        items=items2,
        get_id=lambda x, i: x["passage_id"],
        phase_label="s2",
    )
    msg2 = capsys.readouterr().out.strip()

    assert msg1 == "[resume] s1: 2/2 already present in this shard – skipping those"
    assert msg2 == "[resume] s2: 3/5 already present in this shard – skipping those"
    assert done1 == {0, 1} and shard_ids1 == {0, 1}
    assert done2 == {3, 4, 5} and shard_ids2 == {3, 4, 5, 6, 7}