import json

with open("data/outputs/generated_answers.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10: break  # Show a few examples
        entry = json.loads(line)
        print(f"Q: {entry['question']}")
        print(f"Gold: {entry['gold_answer']}")
        print(f"Generated: {entry['generated_answer']}\n")
