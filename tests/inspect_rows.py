def count_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)









print(count_jsonl("data/processed_datasets/hotpotqa/dev/questions.jsonl"))
print(count_jsonl("data/processed_datasets/hotpotqa/dev/passages.jsonl"))

print(count_jsonl("data/processed_datasets/2wikimultihopqa/dev/questions.jsonl"))
print(count_jsonl("data/processed_datasets/2wikimultihopqa/dev/passages.jsonl"))

print(count_jsonl("data/processed_datasets/musique/dev/questions.jsonl"))
print(count_jsonl("data/processed_datasets/musique/dev/passages.jsonl"))









# print(count_jsonl("data/processed/hotpotqa/train.jsonl"))
# print(count_jsonl("data/processed/hotpotqa/train_passages.jsonl"))
# print(count_jsonl("data/processed/hotpotqa/train_passages_1.jsonl"))
# print(count_jsonl("data/processed/hotpotqa/train_passages_2.jsonl"))





# print(count_jsonl("data/processed/2wikimultihopqa/train.jsonl"))
# print(count_jsonl("data/processed/2wikimultihopqa/train_passages.jsonl"))
# print(count_jsonl("data/processed/2wikimultihopqa/train_passages_1.jsonl"))
# print(count_jsonl("data/processed/2wikimultihopqa/train_passages_2.jsonl"))



# print(count_jsonl("data/processed/musique/train.jsonl"))
# print(count_jsonl("data/processed/musique/train_passages.jsonl"))
# print(count_jsonl("data/processed/musique/train_passages_1.jsonl"))
# print(count_jsonl("data/processed/musique/train_passages_2.jsonl"))





 