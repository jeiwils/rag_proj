# Retrieval-Augmented Graph QA

This project implements a HopRAG-style, multi-hop retrieval-augmented generation (RAG) pipeline that links generated incoming/outgoing questions across passages, builds a passage graph, traverses it with an LLM, and produces final answers and metrics. It supports both graph-based traversal and a dense-retrieval baseline while keeping all intermediate artifacts (IQ/OQ generations, embeddings, graphs, traversal traces, and answers) auditable. 

## Pipeline layout

1. **Dataset preprocessing** – Normalizes raw QA data into `questions.jsonl` and `passages.jsonl`, letting callers provide per-dataset field maps while reusing the shared writer/resume logic.
2. **Question generation & conditioned scoring** – Shards passages by model size, asks LLM servers for conditioned scores plus incoming/outgoing questions (IQ/OQ), and writes per-shard debug logs for baseline/enhanced HopRAG prompts.
3. **Cleaning and explosion** – Merges shard outputs, filters/normalizes IQ/OQ lists, and explodes them into per-question and per-passage JSONLs ready for embedding and graphing.
4. **Dense/sparse representations** – Embeds passages and IQ/OQ items (BGE + FAISS) and enriches them with spaCy keywords; uses GPU acceleration when CUDA is available.
5. **Graph construction** – Connects OQs to candidate IQs with hybrid cosine+Jaccard similarity, applies a global budget, and saves edge lists plus NetworkX graph/diagnostics for each model/dataset/variant.
6. **LLM-guided traversal** – Seeds retrieval, expands through graph edges with an LLM choosing the next hop, and records per-query traces and aggregate traversal metrics for baseline and conditioned-score variants.
7. **Answer generation & reranking** – Loads traversal outputs, fetches supporting passages from the graph, asks the reader LLM for answers, and computes EM/F1 plus token/latency percentiles.
8. **Dense RAG baseline** – Skips graph traversal by retrieving top-k passages directly from FAISS and scoring EM/F1 alongside retrieval metrics for comparison.


## Results (dev splits; averaged across 3 seeds)

Traversal-generated answers (baseline HopRAG) compared with the retrieval-only baseline (dense/FAISS). EM/F1 are averaged over three seeds per model. Separate tables are shown per dataset.【6c388d†L2-L44】

### HotpotQA

| Model | Traversal EM | Traversal F1 | Hybrid retrieval EM | Hybrid retrieval F1 |
| --- | --- | --- | --- | --- |
| deepseek-r1-distill-qwen-14b | 26.3 | 34.5 | 35.0 | 46.9 |
| deepseek-r1-distill-qwen-7b | 26.3 | 33.8 | 9.3 | 23.3 |
| qwen2.5-14b-instruct | 25.0 | 34.0 | 17.3 | 31.9 |
| qwen2.5-2x7b-moe-power-coder-v4 | 23.0 | 32.6 | 3.3 | 10.3 |
| qwen2.5-7b-instruct | 24.7 | 34.1 | 18.3 | 30.9 |
| state-of-the-moe-rp-2x7b | 24.7 | 32.5 | 25.0 | 41.4 |

### Musique

| Model | Traversal EM | Traversal F1 | Hybrid retrieval EM | Hybrid retrieval F1 |
| --- | --- | --- | --- | --- |
| deepseek-r1-distill-qwen-14b | 8.0 | 16.8 | 19.7 | 32.7 |
| deepseek-r1-distill-qwen-7b | 8.3 | 16.5 | 5.0 | 16.8 |
| qwen2.5-14b-instruct | 8.0 | 16.4 | 3.0 | 16.1 |
| qwen2.5-2x7b-moe-power-coder-v4 | 9.7 | 19.1 | 0.7 | 5.0 |
| qwen2.5-7b-instruct | 7.3 | 16.5 | 1.7 | 8.5 |
| state-of-the-moe-rp-2x7b | 7.3 | 15.6 | 6.3 | 23.0 |

### 2WikiMultihopQA

| Model | Traversal EM | Traversal F1 | Hybrid retrieval EM | Hybrid retrieval F1 |
| --- | --- | --- | --- | --- |
| deepseek-r1-distill-qwen-14b | 10.7 | 15.1 | 23.0 | 34.4 |
| deepseek-r1-distill-qwen-7b | 12.0 | 16.4 | 4.0 | 17.9 |
| qwen2.5-14b-instruct | 12.3 | 16.6 | 5.7 | 19.5 |
| qwen2.5-2x7b-moe-power-coder-v4 | 12.3 | 16.4 | 3.0 | 8.8 |
| qwen2.5-7b-instruct | 9.7 | 13.4 | 5.7 | 11.5 |
| state-of-the-moe-rp-2x7b | 12.3 | 16.3 | 22.3 | 31.0 |

## Llama.cpp and CUDA usage

- **LLM serving & determinism** – All generation, traversal edge selection, and answer reading call a local server compatible with llama.cpp’s `/v1/chat/completions` and `/completion` endpoints. Requests forward sampling controls plus optional seeds so llama.cpp responses (including traversal edge choices) are repeatable across runs.
- **GPU-accelerated embeddings** – Passage and IQ/OQ embeddings load the BGE encoder on CUDA when available, accelerating FAISS index construction and search while falling back to CPU otherwise.