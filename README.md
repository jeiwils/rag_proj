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

| Model | Traversal F1 | Traversal APT | Hybrid F1 |
| --- | --- | --- | --- |
| deepseek-r1-distill-qwen-14b | 26.66 | 25,698 | 45.58 |
| deepseek-r1-distill-qwen-7b | 27.07 | 25,533 | 21.86 |
| qwen2.5-14b-instruct | 27.03 | 25,343 | 30.37 |
| qwen2.5-2x7b-moe-power-coder-v4 | 27.21 | 25,572 | 7.49 |
| qwen2.5-7b-instruct | 28.16 | 26,382 | 29.75 |
| state-of-the-moe-rp-2x7b | 25.02 | 25,325 | 30.55 |

### Musique

| Model | Traversal F1 | Traversal APT | Hybrid F1 |
| --- | --- | --- | --- |
| deepseek-r1-distill-qwen-14b | 16.62 | 60,136 | 30.77 |
| deepseek-r1-distill-qwen-7b | 16.20 | 59,566 | 17.42 |
| qwen2.5-14b-instruct | 14.71 | 60,081 | 14.73 |
| qwen2.5-2x7b-moe-power-coder-v4 | 20.40 | 59,456 | 6.68 |
| qwen2.5-7b-instruct | 16.68 | 58,562 | 9.59 |
| state-of-the-moe-rp-2x7b | 15.67 | 58,708 | 19.34 |

### 2WikiMultihopQA

| Model | Traversal F1 | Traversal APT | Hybrid F1 |
| --- | --- | --- | --- |
| deepseek-r1-distill-qwen-14b | 14.48 | 21,602 | 35.30 |
| deepseek-r1-distill-qwen-7b | 17.62 | 21,740 | 17.39 |
| qwen2.5-14b-instruct | 18.84 | 22,236 | 21.45 |
| qwen2.5-2x7b-moe-power-coder-v4 | 16.09 | 21,671 | 10.28 |
| qwen2.5-7b-instruct | 13.74 | 22,045 | 16.27 |
| state-of-the-moe-rp-2x7b | 17.92 | 21,819 | 16.42 |

## Llama.cpp and CUDA usage

- **LLM serving & determinism** – All generation, traversal edge selection, and answer reading call a local server compatible with llama.cpp’s `/v1/chat/completions` and `/completion` endpoints. Requests forward sampling controls plus optional seeds so llama.cpp responses (including traversal edge choices) are repeatable across runs.
- **GPU-accelerated embeddings** – Passage and IQ/OQ embeddings load the BGE encoder on CUDA when available, accelerating FAISS index construction and search while falling back to CPU otherwise.