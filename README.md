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


## Llama.cpp and CUDA usage

- **LLM serving & determinism** – All generation, traversal edge selection, and answer reading call a local server compatible with llama.cpp’s `/v1/chat/completions` and `/completion` endpoints. Requests forward sampling controls plus optional seeds so llama.cpp responses (including traversal edge choices) are repeatable across runs.
- **GPU-accelerated embeddings** – Passage and IQ/OQ embeddings load the BGE encoder on CUDA when available, accelerating FAISS index construction and search while falling back to CPU otherwise.