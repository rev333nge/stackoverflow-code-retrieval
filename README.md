# stackoverflow-code-retrieval

https://github.com/user-attachments/assets/e78e357a-0a1d-4b22-bc6c-23f0e7a0b78a

A code-search engine over Stack Overflow Q&A, built up in stages from a plain BM25 baseline to a hybrid dense/sparse retriever with a learned reranker and a grounded RAG layer on top. Every stage is benchmarked against the one before it on a held-out, time-based test split, using NDCG@10, MRR@10, and Recall@100 against real Stack Overflow signals (accepted answers, scores) as ground truth.

Ships as a desktop app (**PY Retriever**): a FastAPI backend running the retrieval/RAG pipeline locally, wrapped in an Electron + React UI. Generation runs on a local LLM loaded straight from a `.gguf` file you pick (via llama.cpp): no Ollama or other external server to run.

## Results

Corpus scaled from pandas/numpy (537K docs) up to all of Python (3.1M docs); the same pipeline and conclusions hold at both scales. Full test set, Python corpus (56,965 queries):

| Stage | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| BM25 (code-aware tokenizer) | 0.6949 | 0.7105 | 0.9285 |
| + dense retrieval (BGE embeddings) | 0.8129 | 0.8289 | 0.9785 |
| + hybrid fusion (RRF) | 0.8140 | 0.8272 | 0.9855 |
| + learned reranker (LambdaMART) | 0.8397 | 0.8556 | 0.9852 |

Dense retrieval is the single biggest jump: it matches meaning rather than exact terms, so synonyms, typos, and paraphrases that BM25 misses are picked up. BM25 and dense retrieval fail on different queries (correlation ≈ 0.45 between their per-query scores), which is why fusing them and then reranking with a learned model keeps paying off instead of one method just dominating. See [docs/02_results.txt](docs/02_results.txt) and [docs/04_scaling_to_python.txt](docs/04_scaling_to_python.txt) for the full breakdown, including the pandas/numpy-only numbers, hyperparameter sweeps, and a title-ablation study that separates keyword matching from semantic matching.

On top of retrieval, a RAG layer generates synthesized answers from the top reranked results and checks them for grounding with a separate judge model. Across 99 stratified evaluation queries, generated claims were manually checked against their cited sources: about 99% were grounded, with the few hallucinations traced back to a retrieval miss rather than the generator. Retrieval-augmented answers helped most on long-tail questions the base model didn't already know, and added little on common questions it already answered correctly.

## How it works

1. **Ingestion**: Stack Overflow's public data dump (posts, 2008-2024) is filtered by tag, HTML is stripped from question/answer bodies, and documents are assembled as title + answer text.
2. **Ground truth**: relevance judgments are derived from Stack Overflow's own signals: accepted answers, answer score buckets. Queries are split by time (train / validation / test) so the model is never evaluated on data from before it was trained.
3. **Sparse retrieval**: BM25 over a custom tokenizer that keeps code identifiers intact (`np.array`, `read_csv`, `DataFrame`) while also emitting their parts, so both exact-symbol and partial-keyword matches work.
4. **Dense retrieval**: documents and queries are embedded with a sentence-transformer model and searched with FAISS (or a raw GPU matmul at larger scale, where `faiss-gpu` isn't available).
5. **Fusion**: Reciprocal Rank Fusion merges the BM25 and dense result lists by rank rather than raw score, since the two scales aren't comparable.
6. **Learning-to-rank**: a LambdaMART model (LightGBM) reranks the fused candidate pool using retrieval scores/ranks from both methods as features, trained on a pairwise ranking loss tuned toward NDCG@10.
7. **RAG + grounding**: the top reranked answers are passed to a local LLM to generate a synthesized answer with structured, per-claim citations, then a separate judge model checks each claim against its cited source.

## Project layout

```
scripts/    pipeline stages: ingestion, indexing, retrieval, evaluation, RAG
app/        FastAPI backend serving the trained pipeline to the desktop app
desktop/    Electron + React frontend
docs/       design notes, full results, and failure analysis
data/       corpus, indexes, and trained models (gitignored)
```

## Running the pipeline

```bash
pip install -r requirements.txt

python scripts/download_posts.py
python scripts/filter_posts.py
python scripts/clean_text.py
python scripts/build_corpus.py
python scripts/build_qrels.py
python scripts/split_queries.py
python scripts/build_index.py a
python scripts/build_embeddings.py a
python scripts/build_ltr_features.py a train
python scripts/build_ltr_features.py a val
python scripts/build_ltr_features.py a test
python scripts/train_ltr.py a
python scripts/evaluate.py ltr a test
```

Each script is documented at the top of its file; `scripts/evaluate.py` and `scripts/search_demo_ltr.py` are the quickest way to check a build without re-running everything.

## Running the desktop app

```bash
cd desktop
npm install
npm run dev
```

`npm run dev` starts the Vite UI and the Electron shell, and Electron spawns the FastAPI backend itself (bound to localhost only), so this one command runs the whole app. To watch backend logs or use auto-reload during development, run the backend separately instead and it will use that one:

```bash
python -m uvicorn app.server:app --host 127.0.0.1 --port 8756 --reload
```

Once it's up, pick a model with the browse dialog (any local `.gguf`, e.g. from [Hugging Face](https://huggingface.co/models?library=gguf)); the app ships no LLM. `llama-cpp-python` must be installed with a CUDA-enabled build for GPU offload; see the comment in `requirements.txt`.

Each conversation carries its own model and generation settings, adjustable from the gear in the top bar:

- **Temperature** and **max tokens**: take effect on the next message.
- **Context size** (`n_ctx`): a slider bounded by the model's own trained maximum (read from its GGUF metadata). It sets how much of the conversation the model keeps in context: history is packed in up to this budget (tokenized with the actual model), so a bigger window means a longer memory. A ring in the top bar shows how full the window was on the last reply.
- **GPU layers**: how much of the model to offload to the GPU; lower it to fit a large model in limited VRAM.

Changing context size or GPU layers reloads the model (a progress bar covers the load); the other two are free.

The adaptive RAG flow decides per message whether to answer directly or search: the model routes the query, and only when it chooses to search does it retrieve, rerank, and answer grounded in the top Stack Overflow answers (shown as clickable sources), gated by a retrieval-confidence check so off-topic questions fall back to the model's own knowledge.

## Stack

Python, DuckDB, BM25s, FAISS, sentence-transformers, LightGBM, FastAPI, llama.cpp, React, Electron.
