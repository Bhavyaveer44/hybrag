# Hybrid RAG with RAGAS Evaluation

A retrieval-augmented generation system combining **BM25 (lexical)** and **dense vector search (semantic)**, fused via **Reciprocal Rank Fusion**, refined with a **cross-encoder reranker**, and rigorously benchmarked with **RAGAS** — because most RAG projects skip evaluation entirely, and this one doesn't.

## Why hybrid retrieval

Pure vector search is the standard tutorial approach, but it systematically underweights exact identifiers, rare tokens, and numerical values — embeddings optimize for semantic compression, not lexical precision. BM25 patches exactly that blind spot. This system runs both retrievers independently and fuses their rankings with **Reciprocal Rank Fusion (RRF)**, rather than naively averaging incomparable similarity scores.

## Architecture

```
Query
  ├─→ BM25 (rank_bm25)         ─┐
  └─→ FAISS vector search       ├─→ RRF fusion → shortlist (top-20)
      (sentence-transformers)  ─┘
                                        │
                                        ▼
                        Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
                                        │
                                        ▼
                    Focused chunks + neighbor expansion
                       (1,200-token context budget)
                                        │
                                        ▼
                              Groq Qwen 3.8 27B
                                        │
                                        ▼
                              Grounded, cited answer
```

| Stage | Choice | Why |
|---|---|---|
| Chunking | Section-aware recursive splitter (or fixed recursive experiment via `--strategy fixed`) | Preserves paper/section/chunk identity and avoids mid-word cuts |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, no API dependency |
| Vector index | FAISS `IndexFlatIP` (normalized) | Exact search — honest choice at this corpus scale, avoids ANN approximation error |
| Keyword search | `rank_bm25` | Simple, in-memory, standard BM25 |
| Fusion | Reciprocal Rank Fusion (k=60) | Scale-independent — sidesteps BM25/cosine score incomparability |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Joint query-document scoring, corrects first-stage retrieval mistakes |
| Generation | Groq `qwen/qwen3.8-27b` | Separate from the evaluation judge and configured for grounded answers |
| Eval judge | `openai/gpt-oss-120b` (via Groq) | Chosen independently from the generation model for stronger structured-output/tool-calling compliance, which RAGAS's judge metrics depend on |

## Evaluation

Benchmarked with RAGAS across 12 hand-curated questions against a live ~200-paper arXiv (`cs.CL`) corpus, including deliberate **trap questions** — clusters of multiple papers on near-identical sub-topics (e.g. three separate on-policy-distillation papers) — to stress-test whether reranking actually discriminates between similar-but-distinct content, not just topic-matches.

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.958 | Does every claim in the answer trace back to retrieved context? |
| Answer Relevancy | 0.762 | Does the answer directly address the question asked? |
| Context Precision | 0.906 | Of what was retrieved, how much was actually relevant? |
| Context Recall | 1.000 | Of what was needed, how much did we retrieve? |

These are the latest scores after section-aware chunk metadata, neighbor expansion, and a 1,200-token context budget were enabled; the pipeline and scoring checkpoints let the run resume without repeating completed API calls. Compare with earlier runs only when the corpus, generated answers, and evaluation setup are held constant.

**Current finding**: context recall reached 1.000 on this 12-question set, while answer relevancy at 0.762 remains the lowest metric. This suggests all reference information was present in retrieved context for these questions, but some generated answers could address the question more directly. Treat the scores as a small curated evaluation, not a general performance guarantee.

## Engineering notes worth knowing

- **Real dependency conflict, fixed**: the current `ragas` release has a broken dependency declaration against the latest `langchain-community` (an import for a removed `vertexai` submodule). Pinned `langchain-community==0.3.31` to resolve it — verified by reproducing the failure in a clean venv before pinning.
- **Async-native evaluation**: RAGAS's metric classes are async under the hood (faithfulness alone issues multiple judge calls per question — decomposing the answer into claims, then verifying each one). Used `AsyncOpenAI` + `await .ascore(...)` directly rather than the sync wrapper, avoiding event-loop mismatches with the `instructor`-based structured-output adapter.
- **Rate-limit resilience**: the eval runner checkpoints results to disk after every question, so a Groq free-tier rate limit mid-run doesn't lose completed work — rerunning resumes exactly where it left off.

## Project structure

```
hybrid-rag-ragas/
├── data/{raw,processed}/    # fetched corpus, chunks, built indexes (generated; ignored by Git)
├── src/
│   ├── ingest.py            # arXiv corpus fetching
│   ├── chunking.py          # token-aware recursive splitting
│   ├── indexing.py          # BM25 + FAISS index construction
│   ├── retrieval.py         # hybrid search + RRF fusion
│   ├── reranker.py          # cross-encoder reranking
│   ├── generate.py          # grounded LLM generation
│   └── pipeline.py          # end-to-end entrypoint
├── eval/
│   ├── eval_dataset.py      # curated Q&A pairs (incl. trap questions)
│   └── run_ragas.py         # RAGAS scoring harness (async, checkpointed)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows Command Prompt; PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env

python src\ingest.py       # fetch corpus
python src\chunking.py     # chunk it
python src\indexing.py     # build BM25 + FAISS indexes
python src\pipeline.py "your question here"   # ask it something
python eval\run_ragas.py   # run the full RAGAS benchmark
```

## Chunking and context experiments

`python src\chunking.py --strategy section` uses headings/structured sections when available and falls back to the abstract for the current arXiv corpus. Compare it with `python src\chunking.py --strategy fixed`; both accept `--chunk-size` and `--overlap`. Rebuild chunks and indexes after changing strategy or chunk parameters. Each chunk records stable paper, section, parent, and neighbor IDs. At query time the reranker selects focused chunks and expands adjacent chunks by default; callers can choose `none`, `parent`, `neighbors`, or `parent_and_neighbors` and set `context_token_budget` on `HybridRAGPipeline.answer` (default 1,200 embedding-tokenizer tokens). Expansion deduplicates identical text and prioritizes the focused hits within the budget.

For a clean benchmark on a new chunking/context configuration, delete both `eval\pipeline_results_checkpoint.json` (generated answers) and `eval\ragas_results_checkpoint.json` (scores) before running `python eval\run_ragas.py`; otherwise prior answers or scores may be reused. These local result files are not tracked by Git. The scorer resumes completed questions after a rate limit; a non-rate-limit metric failure is reported and retried on the next run. HF Hub authentication is optional for model download rate limits.
