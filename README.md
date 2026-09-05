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
                              Top-5 chunks → Groq Llama 3.3 70B
                                        │
                                        ▼
                              Grounded, cited answer
```

| Stage | Choice | Why |
|---|---|---|
| Chunking | Token-based recursive splitter | Aligned to the embedding model's actual tokenizer — avoids mid-word cuts |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, no API dependency |
| Vector index | FAISS `IndexFlatIP` (normalized) | Exact search — honest choice at this corpus scale, avoids ANN approximation error |
| Keyword search | `rank_bm25` | Simple, in-memory, standard BM25 |
| Fusion | Reciprocal Rank Fusion (k=60) | Scale-independent — sidesteps BM25/cosine score incomparability |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Joint query-document scoring, corrects first-stage retrieval mistakes |
| Generation | Groq Llama 3.3 70B | Free tier, fast inference |
| Eval judge | `openai/gpt-oss-120b` (via Groq) | Chosen independently from the generation model for stronger structured-output/tool-calling compliance, which RAGAS's judge metrics depend on |

## Evaluation

Benchmarked with RAGAS across 12 hand-curated questions against a live ~200-paper arXiv (`cs.CL`) corpus, including deliberate **trap questions** — clusters of multiple papers on near-identical sub-topics (e.g. three separate on-policy-distillation papers) — to stress-test whether reranking actually discriminates between similar-but-distinct content, not just topic-matches.

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.91 | Does every claim in the answer trace back to retrieved context? |
| Context Precision | 0.87 | Of what was retrieved, how much was actually relevant? |
| Context Recall | 0.86 | Of what was needed, how much did we retrieve? |
| Answer Relevancy | 0.64 | Does the answer directly address the question asked? |

**Key finding**: generation stayed well-grounded (high faithfulness) and retrieval found the right material (high precision/recall), but answer relevancy lagged — likely because the citation-heavy, caveat-including prompt style that maximizes faithfulness also produces longer, less tightly-scoped answers, which RAGAS's relevancy metric penalizes. This is a real, known tension between optimizing faithfulness and optimizing relevancy in RAG prompt design.

One question scored Context Precision 1.0 / Context Recall 0.0 simultaneously — retrieval found genuinely on-topic material, but a reference requiring several distinct numeric claims spread across a dense abstract wasn't fully covered by any single retrieved chunk. Points at a chunking granularity issue for claim-dense source text, not a retrieval relevance issue.

## Engineering notes worth knowing

- **Real dependency conflict, fixed**: the current `ragas` release has a broken dependency declaration against the latest `langchain-community` (an import for a removed `vertexai` submodule). Pinned `langchain-community==0.3.31` to resolve it — verified by reproducing the failure in a clean venv before pinning.
- **Async-native evaluation**: RAGAS's metric classes are async under the hood (faithfulness alone issues multiple judge calls per question — decomposing the answer into claims, then verifying each one). Used `AsyncOpenAI` + `await .ascore(...)` directly rather than the sync wrapper, avoiding event-loop mismatches with the `instructor`-based structured-output adapter.
- **Rate-limit resilience**: the eval runner checkpoints results to disk after every question, so a Groq free-tier rate limit mid-run doesn't lose completed work — rerunning resumes exactly where it left off.

## Project structure

```
hybrid-rag-ragas/
├── data/{raw,processed}/    # fetched corpus, chunks, built indexes
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
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env

python src/ingest.py       # fetch corpus
python src/chunking.py     # chunk it
python src/indexing.py     # build BM25 + FAISS indexes
python src/pipeline.py "your question here"   # ask it something
python eval/run_ragas.py   # run the full RAGAS benchmark
```