"""
pipeline.py
End-to-end entrypoint: query -> hybrid retrieve -> rerank -> generate.
This is the single interface a demo or a caller should use -- it
hides the internal staging (RRF fusion, cross-encoder reranking,
prompt construction) behind one method.

Usage:
    python src/pipeline.py "your question here"
"""

import sys

from generate import generate_answer
from reranker import RerankedRetriever


class HybridRAGPipeline:
    def __init__(self):
        self.retriever = RerankedRetriever()

    def answer(self, query: str, shortlist_size: int = 20, final_k: int = 5,
               expansion: str = "neighbors", context_token_budget: int = 1200) -> dict:
        """
        Runs the full pipeline for one query. Returns the answer plus
        the sources actually used, so a caller can display citations
        or log them for later evaluation.
        """
        chunks = self.retriever.retrieve(query, shortlist_size=shortlist_size, final_k=final_k,
                                         expansion=expansion, context_token_budget=context_token_budget)
        answer_text = generate_answer(query, chunks)
        return {
            "query": query,
            "answer": answer_text,
            "sources": [
                {"title": c["title"], "url": c["url"], "paper_id": c.get("paper_id"),
                 "chunk_id": c.get("chunk_id"), "section": c.get("section"),
                 "context_role": c.get("context_role"), "rerank_score": c.get("rerank_score")}
                for c in chunks
            ],
        }


def print_result(result: dict):
    print(f"\nQ: {result['query']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for i, s in enumerate(result["sources"], start=1):
        print(f"  [{i}] {s['title']} (score={s['rerank_score']:.3f})")
        print(f"      {s['url']}")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "How does hybrid search improve retrieval on technical corpora?"

    pipeline = HybridRAGPipeline()
    result = pipeline.answer(query)
    print_result(result)
