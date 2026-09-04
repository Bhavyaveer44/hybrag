"""
Takes hybrid-retrieved shortlist (from RRF fusion) and reranks it with a cross-encoder
that scores (query, document) pairs together rather than comparing precomputed independent vectors.

retrieval(BM25 + FAISS + RRF) optimizes for speed across
whole corpus and returns "probably relevant" candidates.
reranker trades speed for accuracy on just that shortlist.
"""

from sentence_transformers import CrossEncoder

from retrieval import HybridRetriever, TOP_N_PER_RETRIEVER

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankedRetriever:
    def __init__(self, hybrid_retriever: HybridRetriever = None):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    def retrieve(self, query: str, shortlist_size: int = 20, final_k: int = 5) -> list[dict]:
        """
        1. Get shortlist from hybrid (RRF-fused) retrieval wider than
           what will actually be used, so the reranker has real candidates to
           discriminate between.
        2. Score every (query, candidate) pair jointly with the cross-encoder.
        3. Re-sort by that score and keep the top final_k.
        """
        candidates = self.hybrid_retriever.retrieve(query, top_k=shortlist_size)
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)  # one joint forward pass per pair

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:final_k]


if __name__ == "__main__":
    retriever = RerankedRetriever()
    query = "how does hybrid search improve retrieval on technical corpora"
    results = retriever.retrieve(query, shortlist_size=TOP_N_PER_RETRIEVER, final_k=5)
    for r in results:
        print(f"[rerank={r['rerank_score']:.4f}  rrf={r['rrf_score']:.4f}] "
              f"{r['title']} — {r['text'][:100]}...")