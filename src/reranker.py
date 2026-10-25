"""
Takes hybrid-retrieved shortlist (from RRF fusion) and reranks it with a cross-encoder
that scores (query, document) pairs together rather than comparing precomputed independent vectors.

retrieval(BM25 + FAISS + RRF) optimizes for speed across
whole corpus and returns "probably relevant" candidates.
reranker trades speed for accuracy on just that shortlist.
"""

from sentence_transformers import CrossEncoder

from retrieval import HybridRetriever, TOP_N_PER_RETRIEVER
from chunking import token_len

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CONTEXT_TOKEN_BUDGET = 1200


class RerankedRetriever:
    def __init__(self, hybrid_retriever: HybridRetriever = None):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    def retrieve(self, query: str, shortlist_size: int = 20, final_k: int = 5,
                 expansion: str = "neighbors", context_token_budget: int = CONTEXT_TOKEN_BUDGET) -> list[dict]:
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
        focused = candidates[:final_k]
        return self._assemble_context(focused, expansion, context_token_budget)

    def _assemble_context(self, focused: list[dict], expansion: str,
                          token_budget: int) -> list[dict]:
        """Expand focused hits with parents/neighbors, dedup, then fit the token budget."""
        if expansion not in {"none", "parent", "neighbors", "parent_and_neighbors"}:
            raise ValueError("expansion must be none, parent, neighbors, or parent_and_neighbors")
        if token_budget <= 0:
            return []

        chunks_by_id = {c.get("chunk_id"): c for c in self.hybrid_retriever.id_map.values()}
        expanded = []
        seen = set()
        for hit in focused:
            candidates = []
            if expansion in {"parent", "parent_and_neighbors"}:
                parent = hit.get("parent_text")
                if parent:
                    candidates.append({**hit, "chunk_id": hit.get("parent_id"), "text": parent,
                                       "section": "Paper context", "context_role": "parent"})
                elif hit.get("parent_id"):
                    siblings = [c for c in chunks_by_id.values()
                                if c.get("parent_id") == hit.get("parent_id")]
                    parent_text = "\n\n".join(c["text"] for c in siblings)
                    if parent_text:
                        candidates.append({**hit, "chunk_id": hit.get("parent_id"), "text": parent_text,
                                           "section": "Section context", "context_role": "parent"})
            if expansion in {"neighbors", "parent_and_neighbors"}:
                for key in ("prev_chunk_id", "next_chunk_id"):
                    neighbor = chunks_by_id.get(hit.get(key))
                    if neighbor:
                        candidates.append({**neighbor, "context_role": "neighbor"})
            candidates.append({**hit, "context_role": "focused"})
            for candidate in candidates:
                cid = candidate.get("chunk_id")
                if cid in seen:
                    continue
                seen.add(cid)
                expanded.append(candidate)

        # Interleave each focused hit with its expansion; a ranked hit always precedes its neighbors.
        role_order = {"focused": 0, "parent": 1, "neighbor": 2}
        expanded.sort(key=lambda c: (-c.get("rerank_score", 0), role_order.get(c.get("context_role"), 3)))
        result, used = [], set()
        used_tokens = 0
        for candidate in expanded:
            text = candidate.get("text", "").strip()
            if not text:
                continue
            norm = " ".join(text.lower().split())
            if norm in used:
                continue
            size = token_len(text)
            if used_tokens + size > token_budget:
                if candidate.get("context_role") == "focused" and not result:
                    ids = _tokenizer.encode(text, add_special_tokens=False)[:token_budget]
                    candidate["text"] = _tokenizer.decode(ids).strip()
                    size = token_len(candidate["text"])
                else:
                    continue
            if size == 0:
                continue
            used.add(norm)
            used_tokens += size
            result.append(candidate)
        return result


if __name__ == "__main__":
    retriever = RerankedRetriever()
    query = "how does hybrid search improve retrieval on technical corpora"
    results = retriever.retrieve(query, shortlist_size=TOP_N_PER_RETRIEVER, final_k=5)
    for r in results:
        print(f"[rerank={r['rerank_score']:.4f}  rrf={r['rrf_score']:.4f}] "
              f"{r['title']} — {r['text'][:100]}...")
