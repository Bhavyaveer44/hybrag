"""
Queries BM25 index and FAISS index independently then 
fuses their ranked result lists via Reciprocal Rank Fusion(RRF).

RRF combines rank positions, not raw scores.
sidesteps the problem that BM25 scores and cosine similarity scores 
live on totally different, incomparable scales.
"""

import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from indexing import tokenize, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, BM25_INDEX_PATH, ID_MAP_PATH

RRF_K = 60          # standard damping constant from the original RRF paper
TOP_N_PER_RETRIEVER = 20  # how many candidates each retriever contributes before fusion


class HybridRetriever:
    def __init__(self):
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(ID_MAP_PATH, "rb") as f:
            self.id_map = pickle.load(f)  # {row_index: chunk_dict}
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _vector_search(self, query: str, top_n: int) -> list[int]:
        """Returns a ranked list of row indices (best first) from FAISS."""
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        _, indices = self.faiss_index.search(query_vec, top_n)
        return [int(i) for i in indices[0] if i != -1]

    def _bm25_search(self, query: str, top_n: int) -> list[int]:
        """Returns a ranked list of row indices (best first) from BM25."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_n]

    @staticmethod
    def _reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
        """
        ranked_lists: e.g. [[vector_result_ids_in_rank_order], [bm25_result_ids_in_rank_order]]
        Returns: [(row_index, fused_score), ...] sorted best-first.
        """
        fused_scores: dict[int, float] = {}
        for ranked_list in ranked_lists:
            for rank, row_id in enumerate(ranked_list, start=1):  # 1-indexed rank
                fused_scores[row_id] = fused_scores.get(row_id, 0.0) + 1.0 / (k + rank)
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Full hybrid retrieval: query both indexes, fuse via RRF,
        return the top_k chunk dicts (with metadata) plus their fused score.
        """
        vector_hits = self._vector_search(query, TOP_N_PER_RETRIEVER)
        bm25_hits = self._bm25_search(query, TOP_N_PER_RETRIEVER)

        fused = self._reciprocal_rank_fusion([vector_hits, bm25_hits])

        results = []
        for row_id, score in fused[:top_k]:
            chunk = dict(self.id_map[row_id])  # copy so we don't mutate the stored map
            chunk["rrf_score"] = score
            results.append(chunk)
        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "how does hybrid search improve retrieval on technical corpora"
    results = retriever.retrieve(query, top_k=5)
    for r in results:
        print(f"[{r['rrf_score']:.4f}] {r['title']} — {r['text'][:100]}...")