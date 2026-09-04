"""
Builds two independent indexes over the same chunk corpus:
1. BM25 index(lexical/keyword search) via rank_bm25
2. FAISS index(semantic search) via sentence-transformers embeddings

Hybrid search means querying both and fusing the results later
"""

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
INDEX_DIR = Path(__file__).parent.parent / "data" / "processed"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
ID_MAP_PATH = INDEX_DIR / "id_map.pkl"


def tokenize(text: str) -> list[str]:
    """Simple lowercase + word tokenizer -- sufficient for BM25's term-overlap scoring."""
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def build_faiss_index(chunks: list[dict], model: SentenceTransformer) -> faiss.Index:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Normalize to unit length -> inner product becomes equivalent to cosine similarity.
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact search, honest choice at this corpus size
    index.add(embeddings)
    return index


def build_and_save_indexes():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    print("Building BM25 index...")
    bm25 = build_bm25_index(chunks)

    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Building FAISS index (embedding all chunks)...")
    faiss_index = build_faiss_index(chunks, model)

    # id_map lets us go from "row 47 in either index" back to the actual chunk metadata.
    id_map = {i: chunks[i] for i in range(len(chunks))}

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(ID_MAP_PATH, "wb") as f:
        pickle.dump(id_map, f)

    print(f"Saved FAISS index -> {FAISS_INDEX_PATH}")
    print(f"Saved BM25 index -> {BM25_INDEX_PATH}")
    print(f"Saved id_map -> {ID_MAP_PATH}")


if __name__ == "__main__":
    build_and_save_indexes()