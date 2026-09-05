"""
Splits documents into retrieval-sized chunks,measuring size in tokens

recursive separator strategy (paragraph -> line -> sentence -> hard cut)
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS = 220    # covers most full abstracts as 1 chunk, avoids splitting dense,
                           # multi-claim abstracts across chunks, which was hurting context recall
CHUNK_OVERLAP_TOKENS = 20  # smaller overlap needed now as most chunks are the full abstract
SEPARATORS = ["\n\n", "\n", ". ", " "]  # word-level fallback handles the rest

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "arxiv_corpus.json"
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"

# Loaded once, reused everywhere -- loading a tokenizer per call would be wasteful.
_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def recursive_split(text: str, chunk_size: int = CHUNK_SIZE_TOKENS,
                     separators: list[str] = SEPARATORS) -> list[str]:
    if token_len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Absolute last resort: split on whitespace so we never cut mid-word.
        words = text.split(" ")
        chunks, current = [], ""
        for w in words:
            candidate = f"{current} {w}".strip()
            if token_len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = w
        if current:
            chunks.append(current)
        return chunks

    separator, remaining_separators = separators[0], separators[1:]
    pieces = text.split(separator)

    chunks = []
    current = ""
    for piece in pieces:
        candidate = (current + separator + piece) if current else piece
        if token_len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if token_len(piece) > chunk_size:
                chunks.extend(recursive_split(piece, chunk_size, remaining_separators))
                current = ""
            else:
                current = piece
    if current:
        chunks.append(current)
    return chunks


def add_overlap(chunks: list[str], overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """
    Prepends the last `overlap_tokens` tokens of the previous chunk
    onto each chunk, decoded back to text so the overlap always
    lands on whole tokens, never mid-word.
    """
    if not chunks:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_ids = _tokenizer.encode(chunks[i - 1], add_special_tokens=False)
        tail_ids = prev_ids[-overlap_tokens:]
        tail_text = _tokenizer.decode(tail_ids)
        overlapped.append(f"{tail_text.strip()} {chunks[i].strip()}")
    return overlapped


def chunk_corpus(records: list[dict]) -> list[dict]:
    """
    Takes raw paper records ({id, title, abstract, ...}) and returns
    chunk records: {chunk_id, paper_id, title, text, url}
    """
    all_chunks = []
    for paper in records:
        raw_chunks = recursive_split(paper["abstract"])
        raw_chunks = add_overlap(raw_chunks)
        for i, chunk_text in enumerate(raw_chunks):
            all_chunks.append({
                "chunk_id": f"{paper['id']}_chunk{i}",
                "paper_id": paper["id"],
                "title": paper["title"],
                "text": chunk_text,
                "url": paper["url"],
            })
    return all_chunks


if __name__ == "__main__":
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)

    chunks = chunk_corpus(papers)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Chunked {len(papers)} papers into {len(chunks)} chunks")
    print(f"Saved to {PROCESSED_PATH}")