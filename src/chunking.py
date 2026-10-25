"""Section-aware, deterministic chunking for paper abstracts and structured paper text."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS = 220
CHUNK_OVERLAP_TOKENS = 20
SEPARATORS = ["\n\n", "\n", ". ", " "]
RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "arxiv_corpus.json"
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def recursive_split(text: str, chunk_size: int = CHUNK_SIZE_TOKENS,
                    separators: list[str] = SEPARATORS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if token_len(text) <= chunk_size:
        return [text]
    if not separators:
        ids = _tokenizer.encode(text, add_special_tokens=False)
        return [_tokenizer.decode(ids[i:i + chunk_size]).strip()
                for i in range(0, len(ids), chunk_size)]

    separator, rest = separators[0], separators[1:]
    chunks, current = [], ""
    for piece in text.split(separator):
        candidate = f"{current}{separator if current else ''}{piece}"
        if token_len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            if token_len(piece) > chunk_size:
                chunks.extend(recursive_split(piece, chunk_size, rest))
                current = ""
            else:
                current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def add_overlap(chunks: list[str], overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    if overlap_tokens <= 0 or len(chunks) < 2:
        return chunks
    result = [chunks[0]]
    for previous, chunk in zip(chunks, chunks[1:]):
        tail = _tokenizer.decode(_tokenizer.encode(previous, add_special_tokens=False)[-overlap_tokens:]).strip()
        result.append(f"{tail} {chunk}".strip())
    return result


_SECTION_RE = re.compile(r"(?im)^\s*(abstract|introduction|background|method(?:s|ology)?|approach|experiment(?:s|al setup)?|result(?:s)?|discussion|conclusion|related work)\s*:?\s*$")


def split_sections(paper: dict) -> list[tuple[str, str]]:
    """Extract labeled sections from structured input; treat legacy abstracts as one section."""
    for field in ("sections", "full_text"):
        value = paper.get(field)
        if isinstance(value, dict):
            sections = [(str(name).strip(), str(body).strip()) for name, body in value.items()]
            return [(name or "Section", body) for name, body in sections if body]
        if isinstance(value, list):
            sections = []
            for item in value:
                if isinstance(item, dict):
                    name = str(item.get("heading") or item.get("title") or "Section").strip()
                    body = str(item.get("text") or item.get("content") or "").strip()
                    if body:
                        sections.append((name, body))
            if sections:
                return sections
        if isinstance(value, str) and value.strip():
            matches = list(_SECTION_RE.finditer(value))
            if matches:
                result = []
                if value[:matches[0].start()].strip():
                    result.append(("Abstract", value[:matches[0].start()].strip()))
                for i, match in enumerate(matches):
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(value)
                    body = value[match.end():end].strip()
                    if body:
                        result.append((match.group(1).title(), body))
                return result
            return [("Full text", value.strip())]
    abstract = str(paper.get("abstract") or "").strip()
    return [("Abstract", abstract)] if abstract else []


def _stable_paper_id(paper: dict) -> str:
    identity = str(paper.get("id") or paper.get("url") or paper.get("title") or "paper")
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def chunk_corpus(records: list[dict], strategy: str = "section", chunk_size: int = CHUNK_SIZE_TOKENS,
                 overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[dict]:
    """Create stable chunk records; strategies: section-aware or fixed recursive."""
    if strategy not in {"section", "fixed"}:
        raise ValueError("strategy must be 'section' or 'fixed'")
    all_chunks = []
    for paper in records:
        source_id = str(paper.get("id") or paper.get("url") or paper.get("title") or "paper")
        paper_id = _stable_paper_id(paper)
        parent_id = f"paper:{paper_id}"
        sections = split_sections(paper) if strategy == "section" else [("Paper", "\n\n".join(body for _, body in split_sections(paper)))]
        for section_index, (section, body) in enumerate(sections):
            raw_chunks = add_overlap(recursive_split(body, chunk_size), overlap_tokens)
            section_id = f"{parent_id}:section:{section_index}"
            for index, text in enumerate(raw_chunks):
                chunk_id = f"{section_id}:chunk:{index}"
                all_chunks.append({
                    "chunk_id": chunk_id, "paper_id": paper_id, "source_id": source_id,
                    "parent_id": section_id, "paper_parent_id": parent_id,
                    "parent_text": body, "section_id": section_id, "section": section,
                    "chunk_index": index, "section_index": section_index,
                    "prev_chunk_id": f"{section_id}:chunk:{index - 1}" if index else None,
                    "next_chunk_id": None, "title": paper.get("title", "Untitled paper"),
                    "text": text, "url": paper.get("url", ""),
                    "published": paper.get("published"), "authors": paper.get("authors", []),
                    "chunking_strategy": strategy,
                })
                if index:
                    all_chunks[-2]["next_chunk_id"] = chunk_id
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["section", "fixed"], default="section")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_TOKENS)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_TOKENS)
    args = parser.parse_args()
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)
    chunks = chunk_corpus(papers, args.strategy, args.chunk_size, args.overlap)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Chunked {len(papers)} papers into {len(chunks)} chunks using {args.strategy} strategy")
    print(f"Saved to {PROCESSED_PATH}")
