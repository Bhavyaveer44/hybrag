#Takes the final reranked chunks and generates grounded answer

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from reranker import RerankedRetriever

load_dotenv()

LLM_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = """You are a research assistant answering questions using ONLY the provided context passages from academic papers.

Rules:
- Answer using ONLY information found in the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so explicitly instead of guessing.
- When you state a fact, reference which source it came from using [1], [2], etc., matching the numbered context passages.
- Lead with the direct answer to the question in your first sentence. Do not restate or rephrase the question before answering.
- Include only details that directly support answering the question asked -- omit tangential facts from the context even if they're interesting.
- Be concise and precise."""


def format_context(chunks: list[dict]) -> str:
    """Numbers each chunk so the model (and RAGAS later) can trace claims back to a specific source."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (Source: \"{chunk['title']}\")\n{chunk['text']}")
    return "\n\n".join(lines)


def generate_answer(query: str, chunks: list[dict], client: Groq = None) -> str:
    client = client or Groq(api_key=os.environ["GROQ_API_KEY"])
    context = format_context(chunks)

    user_prompt = f"""Context passages:
    {context}

    Question: {query}

    Answer using only the context above, citing sources like [1], [2] as you go."""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,  # deterministic, conservative -- we want grounded answers, not creative ones
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    retriever = RerankedRetriever()
    query = "How does hybrid search improve retrieval on technical corpora?"
    chunks = retriever.retrieve(query, final_k=5)

    answer = generate_answer(query, chunks)
    print("QUESTION:", query)
    print("\nANSWER:\n", answer)
    print("\nSOURCES USED:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['title']} ({c['url']})")