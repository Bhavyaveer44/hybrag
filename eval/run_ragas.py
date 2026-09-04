"""
Runs full RAG pipeline(retrieve -> rerank -> generate) for each question 
in eval_dataset.py,then scores each result with RAGAS's four core metrics:

- Faithfulness(reference-free): does every claim in the answer
                actually trace back to the retrieved context?

- Answer Relevancy(reference-free): does the answer actually
                address the question asked?

- Context Precision(needs reference): of what we retrieved, how much
                was actually necessary/relevant?

- Context Recall(needs reference): of what we NEEDED to answer,
                how much did we actually retrieve?

Note:reference-free metrics catch generation problems (hallucination,
off-topic answers). Reference-based metrics catch retrieval problems
(missing or noisy context)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generate import generate_answer
from indexing import EMBEDDING_MODEL_NAME
from reranker import RerankedRetriever

from eval_dataset import EVAL_QUESTIONS

load_dotenv()

# Groq exposes an OpenAI-compatible API, so we point the standard OpenAI SDK 
# at Groq's base_url,this allows reuse ragas's "openai" adapter path directly 
# instead of needing a Groq-specific integration.
GROQ_JUDGE_MODEL = "openai/gpt-oss-120b"


def build_judge_llm():
    client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    return llm_factory(
        GROQ_JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=4096,
    )


def run_pipeline_on_eval_set(retriever: RerankedRetriever) -> list[dict]:
    """Runs retrieve -> rerank -> generate for each eval question, collecting everything RAGAS needs."""
    results = []
    for item in EVAL_QUESTIONS:
        question = item["question"]
        chunks = retriever.retrieve(question, final_k=5)
        contexts = [c["text"] for c in chunks]
        answer = generate_answer(question, chunks)
        results.append({
            "question": question,
            "reference": item["reference"],
            "contexts": contexts,
            "answer": answer,
        })
    return results


async def score_with_ragas(results: list[dict]) -> list[dict]:
    judge_llm = build_judge_llm()
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME)

    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(
        llm=judge_llm,
        embeddings=embeddings,
    )
    context_precision = ContextPrecision(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)

    scored = []

    for i, r in enumerate(results, 1):
        print(f"\nScoring question {i}/{len(results)}...")

        faith_score = await faithfulness.ascore(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
        )

        relevancy_score = await answer_relevancy.ascore(
            user_input=r["question"],
            response=r["answer"],
        )

        precision_score = await context_precision.ascore(
            user_input=r["question"],
            reference=r["reference"],
            retrieved_contexts=r["contexts"],
        )

        recall_score = await context_recall.ascore(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )

        scored.append({
            **r,
            "faithfulness": faith_score.value,
            "answer_relevancy": relevancy_score.value,
            "context_precision": precision_score.value,
            "context_recall": recall_score.value,
        })

        print(
            f"  Faithfulness:       {faith_score.value:.3f}\n"
            f"  Answer Relevancy:   {relevancy_score.value:.3f}\n"
            f"  Context Precision:  {precision_score.value:.3f}\n"
            f"  Context Recall:     {recall_score.value:.3f}"
        )

    return scored

def print_report(scored: list[dict]):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "=" * 70)
    print("PER-QUESTION SCORES")
    print("=" * 70)
    for r in scored:
        print(f"\nQ: {r['question']}")
        for m in metrics:
            print(f"  {m:20s}: {r[m]:.3f}")

    print("\n" + "=" * 70)
    print("AVERAGE SCORES ACROSS EVAL SET")
    print("=" * 70)
    for m in metrics:
        avg = sum(r[m] for r in scored) / len(scored)
        print(f"  {m:20s}: {avg:.3f}")


if __name__ == "__main__":
    import asyncio

    print(f"Running pipeline on {len(EVAL_QUESTIONS)} eval questions...")

    retriever = RerankedRetriever()
    results = run_pipeline_on_eval_set(retriever)

    print(
        "Scoring with RAGAS "
        "(this calls the judge LLM multiple times per question)..."
    )

    scored = asyncio.run(score_with_ragas(results))

    print_report(scored)