"""
Runs the full RAG pipeline (retrieve -> rerank -> generate) for each
question in eval_dataset.py, then scores each result with RAGAS's
four core metrics:

- Faithfulness (reference-free): does every claim in the answer
actually trace back to the retrieved context?
- Answer Relevancy (reference-free): does the answer actually
address the question asked?
- Context Precision (needs reference): of what we retrieved, how much
was actually necessary/relevant?
- Context Recall (needs reference): of what we NEEDED to answer,
how much did we actually retrieve?

Note: reference-free metrics catch generation problems (hallucination,
off-topic answers). Reference-based metrics catch retrieval problems
(missing or noisy context).

Implementation note: RAGAS's collection-style metrics are natively
async, faithfulness alone issues multiple LLM calls per question
(decompose answer into claims, then verify each claim against
context), so async lets those run concurrently. We use AsyncOpenAI
and await .ascore() directly rather than the sync .score() wrapper,
which avoids event-loop mismatches with the instructor-based
structured-output adapter RAGAS uses for judge calls.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generate import generate_answer
from indexing import EMBEDDING_MODEL_NAME
from reranker import RerankedRetriever

from eval_dataset import EVAL_QUESTIONS

load_dotenv()

# Two separate checkpoints,because these are 2 independent, independently rate-limited stages:
#  1. PIPELINE_CHECKPOINT: retrieve+rerank+generate results (uses the generation model's quota, qwen/qwen3.8-27b)
#  2. SCORING checkpoint: RAGAS judge scores (uses the judge model's quota, openai/gpt-oss-120b)
# Without this split, every rerun regenerates ALL answers even for questions already scored
# wasting generation-model quota and making judge-side debugging burn through both budgets at once.
PIPELINE_CHECKPOINT_PATH = Path(__file__).parent / "pipeline_results_checkpoint.json"
CHECKPOINT_PATH = Path(__file__).parent / "ragas_results_checkpoint.json"


def load_pipeline_checkpoint() -> dict[str, dict]:
    if PIPELINE_CHECKPOINT_PATH.exists():
        with open(PIPELINE_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {r["question"]: r for r in data}
    return {}


def save_pipeline_checkpoint(results: list[dict]):
    with open(PIPELINE_CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_checkpoint() -> dict[str, dict]:
    """Returns {question_text: scored_result_dict} for whatever's already been scored."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {r["question"]: r for r in data}
    return {}


def save_checkpoint(scored: list[dict]):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2)

"""Groq exposes an OpenAI-compatible API, so we point the standard OpenAI SDK at Groq's base_url
this lets us reuse ragas's "openai" adapter path directly instead of needing a Groq-specific integration.

Judge model is chosen independently from the generation model (see
generate.py) because judge metrics need reliable structured/JSON output
(tool-calling compliance), not just good free-text generation quality."""

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
        max_tokens=1024,  # structured claim/verdict JSON doesn't need much -- conserves daily judge quota
    )


def run_pipeline_on_eval_set(retriever: RerankedRetriever) -> list[dict]:
    """
    Runs retrieve -> rerank -> generate for each eval question.
    Checkpointed independently from RAGAS scoring: if a question's
    answer was already generated in a previous run, it is loaded
    from disk instead of calling the generation LLM again.
    """
    already_generated = load_pipeline_checkpoint()
    results = list(already_generated.values())
    if already_generated:
        print(f"Loaded {len(already_generated)} already-generated answer(s) from checkpoint.")

    for item in EVAL_QUESTIONS:
        question = item["question"]
        if question in already_generated:
            continue  # skip -- don't burn generation-model quota re-answering this

        print(f"Generating answer for: {question[:60]}...")
        chunks = retriever.retrieve(question, final_k=5)
        contexts = [c["text"] for c in chunks]
        answer = generate_answer(question, chunks)

        result = {
            "question": question,
            "reference": item["reference"],
            "contexts": contexts,
            "answer": answer,
        }
        results.append(result)
        save_pipeline_checkpoint(results)  # persist immediately -- protects against a mid-loop failure too

    return results


async def score_with_ragas(results: list[dict]) -> list[dict]:
    judge_llm = build_judge_llm()
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL_NAME)

    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)

    checkpointed = load_checkpoint()
    """Only carry forward checkpoint entries for questions still present in
    EVAL_QUESTIONS, if a question was removed or edited since the last
    run (e.g. a placeholder swapped for a real one), its stale checkpoint
    entry must not silently survive into the final averaged report."""
    current_questions = {r["question"] for r in results}
    already_scored = {q: v for q, v in checkpointed.items() if q in current_questions}
    scored = list(already_scored.values())
    if already_scored:
        print(f"Resuming: {len(already_scored)} question(s) already scored in a previous run.")

    for i, r in enumerate(results, 1):
        if r["question"] in already_scored:
            print(f"\nSkipping question {i}/{len(results)} (already scored)")
            continue

        print(f"\nScoring question {i}/{len(results)}...")
        try:
            faith_score = await faithfulness.ascore(
                user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"],
            )
            relevancy_score = await answer_relevancy.ascore(
                user_input=r["question"], response=r["answer"],
            )
            precision_score = await context_precision.ascore(
                user_input=r["question"], reference=r["reference"], retrieved_contexts=r["contexts"],
            )
            recall_score = await context_recall.ascore(
                user_input=r["question"], retrieved_contexts=r["contexts"], reference=r["reference"],
            )
        except RateLimitError as e:
            print(f"\nRate limit hit: {e}")
            print(f"Progress saved -- {len(scored)}/{len(results)} questions scored so far.")
            print(f"Wait for the limit to reset, then rerun this script to resume from where it left off.")
            save_checkpoint(scored)
            return scored

        result = {
            **r,
            "faithfulness": faith_score.value,
            "answer_relevancy": relevancy_score.value,
            "context_precision": precision_score.value,
            "context_recall": recall_score.value,
        }
        scored.append(result)
        save_checkpoint(scored)  # persist immediately -- don't lose progress if the NEXT question fails

        print(
            f"  Faithfulness:       {faith_score.value:.3f}\n"
            f"  Answer Relevancy:   {relevancy_score.value:.3f}\n"
            f"  Context Precision:  {precision_score.value:.3f}\n"
            f"  Context Recall:     {recall_score.value:.3f}"
        )

    return scored


def print_report(scored: list[dict]):
    if not scored:
        print("\nNo questions were scored yet (likely hit a rate limit before the first completed).")
        print("Rerun once quota resets -- checkpointing means nothing already-scored is lost.")
        return

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

    print("Scoring with RAGAS (this calls the judge LLM multiple times per question)...")
    scored = asyncio.run(score_with_ragas(results))

    print_report(scored)