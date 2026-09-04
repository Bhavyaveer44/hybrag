"""
small, hand-curated set of evaluation questions with reference
(ground-truth) answers, used to benchmark the pipeline with RAGAS.

Concept: 
context_precision and context_recall need a "reference"
(ground truth) to compare retrieval against, to check
"did we get the right information," which requires knowing what
right looks like. faithfulness and answer_relevancy are reference-
free, they check internal consistency (does the answer match the
retrieved context, does the answer address the question) without
needing a ground truth at all.

HOW TO FILL THIS IN:
1. Run `python src/ingest.py` then look at data/raw/arxiv_corpus.json
   to see which real papers got fetched into your corpus.
2. Pick 5-8 papers whose abstracts contain clear, checkable facts.
3. Write a question answerable from that abstract, and a reference
   answer using the key fact(s) from it, in your own words.

Keep reference answers short and factual, one or two sentences,
stating the specific claim you expect the system to surface.
"""

EVAL_QUESTIONS = [
    {
        "question": "What method does the paper on hybrid retrieval propose for combining BM25 and dense embeddings?",
        "reference": "It proposes combining BM25 and dense embeddings using reciprocal rank fusion.",
    },
    {
        "question": "What does the RAGAS framework evaluate in a RAG pipeline?",
        "reference": "RAGAS evaluates RAG pipelines using reference-free metrics, separating retrieval quality from generation quality, including faithfulness, context precision, and context recall.",
    },
    {
        "question": "How does ESPO's prompt optimization approach compare to GEPA in terms of accuracy and prompt length?",
        "reference": "ESPO improves average accuracy over GEPA by about 3.76 percentage points (74.67% vs 70.91%) across seven NLP benchmarks, while producing prompts roughly 47% shorter.",
    },
    {
        "question": "What KV cache eviction strategy does the Random Attention paper propose, and how does it perform?",
        "reference": "It proposes evicting cached tokens uniformly at random within each attention head instead of scoring them, and finds this matches the strongest prior selective evictor while serving 32-43% higher throughput.",
    },
    {
        "question": "How does Terminal-Universe generate training environments for terminal-based code agents, and what improvement does fine-tuning on them produce?",
        "reference": "It reconstructs reusable environments from the tool-execution history in existing agent trajectories, producing 37,300 task-sufficient environments; fine-tuning Qwen3.5-27B on this data improves Terminal-Bench 2.1 by 11.9 points.",
    },
    {
        "question": "What competitive programming result did Nemotron-3-Ultra-CC achieve at IOI 2026?",
        "reference": "Under the same constraints as human contestants, the system scored 535.4 out of 600 at IOI 2026, exceeding both the gold-medal threshold and the top human contestant's score of 498.27.",
    },
    {
        "question": "Does the door-in-the-face persuasion technique work on large language models?",
        "reference": "It works on Anthropic's frontier models (e.g., Opus 5 complies with a smaller follow-up request 65.8% of the time after refusing a larger one, vs 29.3% when asked directly), but it backfires on OpenAI's, Google's, and Haiku 4.5's models, lowering compliance instead.",
    },

    # --- Trap questions: corpus has multiple similarly-themed papers ---

    {
        # Trap vs. "Sequential Beats Joint" and "Verify Before You Distill" — both also OPD papers
        "question": "In the on-policy distillation paper that trains on a single query, how much state coverage does one query reach, and how many queries are needed to match full-data training?",
        "reference": "A single query already reaches 71.5% state coverage; 16 semantically distinct queries reach 98.9% coverage and match full-data on-policy distillation.",
    },
    {
        # Trap vs. the "one training example" OPD paper above
        "question": "What two-stage training order does the 'Sequential Beats Joint' paper find outperforms jointly combining on-policy distillation and RLVR?",
        "reference": "It finds that doing on-policy distillation first, then RLVR (OPD-then-RL), consistently outperforms pure OPD, pure RLVR, and joint/fused combinations of the two signals.",
    },
    {
        # Trap vs. the other two OPD papers
        "question": "What problem does Teacher-Gated On-Policy Distillation (TGOPD) address, and how does it decide whether to use dense teacher supervision?",
        "reference": "TGOPD addresses the risk of a confidently wrong teacher misleading training by verifying teacher reliability per prompt using verifier-scored probes, routing reliable prompts to dense OPD and unreliable ones to verifier-grounded GRPO instead.",
    },
    {
        # Trap: VisCAD-M1 (the model) vs RealCADBench (the benchmark it's evaluated on) — same team, easy to conflate
        "question": "How many tasks does RealCADBench contain, and what does its evaluated slice cover?",
        "reference": "RealCADBench contains 12,632 tasks from 19 factory-automation categories; the reported evaluation slice covers 1,745 Part-level tasks across four input regimes plus a 25-task assembly study (RCB-Assm25).",
    },
    {
        # Trap: distinguish from RealCADBench above — this is about the VisCAD-M1 model's own score, not the benchmark's task counts
        "question": "What score does VisCAD-M1 achieve on the combined PubCADBench/RealCADBench part-level evaluation, and how much does using it as a verifier improve that score?",
        "reference": "VisCAD-M1 achieves an average part-level score of 0.5540, the highest among evaluated models; reusing it as a test-time verifier raises this to 0.5797, about a 5% relative improvement.",
    },
    {
        # Trap vs. "Untangling the Mechanisms of Misleading Context in Medical Question Answering" — both are about LLM reliability in medical QA
        "question": "What gap does the medical hallucination detection paper find between first-pass annotators and later adjudication?",
        "reference": "First-pass annotators frequently miss factual errors that are later validated by expert or evidence-based adjudicators, and even LLM-as-a-Judge candidate discovery on its own misses errors that human annotators catch.",
    },
    {
        # Trap vs. the hallucination-adjudication paper above — this one is about susceptibility to specific misleading cue types, not annotation gaps
        "question": "In the study of misleading context in medical question answering, which type of misleading cue are models more susceptible to: fabricated evidence or a bare assertion?",
        "reference": "Models are more susceptible to a bare assertion than to fabricated evidence, adopting the asserted answer 10 to 27 points more often, even though assertions are disclosed in reasoning traces less often than evidence-based cues.",
    },
]