"""Run RAGAS evaluation on expert Q&A pairs.

Questions are read from a JSON file (``--qa-file``); see
evaluation/expert_qa.example.json for the expected format.

Usage:
    Full run (retrieve + score):
    python scripts/run_ragas_eval.py --gold data/gold_llm --silver data/silver_llm --mode llm

    Score existing results:
    python scripts/run_ragas_eval.py --score-only ragas_results_llm.json --output scored.json
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_QA_FILE = "evaluation/expert_qa.example.json"


def load_expert_qa(path: str) -> list[dict]:
    """Load expert question/ground-truth pairs from a JSON list of
    ``{"question": ..., "ground_truth": ...}`` objects."""
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)
    for i, qa in enumerate(pairs):
        missing = {"question", "ground_truth"} - qa.keys()
        if missing:
            raise ValueError(f"{path}: entry {i} is missing {sorted(missing)}")
    return pairs


def _build_ragas_scorers():
    """Build RAGAS metric scorers with Azure OpenAI."""
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    azure_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    emb_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    from openai import AsyncAzureOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        api_version=azure_version,
    )

    llm = llm_factory(model=azure_deployment, provider="openai", client=client)
    ragas_embeddings = embedding_factory(provider="openai", model=emb_deployment, client=client)

    # Patch embed_query onto RAGAS embeddings (AnswerRelevancy expects langchain interface)
    if not hasattr(ragas_embeddings, "embed_query"):
        ragas_embeddings.embed_query = ragas_embeddings.embed_text
    if not hasattr(ragas_embeddings, "embed_documents"):
        ragas_embeddings.embed_documents = ragas_embeddings.embed_texts

    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.metrics._answer_similarity import AnswerSimilarity

    answer_sim = AnswerSimilarity(embeddings=ragas_embeddings)
    answer_correct = AnswerCorrectness(
        llm=llm, embeddings=ragas_embeddings, answer_similarity=answer_sim
    )

    scorers = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=ragas_embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_correctness": answer_correct,
    }

    logger.info(f"RAGAS scorers ready (Azure: {azure_endpoint})")
    return scorers


async def _score_single(scorers, question, answer, contexts, ground_truth):
    """Score one question-answer pair with all RAGAS metrics."""
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts if contexts else ["No context retrieved."],
        reference=ground_truth,
    )

    scores = {}
    metric_names_list = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    for mname in metric_names_list:
        scorer = scorers[mname]
        try:
            r = await scorer.single_turn_ascore(sample)
            scores[mname] = round(float(r), 4)
        except Exception as e:
            logger.warning(f"  {mname} failed: {e}")
            scores[mname] = None

    return scores


async def _run_ragas_scoring_async(all_results, args):
    """Run RAGAS scoring on all results."""
    logger.info("Building RAGAS scorers...")
    scorers = _build_ragas_scorers()

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]
    strategies_evaluated = sorted(set(r["strategy"] for r in all_results))
    ragas_scores = {}

    for strategy_name in strategies_evaluated:
        strategy_results = [r for r in all_results if r["strategy"] == strategy_name]
        logger.info(f"\n  Evaluating {strategy_name} ({len(strategy_results)} questions)...")

        per_question_scores = []
        for i, r in enumerate(strategy_results):
            logger.info(f"    Q{i + 1}: {r['question'][:60]}...")
            ctx = r.get("contexts", [])
            if not ctx:
                ctx = ["No context retrieved."]
            ctx = [str(c) for c in ctx if c]
            if not ctx:
                ctx = ["No context retrieved."]

            scores = await _score_single(
                scorers, r["question"], r["answer"], ctx, r["ground_truth"]
            )
            per_question_scores.append(scores)
            logger.info(f"      {scores}")

        # Average scores (skip None)
        avg = {}
        for mname in metric_names:
            values = [s[mname] for s in per_question_scores if s.get(mname) is not None]
            avg[mname] = round(sum(values) / len(values), 4) if values else None

        avg["num_questions"] = len(strategy_results)
        avg["avg_execution_time"] = round(
            sum(r["execution_time"] for r in strategy_results) / len(strategy_results), 2
        )
        avg["avg_chunks"] = round(
            sum(r["num_chunks"] for r in strategy_results) / len(strategy_results), 1
        )
        avg["per_question"] = per_question_scores
        ragas_scores[strategy_name] = avg

        for mname in metric_names:
            logger.info(f"    AVG {mname}: {avg.get(mname, '--')}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "mode": getattr(args, "mode", "unknown"),
        "num_questions": len(set(r["question"] for r in all_results)),
        "strategies_evaluated": strategies_evaluated,
        "ragas_scores": ragas_scores,
        "detailed_results": all_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nResults saved to {args.output}")

    # Print table
    print(f"\n{'=' * 90}")
    print("RAGAS EVALUATION RESULTS")
    print(f"{'=' * 90}")
    print(
        f"{'Strategy':<12} {'Faithful':>10} {'Relevancy':>10} {'Precision':>10} {'Recall':>10} {'Correct':>10} {'Time(s)':>8} {'Chunks':>7}"
    )
    print(f"{'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 7}")

    for sn in strategies_evaluated:
        s = ragas_scores.get(sn, {})
        if "error" in s:
            print(f"{sn:<12} {'ERROR':>10}")
            continue
        f_val = f"{s['faithfulness']:.4f}" if s.get("faithfulness") is not None else "--"
        r_val = f"{s['answer_relevancy']:.4f}" if s.get("answer_relevancy") is not None else "--"
        p_val = f"{s['context_precision']:.4f}" if s.get("context_precision") is not None else "--"
        c_val = f"{s['context_recall']:.4f}" if s.get("context_recall") is not None else "--"
        a_val = (
            f"{s['answer_correctness']:.4f}" if s.get("answer_correctness") is not None else "--"
        )
        print(
            f"{sn:<12} {f_val:>10} {r_val:>10} {p_val:>10} {c_val:>10} {a_val:>10} {s.get('avg_execution_time', '--'):>8} {s.get('avg_chunks', '--'):>7}"
        )

    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--gold", default="", help="Path to Gold layer")
    parser.add_argument("--silver", default="", help="Path to Silver layer")
    parser.add_argument("--mode", default="llm", help="Processing mode")
    parser.add_argument("--output", default="ragas_results.json", help="Output file")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=["vector", "pathrag", "graphrag", "hybrid", "react"],
        help="Strategies to evaluate",
    )
    parser.add_argument(
        "--qa-file",
        default=DEFAULT_QA_FILE,
        help="JSON file with expert question/ground_truth pairs",
    )
    parser.add_argument(
        "--score-only", type=str, default="", help="Path to existing results JSON — skip retrieval"
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    if args.score_only:
        logger.info(f"Score-only mode: loading from {args.score_only}")
        with open(args.score_only, encoding="utf-8") as f:
            existing = json.load(f)
        all_results = existing.get("detailed_results", [])
        if not all_results:
            logger.error("No detailed_results found")
            return
        asyncio.run(_run_ragas_scoring_async(all_results, args))
        return

    if not args.gold or not args.silver:
        logger.error("--gold and --silver required (or use --score-only)")
        return

    from tacitgraph.retrieval import HybridRetriever, RetrievalStrategy

    strategy_map = {
        "vector": RetrievalStrategy.VECTOR,
        "pathrag": RetrievalStrategy.PATHRAG,
        "graphrag": RetrievalStrategy.GRAPHRAG,
        "hybrid": RetrievalStrategy.HYBRID,
        "react": RetrievalStrategy.REACT,
    }

    logger.info("Initializing retriever...")
    retriever = HybridRetriever(args.gold, args.silver, mode=args.mode)

    all_results = []
    expert_qa = load_expert_qa(args.qa_file)
    logger.info(f"Loaded {len(expert_qa)} Q&A pairs from {args.qa_file}")
    for qa in expert_qa:
        logger.info(f"\nQuestion: {qa['question']}")
        for sn in args.strategies:
            strategy = strategy_map.get(sn)
            if not strategy:
                continue
            logger.info(f"  Strategy: {sn}")
            try:
                result = retriever.retrieve(qa["question"], strategy)
                all_results.append(
                    {
                        "question": qa["question"],
                        "ground_truth": qa["ground_truth"],
                        "answer": result.answer,
                        "contexts": [c.get("text", "") for c in result.chunks if c.get("text")],
                        "strategy": sn,
                        "execution_time": result.execution_time,
                        "num_chunks": len(result.chunks),
                    }
                )
                logger.info(f"    Answer: {result.answer[:100]}...")
            except Exception as e:
                logger.error(f"    Failed: {e}")
                all_results.append(
                    {
                        "question": qa["question"],
                        "ground_truth": qa["ground_truth"],
                        "answer": f"ERROR: {e}",
                        "contexts": [],
                        "strategy": sn,
                        "execution_time": 0,
                        "num_chunks": 0,
                    }
                )

    asyncio.run(_run_ragas_scoring_async(all_results, args))


if __name__ == "__main__":
    main()
