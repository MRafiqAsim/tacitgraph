"""
RAGAS Evaluation Module
=======================
Evaluate RAG systems using the RAGAS framework.

Metrics:
- Faithfulness: How factually accurate is the answer based on context
- Answer Relevancy: How relevant is the answer to the question
- Context Precision: How much of the retrieved context is relevant
- Context Recall: How much of the ground truth is covered by context
- Answer Correctness: How correct is the answer compared to ground truth

"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Single evaluation sample."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result for a single sample."""

    sample_id: str
    question: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResults:
    """Aggregated evaluation results."""

    num_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_answer_correctness: float | None
    std_faithfulness: float
    std_answer_relevancy: float
    std_context_precision: float
    std_context_recall: float
    individual_results: list[EvaluationResult]
    evaluated_at: str


@dataclass
class RAGASConfig:
    """Configuration for RAGAS evaluation."""

    # Model settings
    model_deployment: str = "gpt-4o"
    embedding_deployment: str = "text-embedding-3-large"

    # Evaluation settings
    batch_size: int = 10
    max_retries: int = 3

    # Metric weights for composite score
    faithfulness_weight: float = 0.25
    answer_relevancy_weight: float = 0.25
    context_precision_weight: float = 0.25
    context_recall_weight: float = 0.25


class RAGASEvaluator:
    """
    Evaluate RAG systems using RAGAS metrics.

    Usage:
        evaluator = RAGASEvaluator(llm, embeddings, config)
        results = evaluator.evaluate(samples)
    """

    def __init__(self, llm, embeddings, config: RAGASConfig | None = None):
        """
        Initialize RAGAS evaluator.

        Args:
            llm: Language model for evaluation
            embeddings: Embedding model for similarity
            config: Evaluation configuration
        """
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or RAGASConfig()

    def evaluate(
        self,
        samples: list[EvaluationSample],
        include_correctness: bool = False,
        progress_callback: Callable[..., Any] | None = None,
    ) -> AggregatedResults:
        """
        Evaluate samples using RAGAS metrics.

        Args:
            samples: List of evaluation samples
            include_correctness: Whether to evaluate answer correctness (requires ground truth)
            progress_callback: Optional callback(current, total)

        Returns:
            AggregatedResults with individual and aggregated scores
        """
        results = []
        total = len(samples)

        for i, sample in enumerate(samples):
            try:
                result = self._evaluate_single(sample, include_correctness)
                results.append(result)
            except Exception as e:
                logger.warning(f"Evaluation failed for sample {i}: {e}")
                results.append(
                    EvaluationResult(
                        sample_id=f"sample_{i}",
                        question=sample.question,
                        faithfulness=0.0,
                        answer_relevancy=0.0,
                        context_precision=0.0,
                        context_recall=0.0,
                        metadata={"error": str(e)},
                    )
                )

            if progress_callback:
                progress_callback(i + 1, total)

        return self._aggregate_results(results, include_correctness)

    def _evaluate_single(
        self, sample: EvaluationSample, include_correctness: bool
    ) -> EvaluationResult:
        """Evaluate a single sample."""
        # Calculate each metric
        faithfulness = self._calculate_faithfulness(sample.answer, sample.contexts)

        answer_relevancy = self._calculate_answer_relevancy(sample.question, sample.answer)

        context_precision = self._calculate_context_precision(sample.question, sample.contexts)

        context_recall = self._calculate_context_recall(
            sample.answer, sample.contexts, sample.ground_truth
        )

        answer_correctness = None
        if include_correctness and sample.ground_truth:
            answer_correctness = self._calculate_answer_correctness(
                sample.answer, sample.ground_truth
            )

        return EvaluationResult(
            sample_id=sample.metadata.get("id", f"sample_{hash(sample.question)}"),
            question=sample.question,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            answer_correctness=answer_correctness,
            metadata=sample.metadata,
        )

    def _calculate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        Calculate faithfulness: whether answer is supported by context.

        Uses LLM to extract claims from answer and verify each against context.
        """
        if not answer or not contexts:
            return 0.0

        # Extract claims from answer
        claims_prompt = f"""Extract factual claims from this answer. List each claim on a new line.

Answer: {answer}

Claims:"""

        claims_response = self.llm.invoke(claims_prompt)
        claims = [c.strip() for c in claims_response.content.split("\n") if c.strip()]

        if not claims:
            return 1.0  # No claims = fully faithful (vacuously true)

        # Verify each claim against context
        context_text = "\n\n".join(contexts)
        supported_count = 0

        for claim in claims:
            verify_prompt = f"""Is this claim supported by the context? Answer only 'yes' or 'no'.

Context:
{context_text[:4000]}

Claim: {claim}

Supported:"""

            verify_response = self.llm.invoke(verify_prompt)
            if "yes" in verify_response.content.lower():
                supported_count += 1

        return supported_count / len(claims)

    def _calculate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Calculate answer relevancy: how relevant is answer to question.

        Uses embedding similarity between question and answer.
        """
        if not question or not answer:
            return 0.0

        import numpy as np

        # Get embeddings
        q_embedding = np.array(self.embeddings.embed_query(question))
        a_embedding = np.array(self.embeddings.embed_query(answer))

        # Cosine similarity
        similarity = np.dot(q_embedding, a_embedding) / (
            np.linalg.norm(q_embedding) * np.linalg.norm(a_embedding)
        )

        # Normalize to 0-1 range (similarity can be negative)
        return max(0.0, min(1.0, (similarity + 1) / 2))

    def _calculate_context_precision(self, question: str, contexts: list[str]) -> float:
        """
        Calculate context precision: how much context is relevant to question.

        Uses LLM to judge relevance of each context chunk.
        """
        if not question or not contexts:
            return 0.0

        relevant_count = 0

        for context in contexts:
            relevance_prompt = f"""Is this context relevant to answering the question? Answer only 'yes' or 'no'.

Question: {question}

Context: {context[:1000]}

Relevant:"""

            response = self.llm.invoke(relevance_prompt)
            if "yes" in response.content.lower():
                relevant_count += 1

        return relevant_count / len(contexts)

    def _calculate_context_recall(
        self, answer: str, contexts: list[str], ground_truth: str | None
    ) -> float:
        """
        Calculate context recall: how much of the answer is covered by context.

        If ground truth is provided, uses that instead.
        """
        reference = ground_truth if ground_truth else answer

        if not reference or not contexts:
            return 0.0

        context_text = "\n\n".join(contexts)

        # Extract key facts from reference
        facts_prompt = f"""Extract the key facts from this text. List each fact on a new line.

Text: {reference}

Key facts:"""

        facts_response = self.llm.invoke(facts_prompt)
        facts = [f.strip() for f in facts_response.content.split("\n") if f.strip()]

        if not facts:
            return 1.0  # No facts = full recall (vacuously true)

        # Check how many facts are covered by context
        covered_count = 0

        for fact in facts:
            coverage_prompt = f"""Is this fact mentioned or implied in the context? Answer only 'yes' or 'no'.

Context:
{context_text[:4000]}

Fact: {fact}

Covered:"""

            response = self.llm.invoke(coverage_prompt)
            if "yes" in response.content.lower():
                covered_count += 1

        return covered_count / len(facts)

    def _calculate_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Calculate answer correctness: semantic similarity to ground truth.
        """
        if not answer or not ground_truth:
            return 0.0

        import numpy as np

        # Get embeddings
        answer_embedding = np.array(self.embeddings.embed_query(answer))
        truth_embedding = np.array(self.embeddings.embed_query(ground_truth))

        # Cosine similarity
        similarity = np.dot(answer_embedding, truth_embedding) / (
            np.linalg.norm(answer_embedding) * np.linalg.norm(truth_embedding)
        )

        return max(0.0, min(1.0, similarity))

    def _aggregate_results(
        self, results: list[EvaluationResult], include_correctness: bool
    ) -> AggregatedResults:
        """Aggregate individual results."""
        import numpy as np

        faithfulness_scores = [r.faithfulness for r in results]
        relevancy_scores = [r.answer_relevancy for r in results]
        precision_scores = [r.context_precision for r in results]
        recall_scores = [r.context_recall for r in results]

        correctness_avg = None
        if include_correctness:
            correctness_scores = [
                r.answer_correctness for r in results if r.answer_correctness is not None
            ]
            if correctness_scores:
                correctness_avg = np.mean(correctness_scores)

        return AggregatedResults(
            num_samples=len(results),
            avg_faithfulness=np.mean(faithfulness_scores),
            avg_answer_relevancy=np.mean(relevancy_scores),
            avg_context_precision=np.mean(precision_scores),
            avg_context_recall=np.mean(recall_scores),
            avg_answer_correctness=correctness_avg,
            std_faithfulness=np.std(faithfulness_scores),
            std_answer_relevancy=np.std(relevancy_scores),
            std_context_precision=np.std(precision_scores),
            std_context_recall=np.std(recall_scores),
            individual_results=results,
            evaluated_at=datetime.now().isoformat(),
        )

    def calculate_composite_score(self, results: AggregatedResults) -> float:
        """
        Calculate weighted composite score.
        """
        score = (
            self.config.faithfulness_weight * results.avg_faithfulness
            + self.config.answer_relevancy_weight * results.avg_answer_relevancy
            + self.config.context_precision_weight * results.avg_context_precision
            + self.config.context_recall_weight * results.avg_context_recall
        )
        return score


class RAGASDatasetBuilder:
    """
    Build evaluation datasets for RAGAS.
    """

    @staticmethod
    def from_qa_pairs(
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> list[EvaluationSample]:
        """
        Create evaluation samples from QA pairs.
        """
        samples = []

        for i, (q, a, c) in enumerate(zip(questions, answers, contexts, strict=False)):
            gt = ground_truths[i] if ground_truths else None
            samples.append(
                EvaluationSample(
                    question=q,
                    answer=a,
                    contexts=c,
                    ground_truth=gt,
                    metadata={"id": f"sample_{i}"},
                )
            )

        return samples

    @staticmethod
    def from_rag_results(rag_results: list[dict[str, Any]]) -> list[EvaluationSample]:
        """
        Create evaluation samples from RAG system results.

        Expects dicts with 'question', 'answer', 'contexts' keys.
        """
        samples = []

        for i, result in enumerate(rag_results):
            samples.append(
                EvaluationSample(
                    question=result.get("question", ""),
                    answer=result.get("answer", ""),
                    contexts=result.get("contexts", []),
                    ground_truth=result.get("ground_truth"),
                    metadata={"id": result.get("id", f"sample_{i}")},
                )
            )

        return samples


# Export
__all__ = [
    "AggregatedResults",
    "EvaluationResult",
    "EvaluationSample",
    "RAGASConfig",
    "RAGASDatasetBuilder",
    "RAGASEvaluator",
]
