"""
Comparative Analysis Module
===========================
Cross-system comparison for RAG evaluation.

Compares:
- Baseline RAG (vector search only)
- GraphRAG (vector + graph + communities)
- ReAct Agent (tool-augmented reasoning)
- Full System (GraphRAG + ReAct)

Metrics:
- RAGAS metrics (faithfulness, relevancy, precision, recall)
- Retrieval metrics (MRR, NDCG, Precision@K)
- Efficiency metrics (latency, token usage)
- Quality metrics (answer completeness, citation accuracy)

"""

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SystemType(Enum):
    """RAG system types for comparison."""

    BASELINE = "baseline_rag"
    GRAPHRAG = "graphrag"
    REACT = "react_agent"
    FULL_SYSTEM = "full_system"


@dataclass
class SystemResult:
    """Result from a single system for one question."""

    system: SystemType
    question: str
    answer: str
    contexts: list[str]
    sources: list[dict[str, Any]]
    latency_ms: float
    tokens_used: int
    tools_used: list[str] = field(default_factory=list)
    reasoning_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison result for one question across systems."""

    question: str
    question_type: str
    ground_truth: str | None
    system_results: dict[SystemType, SystemResult]
    ragas_scores: dict[SystemType, dict[str, float]]
    retrieval_scores: dict[SystemType, dict[str, float]]
    best_system: SystemType
    analysis: str


@dataclass
class AggregatedComparison:
    """Aggregated comparison across all questions."""

    total_questions: int
    by_system: dict[SystemType, dict[str, float]]
    by_question_type: dict[str, dict[SystemType, dict[str, float]]]
    statistical_significance: dict[tuple[SystemType, SystemType], dict[str, dict[str, Any]]]
    recommendations: list[str]
    generated_at: str


class ComparativeAnalyzer:
    """
    Compare multiple RAG systems.

    Usage:
        analyzer = ComparativeAnalyzer(ragas_evaluator)
        comparison = analyzer.compare(results_by_system)
    """

    def __init__(self, ragas_evaluator=None, significance_threshold: float = 0.05):
        """
        Initialize comparative analyzer.

        Args:
            ragas_evaluator: RAGAS evaluator instance
            significance_threshold: P-value threshold for significance tests
        """
        self.ragas_evaluator = ragas_evaluator
        self.significance_threshold = significance_threshold

    def compare_single_question(
        self,
        question: str,
        question_type: str,
        system_results: dict[SystemType, SystemResult],
        ground_truth: str | None = None,
    ) -> ComparisonResult:
        """
        Compare systems for a single question.

        Args:
            question: The question
            question_type: Type classification (single-hop, multi-hop, etc.)
            system_results: Results from each system
            ground_truth: Optional ground truth answer

        Returns:
            ComparisonResult with scores and analysis
        """
        ragas_scores = {}
        retrieval_scores = {}

        for system_type, result in system_results.items():
            # Calculate RAGAS scores if evaluator available
            if self.ragas_evaluator:
                from .ragas_evaluator import EvaluationSample

                sample = EvaluationSample(
                    question=question,
                    answer=result.answer,
                    contexts=result.contexts,
                    ground_truth=ground_truth,
                )

                eval_result = self.ragas_evaluator._evaluate_single(
                    sample, include_correctness=ground_truth is not None
                )

                ragas_scores[system_type] = {
                    "faithfulness": eval_result.faithfulness,
                    "answer_relevancy": eval_result.answer_relevancy,
                    "context_precision": eval_result.context_precision,
                    "context_recall": eval_result.context_recall,
                }

                if eval_result.answer_correctness is not None:
                    ragas_scores[system_type]["answer_correctness"] = eval_result.answer_correctness

            # Calculate retrieval metrics
            retrieval_scores[system_type] = {
                "num_contexts": len(result.contexts),
                "num_sources": len(result.sources),
                "latency_ms": result.latency_ms,
                "tokens_used": result.tokens_used,
            }

        # Determine best system (composite score)
        best_system = self._determine_best_system(ragas_scores)

        # Generate analysis
        analysis = self._generate_analysis(question, question_type, system_results, ragas_scores)

        return ComparisonResult(
            question=question,
            question_type=question_type,
            ground_truth=ground_truth,
            system_results=system_results,
            ragas_scores=ragas_scores,
            retrieval_scores=retrieval_scores,
            best_system=best_system,
            analysis=analysis,
        )

    def compare_batch(self, comparisons: list[ComparisonResult]) -> AggregatedComparison:
        """
        Aggregate comparison results.

        Args:
            comparisons: List of individual comparison results

        Returns:
            AggregatedComparison with statistics
        """
        # Aggregate by system
        by_system: defaultdict[SystemType, defaultdict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for comp in comparisons:
            for system_type, scores in comp.ragas_scores.items():
                for metric, value in scores.items():
                    by_system[system_type][metric].append(value)

            for system_type, scores in comp.retrieval_scores.items():
                for metric, value in scores.items():
                    by_system[system_type][f"retrieval_{metric}"].append(value)

        # Calculate means
        by_system_means = {}
        for system_type, metrics in by_system.items():
            by_system_means[system_type] = {
                metric: float(np.mean(values)) for metric, values in metrics.items()
            }
            # Add standard deviations
            for metric, values in metrics.items():
                by_system_means[system_type][f"{metric}_std"] = float(np.std(values))

        # Aggregate by question type
        by_question_type: defaultdict[
            str, defaultdict[SystemType, defaultdict[str, list[float]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for comp in comparisons:
            qtype = comp.question_type
            for system_type, scores in comp.ragas_scores.items():
                for metric, value in scores.items():
                    by_question_type[qtype][system_type][metric].append(value)

        by_question_type_means: dict[str, dict[SystemType, dict[str, float]]] = {}
        for qtype, systems in by_question_type.items():
            by_question_type_means[qtype] = {}
            for system_type, metrics in systems.items():
                by_question_type_means[qtype][system_type] = {
                    metric: float(np.mean(values)) for metric, values in metrics.items()
                }

        # Statistical significance tests
        significance = self._calculate_significance(by_system)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            by_system_means, by_question_type_means, comparisons
        )

        return AggregatedComparison(
            total_questions=len(comparisons),
            by_system=by_system_means,
            by_question_type=by_question_type_means,
            statistical_significance=significance,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat(),
        )

    def _determine_best_system(
        self, ragas_scores: dict[SystemType, dict[str, float]]
    ) -> SystemType:
        """Determine best system based on composite score."""
        if not ragas_scores:
            return SystemType.BASELINE

        composite_scores = {}

        for system_type, scores in ragas_scores.items():
            # Equal weight composite
            composite = np.mean(
                [
                    scores.get("faithfulness", 0),
                    scores.get("answer_relevancy", 0),
                    scores.get("context_precision", 0),
                    scores.get("context_recall", 0),
                ]
            )
            composite_scores[system_type] = composite

        return max(composite_scores, key=lambda s: composite_scores[s])

    def _generate_analysis(
        self,
        question: str,
        question_type: str,
        system_results: dict[SystemType, SystemResult],
        ragas_scores: dict[SystemType, dict[str, float]],
    ) -> str:
        """Generate textual analysis of comparison."""
        analysis_parts = []

        # Question classification
        analysis_parts.append(f"Question Type: {question_type}")

        # Best performers by metric
        if ragas_scores:
            for metric in [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]:
                best = max(ragas_scores.items(), key=lambda x: x[1].get(metric, 0))
                analysis_parts.append(
                    f"Best {metric}: {best[0].value} ({best[1].get(metric, 0):.3f})"
                )

        # Efficiency comparison
        latencies = {sys: res.latency_ms for sys, res in system_results.items()}
        if latencies:
            fastest = min(latencies, key=lambda s: latencies[s])
            analysis_parts.append(f"Fastest: {fastest.value} ({latencies[fastest]:.0f}ms)")

        return " | ".join(analysis_parts)

    def _calculate_significance(
        self, by_system: Mapping[SystemType, Mapping[str, list[float]]]
    ) -> dict[tuple[SystemType, SystemType], dict[str, dict[str, Any]]]:
        """Calculate statistical significance between system pairs."""
        from scipy import stats

        significance: dict[tuple[SystemType, SystemType], dict[str, dict[str, Any]]] = {}
        systems = list(by_system.keys())

        for i, sys1 in enumerate(systems):
            for sys2 in systems[i + 1 :]:
                pair_significance: dict[str, dict[str, Any]] = {}

                for metric in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]:
                    values1 = by_system[sys1].get(metric, [])
                    values2 = by_system[sys2].get(metric, [])

                    if len(values1) >= 2 and len(values2) >= 2:
                        try:
                            # Paired t-test
                            min_len = min(len(values1), len(values2))
                            t_stat, p_value = stats.ttest_rel(values1[:min_len], values2[:min_len])
                            pair_significance[metric] = {
                                "t_statistic": t_stat,
                                "p_value": p_value,
                                "significant": p_value < self.significance_threshold,
                            }
                        except Exception:
                            pass

                significance[(sys1, sys2)] = pair_significance

        return significance

    def _generate_recommendations(
        self,
        by_system_means: dict[SystemType, dict[str, float]],
        by_question_type_means: dict[str, dict[SystemType, dict[str, float]]],
        comparisons: list[ComparisonResult],
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if not by_system_means:
            return ["Insufficient data for recommendations"]

        # Overall best system
        composite_scores = {}
        for system_type, metrics in by_system_means.items():
            composite_scores[system_type] = np.mean(
                [
                    metrics.get("faithfulness", 0),
                    metrics.get("answer_relevancy", 0),
                    metrics.get("context_precision", 0),
                    metrics.get("context_recall", 0),
                ]
            )

        overall_best = max(composite_scores, key=lambda s: composite_scores[s])
        recommendations.append(
            f"Overall best system: {overall_best.value} (composite: {composite_scores[overall_best]:.3f})"
        )

        # Best by question type
        for qtype, systems in by_question_type_means.items():
            if systems:
                qtype_composite = {}
                for system_type, metrics in systems.items():
                    qtype_composite[system_type] = np.mean(
                        [metrics.get("faithfulness", 0), metrics.get("answer_relevancy", 0)]
                    )
                if qtype_composite:
                    best_for_type = max(qtype_composite, key=lambda s: qtype_composite[s])
                    recommendations.append(f"Best for {qtype}: {best_for_type.value}")

        # Efficiency vs quality trade-off
        if SystemType.BASELINE in by_system_means and SystemType.FULL_SYSTEM in by_system_means:
            baseline_latency = by_system_means[SystemType.BASELINE].get("retrieval_latency_ms", 0)
            full_latency = by_system_means[SystemType.FULL_SYSTEM].get("retrieval_latency_ms", 0)

            baseline_quality = composite_scores.get(SystemType.BASELINE, 0)
            full_quality = composite_scores.get(SystemType.FULL_SYSTEM, 0)

            if full_latency > baseline_latency and full_quality > baseline_quality:
                quality_gain = (
                    (full_quality - baseline_quality) / baseline_quality * 100
                    if baseline_quality > 0
                    else 0
                )
                latency_increase = (
                    (full_latency - baseline_latency) / baseline_latency * 100
                    if baseline_latency > 0
                    else 0
                )
                recommendations.append(
                    f"Full system: +{quality_gain:.1f}% quality, +{latency_increase:.1f}% latency"
                )

        return recommendations


class PerformanceProfiler:
    """
    Profile system performance characteristics.
    """

    @staticmethod
    def profile_latency(results: list[SystemResult]) -> dict[str, float]:
        """Calculate latency statistics."""
        latencies = [r.latency_ms for r in results]

        return {
            "mean_ms": np.mean(latencies),
            "median_ms": np.median(latencies),
            "std_ms": np.std(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
        }

    @staticmethod
    def profile_token_usage(results: list[SystemResult]) -> dict[str, float]:
        """Calculate token usage statistics."""
        tokens = [r.tokens_used for r in results]

        return {
            "mean_tokens": np.mean(tokens),
            "total_tokens": np.sum(tokens),
            "std_tokens": np.std(tokens),
            "min_tokens": np.min(tokens),
            "max_tokens": np.max(tokens),
        }

    @staticmethod
    def profile_tool_usage(results: list[SystemResult]) -> dict[str, Any]:
        """Analyze tool usage patterns (for ReAct systems)."""
        from collections import Counter

        all_tools = []
        steps_per_query = []

        for r in results:
            all_tools.extend(r.tools_used)
            steps_per_query.append(r.reasoning_steps)

        tool_counts = Counter(all_tools)

        return {
            "tool_distribution": dict(tool_counts),
            "mean_tools_per_query": len(all_tools) / len(results) if results else 0,
            "mean_reasoning_steps": np.mean(steps_per_query) if steps_per_query else 0,
            "max_reasoning_steps": max(steps_per_query) if steps_per_query else 0,
        }


# Export
__all__ = [
    "AggregatedComparison",
    "ComparativeAnalyzer",
    "ComparisonResult",
    "PerformanceProfiler",
    "SystemResult",
    "SystemType",
]
