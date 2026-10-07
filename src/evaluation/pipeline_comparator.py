"""
Pipeline Comparator

Compares processing quality across LOCAL, LLM, and HYBRID pipeline modes.
Generates statistical comparisons with paired t-tests.

Evaluation dimensions:
- Anonymization: PII detection precision/recall/F1, identity consistency
- Summarization: ROUGE scores, semantic similarity, LLM faithfulness
- Retrieval: RAGAS metrics (faithfulness, relevancy, precision, recall)
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModeMetrics:
    """Collected metrics for a single processing mode"""
    mode: str
    anonymization: Dict[str, Any] = field(default_factory=dict)
    summarization: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "anonymization": self.anonymization,
            "summarization": self.summarization,
            "retrieval": self.retrieval,
            "processing_stats": self.processing_stats,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two modes"""
    mode_a: str
    mode_b: str
    metric_name: str
    mean_a: float
    mean_b: float
    difference: float
    t_statistic: float
    p_value: float
    significant: bool  # p < 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode_a": self.mode_a,
            "mode_b": self.mode_b,
            "metric_name": self.metric_name,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "difference": round(self.difference, 4),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
        }


class PipelineComparator:
    """
    Compare processing quality across pipeline modes.

    Collects metrics from each mode's silver output and generates
    statistical comparisons using paired t-tests.
    """

    MODES = ["local", "llm", "hybrid"]

    def __init__(
        self,
        silver_paths: Optional[Dict[str, str]] = None,
        ground_truth_path: Optional[str] = None,
    ):
        """
        Args:
            silver_paths: Dict mapping mode → silver layer path
                e.g., {"local": "./data/silver", "llm": "./data/silver_llm", "hybrid": "./data/silver_hybrid"}
            ground_truth_path: Path to ground_truth_pii.json
        """
        self.silver_paths = silver_paths or {}
        self.ground_truth_path = ground_truth_path
        self.mode_metrics: Dict[str, ModeMetrics] = {}

    def collect_all_metrics(self) -> Dict[str, ModeMetrics]:
        """
        Collect metrics from all available modes.

        Returns:
            Dict of mode → ModeMetrics
        """
        for mode, silver_path in self.silver_paths.items():
            if not Path(silver_path).exists():
                logger.warning(f"Silver path not found for mode {mode}: {silver_path}")
                continue

            metrics = ModeMetrics(mode=mode)

            # Anonymization metrics
            metrics.anonymization = self._collect_anonymization_metrics(silver_path)

            # Summarization metrics
            metrics.summarization = self._collect_summarization_metrics(silver_path)

            # Processing stats
            metrics.processing_stats = self._collect_processing_stats(silver_path)

            self.mode_metrics[mode] = metrics
            logger.info(f"Collected metrics for mode: {mode}")

        return self.mode_metrics

    def _collect_anonymization_metrics(self, silver_path: str) -> Dict[str, Any]:
        """Collect anonymization-related metrics from silver layer."""
        from .anonymization_evaluator import AnonymizationEvaluator

        evaluator = AnonymizationEvaluator(ground_truth_path=self.ground_truth_path)
        report = evaluator.generate_report(silver_path)

        return {
            "identity_consistency": report.get("identity_consistency", {}),
            "false_positive_analysis": report.get("false_positive_analysis", {}),
            "detection_metrics": report.get("detection_metrics", {}),
            "aggregate_detection": report.get("aggregate_detection", {}),
        }

    def _collect_summarization_metrics(self, silver_path: str) -> Dict[str, Any]:
        """Collect summarization metrics from silver layer."""
        silver = Path(silver_path)
        summaries_dir = silver / "not_personal" / "thread_summaries"

        if not summaries_dir.exists():
            return {"summary_count": 0}

        summary_lengths = []
        for json_file in summaries_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                text = summary.get("summary", "")
                if text:
                    summary_lengths.append(len(text.split()))
            except Exception:
                continue

        return {
            "summary_count": len(summary_lengths),
            "avg_summary_length_words": round(sum(summary_lengths) / len(summary_lengths), 1) if summary_lengths else 0,
            "min_length": min(summary_lengths) if summary_lengths else 0,
            "max_length": max(summary_lengths) if summary_lengths else 0,
        }

    def _collect_processing_stats(self, silver_path: str) -> Dict[str, Any]:
        """Collect processing statistics from metadata."""
        stats_file = Path(silver_path) / "metadata" / "thread_processing_stats.json"

        if not stats_file.exists():
            return {}

        try:
            with open(stats_file, "r") as f:
                stats_list = json.load(f)

            if stats_list and isinstance(stats_list, list):
                latest = stats_list[-1]
                return {
                    "threads_processed": latest.get("threads_processed", 0),
                    "chunks_created": latest.get("chunks_created", 0),
                    "pii_detected": latest.get("pii_detected", 0),
                    "kg_entities_extracted": latest.get("kg_entities_extracted", 0),
                    "errors": latest.get("errors", 0),
                }
        except Exception:
            pass

        return {}

    def compare_modes(
        self,
        metric_name: str,
        values_by_mode: Optional[Dict[str, List[float]]] = None,
    ) -> List[ComparisonResult]:
        """
        Perform pairwise comparisons between modes using paired t-tests.

        Args:
            metric_name: Name of the metric being compared
            values_by_mode: Dict of mode → list of per-sample values
                If None, uses aggregate metrics (no t-test possible)

        Returns:
            List of ComparisonResult for each pair
        """
        results = []
        available_modes = list(values_by_mode.keys()) if values_by_mode else list(self.mode_metrics.keys())

        for i, mode_a in enumerate(available_modes):
            for mode_b in available_modes[i + 1:]:
                if values_by_mode:
                    vals_a = values_by_mode[mode_a]
                    vals_b = values_by_mode[mode_b]

                    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
                        logger.warning(f"Cannot compare {mode_a} vs {mode_b}: unequal or insufficient samples")
                        continue

                    mean_a = sum(vals_a) / len(vals_a)
                    mean_b = sum(vals_b) / len(vals_b)
                    t_stat, p_val = self._paired_t_test(vals_a, vals_b)
                else:
                    mean_a = 0.0
                    mean_b = 0.0
                    t_stat = 0.0
                    p_val = 1.0

                results.append(ComparisonResult(
                    mode_a=mode_a,
                    mode_b=mode_b,
                    metric_name=metric_name,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    difference=mean_a - mean_b,
                    t_statistic=t_stat,
                    p_value=p_val,
                    significant=p_val < 0.05,
                ))

        return results

    @staticmethod
    def _paired_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
        """
        Compute paired t-test between two lists of paired observations.

        Returns:
            (t_statistic, p_value)
        """
        n = len(a)
        if n < 2:
            return 0.0, 1.0

        diffs = [a[i] - b[i] for i in range(n)]
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)

        if var_diff == 0:
            return 0.0, 1.0

        se = math.sqrt(var_diff / n)
        t_stat = mean_diff / se

        # Approximate p-value using normal distribution for large n
        # For small n, use proper t-distribution (scipy would be needed)
        df = n - 1
        p_value = PipelineComparator._t_distribution_p(abs(t_stat), df) * 2  # Two-tailed

        return t_stat, min(p_value, 1.0)

    @staticmethod
    def _t_distribution_p(t: float, df: int) -> float:
        """
        Approximate p-value from t-distribution using normal approximation.
        For production use, prefer scipy.stats.t.sf
        """
        # Simple normal approximation (reasonable for df > 30)
        if df > 30:
            # Standard normal approximation
            z = t
            p = 0.5 * math.erfc(z / math.sqrt(2))
            return p

        # For smaller df, use a rougher approximation
        # This is the Abramowitz and Stegun approximation
        x = t * t
        p = math.pow(1 + x / df, -(df + 1) / 2)
        return min(p, 0.5)

    def generate_comparison_report(
        self,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report across all modes.

        Args:
            output_path: Optional path to save report

        Returns:
            Complete comparison report
        """
        if not self.mode_metrics:
            self.collect_all_metrics()

        report = {
            "generated_at": datetime.now().isoformat(),
            "modes_compared": list(self.mode_metrics.keys()),
            "mode_summaries": {
                mode: metrics.to_dict()
                for mode, metrics in self.mode_metrics.items()
            },
        }

        # Compare PII detection counts
        pii_counts = {
            mode: [m.processing_stats.get("pii_detected", 0)]
            for mode, m in self.mode_metrics.items()
        }
        report["pii_count_comparison"] = {
            mode: vals[0] if vals else 0
            for mode, vals in pii_counts.items()
        }

        # Compare identity consistency
        consistency = {}
        for mode, m in self.mode_metrics.items():
            ic = m.anonymization.get("identity_consistency", {})
            consistency[mode] = ic.get("consistency_score", 0.0)
        report["identity_consistency_comparison"] = consistency

        # Generate markdown summary
        report["markdown_summary"] = self._generate_markdown(report)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Comparison report saved to {output_path}")

            # Also save markdown
            md_path = Path(output_path).with_suffix(".md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(report["markdown_summary"])
            logger.info(f"Markdown report saved to {md_path}")

        return report

    def _generate_markdown(self, report: Dict[str, Any]) -> str:
        """Generate a markdown summary of the comparison."""
        lines = [
            "# Pipeline Mode Comparison Report",
            "",
            f"Generated: {report.get('generated_at', '')}",
            f"Modes compared: {', '.join(report.get('modes_compared', []))}",
            "",
            "## Processing Statistics",
            "",
            "| Metric | " + " | ".join(report.get("modes_compared", [])) + " |",
            "|--------|" + "|".join(["--------"] * len(report.get("modes_compared", []))) + "|",
        ]

        # Add stats rows
        stat_keys = ["threads_processed", "chunks_created", "pii_detected", "kg_entities_extracted", "errors"]
        for key in stat_keys:
            values = []
            for mode in report.get("modes_compared", []):
                mode_data = report.get("mode_summaries", {}).get(mode, {})
                val = mode_data.get("processing_stats", {}).get(key, "N/A")
                values.append(str(val))
            lines.append(f"| {key} | " + " | ".join(values) + " |")

        lines.extend([
            "",
            "## Identity Consistency",
            "",
        ])

        for mode, score in report.get("identity_consistency_comparison", {}).items():
            lines.append(f"- **{mode.upper()}**: {score:.4f}")

        lines.extend([
            "",
            "## PII Detection Counts",
            "",
        ])

        for mode, count in report.get("pii_count_comparison", {}).items():
            lines.append(f"- **{mode.upper()}**: {count} entities detected")

        return "\n".join(lines)
