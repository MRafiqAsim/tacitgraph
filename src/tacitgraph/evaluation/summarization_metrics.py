"""
Summarization Quality Metrics

Evaluates the quality of text summarization using multiple metrics:
- ROUGE scores (lexical overlap)
- Semantic similarity (embedding-based)
- Faithfulness (factual consistency)
- Coverage (key information retention)
- Compression ratio

For LLM-generated summaries, we also use LLM-as-judge evaluation.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SummarizationMetrics:
    """Complete summarization evaluation metrics"""

    # ROUGE scores (0-1, higher is better)
    rouge_1: float = 0.0  # Unigram overlap
    rouge_2: float = 0.0  # Bigram overlap
    rouge_l: float = 0.0  # Longest common subsequence

    # Semantic similarity (0-1, higher is better)
    semantic_similarity: float = 0.0

    # Faithfulness (0-1, higher is better) - no hallucinations
    faithfulness: float = 0.0

    # Coverage (0-1, higher is better) - key info retained
    coverage: float = 0.0

    # Compression metrics
    compression_ratio: float = 0.0  # summary_len / source_len

    # LLM-as-judge scores (1-5 scale)
    llm_relevance: float = 0.0
    llm_coherence: float = 0.0
    llm_fluency: float = 0.0
    llm_overall: float = 0.0

    # Metadata
    source_length: int = 0
    summary_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rouge": {
                "rouge_1": round(self.rouge_1, 4),
                "rouge_2": round(self.rouge_2, 4),
                "rouge_l": round(self.rouge_l, 4),
            },
            "semantic_similarity": round(self.semantic_similarity, 4),
            "faithfulness": round(self.faithfulness, 4),
            "coverage": round(self.coverage, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "llm_scores": {
                "relevance": round(self.llm_relevance, 2),
                "coherence": round(self.llm_coherence, 2),
                "fluency": round(self.llm_fluency, 2),
                "overall": round(self.llm_overall, 2),
            },
            "lengths": {
                "source": self.source_length,
                "summary": self.summary_length,
            },
        }

    def to_report(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║              SUMMARIZATION QUALITY REPORT                        ║
╠══════════════════════════════════════════════════════════════════╣

LEXICAL METRICS (ROUGE)
───────────────────────
ROUGE-1 (unigram):     {self.rouge_1:.4f}
ROUGE-2 (bigram):      {self.rouge_2:.4f}
ROUGE-L (LCS):         {self.rouge_l:.4f}

SEMANTIC METRICS
────────────────
Semantic Similarity:   {self.semantic_similarity:.4f}
Faithfulness:          {self.faithfulness:.4f}
Coverage:              {self.coverage:.4f}

COMPRESSION
───────────
Source length:         {self.source_length} chars
Summary length:        {self.summary_length} chars
Compression ratio:     {self.compression_ratio:.2%}

LLM-AS-JUDGE (1-5 scale)
────────────────────────
Relevance:             {self.llm_relevance:.1f}/5
Coherence:             {self.llm_coherence:.1f}/5
Fluency:               {self.llm_fluency:.1f}/5
Overall:               {self.llm_overall:.1f}/5
"""


class SummarizationEvaluator:
    """
    Evaluate summarization quality using multiple metrics.

    Supports both reference-based (with gold summaries) and
    reference-free evaluation (using LLM-as-judge).
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o",
        use_embeddings: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            openai_api_key: OpenAI API key for LLM-as-judge and embeddings
            openai_model: Model for LLM-as-judge
            use_embeddings: Use embeddings for semantic similarity
        """
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.use_embeddings = use_embeddings
        self._openai_client = None

        if openai_api_key:
            try:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=openai_api_key)
            except ImportError:
                logger.warning("OpenAI package not installed")

    def evaluate(
        self, source_text: str, summary: str, reference_summary: str | None = None
    ) -> SummarizationMetrics:
        """
        Evaluate a summary.

        Args:
            source_text: Original source text
            summary: Generated summary to evaluate
            reference_summary: Optional gold/reference summary

        Returns:
            SummarizationMetrics with all scores
        """
        metrics = SummarizationMetrics(
            source_length=len(source_text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(source_text) if source_text else 0,
        )

        # ROUGE scores
        if reference_summary:
            rouge_scores = self._calculate_rouge(summary, reference_summary)
            metrics.rouge_1 = rouge_scores["rouge_1"]
            metrics.rouge_2 = rouge_scores["rouge_2"]
            metrics.rouge_l = rouge_scores["rouge_l"]

        # Semantic similarity (summary vs source)
        if self._openai_client and self.use_embeddings:
            metrics.semantic_similarity = self._calculate_semantic_similarity(source_text, summary)

        # Faithfulness and coverage using LLM
        if self._openai_client:
            faith_cov = self._evaluate_faithfulness_coverage(source_text, summary)
            metrics.faithfulness = faith_cov["faithfulness"]
            metrics.coverage = faith_cov["coverage"]

            # LLM-as-judge evaluation
            llm_scores = self._llm_judge_evaluation(source_text, summary)
            metrics.llm_relevance = llm_scores["relevance"]
            metrics.llm_coherence = llm_scores["coherence"]
            metrics.llm_fluency = llm_scores["fluency"]
            metrics.llm_overall = llm_scores["overall"]

        return metrics

    def evaluate_batch(
        self,
        source_texts: list[str],
        summaries: list[str],
        reference_summaries: list[str] | None = None,
    ) -> tuple[list[SummarizationMetrics], SummarizationMetrics]:
        """
        Evaluate multiple summaries and return individual + aggregate metrics.

        Returns:
            (list of individual metrics, aggregate metrics)
        """
        if reference_summaries is None:
            reference_summaries = [None] * len(summaries)

        individual_metrics = []
        for source, summary, ref in zip(source_texts, summaries, reference_summaries, strict=False):
            metrics = self.evaluate(source, summary, ref)
            individual_metrics.append(metrics)

        # Calculate aggregate
        aggregate = self._aggregate_metrics(individual_metrics)

        return individual_metrics, aggregate

    def _calculate_rouge(self, summary: str, reference: str) -> dict[str, float]:
        """Calculate ROUGE scores"""
        # Tokenize
        summary_tokens = self._tokenize(summary)
        reference_tokens = self._tokenize(reference)

        # ROUGE-1 (unigram)
        rouge_1 = self._rouge_n(summary_tokens, reference_tokens, n=1)

        # ROUGE-2 (bigram)
        rouge_2 = self._rouge_n(summary_tokens, reference_tokens, n=2)

        # ROUGE-L (longest common subsequence)
        rouge_l = self._rouge_l(summary_tokens, reference_tokens)

        return {"rouge_1": rouge_1, "rouge_2": rouge_2, "rouge_l": rouge_l}

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization"""
        # Lowercase and split on whitespace/punctuation
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _get_ngrams(self, tokens: list[str], n: int) -> list[tuple[str, ...]]:
        """Get n-grams from tokens"""
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _rouge_n(self, summary_tokens: list[str], reference_tokens: list[str], n: int) -> float:
        """Calculate ROUGE-N score"""
        summary_ngrams = Counter(self._get_ngrams(summary_tokens, n))
        reference_ngrams = Counter(self._get_ngrams(reference_tokens, n))

        # Overlap
        overlap = sum((summary_ngrams & reference_ngrams).values())

        # Recall-based ROUGE
        total_ref = sum(reference_ngrams.values())
        if total_ref == 0:
            return 0.0

        return overlap / total_ref

    def _rouge_l(self, summary_tokens: list[str], reference_tokens: list[str]) -> float:
        """Calculate ROUGE-L using longest common subsequence"""
        lcs_length = self._lcs_length(summary_tokens, reference_tokens)

        if len(reference_tokens) == 0:
            return 0.0

        # F1-based ROUGE-L
        precision = lcs_length / len(summary_tokens) if summary_tokens else 0
        recall = lcs_length / len(reference_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _lcs_length(self, x: list[str], y: list[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _calculate_semantic_similarity(self, source: str, summary: str) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            # Get embeddings
            response = self._openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[source[:8000], summary],  # Truncate if too long
            )

            source_emb = response.data[0].embedding
            summary_emb = response.data[1].embedding

            # Cosine similarity
            import math

            dot_product = sum(a * b for a, b in zip(source_emb, summary_emb, strict=False))
            norm_a = math.sqrt(sum(a * a for a in source_emb))
            norm_b = math.sqrt(sum(b * b for b in summary_emb))

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return dot_product / (norm_a * norm_b)

        except Exception as e:
            logger.warning(f"Embedding calculation failed: {e}")
            return 0.0

    def _evaluate_faithfulness_coverage(self, source: str, summary: str) -> dict[str, float]:
        """Evaluate faithfulness and coverage using LLM"""
        import json

        prompt = f"""Evaluate this summary for faithfulness and coverage.

SOURCE TEXT:
\"\"\"
{source[:4000]}
\"\"\"

SUMMARY:
\"\"\"
{summary}
\"\"\"

Evaluate:
1. FAITHFULNESS (0-1): Does the summary only contain information from the source?
   - 1.0 = Perfect, no hallucinations
   - 0.0 = Contains made-up information

2. COVERAGE (0-1): Does the summary capture the key information?
   - 1.0 = All important points included
   - 0.0 = Missing critical information

Return JSON only:
{{"faithfulness": 0.X, "coverage": 0.X, "reasoning": "brief explanation"}}"""

        try:
            response = self._openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            content = extract_llm_content(response, context="faithfulness evaluation")
            result = json.loads(content)
            return {
                "faithfulness": float(result.get("faithfulness", 0)),
                "coverage": float(result.get("coverage", 0)),
            }

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # JSONDecodeError, etc. Returns zero scores so evaluation can continue.
            logger.warning(f"Faithfulness/coverage evaluation failed: {e}")
            return {"faithfulness": 0.0, "coverage": 0.0}

    def _llm_judge_evaluation(self, source: str, summary: str) -> dict[str, float]:
        """LLM-as-judge evaluation for quality dimensions"""
        import json

        prompt = f"""Rate this summary on a scale of 1-5 for each dimension.

SOURCE TEXT:
\"\"\"
{source[:4000]}
\"\"\"

SUMMARY:
\"\"\"
{summary}
\"\"\"

Rate each dimension (1=Poor, 5=Excellent):

1. RELEVANCE: Does the summary focus on the most important information?
2. COHERENCE: Is the summary well-organized and logically structured?
3. FLUENCY: Is the summary grammatically correct and easy to read?
4. OVERALL: Overall quality of the summary

Return JSON only:
{{"relevance": X, "coherence": X, "fluency": X, "overall": X}}"""

        try:
            response = self._openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            content = extract_llm_content(response, context="LLM judge evaluation")
            result = json.loads(content)
            return {
                "relevance": float(result.get("relevance", 0)),
                "coherence": float(result.get("coherence", 0)),
                "fluency": float(result.get("fluency", 0)),
                "overall": float(result.get("overall", 0)),
            }

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # JSONDecodeError, etc. Returns zero scores so evaluation can continue.
            logger.warning(f"LLM judge evaluation failed: {e}")
            return {"relevance": 0, "coherence": 0, "fluency": 0, "overall": 0}

    def _aggregate_metrics(self, metrics_list: list[SummarizationMetrics]) -> SummarizationMetrics:
        """Calculate aggregate metrics from a list"""
        if not metrics_list:
            return SummarizationMetrics()

        n = len(metrics_list)

        return SummarizationMetrics(
            rouge_1=sum(m.rouge_1 for m in metrics_list) / n,
            rouge_2=sum(m.rouge_2 for m in metrics_list) / n,
            rouge_l=sum(m.rouge_l for m in metrics_list) / n,
            semantic_similarity=sum(m.semantic_similarity for m in metrics_list) / n,
            faithfulness=sum(m.faithfulness for m in metrics_list) / n,
            coverage=sum(m.coverage for m in metrics_list) / n,
            compression_ratio=sum(m.compression_ratio for m in metrics_list) / n,
            llm_relevance=sum(m.llm_relevance for m in metrics_list) / n,
            llm_coherence=sum(m.llm_coherence for m in metrics_list) / n,
            llm_fluency=sum(m.llm_fluency for m in metrics_list) / n,
            llm_overall=sum(m.llm_overall for m in metrics_list) / n,
            source_length=sum(m.source_length for m in metrics_list),
            summary_length=sum(m.summary_length for m in metrics_list),
        )


# Convenience function
def evaluate_summary(
    source_text: str,
    summary: str,
    reference_summary: str | None = None,
    openai_api_key: str | None = None,
) -> SummarizationMetrics:
    """
    Evaluate a summary's quality.

    Args:
        source_text: Original source text
        summary: Generated summary
        reference_summary: Optional gold/reference summary
        openai_api_key: OpenAI API key for advanced metrics

    Returns:
        SummarizationMetrics
    """
    evaluator = SummarizationEvaluator(openai_api_key=openai_api_key)
    return evaluator.evaluate(source_text, summary, reference_summary)
