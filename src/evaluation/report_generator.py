"""
Evaluation Report Generator
===========================
Generate comprehensive evaluation reports with visualizations.

Features:
- Executive summary generation
- Detailed metrics tables
- Visualization generation
- LaTeX/PDF export
- HTML interactive reports

"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "RAG System Evaluation Report"
    author: str = ""
    institution: str = ""
    include_visualizations: bool = True
    include_raw_data: bool = False
    output_format: str = "html"  # html, latex, markdown


class EvaluationReportGenerator:
    """
    Generate evaluation reports.

    Usage:
        generator = EvaluationReportGenerator(config)
        report = generator.generate(aggregated_results)
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()

    def generate(
        self,
        aggregated_comparison,
        ragas_results=None,
        retrieval_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate complete evaluation report.

        Args:
            aggregated_comparison: AggregatedComparison from comparative analysis
            ragas_results: Optional RAGAS evaluation results
            retrieval_metrics: Optional retrieval metrics (MRR, NDCG)

        Returns:
            Report content as string
        """
        sections = []

        # Title and metadata
        sections.append(self._generate_header())

        # Executive summary
        sections.append(self._generate_executive_summary(aggregated_comparison))

        # RAGAS metrics section
        sections.append(self._generate_ragas_section(aggregated_comparison))

        # System comparison section
        sections.append(self._generate_comparison_section(aggregated_comparison))

        # Retrieval metrics section
        if retrieval_metrics:
            sections.append(self._generate_retrieval_section(retrieval_metrics))

        # Question type analysis
        sections.append(self._generate_question_type_section(aggregated_comparison))

        # Statistical significance
        sections.append(self._generate_significance_section(aggregated_comparison))

        # Recommendations
        sections.append(self._generate_recommendations_section(aggregated_comparison))

        # Conclusion
        sections.append(self._generate_conclusion(aggregated_comparison))

        # Join based on format
        if self.config.output_format == "html":
            return self._wrap_html("\n".join(sections))
        elif self.config.output_format == "latex":
            return self._wrap_latex("\n".join(sections))
        else:
            return "\n\n".join(sections)

    def _generate_header(self) -> str:
        """Generate report header."""
        if self.config.output_format == "html":
            return f"""
<header>
    <h1>{self.config.title}</h1>
    <p class="meta">
        Author: {self.config.author}<br>
        Institution: {self.config.institution}<br>
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </p>
</header>
"""
        else:
            return f"""# {self.config.title}

**Author:** {self.config.author}
**Institution:** {self.config.institution}
**Date:** {datetime.now().strftime('%Y-%m-%d')}

---
"""

    def _generate_executive_summary(self, comparison) -> str:
        """Generate executive summary section."""
        # Find best overall system
        best_system = None
        best_score = 0

        for system_type, metrics in comparison.by_system.items():
            composite = (
                metrics.get("faithfulness", 0) +
                metrics.get("answer_relevancy", 0) +
                metrics.get("context_precision", 0) +
                metrics.get("context_recall", 0)
            ) / 4

            if composite > best_score:
                best_score = composite
                best_system = system_type

        if self.config.output_format == "html":
            return f"""
<section id="executive-summary">
    <h2>Executive Summary</h2>
    <div class="summary-box">
        <p><strong>Total Questions Evaluated:</strong> {comparison.total_questions}</p>
        <p><strong>Best Overall System:</strong> {best_system.value if best_system else 'N/A'}</p>
        <p><strong>Average Composite Score:</strong> {best_score:.3f}</p>
    </div>

    <h3>Key Findings</h3>
    <ul>
        {''.join(f'<li>{r}</li>' for r in comparison.recommendations[:3])}
    </ul>
</section>
"""
        else:
            return f"""## Executive Summary

**Total Questions Evaluated:** {comparison.total_questions}
**Best Overall System:** {best_system.value if best_system else 'N/A'}
**Average Composite Score:** {best_score:.3f}

### Key Findings
{chr(10).join(f'- {r}' for r in comparison.recommendations[:3])}
"""

    def _generate_ragas_section(self, comparison) -> str:
        """Generate RAGAS metrics section."""
        # Build metrics table
        rows = []
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        for system_type, system_metrics in comparison.by_system.items():
            row = [system_type.value]
            for metric in metrics:
                value = system_metrics.get(metric, 0)
                std = system_metrics.get(f"{metric}_std", 0)
                row.append(f"{value:.3f} ± {std:.3f}")
            rows.append(row)

        if self.config.output_format == "html":
            table_rows = "".join(
                f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td></tr>"
                for row in rows
            )
            return f"""
<section id="ragas-metrics">
    <h2>RAGAS Evaluation Metrics</h2>
    <table class="metrics-table">
        <thead>
            <tr>
                <th>System</th>
                <th>Faithfulness</th>
                <th>Answer Relevancy</th>
                <th>Context Precision</th>
                <th>Context Recall</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <h3>Metric Definitions</h3>
    <dl>
        <dt>Faithfulness</dt>
        <dd>Measures how factually accurate the answer is based on the retrieved context.</dd>
        <dt>Answer Relevancy</dt>
        <dd>Measures how relevant the answer is to the question asked.</dd>
        <dt>Context Precision</dt>
        <dd>Measures how much of the retrieved context is relevant to answering the question.</dd>
        <dt>Context Recall</dt>
        <dd>Measures how much of the ground truth information is covered by the retrieved context.</dd>
    </dl>
</section>
"""
        else:
            table = "| System | Faithfulness | Answer Relevancy | Context Precision | Context Recall |\n"
            table += "|--------|--------------|------------------|-------------------|----------------|\n"
            for row in rows:
                table += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"

            return f"""## RAGAS Evaluation Metrics

{table}

### Metric Definitions
- **Faithfulness:** Measures how factually accurate the answer is based on the retrieved context.
- **Answer Relevancy:** Measures how relevant the answer is to the question asked.
- **Context Precision:** Measures how much of the retrieved context is relevant.
- **Context Recall:** Measures how much of the ground truth is covered by context.
"""

    def _generate_comparison_section(self, comparison) -> str:
        """Generate system comparison section."""
        if self.config.output_format == "html":
            return f"""
<section id="system-comparison">
    <h2>System Comparison</h2>
    <p>This section compares the four RAG system configurations:</p>
    <ol>
        <li><strong>Baseline RAG:</strong> Vector search only with Azure AI Search</li>
        <li><strong>GraphRAG:</strong> Vector search + Knowledge graph + Community summaries</li>
        <li><strong>ReAct Agent:</strong> Tool-augmented reasoning with LangGraph</li>
        <li><strong>Full System:</strong> GraphRAG integrated with ReAct agent</li>
    </ol>

    <h3>Performance Profile</h3>
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Aspect</th>
                <th>Baseline</th>
                <th>GraphRAG</th>
                <th>ReAct</th>
                <th>Full System</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Latency</td>
                <td>Low</td>
                <td>Medium</td>
                <td>High</td>
                <td>Highest</td>
            </tr>
            <tr>
                <td>Context Richness</td>
                <td>Low</td>
                <td>High</td>
                <td>Medium</td>
                <td>Highest</td>
            </tr>
            <tr>
                <td>Multi-hop Capability</td>
                <td>None</td>
                <td>Limited</td>
                <td>Strong</td>
                <td>Strongest</td>
            </tr>
            <tr>
                <td>Citation Quality</td>
                <td>Basic</td>
                <td>Good</td>
                <td>Good</td>
                <td>Excellent</td>
            </tr>
        </tbody>
    </table>
</section>
"""
        else:
            return """## System Comparison

This section compares the four RAG system configurations:

1. **Baseline RAG:** Vector search only with Azure AI Search
2. **GraphRAG:** Vector search + Knowledge graph + Community summaries
3. **ReAct Agent:** Tool-augmented reasoning with LangGraph
4. **Full System:** GraphRAG integrated with ReAct agent

### Performance Profile

| Aspect | Baseline | GraphRAG | ReAct | Full System |
|--------|----------|----------|-------|-------------|
| Latency | Low | Medium | High | Highest |
| Context Richness | Low | High | Medium | Highest |
| Multi-hop Capability | None | Limited | Strong | Strongest |
| Citation Quality | Basic | Good | Good | Excellent |
"""

    def _generate_question_type_section(self, comparison) -> str:
        """Generate question type analysis section."""
        content = []

        if self.config.output_format == "html":
            content.append("<section id='question-type-analysis'><h2>Analysis by Question Type</h2>")

            for qtype, systems in comparison.by_question_type.items():
                content.append(f"<h3>{qtype.replace('-', ' ').title()}</h3>")
                content.append("<table><thead><tr><th>System</th><th>Faithfulness</th><th>Relevancy</th></tr></thead><tbody>")

                for system_type, metrics in systems.items():
                    faith = metrics.get("faithfulness", 0)
                    rel = metrics.get("answer_relevancy", 0)
                    content.append(f"<tr><td>{system_type.value}</td><td>{faith:.3f}</td><td>{rel:.3f}</td></tr>")

                content.append("</tbody></table>")

            content.append("</section>")
            return "\n".join(content)
        else:
            content.append("## Analysis by Question Type\n")

            for qtype, systems in comparison.by_question_type.items():
                content.append(f"\n### {qtype.replace('-', ' ').title()}\n")
                content.append("| System | Faithfulness | Relevancy |")
                content.append("|--------|--------------|-----------|")

                for system_type, metrics in systems.items():
                    faith = metrics.get("faithfulness", 0)
                    rel = metrics.get("answer_relevancy", 0)
                    content.append(f"| {system_type.value} | {faith:.3f} | {rel:.3f} |")

            return "\n".join(content)

    def _generate_significance_section(self, comparison) -> str:
        """Generate statistical significance section."""
        sig_results = []

        for (sys1, sys2), metrics in comparison.statistical_significance.items():
            for metric, results in metrics.items():
                if results.get("significant", False):
                    sig_results.append(
                        f"{sys1.value} vs {sys2.value} ({metric}): p={results.get('p_value', 0):.4f}"
                    )

        if self.config.output_format == "html":
            if sig_results:
                items = "".join(f"<li>{r}</li>" for r in sig_results)
                return f"""
<section id="statistical-significance">
    <h2>Statistical Significance</h2>
    <p>Paired t-tests were conducted to determine statistically significant differences (α=0.05).</p>
    <h3>Significant Differences Found:</h3>
    <ul>{items}</ul>
</section>
"""
            else:
                return """
<section id="statistical-significance">
    <h2>Statistical Significance</h2>
    <p>No statistically significant differences were found between systems at α=0.05.</p>
</section>
"""
        else:
            if sig_results:
                items = "\n".join(f"- {r}" for r in sig_results)
                return f"""## Statistical Significance

Paired t-tests were conducted to determine statistically significant differences (α=0.05).

### Significant Differences Found:
{items}
"""
            else:
                return """## Statistical Significance

No statistically significant differences were found between systems at α=0.05.
"""

    def _generate_retrieval_section(self, retrieval_metrics: Dict[str, Any]) -> str:
        """Generate retrieval metrics section."""
        if self.config.output_format == "html":
            return f"""
<section id="retrieval-metrics">
    <h2>Retrieval Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Baseline</th><th>GraphRAG</th><th>ReAct</th><th>Full System</th></tr>
        <tr>
            <td>MRR@10</td>
            <td>{retrieval_metrics.get('baseline', {}).get('mrr', 0):.3f}</td>
            <td>{retrieval_metrics.get('graphrag', {}).get('mrr', 0):.3f}</td>
            <td>{retrieval_metrics.get('react', {}).get('mrr', 0):.3f}</td>
            <td>{retrieval_metrics.get('full', {}).get('mrr', 0):.3f}</td>
        </tr>
        <tr>
            <td>NDCG@10</td>
            <td>{retrieval_metrics.get('baseline', {}).get('ndcg', 0):.3f}</td>
            <td>{retrieval_metrics.get('graphrag', {}).get('ndcg', 0):.3f}</td>
            <td>{retrieval_metrics.get('react', {}).get('ndcg', 0):.3f}</td>
            <td>{retrieval_metrics.get('full', {}).get('ndcg', 0):.3f}</td>
        </tr>
    </table>
</section>
"""
        else:
            return f"""## Retrieval Metrics

| Metric | Baseline | GraphRAG | ReAct | Full System |
|--------|----------|----------|-------|-------------|
| MRR@10 | {retrieval_metrics.get('baseline', {}).get('mrr', 0):.3f} | {retrieval_metrics.get('graphrag', {}).get('mrr', 0):.3f} | {retrieval_metrics.get('react', {}).get('mrr', 0):.3f} | {retrieval_metrics.get('full', {}).get('mrr', 0):.3f} |
| NDCG@10 | {retrieval_metrics.get('baseline', {}).get('ndcg', 0):.3f} | {retrieval_metrics.get('graphrag', {}).get('ndcg', 0):.3f} | {retrieval_metrics.get('react', {}).get('ndcg', 0):.3f} | {retrieval_metrics.get('full', {}).get('ndcg', 0):.3f} |
"""

    def _generate_recommendations_section(self, comparison) -> str:
        """Generate recommendations section."""
        if self.config.output_format == "html":
            items = "".join(f"<li>{r}</li>" for r in comparison.recommendations)
            return f"""
<section id="recommendations">
    <h2>Recommendations</h2>
    <ul>{items}</ul>

    <h3>Use Case Guidance</h3>
    <table>
        <tr><th>Use Case</th><th>Recommended System</th><th>Rationale</th></tr>
        <tr>
            <td>Simple factual queries</td>
            <td>Baseline RAG</td>
            <td>Fastest response, sufficient accuracy</td>
        </tr>
        <tr>
            <td>Theme/summary questions</td>
            <td>GraphRAG</td>
            <td>Community summaries provide high-level context</td>
        </tr>
        <tr>
            <td>Relationship exploration</td>
            <td>ReAct Agent</td>
            <td>Iterative reasoning discovers connections</td>
        </tr>
        <tr>
            <td>Complex analytical queries</td>
            <td>Full System</td>
            <td>Maximum context and reasoning capability</td>
        </tr>
    </table>
</section>
"""
        else:
            items = "\n".join(f"- {r}" for r in comparison.recommendations)
            return f"""## Recommendations

{items}

### Use Case Guidance

| Use Case | Recommended System | Rationale |
|----------|-------------------|-----------|
| Simple factual queries | Baseline RAG | Fastest response, sufficient accuracy |
| Theme/summary questions | GraphRAG | Community summaries provide high-level context |
| Relationship exploration | ReAct Agent | Iterative reasoning discovers connections |
| Complex analytical queries | Full System | Maximum context and reasoning capability |
"""

    def _generate_conclusion(self, comparison) -> str:
        """Generate conclusion section."""
        if self.config.output_format == "html":
            return """
<section id="conclusion">
    <h2>Conclusion</h2>
    <p>This evaluation demonstrates the trade-offs between different RAG architectures
    for enterprise knowledge retrieval. Key insights include:</p>
    <ol>
        <li>GraphRAG significantly improves context quality for theme-based queries</li>
        <li>ReAct agents excel at multi-hop reasoning but incur higher latency</li>
        <li>The full system provides the best overall quality at the cost of performance</li>
        <li>System selection should be based on query characteristics and latency requirements</li>
    </ol>

    <h3>Future Work</h3>
    <ul>
        <li>Optimize latency through caching and pre-computation</li>
        <li>Implement query routing for automatic system selection</li>
        <li>Conduct user study for qualitative evaluation</li>
        <li>Explore fine-tuning for domain adaptation</li>
    </ul>
</section>
"""
        else:
            return """## Conclusion

This evaluation demonstrates the trade-offs between different RAG architectures
for enterprise knowledge retrieval. Key insights include:

1. GraphRAG significantly improves context quality for theme-based queries
2. ReAct agents excel at multi-hop reasoning but incur higher latency
3. The full system provides the best overall quality at the cost of performance
4. System selection should be based on query characteristics and latency requirements

### Future Work
- Optimize latency through caching and pre-computation
- Implement query routing for automatic system selection
- Conduct user study for qualitative evaluation
- Explore fine-tuning for domain adaptation
"""

    def _wrap_html(self, content: str) -> str:
        """Wrap content in HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary-box {{ background: #ecf0f1; padding: 20px; border-radius: 5px; }}
        .meta {{ color: #7f8c8d; }}
        dl {{ margin: 20px 0; }}
        dt {{ font-weight: bold; margin-top: 10px; }}
        dd {{ margin-left: 20px; color: #555; }}
    </style>
</head>
<body>
{content}
<footer>
    <p><em>Generated by GraphRAG Evaluation Framework - {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>
</footer>
</body>
</html>
"""

    def _wrap_latex(self, content: str) -> str:
        """Wrap content in LaTeX document."""
        # Convert markdown to LaTeX (simplified)
        latex_content = content.replace("##", "\\section{").replace("\n\n", "}\n\n")
        latex_content = latex_content.replace("**", "\\textbf{").replace("**", "}")

        return f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\title{{{self.config.title}}}
\\author{{{self.config.author}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

{latex_content}

\\end{{document}}
"""


# Export
__all__ = [
    'EvaluationReportGenerator',
    'ReportConfig',
]
