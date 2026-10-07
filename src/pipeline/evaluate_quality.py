#!/usr/bin/env python3
"""
Comprehensive Quality Evaluation Script

Evaluates both anonymization and summarization quality.

Usage:
    # Evaluate anonymization only
    python evaluate_quality.py --anonymization --ground-truth ground_truth.json

    # Evaluate summarization only
    python evaluate_quality.py --summarization --silver ./data/silver

    # Evaluate both
    python evaluate_quality.py --all --silver ./data/silver --ground-truth ground_truth.json

    # Quick demo
    python evaluate_quality.py --demo
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config, init_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_anonymization_quality(
    ground_truth_path: str,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate anonymization/PII detection quality against ground truth.

    Returns metrics: precision, recall, F1, per-type breakdown
    """
    from silver.pii_detector import PIIDetector
    from conflict_handling.pii_evaluation import PIIEvaluator, GroundTruthLoader
    from conflict_handling.models import PIIAnnotation, PIIType

    print("\n" + "=" * 70)
    print("ANONYMIZATION QUALITY EVALUATION")
    print("=" * 70)

    # Load ground truth
    ground_truth = GroundTruthLoader.load_from_json(ground_truth_path)
    logger.info(f"Loaded {len(ground_truth)} documents from ground truth")

    # Initialize detector based on config
    config = get_config()

    if config.mode.value == "openai":
        from silver.openai_pii_detector import OpenAIPIIDetector

        class DetectorWrapper:
            def __init__(self):
                self.detector = OpenAIPIIDetector(
                    api_key=config.openai.api_key,
                    model=config.openai.model,
                    confidence_threshold=confidence_threshold
                )

            def detect(self, text, language="en"):
                entities = self.detector.detect(text, language)
                return [
                    PIIAnnotation(
                        text=e.text,
                        pii_type=PIIType(e.pii_type.value),
                        start_char=e.start,
                        end_char=e.end,
                        confidence=e.confidence,
                        is_sensitive=True
                    )
                    for e in entities
                ]

        detector = DetectorWrapper()
        print(f"Using: OpenAI mode ({config.openai.model})")

    else:
        # Local mode
        class DetectorWrapper:
            def __init__(self):
                self.detector = PIIDetector(
                    confidence_threshold=confidence_threshold
                )

            def detect(self, text, language="en"):
                entities = self.detector.detect(text, language)
                return [
                    PIIAnnotation(
                        text=e.text,
                        pii_type=PIIType(e.pii_type.value),
                        start_char=e.start,
                        end_char=e.end,
                        confidence=e.confidence,
                        is_sensitive=True
                    )
                    for e in entities
                ]

        detector = DetectorWrapper()
        print(f"Using: Local mode (Presidio/spaCy/regex)")

    # Evaluate
    evaluator = PIIEvaluator(overlap_threshold=0.5, require_type_match=True)
    result = evaluator.evaluate(ground_truth, detector)

    # Print report
    print(result.to_report())

    return {
        "mode": config.mode.value,
        "total_ground_truth": result.total_ground_truth,
        "total_detected": result.total_detected,
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "metrics_by_type": result.metrics_by_type,
    }


def evaluate_summarization_quality(
    silver_path: str,
    sample_size: int = 10
) -> Dict[str, Any]:
    """
    Evaluate summarization quality on Silver layer data.

    Uses LLM-as-judge for faithfulness, coverage, and quality scores.
    """
    from evaluation.summarization_metrics import SummarizationEvaluator

    print("\n" + "=" * 70)
    print("SUMMARIZATION QUALITY EVALUATION")
    print("=" * 70)

    config = get_config()

    if not config.openai.api_key:
        print("\n⚠️  OpenAI API key required for summarization evaluation")
        print("   Set OPENAI_API_KEY environment variable")
        return {"error": "API key not set"}

    # Load Silver layer chunks
    chunks_dir = Path(silver_path) / "chunks"
    if not chunks_dir.exists():
        print(f"\n⚠️  Chunks directory not found: {chunks_dir}")
        return {"error": "No chunks found"}

    # Load sample chunks
    chunks = []
    for json_file in list(chunks_dir.glob("*.json"))[:sample_size]:
        with open(json_file) as f:
            chunk = json.load(f)
            if chunk.get("summary") and chunk.get("text_anonymized"):
                chunks.append(chunk)

    if not chunks:
        print("\n⚠️  No chunks with summaries found")
        print("   Make sure summarization is enabled in the pipeline")
        return {"error": "No summaries found"}

    print(f"Evaluating {len(chunks)} summaries...")

    # Initialize evaluator
    evaluator = SummarizationEvaluator(
        openai_api_key=config.openai.api_key,
        openai_model=config.openai.model
    )

    # Evaluate each summary
    all_metrics = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("text_anonymized", "")
        summary = chunk.get("summary", "")

        print(f"  Evaluating {i+1}/{len(chunks)}...", end="\r")

        metrics = evaluator.evaluate(source, summary)
        all_metrics.append(metrics)

    # Aggregate metrics
    _, aggregate = evaluator.evaluate_batch(
        [c.get("text_anonymized", "") for c in chunks],
        [c.get("summary", "") for c in chunks]
    )

    # Print report
    print(aggregate.to_report())

    return aggregate.to_dict()


def run_demo():
    """Run demonstration with sample data"""
    print("\n" + "=" * 70)
    print("QUALITY EVALUATION DEMO")
    print("=" * 70)

    config = get_config()

    # Demo text
    demo_text = """
    Dear Dr. Sarah Johnson,

    Thank you for your email regarding patient Michael Brown (DOB: 03/15/1985).
    I've reviewed the medical records you sent for case #MED-2024-0542.

    The patient was admitted to Memorial Hospital on December 28, 2023.
    Please contact me at sarah.j@hospital.org or call (555) 987-6543.

    Best regards,
    Dr. Robert Williams
    Chief Medical Officer
    """

    print("\n--- Original Text ---")
    print(demo_text)

    # Process with unified processor
    from silver.unified_processor import UnifiedProcessor

    try:
        processor = UnifiedProcessor(config)
        result = processor.process(demo_text, include_summary=True)

        print("\n--- Anonymized Text ---")
        print(result.anonymized_text)

        print(f"\n--- Detected PII ({len(result.pii_entities)} entities) ---")
        for entity in result.pii_entities:
            print(f"  • {entity.text} ({entity.pii_type.value})")

        if result.summary:
            print("\n--- Summary ---")
            print(result.summary)

            # Evaluate summary quality
            if config.openai.api_key:
                print("\n--- Summary Quality Evaluation ---")
                from evaluation.summarization_metrics import SummarizationEvaluator

                evaluator = SummarizationEvaluator(
                    openai_api_key=config.openai.api_key
                )
                metrics = evaluator.evaluate(result.anonymized_text, result.summary)
                print(metrics.to_report())

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate anonymization and summarization quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo
  python evaluate_quality.py --demo

  # Evaluate anonymization against ground truth
  python evaluate_quality.py --anonymization --ground-truth ground_truth.json

  # Evaluate summarization quality
  python evaluate_quality.py --summarization --silver ./data/silver

  # Evaluate both
  python evaluate_quality.py --all --silver ./data/silver --ground-truth ground_truth.json

Environment:
  OPENAI_API_KEY - Required for OpenAI mode and summarization evaluation
  PIPELINE_MODE  - Processing mode (openai, local, hybrid)
        """
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration"
    )

    parser.add_argument(
        "--anonymization", "-a",
        action="store_true",
        help="Evaluate anonymization quality"
    )

    parser.add_argument(
        "--summarization", "-s",
        action="store_true",
        help="Evaluate summarization quality"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate both anonymization and summarization"
    )

    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        help="Path to ground truth JSON for anonymization evaluation"
    )

    parser.add_argument(
        "--silver",
        type=str,
        default="./data/silver",
        help="Path to Silver layer for summarization evaluation"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save evaluation results"
    )

    parser.add_argument(
        "--mode",
        choices=["openai", "local", "hybrid"],
        help="Override processing mode"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of summaries to evaluate (default: 10)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize config with override if specified
    if args.mode:
        init_config(mode=args.mode)

    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": get_config().mode.value,
    }

    if args.demo:
        run_demo()
        return

    if args.all or args.anonymization:
        if not args.ground_truth:
            # Try default path
            default_gt = Path(__file__).parent.parent.parent / "data/evaluation/ground_truth_sample.json"
            if default_gt.exists():
                args.ground_truth = str(default_gt)
            else:
                parser.error("--ground-truth required for anonymization evaluation")

        results["anonymization"] = evaluate_anonymization_quality(args.ground_truth)

    if args.all or args.summarization:
        results["summarization"] = evaluate_summarization_quality(
            args.silver,
            sample_size=args.sample_size
        )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    if "anonymization" in results:
        a = results["anonymization"]
        print(f"\nAnonymization (Mode: {a.get('mode', 'N/A')}):")
        print(f"  Precision: {a.get('precision', 0):.2%}")
        print(f"  Recall:    {a.get('recall', 0):.2%}")
        print(f"  F1 Score:  {a.get('f1_score', 0):.2%}")

    if "summarization" in results and "error" not in results["summarization"]:
        s = results["summarization"]
        print(f"\nSummarization:")
        print(f"  Faithfulness: {s.get('faithfulness', 0):.2%}")
        print(f"  Coverage:     {s.get('coverage', 0):.2%}")
        print(f"  LLM Overall:  {s.get('llm_scores', {}).get('overall', 0):.1f}/5")


if __name__ == "__main__":
    main()
