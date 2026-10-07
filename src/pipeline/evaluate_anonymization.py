#!/usr/bin/env python3
"""
Anonymization Evaluation Script

Evaluates PII detection accuracy against ground truth.

Usage:
    python evaluate_anonymization.py --ground-truth ../data/evaluation/ground_truth_sample.json
    python evaluate_anonymization.py --ground-truth ground_truth.json --output results.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from silver.pii_detector import PIIDetector, PIIType as DetectorPIIType
from conflict_handling.pii_evaluation import (
    PIIEvaluator,
    GroundTruthLoader,
    AnnotatedDocument,
)
from conflict_handling.models import PIIAnnotation, PIIType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectorWrapper:
    """
    Wrapper to adapt PIIDetector to the evaluation interface.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.detector = PIIDetector(
            confidence_threshold=confidence_threshold,
            use_presidio=True,
            use_spacy=True,
            use_regex=True
        )

    def detect(self, text: str, language: str = "en") -> list:
        """
        Detect PII and convert to evaluation format.
        """
        # Detect using our detector
        entities = self.detector.detect(text, language)

        # Convert to evaluation format
        eval_entities = []
        for e in entities:
            try:
                # Map detector PIIType to evaluation PIIType
                pii_type = PIIType(e.pii_type.value)

                eval_entity = PIIAnnotation(
                    text=e.text,
                    pii_type=pii_type,
                    start_char=e.start,
                    end_char=e.end,
                    confidence=e.confidence,
                    is_sensitive=True
                )
                eval_entities.append(eval_entity)

            except (ValueError, KeyError) as err:
                # Skip if type mapping fails
                logger.debug(f"Skipping entity with unknown type: {e.pii_type}")

        return eval_entities


def run_evaluation(
    ground_truth_path: str,
    confidence_threshold: float = 0.5,
    output_path: str = None
) -> dict:
    """
    Run the anonymization evaluation.

    Args:
        ground_truth_path: Path to ground truth JSON
        confidence_threshold: Minimum confidence for detection
        output_path: Optional path to save results

    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Loading ground truth from: {ground_truth_path}")

    # Load ground truth
    ground_truth = GroundTruthLoader.load_from_json(ground_truth_path)
    logger.info(f"Loaded {len(ground_truth)} documents")

    # Initialize detector
    logger.info("Initializing PII detector...")
    detector = DetectorWrapper(confidence_threshold=confidence_threshold)

    # Initialize evaluator
    evaluator = PIIEvaluator(
        overlap_threshold=0.5,
        require_type_match=True
    )

    # Run evaluation
    logger.info("Running evaluation...")
    result = evaluator.evaluate(ground_truth, detector)

    # Print report
    print("\n" + "=" * 70)
    print(result.to_report())
    print("=" * 70)

    # Prepare results dictionary
    results = {
        "timestamp": datetime.now().isoformat(),
        "ground_truth_path": ground_truth_path,
        "confidence_threshold": confidence_threshold,
        "document_count": len(ground_truth),
        "metrics": {
            "total_ground_truth": result.total_ground_truth,
            "total_detected": result.total_detected,
            "true_positives": result.true_positives,
            "false_positives": result.false_positives,
            "false_negatives": result.false_negatives,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
        },
        "metrics_by_type": result.metrics_by_type,
        "thresholds_met": {
            "recall_95": result.recall >= 0.95,
            "precision_90": result.precision >= 0.90,
            "f1_92": result.f1_score >= 0.92,
        }
    }

    # Add false negative details (for debugging)
    if result.fn_details:
        results["false_negatives_sample"] = [
            {
                "text": fn.text,
                "type": fn.pii_type.value if hasattr(fn.pii_type, 'value') else str(fn.pii_type),
            }
            for fn in result.fn_details[:10]  # First 10
        ]

    # Add false positive details
    if result.fp_details:
        results["false_positives_sample"] = [
            {
                "text": fp.text,
                "type": fp.pii_type.value if hasattr(fp.pii_type, 'value') else str(fp.pii_type),
            }
            for fp in result.fp_details[:10]
        ]

    # Save results if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    return results


def run_quick_test():
    """
    Run a quick test with sample text.
    """
    print("\n" + "=" * 70)
    print("QUICK TEST - PII Detection")
    print("=" * 70)

    test_texts = [
        {
            "text": "Contact John Smith at john.smith@email.com or call +1 555-123-4567.",
            "language": "en",
            "expected": ["John Smith", "john.smith@email.com", "+1 555-123-4567"]
        },
        {
            "text": "Beste Jan, mijn IBAN is NL91 ABNA 0417 1643 00. Groeten, Pieter.",
            "language": "nl",
            "expected": ["Jan", "NL91 ABNA 0417 1643 00", "Pieter"]
        }
    ]

    detector = PIIDetector()

    for test in test_texts:
        print(f"\nInput ({test['language']}): {test['text']}")
        print(f"Expected: {test['expected']}")

        entities = detector.detect(test['text'], test['language'])

        print(f"Detected: {[e.text for e in entities]}")
        print(f"Details:")
        for e in entities:
            print(f"  - {e.text} ({e.pii_type.value}, conf={e.confidence:.2f}, method={e.detection_method})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PII anonymization accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate against ground truth
  python evaluate_anonymization.py --ground-truth ground_truth.json

  # Save results to file
  python evaluate_anonymization.py --ground-truth ground_truth.json --output results.json

  # Quick test without ground truth
  python evaluate_anonymization.py --quick-test
        """
    )

    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        help="Path to ground truth JSON file"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save evaluation results"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detection (default: 0.5)"
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with sample text"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.quick_test:
        run_quick_test()
        return

    if not args.ground_truth:
        # Default to sample ground truth
        default_path = Path(__file__).parent.parent.parent / "data/evaluation/ground_truth_sample.json"
        if default_path.exists():
            args.ground_truth = str(default_path)
            logger.info(f"Using default ground truth: {args.ground_truth}")
        else:
            parser.error("Please specify --ground-truth or use --quick-test")

    try:
        results = run_evaluation(
            ground_truth_path=args.ground_truth,
            confidence_threshold=args.confidence,
            output_path=args.output
        )

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Precision: {results['metrics']['precision']:.2%} (target: >90%)")
        print(f"Recall:    {results['metrics']['recall']:.2%} (target: >95%)")
        print(f"F1 Score:  {results['metrics']['f1_score']:.2%} (target: >92%)")

        # Check thresholds
        all_met = all(results['thresholds_met'].values())
        if all_met:
            print("\n✓ All quality thresholds MET!")
        else:
            print("\n✗ Some thresholds NOT met:")
            for threshold, met in results['thresholds_met'].items():
                status = "✓" if met else "✗"
                print(f"  {status} {threshold}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
