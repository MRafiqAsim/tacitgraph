#!/usr/bin/env python3
"""
Privacy Metrics Evaluation Script

Evaluates advanced privacy metrics for anonymized data:
- K-anonymity
- L-diversity
- T-closeness
- Re-identification Risk
- Closest Distances Ratio (CDR)

Usage:
    python evaluate_privacy.py --silver ./data/silver
    python evaluate_privacy.py --silver ./data/silver --output privacy_report.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from silver.privacy_metrics import (
    PrivacyMetricsCalculator,
    PrivacyMetricsResult,
    TextPrivacyAnalyzer,
    AnonymizedRecord,
    QuasiIdentifier,
    SensitiveAttribute,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_silver_layer_chunks(silver_path: str) -> List[Dict[str, Any]]:
    """
    Load processed chunks from Silver layer.

    Args:
        silver_path: Path to Silver layer

    Returns:
        List of chunk dictionaries
    """
    chunks_dir = Path(silver_path) / "chunks"

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    chunks = []
    for json_file in chunks_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                chunk = json.load(f)
                chunks.append(chunk)
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")

    logger.info(f"Loaded {len(chunks)} chunks from Silver layer")
    return chunks


def chunks_to_anonymized_records(chunks: List[Dict[str, Any]]) -> List[AnonymizedRecord]:
    """
    Convert Silver layer chunks to AnonymizedRecord objects.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        List of AnonymizedRecord objects
    """
    records = []

    for chunk in chunks:
        # Extract quasi-identifiers from chunk metadata
        quasi_identifiers = []

        # Language as QI
        if chunk.get("language"):
            quasi_identifiers.append(
                QuasiIdentifier(name="language", value=chunk["language"])
            )

        # Document ID prefix as QI (first 4 chars)
        if chunk.get("doc_id"):
            doc_prefix = chunk["doc_id"][:4] if len(chunk["doc_id"]) >= 4 else chunk["doc_id"]
            quasi_identifiers.append(
                QuasiIdentifier(name="doc_prefix", value=doc_prefix)
            )

        # PII count bucket as QI
        pii_count = chunk.get("pii_count", 0)
        pii_bucket = "none" if pii_count == 0 else "low" if pii_count <= 2 else "medium" if pii_count <= 5 else "high"
        quasi_identifiers.append(
            QuasiIdentifier(name="pii_level", value=pii_bucket)
        )

        # Token count bucket as QI
        token_count = chunk.get("token_count", 0)
        token_bucket = token_count // 100 * 100  # Round to nearest 100
        quasi_identifiers.append(
            QuasiIdentifier(name="token_bucket", value=token_bucket)
        )

        # Entity types as QI
        entity_types = set()
        for entity in chunk.get("entities", []):
            if entity.get("type"):
                entity_types.add(entity["type"])
        entity_signature = "_".join(sorted(entity_types)) if entity_types else "none"
        quasi_identifiers.append(
            QuasiIdentifier(name="entity_types", value=entity_signature)
        )

        # PII types present as QI
        pii_types = set()
        for pii in chunk.get("pii_entities", []):
            if pii.get("type"):
                pii_types.add(pii["type"])
        pii_signature = "_".join(sorted(pii_types)) if pii_types else "none"
        quasi_identifiers.append(
            QuasiIdentifier(name="pii_types", value=pii_signature)
        )

        # Sensitive attributes (original text characteristics)
        sensitive_attributes = []

        # Content hash as sensitive attribute
        import hashlib
        original_text = chunk.get("text_original", "")
        content_hash = hashlib.md5(original_text.encode()).hexdigest()[:8]
        sensitive_attributes.append(
            SensitiveAttribute(name="content_hash", value=content_hash)
        )

        # Source file as sensitive attribute
        if chunk.get("source_file"):
            sensitive_attributes.append(
                SensitiveAttribute(name="source", value=chunk["source_file"])
            )

        record = AnonymizedRecord(
            record_id=chunk.get("chunk_id", str(len(records))),
            quasi_identifiers=quasi_identifiers,
            sensitive_attributes=sensitive_attributes,
            original_text=chunk.get("text_original"),
            anonymized_text=chunk.get("text_anonymized")
        )
        records.append(record)

    return records


def evaluate_silver_layer(
    silver_path: str,
    k_threshold: int = 5,
    l_threshold: int = 3,
    t_threshold: float = 0.2,
    risk_threshold: float = 0.1,
    cdr_threshold: float = 0.5,
    output_path: Optional[str] = None
) -> PrivacyMetricsResult:
    """
    Evaluate privacy metrics for Silver layer data.

    Args:
        silver_path: Path to Silver layer
        k_threshold: Minimum k for k-anonymity
        l_threshold: Minimum l for l-diversity
        t_threshold: Maximum t for t-closeness
        risk_threshold: Maximum re-identification risk
        cdr_threshold: Minimum CDR score
        output_path: Optional path to save results

    Returns:
        PrivacyMetricsResult
    """
    # Load chunks
    chunks = load_silver_layer_chunks(silver_path)

    if not chunks:
        logger.error("No chunks found in Silver layer")
        return None

    # Convert to records
    records = chunks_to_anonymized_records(chunks)
    logger.info(f"Converted {len(records)} chunks to anonymized records")

    # Initialize calculator
    calculator = PrivacyMetricsCalculator(
        k_threshold=k_threshold,
        l_threshold=l_threshold,
        t_threshold=t_threshold,
        risk_threshold=risk_threshold,
        cdr_threshold=cdr_threshold
    )

    # Calculate metrics
    logger.info("Calculating privacy metrics...")
    result = calculator.calculate_all_metrics(
        records,
        sensitive_attribute="content_hash"
    )

    # Print report
    print(result.to_report())

    # Save results if output path provided
    if output_path:
        save_results(result, output_path, silver_path)

    return result


def evaluate_text_anonymization(
    original_texts: List[str],
    anonymized_texts: List[str],
    output_path: Optional[str] = None
) -> PrivacyMetricsResult:
    """
    Evaluate privacy metrics for anonymized texts.

    Args:
        original_texts: List of original texts
        anonymized_texts: List of anonymized texts
        output_path: Optional path to save results

    Returns:
        PrivacyMetricsResult
    """
    analyzer = TextPrivacyAnalyzer()
    result = analyzer.analyze_anonymized_texts(original_texts, anonymized_texts)

    print(result.to_report())

    if output_path:
        save_results(result, output_path, "text_analysis")

    return result


def save_results(
    result: PrivacyMetricsResult,
    output_path: str,
    source: str
) -> None:
    """Save evaluation results to JSON file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "results": result.to_dict()
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def run_demo():
    """
    Run a demonstration with sample data.
    """
    print("\n" + "=" * 70)
    print("PRIVACY METRICS DEMONSTRATION")
    print("=" * 70)

    # Create sample records
    records = []

    # Create equivalence class 1: 5 similar records (k=5)
    for i in range(5):
        record = AnonymizedRecord(
            record_id=f"rec_a{i}",
            quasi_identifiers=[
                QuasiIdentifier(name="age_group", value="30-40"),
                QuasiIdentifier(name="zip_prefix", value="100"),
                QuasiIdentifier(name="gender", value="M"),
            ],
            sensitive_attributes=[
                SensitiveAttribute(name="salary_range", value="50k-75k" if i < 3 else "75k-100k"),
            ]
        )
        records.append(record)

    # Create equivalence class 2: 3 similar records (k=3)
    for i in range(3):
        record = AnonymizedRecord(
            record_id=f"rec_b{i}",
            quasi_identifiers=[
                QuasiIdentifier(name="age_group", value="40-50"),
                QuasiIdentifier(name="zip_prefix", value="100"),
                QuasiIdentifier(name="gender", value="F"),
            ],
            sensitive_attributes=[
                SensitiveAttribute(name="salary_range", value=f"salary_{i}"),  # Each unique for l-diversity demo
            ]
        )
        records.append(record)

    # Create a unique record (k=1, high risk)
    records.append(AnonymizedRecord(
        record_id="rec_unique",
        quasi_identifiers=[
            QuasiIdentifier(name="age_group", value="60-70"),
            QuasiIdentifier(name="zip_prefix", value="999"),
            QuasiIdentifier(name="gender", value="M"),
        ],
        sensitive_attributes=[
            SensitiveAttribute(name="salary_range", value="150k+"),
        ]
    ))

    print(f"\nSample dataset: {len(records)} records")
    print("- Equivalence class 1: 5 records (age 30-40, zip 100, M)")
    print("- Equivalence class 2: 3 records (age 40-50, zip 100, F)")
    print("- Unique record: 1 record (age 60-70, zip 999, M)")

    # Calculate metrics
    calculator = PrivacyMetricsCalculator(
        k_threshold=5,
        l_threshold=2,
        t_threshold=0.3,
        risk_threshold=0.2,
        cdr_threshold=0.3
    )

    result = calculator.calculate_all_metrics(records, "salary_range")

    print(result.to_report())

    # Interpretation
    print("INTERPRETATION")
    print("─" * 50)
    print(f"• K-anonymity = {result.k_anonymity}: The unique record makes k=1")
    print(f"  → To fix: Generalize or suppress the unique record")
    print(f"\n• L-diversity = {result.l_diversity}: Class 1 has only 2 distinct salaries")
    print(f"  → To fix: Ensure more variety in sensitive attributes")
    print(f"\n• Prosecutor risk = {result.prosecutor_risk:.2%}: High due to unique record")
    print(f"  → The attacker can definitely identify the 60-70 age person")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate privacy metrics for anonymized data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Silver layer
  python evaluate_privacy.py --silver ./data/silver

  # With custom thresholds
  python evaluate_privacy.py --silver ./data/silver -k 5 -l 3 -t 0.2

  # Save results
  python evaluate_privacy.py --silver ./data/silver --output privacy_report.json

  # Run demonstration
  python evaluate_privacy.py --demo

Metrics explained:
  K-anonymity:  Each record indistinguishable from k-1 others
  L-diversity:  At least l distinct sensitive values per group
  T-closeness:  Distribution close to overall (t = max distance)
  Re-id Risk:   Probability of identifying an individual
  CDR:          Ratio of inter-class to intra-class distances
        """
    )

    parser.add_argument(
        "--silver",
        type=str,
        help="Path to Silver layer directory"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save results JSON"
    )

    # Thresholds
    parser.add_argument(
        "-k", "--k-threshold",
        type=int,
        default=5,
        help="Minimum k for k-anonymity (default: 5)"
    )

    parser.add_argument(
        "-l", "--l-threshold",
        type=int,
        default=3,
        help="Minimum l for l-diversity (default: 3)"
    )

    parser.add_argument(
        "-t", "--t-threshold",
        type=float,
        default=0.2,
        help="Maximum t for t-closeness (default: 0.2)"
    )

    parser.add_argument(
        "-r", "--risk-threshold",
        type=float,
        default=0.1,
        help="Maximum re-identification risk (default: 0.1)"
    )

    parser.add_argument(
        "--cdr-threshold",
        type=float,
        default=0.5,
        help="Minimum CDR score (default: 0.5)"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with sample data"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        run_demo()
        return

    if not args.silver:
        parser.error("Please specify --silver path or use --demo")

    try:
        result = evaluate_silver_layer(
            silver_path=args.silver,
            k_threshold=args.k_threshold,
            l_threshold=args.l_threshold,
            t_threshold=args.t_threshold,
            risk_threshold=args.risk_threshold,
            cdr_threshold=args.cdr_threshold,
            output_path=args.output
        )

        if result:
            # Summary
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)

            all_satisfied = all([
                result.k_anonymity_satisfied,
                result.l_diversity_satisfied,
                result.t_closeness_satisfied,
                result.risk_satisfied,
                result.cdr_satisfied
            ])

            if all_satisfied:
                print("\n✓ All privacy thresholds MET!")
            else:
                print("\n✗ Some privacy thresholds NOT met:")
                if not result.k_anonymity_satisfied:
                    print(f"  ✗ K-anonymity: {result.k_anonymity} < {result.k_threshold}")
                if not result.l_diversity_satisfied:
                    print(f"  ✗ L-diversity: {result.l_diversity} < {result.l_threshold}")
                if not result.t_closeness_satisfied:
                    print(f"  ✗ T-closeness: {result.t_closeness:.4f} > {result.t_threshold}")
                if not result.risk_satisfied:
                    print(f"  ✗ Re-id Risk: {result.reidentification_risk:.4f} > {result.risk_threshold}")
                if not result.cdr_satisfied:
                    print(f"  ✗ CDR: {result.cdr_score:.4f} < {result.cdr_threshold}")

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
