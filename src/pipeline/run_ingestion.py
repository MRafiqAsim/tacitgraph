#!/usr/bin/env python3
"""
Ingestion and Anonymization Pipeline

Main entry point for processing raw data through Bronze and Silver layers.

Usage:
    python run_ingestion.py --pst /path/to/emails.pst --output /path/to/data
    python run_ingestion.py --documents /path/to/docs --output /path/to/data
    python run_ingestion.py --bronze /path/to/bronze --silver /path/to/silver

Steps:
    1.   Extract emails from PST files → Bronze layer
    1.5  Classify attachments (extract, classify, move, enrich email JSON)
    2.   Parse documents (PDF, DOCX, etc.) → Bronze layer
    3.   Chunk, detect PII, anonymize → Silver layer
    4.   Evaluate anonymization accuracy (if ground truth provided)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bronze.pst_extractor import PSTExtractor
from bronze.document_parser import DocumentParser
from bronze.bronze_loader import BronzeLayerLoader
from bronze.attachment_processor import AttachmentProcessor
from silver.silver_processor import SilverLayerProcessor
try:
    from conflict_handling.pii_evaluation import PIIEvaluator, GroundTruthLoader
except ImportError:
    PIIEvaluator = None
    GroundTruthLoader = None

# Configure logging — console + file
_project_root = Path(__file__).resolve().parent.parent.parent
_log_dir = _project_root / "logs"
_log_dir.mkdir(exist_ok=True)
_log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not any(isinstance(h, logging.FileHandler) and 'ingestion.log' in getattr(h, 'baseFilename', '')
           for h in _root.handlers):
    _root.addHandler(logging.FileHandler(str(_log_dir / "ingestion.log"), mode="a", encoding="utf-8"))
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
           for h in _root.handlers):
    _root.addHandler(logging.StreamHandler())
for h in _root.handlers:
    h.setFormatter(logging.Formatter(_log_format))

logger = logging.getLogger(__name__)

# Suppress noisy third-party warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("openpyxl").setLevel(logging.ERROR)
logging.getLogger("fitz").setLevel(logging.ERROR)   # MuPDF/PyMuPDF
import warnings
warnings.filterwarnings("ignore", message="Sparkline Group extension")


def extract_pst_to_bronze(
    pst_path: str,
    bronze_path: str,
    attachment_dir: Optional[str] = None,
    max_emails: Optional[int] = None,
) -> dict:
    """
    Extract emails from PST file to Bronze layer.

    Bronze is pure extraction — no classification or filtering.
    Classification happens in Silver layer.

    Args:
        pst_path: Path to PST file
        bronze_path: Output Bronze layer path
        attachment_dir: Directory for attachments
        max_emails: Maximum number of emails to extract

    Returns:
        Extraction statistics
    """
    logger.info(f"Extracting PST: {pst_path}")

    # Initialize extractor
    extractor = PSTExtractor(
        extract_attachments=True,
        attachment_output_dir=attachment_dir or f"{bronze_path}/attachments"
    )

    # Initialize loader (pure extraction, no classification)
    loader = BronzeLayerLoader(bronze_path=bronze_path)

    # Progress callback
    def progress(count, message):
        if count % 500 == 0:
            logger.info(f"Progress: {count} emails - {message}")

    # Extract and load
    try:
        emails = extractor.extract(pst_path, progress_callback=progress, max_emails=max_emails)
        stats = loader.load_emails_batch(emails, batch_size=100)
        loader.save_metadata()

        logger.info(f"PST extraction complete: {stats}")
        return stats

    except Exception as e:
        logger.error(f"PST extraction failed: {e}")
        raise


def extract_bronze_attachments(bronze_path: str) -> dict:
    """
    Extract text from all attachments in the Bronze layer.

    Bronze is pure extraction — text is extracted and stored alongside
    binary files. No classification or filtering happens here.
    Classification moves to Silver layer.

    Args:
        bronze_path: Bronze layer path

    Returns:
        Extraction statistics
    """
    logger.info(f"Extracting Bronze attachment text: {bronze_path}")

    processor = AttachmentProcessor(bronze_path=bronze_path)

    def progress(count, message):
        logger.info(f"Progress: {message}")

    stats = processor.process_all_attachments(progress_callback=progress)

    logger.info(f"Attachment text extraction complete: {stats}")
    return stats


def cleanup_bronze_attachments(bronze_path: str) -> dict:
    """
    Remove legacy duplicate storage from Bronze attachments.

    Removes 32-char email_id directories (BronzeLayerLoader duplicates),
    migrates old attachments_cache/ to co-located metadata, and removes
    empty directories.

    Args:
        bronze_path: Bronze layer path

    Returns:
        Cleanup statistics
    """
    logger.info(f"Cleaning up legacy attachment storage: {bronze_path}")

    processor = AttachmentProcessor(bronze_path=bronze_path)
    stats = processor.cleanup_legacy_storage()

    logger.info(f"Attachment cleanup complete: {stats}")
    return stats


def parse_documents_to_bronze(
    docs_path: str,
    bronze_path: str,
    extensions: Optional[list] = None
) -> dict:
    """
    Parse documents to Bronze layer.

    Args:
        docs_path: Path to documents directory
        bronze_path: Output Bronze layer path
        extensions: File extensions to process

    Returns:
        Processing statistics
    """
    logger.info(f"Parsing documents from: {docs_path}")

    extensions = extensions or [".pdf", ".docx", ".xlsx", ".pptx", ".txt"]

    # Initialize parser and loader
    parser = DocumentParser(extract_tables=True)
    loader = BronzeLayerLoader(bronze_path=bronze_path)

    # Find documents
    docs_dir = Path(docs_path)
    doc_files = []
    for ext in extensions:
        doc_files.extend(docs_dir.rglob(f"*{ext}"))

    logger.info(f"Found {len(doc_files)} documents to process")

    # Process documents
    for i, doc_path in enumerate(doc_files):
        try:
            document = parser.parse(doc_path)
            loader.load_document(document)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(doc_files)} documents")

        except Exception as e:
            logger.warning(f"Failed to process {doc_path}: {e}")

    stats = loader.get_stats()
    loader.save_metadata()

    logger.info(f"Document parsing complete: {stats}")
    return stats


def process_bronze_to_silver(
    bronze_path: str,
    silver_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> dict:
    """
    Process Bronze layer to Silver layer.

    Args:
        bronze_path: Bronze layer path
        silver_path: Silver layer output path
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks

    Returns:
        Processing statistics
    """
    logger.info(f"Processing Bronze → Silver: {bronze_path} → {silver_path}")

    # Initialize processor
    processor = SilverLayerProcessor(
        bronze_path=bronze_path,
        silver_path=silver_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Progress callback
    def progress(count, message):
        logger.info(f"Progress: {message}")

    # Process
    stats = processor.process_bronze_layer(progress_callback=progress)

    logger.info(f"Silver layer processing complete: {stats}")
    return stats


def evaluate_anonymization(
    silver_path: str,
    ground_truth_path: str
) -> dict:
    """
    Evaluate anonymization accuracy against ground truth.

    Args:
        silver_path: Silver layer path with processed chunks
        ground_truth_path: Path to ground truth JSON

    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating anonymization accuracy")

    # Load ground truth
    ground_truth = GroundTruthLoader.load_from_json(ground_truth_path)

    # Create detector wrapper for evaluation
    from silver.pii_detector import PIIDetector

    class DetectorWrapper:
        def __init__(self):
            self.detector = PIIDetector()

        def detect(self, text):
            from conflict_handling.models import PIIAnnotation as EvalPIIAnnotation
            from conflict_handling.models import PIIType as EvalPIIType

            entities = self.detector.detect(text)

            # Convert to evaluation format
            return [
                EvalPIIAnnotation(
                    text=e.text,
                    pii_type=EvalPIIType(e.pii_type.value),
                    start_char=e.start,
                    end_char=e.end,
                    confidence=e.confidence,
                    is_sensitive=True
                )
                for e in entities
            ]

    # Evaluate
    evaluator = PIIEvaluator()
    detector = DetectorWrapper()
    result = evaluator.evaluate(ground_truth, detector)

    # Print report
    print(result.to_report())

    # Save results
    eval_output = Path(silver_path) / "metadata" / "pii_evaluation.json"
    eval_data = {
        "timestamp": datetime.now().isoformat(),
        "ground_truth_path": ground_truth_path,
        "total_ground_truth": result.total_ground_truth,
        "total_detected": result.total_detected,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "metrics_by_type": result.metrics_by_type,
    }

    with open(eval_output, "w") as f:
        json.dump(eval_data, f, indent=2)

    logger.info(f"Evaluation results saved to: {eval_output}")

    return eval_data


def run_full_pipeline(
    pst_path: Optional[str] = None,
    docs_path: Optional[str] = None,
    output_path: str = "./data",
    ground_truth_path: Optional[str] = None,
    chunk_size: int = 512,
    max_emails: Optional[int] = None,
) -> dict:
    """
    Run the full ingestion and anonymization pipeline.

    Args:
        pst_path: Path to PST file (optional)
        docs_path: Path to documents directory (optional)
        output_path: Output directory
        ground_truth_path: Path to ground truth for evaluation (optional)
        chunk_size: Chunk size in tokens

    Returns:
        Combined statistics
    """
    bronze_path = f"{output_path}/bronze"
    silver_path = f"{output_path}/silver"

    all_stats = {
        "start_time": datetime.now().isoformat(),
        "bronze_stats": {},
        "silver_stats": {},
        "evaluation_stats": {},
    }

    # Step 1: Extract PST to Bronze
    if pst_path:
        logger.info("=" * 60)
        logger.info("STEP 1: PST Extraction → Bronze Layer")
        logger.info("=" * 60)
        pst_stats = extract_pst_to_bronze(
            pst_path, bronze_path, max_emails=max_emails,
        )
        all_stats["bronze_stats"]["pst"] = pst_stats

        # Step 1.5: Extract text from all attachments (no classification)
        logger.info("=" * 60)
        logger.info("STEP 1.5: Attachment Text Extraction → Bronze Layer")
        logger.info("=" * 60)
        att_stats = extract_bronze_attachments(bronze_path)
        all_stats["bronze_stats"]["attachment_classification"] = att_stats

    # Step 2: Parse documents to Bronze
    if docs_path:
        logger.info("=" * 60)
        logger.info("STEP 2: Document Parsing → Bronze Layer")
        logger.info("=" * 60)
        doc_stats = parse_documents_to_bronze(docs_path, bronze_path)
        all_stats["bronze_stats"]["documents"] = doc_stats

    # Silver processing is handled separately by run_thread_processing.py
    # Run: python run_thread_processing.py --mode llm --bronze ./data/bronze --silver ./data/silver

    all_stats["end_time"] = datetime.now().isoformat()

    # Save combined stats
    stats_file = f"{output_path}/pipeline_stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Stats saved to: {stats_file}")

    return all_stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Ingestion and Anonymization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PST file
  python run_ingestion.py --pst emails.pst --output ./data

  # Process documents
  python run_ingestion.py --documents ./docs --output ./data

  # Process both
  python run_ingestion.py --pst emails.pst --documents ./docs --output ./data

  # Process Bronze to Silver only
  python run_ingestion.py --bronze ./data/bronze --silver ./data/silver

  # With evaluation
  python run_ingestion.py --bronze ./data/bronze --silver ./data/silver --evaluate ground_truth.json

  # Extract text from attachments on existing Bronze
  python run_ingestion.py --bronze ./data/bronze --extract-attachments

  # Clean up legacy duplicate storage
  python run_ingestion.py --bronze ./data/bronze --cleanup-attachments
        """
    )

    # Input options
    parser.add_argument(
        "--pst",
        type=str,
        help="Path to PST file for email extraction"
    )
    parser.add_argument(
        "--documents", "--docs",
        type=str,
        help="Path to documents directory"
    )
    parser.add_argument(
        "--bronze",
        type=str,
        help="Path to existing Bronze layer (skip extraction)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data",
        help="Output directory (default: ./data)"
    )
    parser.add_argument(
        "--silver",
        type=str,
        help="Silver layer output path (default: {output}/silver)"
    )

    # Processing options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens (default: 512)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)"
    )

    # Evaluation
    parser.add_argument(
        "--evaluate",
        type=str,
        help="Path to ground truth JSON for evaluation"
    )

    # Attachment management
    parser.add_argument(
        "--extract-attachments",
        action="store_true",
        help="Extract text from all Bronze attachments (standalone operation)"
    )
    parser.add_argument(
        "--cleanup-attachments",
        action="store_true",
        help="Remove legacy duplicate storage (32-char dirs, old cache)"
    )

    # NOTE: Email sensitivity and attachment classification moved to Silver layer.
    # Use run_thread_processing.py --mode local|llm|hybrid for classification.

    # Limit
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of emails to extract (default: all)"
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    if not any([args.pst, args.documents, args.bronze, args.extract_attachments, args.cleanup_attachments]):
        parser.error("Must specify at least one of: --pst, --documents, --bronze, --extract-attachments, or --cleanup-attachments")

    # Determine paths
    bronze_path = args.bronze or f"{args.output}/bronze"
    silver_path = args.silver or f"{args.output}/silver"

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Run pipeline
    try:
        # Standalone attachment operations (no Silver processing needed)
        if args.extract_attachments:
            extract_bronze_attachments(bronze_path)

        if args.cleanup_attachments:
            cleanup_bronze_attachments(bronze_path)

        # If only attachment flags were set, we're done
        if not any([args.pst, args.documents, args.bronze]):
            print("\n" + "=" * 60)
            print("SUCCESS! Attachment operations completed.")
            print("=" * 60)
            print(f"\nBronze layer: {bronze_path}")
            sys.exit(0)

        if args.bronze:
            # Bronze already exists — run attachment extraction only
            stats = extract_bronze_attachments(bronze_path)

        else:
            # Full Bronze pipeline (PST + documents + attachment extraction)
            stats = run_full_pipeline(
                pst_path=args.pst,
                docs_path=args.documents,
                output_path=args.output,
                chunk_size=args.chunk_size,
                max_emails=args.limit,
            )

        # Print summary
        logger.info(f"Bronze layer: {bronze_path}")
        if isinstance(stats, dict):
            pst_stats = stats.get("bronze_stats", {}).get("pst", stats)
            logger.info(f"Emails: {pst_stats.get('emails_saved', pst_stats.get('success', 'N/A'))}")
            att_stats = stats.get("bronze_stats", {}).get("attachment_classification", {})
            if att_stats:
                logger.info(f"Attachments: {att_stats.get('processed', 'N/A')} processed, "
                            f"{att_stats.get('success', 'N/A')} text extracted, "
                            f"{att_stats.get('unsupported', 0)} unsupported format, "
                            f"{att_stats.get('failed', 0)} failed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
