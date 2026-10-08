#!/usr/bin/env python3
"""
Thread-Aware Processing Pipeline

Processes emails with thread context preservation:
- Groups related emails into conversation threads
- Chunks threads while preserving semantic context
- Extracts entities and relationships for PathRAG
- Generates thread summaries for high-level retrieval

Supports three processing modes:
    --mode local   : Free, uses Presidio+spaCy+regex, extractive summaries
    --mode llm     : Azure GPT-4o for PII detection + summarization
    --mode hybrid  : Local first, LLM verifies low-confidence, LLM summarization

Usage:
    # LOCAL mode (free, fast)
    python run_thread_processing.py --bronze ./data/bronze --silver ./data/silver --mode local

    # LLM mode (Azure OpenAI)
    python run_thread_processing.py --bronze ./data/bronze --silver ./data/silver --mode llm \
        --use-azure --azure-key YOUR_KEY --azure-endpoint YOUR_ENDPOINT

    # HYBRID mode
    python run_thread_processing.py --bronze ./data/bronze --silver ./data/silver --mode hybrid \
        --use-azure --azure-key YOUR_KEY --azure-endpoint YOUR_ENDPOINT
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Load .env before anything else
from dotenv import load_dotenv

from tacitgraph.paths import PROJECT_ROOT

load_dotenv()


# Configure logging — console + file

from tacitgraph.silver.identity_registry import IdentityRegistry
from tacitgraph.silver.thread_aware_processor import ThreadAwareProcessor

_log_dir = PROJECT_ROOT / "logs"
_log_dir.mkdir(exist_ok=True)
_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Force-attach handlers to root logger (basicConfig is a no-op if a library
# like transformers/spacy already configured logging on import).
_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not any(
    isinstance(h, logging.FileHandler) and "silver.log" in getattr(h, "baseFilename", "")
    for h in _root.handlers
):
    _root.addHandler(logging.FileHandler(str(_log_dir / "silver.log"), mode="a", encoding="utf-8"))
if not any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    for h in _root.handlers
):
    _root.addHandler(logging.StreamHandler())
for h in _root.handlers:
    h.setFormatter(logging.Formatter(_log_format))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Process emails with thread-aware chunking and PathRAG entity extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic thread processing (free, uses spaCy)
  python run_thread_processing.py --bronze ./data/bronze --silver ./data/silver

  # With Azure OpenAI for LLM extraction
  python run_thread_processing.py --bronze ./data/bronze --silver ./data/silver \\
      --use-azure \\
      --azure-key YOUR_KEY \\
      --azure-endpoint https://your-endpoint.cognitiveservices.azure.com/ \\
      --azure-deployment structexp-4o \\
      --kg-strategy llm \\
      --rel-strategy llm

  # Hybrid mode (spaCy + LLM enhancement)
  python run_thread_processing.py --bronze ./data/bronze --silver ./data/silver \\
      --use-azure --azure-key YOUR_KEY --azure-endpoint YOUR_ENDPOINT --azure-deployment structexp-4o \\
      --kg-strategy hybrid --rel-strategy hybrid
        """,
    )

    # Processing mode
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="local",
        choices=["local", "llm", "hybrid"],
        help="Processing mode: local (free), llm (Azure GPT-4o), hybrid (local+LLM verify)",
    )

    # Path arguments
    parser.add_argument(
        "--bronze", "-b", type=str, default="./data/bronze", help="Path to Bronze layer"
    )

    parser.add_argument(
        "--silver", "-s", type=str, default="./data/silver", help="Path for Silver layer output"
    )

    # Identity registry
    parser.add_argument(
        "--registry-path",
        type=str,
        default="./data/identity_registry.json",
        help="Path to identity registry JSON (built from bronze if not found)",
    )

    parser.add_argument("--no-registry", action="store_true", help="Disable identity registry")

    # Chunking arguments
    parser.add_argument(
        "--chunk-size", type=int, default=1024, help="Chunk size in tokens (default: 1024)"
    )

    parser.add_argument(
        "--chunk-overlap", type=int, default=50, help="Chunk overlap in tokens (default: 50)"
    )

    # Extraction strategy arguments
    parser.add_argument(
        "--kg-strategy",
        type=str,
        default="spacy",
        choices=["spacy", "llm", "hybrid"],
        help="KG entity extraction strategy (default: spacy)",
    )

    parser.add_argument(
        "--rel-strategy",
        type=str,
        default="cooccurrence",
        choices=["cooccurrence", "llm", "hybrid"],
        help="Relationship extraction strategy (default: cooccurrence)",
    )

    parser.add_argument(
        "--no-relationships", action="store_true", help="Skip relationship extraction"
    )

    # OpenAI arguments
    parser.add_argument(
        "--openai-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )

    # Azure OpenAI arguments
    parser.add_argument(
        "--use-azure", action="store_true", help="Use Azure OpenAI instead of OpenAI"
    )

    parser.add_argument("--azure-key", type=str, help="Azure OpenAI API key")

    parser.add_argument(
        "--azure-endpoint",
        type=str,
        default="https://muham-mll3ne3p-eastus2.cognitiveservices.azure.com/",
        help="Azure OpenAI endpoint URL",
    )

    parser.add_argument(
        "--azure-deployment",
        type=str,
        default="gpt-4o",
        help="Azure OpenAI deployment name (default: structexp-4o)",
    )

    parser.add_argument(
        "--azure-api-version", type=str, default="2024-12-01-preview", help="Azure API version"
    )

    # Other arguments
    parser.add_argument(
        "--with-summaries",
        action="store_true",
        default=True,
        help="Generate thread summaries (BART for local, GPT-4o for LLM)",
    )

    # Attachment processing arguments
    parser.add_argument(
        "--process-attachments",
        action="store_true",
        default=True,
        help="Process email attachments (default: True)",
    )

    parser.add_argument("--no-attachments", action="store_true", help="Skip attachment processing")

    parser.add_argument(
        "--include-attachment-text",
        action="store_true",
        default=True,
        help="Include attachment text in chunks for KG extraction (default: True)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of threads to process (default: all)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip threads already processed (reads checkpoint.json from silver dir)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine API key and auto-detect Azure from env
    api_key = (
        args.azure_key
        or args.openai_key
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    # Auto-enable Azure if env vars are set and --use-azure wasn't explicitly given
    if (
        not args.use_azure
        and os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_ENDPOINT")
    ):
        args.use_azure = True
        if (
            not args.azure_endpoint
            or args.azure_endpoint == "https://muham-mll3ne3p-eastus2.cognitiveservices.azure.com/"
        ):
            args.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", args.azure_endpoint)
        logger.info("Auto-detected Azure OpenAI from environment variables")

    # --mode llm/hybrid implies LLM for all sub-strategies (unless explicitly overridden)
    if args.mode == "llm":
        if args.kg_strategy == "spacy":
            args.kg_strategy = "llm"
        if args.rel_strategy == "cooccurrence":
            args.rel_strategy = "llm"
        if not args.with_summaries:
            args.with_summaries = True
    elif args.mode == "hybrid":
        if args.kg_strategy == "spacy":
            args.kg_strategy = "hybrid"
        if args.rel_strategy == "cooccurrence":
            args.rel_strategy = "hybrid"
        if not args.with_summaries:
            args.with_summaries = True

    # Check if LLM/hybrid mode or strategy requires API key
    if (
        args.mode in ["llm", "hybrid"]
        or args.kg_strategy in ["llm", "hybrid"]
        or args.rel_strategy in ["llm", "hybrid"]
    ) and not api_key:
        print("ERROR: LLM/hybrid mode requires API key.")
        print("   Use --azure-key, --openai-key, or set AZURE_OPENAI_API_KEY env var")
        sys.exit(1)

    # Build or load Identity Registry
    identity_registry = None
    if not args.no_registry:
        identity_registry = IdentityRegistry()
        registry_path = Path(args.registry_path)

        if registry_path.exists():
            print(f"\nLoading identity registry from {registry_path}")
            identity_registry.load(str(registry_path))
        else:
            print(f"\nBuilding identity registry from {args.bronze}...")
            build_stats = identity_registry.build_from_bronze(args.bronze)
            print(
                f"  Found {build_stats.get('total_identities', 0)} identities, "
                f"{build_stats.get('total_aliases', 0)} aliases"
            )
            identity_registry.save(str(registry_path))
            print(f"  Saved to {registry_path}")

    # Determine silver output path (mode-specific for isolation)
    if args.silver.endswith(f"_{args.mode}"):
        silver_path = args.silver
    else:
        silver_path = f"{args.silver}_{args.mode}"

    print("\n" + "=" * 60)
    print("THREAD-AWARE EMAIL PROCESSING WITH PATHRAG")
    print("=" * 60)
    print(f"\nProcessing mode: {args.mode.upper()}")
    print(f"Bronze layer: {args.bronze}")
    print(f"Silver layer: {silver_path}")
    print(f"Chunk size: {args.chunk_size} tokens")
    if identity_registry:
        print(f"Identity registry: {identity_registry.identity_count} identities")
    print("\nExtraction strategies:")
    print(f"  - KG entities: {args.kg_strategy}")
    print(f"  - Relationships: {args.rel_strategy}")
    if args.use_azure:
        print("\nUsing Azure OpenAI:")
        print(f"  - Endpoint: {args.azure_endpoint}")
        print(f"  - Deployment: {args.azure_deployment}")
    elif api_key:
        print("\nUsing OpenAI API")

    # Determine attachment processing settings
    process_attachments = args.process_attachments and not args.no_attachments
    include_attachment_text = args.include_attachment_text and process_attachments

    if process_attachments:
        print("\nAttachment processing: ENABLED")
        print(f"  - Include text in KG: {include_attachment_text}")
    else:
        print("\nAttachment processing: DISABLED")

    # Initialize processor
    processor = ThreadAwareProcessor(
        bronze_path=args.bronze,
        silver_path=silver_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        generate_summaries=args.with_summaries,
        openai_api_key=api_key,
        kg_extractor_strategy=args.kg_strategy,
        relationship_extractor_strategy=args.rel_strategy,
        extract_relationships=not args.no_relationships,
        use_azure=args.use_azure,
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version,
        azure_deployment=args.azure_deployment,
        process_attachments=process_attachments,
        include_attachment_text=include_attachment_text,
        processing_mode=args.mode,
        identity_registry=identity_registry,
    )

    # Progress callback
    def progress(count, message):
        print(f"  {message}")

    # Process
    print("\nProcessing...")
    print("-" * 40)

    stats = processor.process(
        progress_callback=progress, max_threads=args.limit, resume=args.resume
    )

    # Print results
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nThreads processed:     {stats['threads_processed']}")
    print(f"  - Multi-email:       {stats['multi_email_threads']}")
    print(f"  - Single email:      {stats['single_emails']}")
    print(f"Chunks created:        {stats['chunks_created']}")
    print(f"Thread summaries:      {stats['summaries_generated']}")
    print(f"Email summaries:       {stats.get('email_summaries_generated', 0)}")
    print(f"PII entities detected: {stats['pii_detected']}")
    print(f"KG entities extracted: {stats['kg_entities_extracted']}")
    print(f"Relationships extracted: {stats['kg_relationships_extracted']}")
    print(f"Attachments processed: {stats.get('attachments_processed', 0)}")
    print(f"Attachments with text: {stats.get('attachments_with_text', 0)}")
    print(f"Attachment summaries:  {stats.get('attachment_summaries_generated', 0)}")
    print(f"Threads not personal:  {stats.get('threads_not_personal', 0)}")
    print(f"Threads personal:      {stats.get('threads_skipped_personal', 0)} (skipped)")
    print(f"Emails not personal:   {stats.get('emails_not_personal', 0)}")
    print(f"Emails personal:       {stats.get('emails_skipped_personal', 0)} (skipped)")
    print(f"Emails empty body:     {stats.get('emails_skipped_empty', 0)} (skipped)")
    print(f"Errors:                {stats['errors']}")

    print("\nOutput:")
    print(f"  Not personal:     {silver_path}/not_personal/")
    print("    Email chunks:     not_personal/email_chunks/ (summary in chunk)")
    print("    Thread summaries: not_personal/thread_summaries/")
    print(f"  Personal:         {silver_path}/personal/")
    print(f"  Processing mode:  {args.mode.upper()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
