#!/usr/bin/env python3
"""
Query Pipeline

Run queries against the knowledge graph using various retrieval strategies.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.retrieval import (
    HybridRetriever,
    RetrievalStrategy,
    ReActRetriever
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the knowledge graph using various retrieval strategies"
    )

    # --- Mode selection (auto-derives paths) ---
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["local", "llm", "hybrid"],
        help="Processing mode — auto-derives silver/gold paths (e.g., gold_llm, silver_llm)"
    )

    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Run the same query against ALL available mode-specific gold/silver layers and compare"
    )

    # --- Explicit paths (override --mode) ---
    parser.add_argument(
        "--gold",
        type=str,
        help="Path to Gold layer (default: ./data/gold_{mode})"
    )

    parser.add_argument(
        "--silver",
        type=str,
        help="Path to Silver layer (default: ./data/silver_{mode})"
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        help="The query to run (interactive mode if not provided)"
    )

    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["vector", "pathrag", "graphrag", "hybrid", "react"],
        default="hybrid",
        help="Retrieval strategy (default: hybrid)"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all strategies for the query (within the selected mode)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def run_interactive(retriever: HybridRetriever, strategy: RetrievalStrategy):
    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE QUERY MODE")
    print("=" * 60)
    print(f"Strategy: {strategy.value}")
    print("Type 'quit' or 'exit' to exit")
    print("Type 'strategy <name>' to change strategy")
    print("=" * 60 + "\n")

    current_strategy = strategy

    while True:
        try:
            query = input("\nQuery: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if query.lower().startswith('strategy '):
                new_strategy = query.split(' ', 1)[1].lower()
                try:
                    current_strategy = RetrievalStrategy(new_strategy)
                    print(f"Strategy changed to: {current_strategy.value}")
                except ValueError:
                    print(f"Invalid strategy. Choose from: {[s.value for s in RetrievalStrategy]}")
                continue

            print(f"\nSearching with {current_strategy.value}...")
            start_time = datetime.now()

            result = retriever.retrieve(query, current_strategy)

            print("\n" + "-" * 60)
            print("ANSWER:")
            print("-" * 60)
            print(result.answer or "(No answer generated)")

            print("\n" + "-" * 60)
            print(f"SOURCES ({len(result.sources)}):")
            print("-" * 60)
            for source in result.sources[:5]:
                print(f"  - {source}")

            print(f"\nConfidence: {result.confidence:.2f}")
            print(f"Time: {result.execution_time:.2f}s")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_single_query(
    retriever: HybridRetriever,
    query: str,
    strategy: RetrievalStrategy,
    verbose: bool = False
) -> dict:
    """Run a single query."""
    print(f"\nQuery: {query}")
    print(f"Strategy: {strategy.value}")
    print("-" * 60)

    result = retriever.retrieve(query, strategy)

    print("\nANSWER:")
    print("-" * 40)
    print(result.answer or "(No answer generated)")

    if verbose:
        print("\n" + "-" * 40)
        print(f"Chunks retrieved: {len(result.chunks)}")
        print(f"Sources: {result.sources[:5]}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Execution time: {result.execution_time:.2f}s")

        if result.metadata:
            print(f"Metadata: {json.dumps(result.metadata, indent=2, default=str)}")

    return result.to_dict()


def compare_across_modes(query: str, strategy: RetrievalStrategy, verbose: bool = False) -> dict:
    """Run the same query+strategy against every available mode's data and compare."""
    from pathlib import Path as P

    MODES = ["local", "llm", "hybrid"]
    data_root = P("./data")

    mode_results = {}

    print(f"\nQuery: {query}")
    print(f"Strategy: {strategy.value}")
    print("=" * 70)
    print("CROSS-MODE COMPARISON")
    print("=" * 70)

    for mode in MODES:
        gold = data_root / f"gold_{mode}"
        silver = data_root / f"silver_{mode}"

        if not gold.exists():
            print(f"\n--- {mode.upper()} --- (skipped, gold not found at {gold})")
            continue

        print(f"\n--- {mode.upper()} ---")
        try:
            retriever = HybridRetriever(str(gold), str(silver) if silver.exists() else None)
            result = retriever.retrieve(query, strategy)

            print(f"Answer preview: {(result.answer or '(none)')[:200]}...")
            print(f"Chunks: {len(result.chunks)}, Confidence: {result.confidence:.2f}, Time: {result.execution_time:.2f}s")

            mode_results[mode] = result.to_dict()
        except Exception as e:
            print(f"Error: {e}")
            mode_results[mode] = {"error": str(e)}

    # Summary table
    if mode_results:
        print("\n" + "=" * 70)
        print("CROSS-MODE SUMMARY")
        print("=" * 70)
        print(f"{'Mode':<10} {'Chunks':<8} {'Confidence':<12} {'Time (s)':<10}")
        print("-" * 40)
        for mode, data in mode_results.items():
            if "error" in data:
                print(f"{mode:<10} {'ERROR':<8}")
            else:
                chunks = len(data.get("chunks", []))
                conf = data.get("confidence", 0)
                t = data.get("execution_time", 0)
                print(f"{mode:<10} {chunks:<8} {conf:<12.2f} {t:<10.2f}")
        print("=" * 70)

    return mode_results


def compare_strategies(retriever: HybridRetriever, query: str) -> dict:
    """Compare all retrieval strategies."""
    print(f"\nQuery: {query}")
    print("=" * 60)
    print("COMPARING ALL STRATEGIES")
    print("=" * 60)

    results = {}

    for strategy in RetrievalStrategy:
        print(f"\n--- {strategy.value.upper()} ---")
        result = retriever.retrieve(query, strategy)

        print(f"Answer preview: {result.answer[:200] if result.answer else '(none)'}...")
        print(f"Chunks: {len(result.chunks)}, Confidence: {result.confidence:.2f}, Time: {result.execution_time:.2f}s")

        results[strategy.value] = result.to_dict()

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<12} {'Chunks':<8} {'Confidence':<12} {'Time (s)':<10}")
    print("-" * 42)

    for strategy_name, result_data in results.items():
        chunks = len(result_data.get('chunks', []))
        confidence = result_data.get('confidence', 0)
        time_taken = result_data.get('execution_time', 0)
        print(f"{strategy_name:<12} {chunks:<8} {confidence:<12.2f} {time_taken:<10.2f}")

    return results


def main():
    args = parse_args()

    strategy = RetrievalStrategy(args.strategy)

    # --- Cross-mode comparison ---
    if args.compare_modes:
        if not args.query:
            logger.error("--compare-modes requires --query")
            sys.exit(1)
        results = compare_across_modes(args.query, strategy, args.verbose)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
        return

    # --- Resolve gold/silver paths ---
    data_root = Path("./data")

    if args.gold:
        gold_path = Path(args.gold)
    elif args.mode:
        gold_path = data_root / f"gold_{args.mode}"
    else:
        logger.error("Provide --mode or --gold")
        sys.exit(1)

    if not gold_path.exists():
        logger.error(f"Gold path does not exist: {gold_path}")
        sys.exit(1)

    if args.silver:
        silver_path = args.silver
    elif args.mode:
        silver_path = str(data_root / f"silver_{args.mode}")
    else:
        silver_path = None

    # Initialize retriever
    print(f"Initializing retriever (mode: {args.mode or 'custom'})...")
    retriever = HybridRetriever(
        str(gold_path),
        silver_path
    )

    if args.query:
        if args.compare:
            results = compare_strategies(retriever, args.query)
        else:
            results = run_single_query(retriever, args.query, strategy, args.verbose)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
    else:
        # Interactive mode
        run_interactive(retriever, strategy)


if __name__ == "__main__":
    main()
