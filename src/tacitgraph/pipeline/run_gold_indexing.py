#!/usr/bin/env python3
"""
Gold Layer Indexing Pipeline

Builds knowledge graph, detects communities, computes paths,
and generates embeddings for PathRAG/GraphRAG retrieval.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

from tacitgraph.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


from tacitgraph.gold.community_detector import CommunityConfig, CommunityDetector
from tacitgraph.gold.embedding_generator import EmbeddingConfig, EmbeddingGenerator
from tacitgraph.gold.graph_builder import GraphBuilder
from tacitgraph.gold.path_indexer import PathIndexConfig, PathIndexer

_log_dir = PROJECT_ROOT / "logs"
_log_dir.mkdir(exist_ok=True)
_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not any(
    isinstance(h, logging.FileHandler) and "gold.log" in getattr(h, "baseFilename", "")
    for h in _root.handlers
):
    _root.addHandler(logging.FileHandler(str(_log_dir / "gold.log"), mode="a", encoding="utf-8"))
if not any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    for h in _root.handlers
):
    _root.addHandler(logging.StreamHandler())
for h in _root.handlers:
    h.setFormatter(logging.Formatter(_log_format))

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Gold layer indexes for PathRAG/GraphRAG retrieval"
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["local", "llm", "hybrid"],
        help="Processing mode — auto-derives silver/gold paths (e.g., silver_llm → gold_llm). "
        "Overridden by explicit --silver / --gold.",
    )

    parser.add_argument(
        "--silver",
        type=str,
        help="Path to Silver layer with processed chunks (default: ./data/silver_{mode})",
    )

    parser.add_argument(
        "--gold", type=str, help="Path to Gold layer output directory (default: ./data/gold_{mode})"
    )

    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Build gold layer for ALL available modes (local, llm, hybrid)",
    )

    # Graph building
    parser.add_argument(
        "--build-graph", action="store_true", default=False, help="Build knowledge graph"
    )

    parser.add_argument(
        "--skip-graph", action="store_true", help="Skip graph building (use existing)"
    )

    # Community detection (GraphRAG)
    parser.add_argument(
        "--build-communities", action="store_true", help="Detect communities for GraphRAG"
    )

    parser.add_argument(
        "--community-levels", type=int, default=3, help="Number of hierarchy levels (default: 3)"
    )

    parser.add_argument(
        "--community-resolution",
        type=float,
        default=1.0,
        help="Leiden resolution parameter (default: 1.0)",
    )

    # Path indexing (PathRAG)
    parser.add_argument("--build-paths", action="store_true", help="Build path index for PathRAG")

    parser.add_argument(
        "--max-path-length", type=int, default=5, help="Maximum path length (default: 5)"
    )

    # Embeddings
    parser.add_argument(
        "--generate-embeddings", action="store_true", help="Generate vector embeddings"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model (default: text-embedding-3-small)",
    )

    parser.add_argument(
        "--embed-communities",
        action="store_true",
        help="Embed community summaries only (for GraphRAG global search)",
    )

    # All steps
    parser.add_argument("--all", action="store_true", help="Run all indexing steps")

    return parser.parse_args()


def resolve_paths(args) -> list:
    """
    Resolve silver/gold paths from --mode, --silver, --gold, or --all-modes.

    Returns list of (mode_label, silver_path, gold_path) tuples.
    """
    MODES = ["local", "llm", "hybrid"]
    data_root = Path("./data")

    if args.all_modes:
        # Build gold for every mode that has a silver directory
        pairs = []
        for mode in MODES:
            silver = data_root / f"silver_{mode}"
            gold = data_root / f"gold_{mode}"
            if silver.exists():
                pairs.append((mode, silver, gold))
            else:
                logger.info(f"Skipping mode '{mode}' — silver not found at {silver}")
        if not pairs:
            logger.error("No silver_{mode} directories found under ./data/")
            sys.exit(1)
        return pairs

    # Single mode
    if args.silver and args.gold:
        label = args.mode or "custom"
        return [(label, Path(args.silver), Path(args.gold))]

    if args.mode:
        silver = Path(args.silver) if args.silver else data_root / f"silver_{args.mode}"
        gold = Path(args.gold) if args.gold else data_root / f"gold_{args.mode}"
        return [(args.mode, silver, gold)]

    # Fallback: require explicit paths
    logger.error("Provide --mode, --all-modes, or both --silver and --gold.")
    sys.exit(1)


def run_gold_indexing(args, mode_label: str, silver_path: Path, gold_path: Path) -> dict:
    """Run gold indexing for a single mode. Returns stats dict."""

    if not silver_path.exists():
        logger.error(f"Silver path does not exist: {silver_path}")
        return {"error": f"Silver path not found: {silver_path}"}

    gold_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"GOLD LAYER INDEXING — mode: {mode_label.upper()}")
    print("=" * 60)
    print(f"Silver layer: {silver_path}")
    print(f"Gold layer:   {gold_path}")
    print()

    start_time = datetime.now()
    stats = {"mode": mode_label}

    # Determine what to run
    run_graph = args.all or (args.build_graph and not args.skip_graph)
    run_communities = args.all or args.build_communities
    run_paths = args.all or args.build_paths
    run_embeddings = args.all or args.generate_embeddings
    run_community_embeddings = args.all or args.embed_communities

    print("Steps to run:")
    print(f"  - Build graph: {run_graph}")
    print(f"  - Detect communities: {run_communities}")
    print(f"  - Build path index: {run_paths}")
    print(f"  - Generate embeddings: {run_embeddings}")
    print("-" * 60)

    # Step 1: Build Knowledge Graph
    knowledge_graph = None

    if run_graph:
        print("\n[1/4] Building Knowledge Graph...")
        print("-" * 40)

        builder = GraphBuilder(str(silver_path), str(gold_path))

        def graph_progress(current, total):
            if current % 500 == 0:
                print(f"  Processed {current}/{total} chunks...")

        knowledge_graph = builder.build_from_chunks(progress_callback=graph_progress)
        builder.save()

        graph_stats = knowledge_graph.stats()
        stats["graph"] = graph_stats

        print("\nGraph built:")
        print(f"  Nodes: {graph_stats['total_nodes']}")
        print(f"  Edges: {graph_stats['total_edges']}")
        print(f"  Node types: {graph_stats['nodes_by_type']}")

    else:
        # Auto-load existing graph if other steps need it
        needs_graph = run_communities or run_paths or run_embeddings
        if needs_graph:
            print("\n[1/4] Loading existing Knowledge Graph...")
            builder = GraphBuilder(str(silver_path), str(gold_path))
            try:
                knowledge_graph = builder.load()
                print(f"  Loaded graph with {len(knowledge_graph.nodes)} nodes")
            except FileNotFoundError:
                logger.error("No existing graph found. Run with --build-graph first.")
                return stats

    # Step 2: Community Detection (GraphRAG)
    if run_communities:
        print("\n[2/4] Detecting Communities (GraphRAG)...")
        print("-" * 40)

        if knowledge_graph is None:
            logger.error("Knowledge graph required. Run with --build-graph or --skip-graph.")
            return stats

        config = CommunityConfig(
            resolution=args.community_resolution,
            num_levels=args.community_levels,
            use_llm_summarization=(mode_label != "local"),
        )

        detector = CommunityDetector(knowledge_graph, str(gold_path), config)

        def community_progress(current, total):
            print(f"  Progress: {current}/{total}...")

        communities = detector.detect_communities(progress_callback=community_progress)
        detector.save()

        total_communities = sum(len(c) for c in communities.values())
        stats["communities"] = {
            "total": total_communities,
            "by_level": {level: len(comms) for level, comms in communities.items()},
        }

        print("\nCommunities detected:")
        print(f"  Total: {total_communities}")
        for level, comms in communities.items():
            print(f"  Level {level}: {len(comms)} communities")

    # Step 3: Path Indexing (PathRAG)
    if run_paths:
        print("\n[3/4] Building Path Index (PathRAG)...")
        print("-" * 40)

        if knowledge_graph is None:
            logger.error("Knowledge graph required. Run with --build-graph or --skip-graph.")
            return stats

        config = PathIndexConfig(max_path_length=args.max_path_length, min_path_weight=0.3)

        indexer = PathIndexer(knowledge_graph, str(gold_path), config)

        def path_progress(current, total):
            if current % 5000 == 0:
                print(f"  Processed {current}/{total} entity pairs...")

        paths = indexer.build_index(progress_callback=path_progress)
        indexer.save()

        path_stats = indexer.get_path_statistics()
        stats["paths"] = path_stats

        print("\nPaths indexed:")
        print(f"  Total paths: {path_stats['total_paths']}")
        print(f"  Unique sources: {path_stats['unique_sources']}")
        print(f"  Avg path length: {path_stats['avg_path_length']:.2f}")

    # Step 4: Generate Embeddings
    if run_embeddings:
        print("\n[4/4] Generating Embeddings...")
        print("-" * 40)

        config = EmbeddingConfig(model=args.embedding_model)
        generator = EmbeddingGenerator(str(gold_path), config, mode=mode_label)

        if not generator.is_available():
            logger.warning("Embedding generation not available (no API key)")
        else:
            # Embed chunks
            print("  Embedding chunks...")
            chunk_ids, chunk_embeddings = generator.embed_chunks(str(silver_path))
            generator.save_embeddings(chunk_ids, chunk_embeddings, "chunks")

            # Embed entities
            if knowledge_graph:
                print("  Embedding entities...")
                entities = [
                    {"id": n.node_id, "name": n.name, "type": n.node_type}
                    for n in knowledge_graph.nodes.values()
                    if n.node_type not in ["CHUNK", "THREAD"]
                ]
                entity_ids, entity_embeddings = generator.embed_entities(entities)
                generator.save_embeddings(entity_ids, entity_embeddings, "entities")

                stats["embeddings"] = {"chunks": len(chunk_ids), "entities": len(entity_ids)}

            print("\nEmbeddings generated:")
            print(f"  Chunks: {len(chunk_ids)}")
            if knowledge_graph:
                print(f"  Entities: {len(entity_ids)}")

    # Step 5: Embed community summaries
    if run_community_embeddings:
        print("\n[5] Embedding Community Summaries...")
        print("-" * 40)

        config = EmbeddingConfig(model=args.embedding_model)
        generator = EmbeddingGenerator(str(gold_path), config, mode=mode_label)

        if not generator.is_available():
            logger.warning("Embedding generation not available (no API key)")
        else:
            # Load all community summaries across all levels
            summaries = []
            communities_dir = gold_path / "communities"
            if communities_dir.exists():
                for level_dir in sorted(communities_dir.glob("level_*")):
                    for comm_file in sorted(level_dir.glob("*.json")):
                        with open(comm_file, encoding="utf-8") as f:
                            comm_data = json.load(f)
                        comm_id = comm_data.get("community_id", comm_file.stem)
                        summary = comm_data.get("summary", "")
                        if summary:
                            summaries.append({"id": comm_id, "summary": summary})

            if summaries:
                print(f"  Embedding {len(summaries)} community summaries...")
                summary_ids, summary_embeddings = generator.embed_summaries(summaries)
                generator.save_embeddings(summary_ids, summary_embeddings, "community_summaries")
                print(f"  Community summaries embedded: {len(summary_ids)}")
                stats["community_embeddings"] = len(summary_ids)
            else:
                print("  No community summaries found")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    stats["duration_seconds"] = duration

    print("\n" + "=" * 60)
    print(f"GOLD INDEXING COMPLETE — {mode_label.upper()}")
    print("=" * 60)
    print(f"\nDuration: {duration:.1f} seconds")
    print(f"Output: {gold_path}")

    if "graph" in stats:
        print("\nKnowledge Graph:")
        print(f"  {stats['graph']['total_nodes']} nodes, {stats['graph']['total_edges']} edges")

    if "communities" in stats:
        print(f"\nCommunities: {stats['communities']['total']} total")

    if "paths" in stats:
        print(f"\nPaths: {stats['paths']['total_paths']} indexed")

    if "embeddings" in stats:
        print(
            f"\nEmbeddings: {stats['embeddings'].get('chunks', 0)} chunks, "
            f"{stats['embeddings'].get('entities', 0)} entities"
        )

    print("=" * 60)
    return stats


def main():
    args = parse_args()

    mode_pairs = resolve_paths(args)

    all_stats = {}
    for mode_label, silver_path, gold_path in mode_pairs:
        stats = run_gold_indexing(args, mode_label, silver_path, gold_path)
        all_stats[mode_label] = stats
        print()

    # Print cross-mode summary if multiple modes were processed
    if len(all_stats) > 1:
        print("\n" + "=" * 60)
        print("CROSS-MODE GOLD INDEXING SUMMARY")
        print("=" * 60)
        print(
            f"{'Mode':<10} {'Nodes':<10} {'Edges':<10} {'Communities':<13} {'Paths':<10} {'Time (s)':<10}"
        )
        print("-" * 63)
        for mode, st in all_stats.items():
            g = st.get("graph", {})
            c = st.get("communities", {})
            p = st.get("paths", {})
            dur = st.get("duration_seconds", 0)
            print(
                f"{mode:<10} "
                f"{g.get('total_nodes', 'N/A'):<10} "
                f"{g.get('total_edges', 'N/A'):<10} "
                f"{c.get('total', 'N/A'):<13} "
                f"{p.get('total_paths', 'N/A'):<10} "
                f"{dur:<10.1f}"
            )
        print("=" * 60)


if __name__ == "__main__":
    main()
