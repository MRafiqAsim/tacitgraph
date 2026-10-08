#!/usr/bin/env python3
"""
Display knowledge graph statistics — node types, edge types, entity catalog, and community info.

Usage:
    python run_graph_stats.py --gold ./data/gold_llm
    python run_graph_stats.py --gold ./data/gold_llm --show-entities FACILITY
    python run_graph_stats.py --gold ./data/gold_llm --unknown-types
"""

import argparse
import json
import sys
from pathlib import Path

from tacitgraph.paths import CONFIG_DIR


def main():
    parser = argparse.ArgumentParser(description="Display knowledge graph statistics")
    parser.add_argument("--gold", required=True, help="Path to Gold layer")
    parser.add_argument("--show-entities", type=str, help="Show entities of a specific type")
    parser.add_argument(
        "--unknown-types", action="store_true", help="Show only types not in entity_config.json"
    )
    parser.add_argument("--catalog", action="store_true", help="Show entity catalog summary")
    args = parser.parse_args()

    gold = Path(args.gold)

    # Load graph stats
    stats_file = gold / "knowledge_graph" / "graph_stats.json"
    if not stats_file.exists():
        print(f"ERROR: {stats_file} not found")
        sys.exit(1)

    with open(stats_file) as f:
        stats = json.load(f)

    # Load config for known types
    config_path = CONFIG_DIR / "entity_config.json"
    known_types = set()
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        known_types = set(cfg.get("entity_types", {}).keys())
        # Add aliases
        for info in cfg.get("entity_types", {}).values():
            for alias in info.get("llm_aliases", []):
                known_types.add(alias.upper())
            for label in info.get("spacy_labels", []):
                known_types.add(label.upper())

    # Node types
    node_types = stats.get("nodes_by_type", {})
    edge_types = stats.get("edges_by_type", {})

    print("=" * 60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("=" * 60)
    print(f"  Total nodes: {stats.get('total_nodes', 0)}")
    print(f"  Total edges: {stats.get('total_edges', 0)}")

    print(f"\nNode types ({len(node_types)}):")
    for t, c in sorted(node_types.items(), key=lambda x: -x[1]):
        marker = ""
        if args.unknown_types:
            if t in known_types or t in ("CHUNK", "THREAD", "EMAIL"):
                continue
            marker = " ← UNKNOWN"
        elif t not in known_types and t not in ("CHUNK", "THREAD", "EMAIL"):
            marker = " ← not in config"
        print(f"  {t:25s} {c:>6}{marker}")

    print(f"\nEdge types ({len(edge_types)}):")
    for t, c in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {t:30s} {c:>6}")

    # Show entities of a specific type
    if args.show_entities:
        nodes_file = gold / "knowledge_graph" / "nodes.json"
        if nodes_file.exists():
            with open(nodes_file) as f:
                nodes = json.load(f)
            print(f"\nEntities of type '{args.show_entities}':")
            count = 0
            for nid, node in sorted(nodes.items(), key=lambda x: -x[1].get("mention_count", 0)):
                ntype = node.get("node_type", node.get("type", ""))
                if ntype == args.show_entities:
                    name = node.get("name", nid)
                    mentions = node.get("mention_count", 0)
                    print(f"  {name:50s} mentions={mentions}")
                    count += 1
                    if count >= 50:
                        remaining = (
                            sum(
                                1
                                for n in nodes.values()
                                if n.get("node_type", n.get("type", "")) == args.show_entities
                            )
                            - 50
                        )
                        if remaining > 0:
                            print(f"  ... and {remaining} more")
                        break

    # Catalog summary
    if args.catalog:
        catalog_file = gold / "entity_catalog.json"
        if catalog_file.exists():
            with open(catalog_file) as f:
                catalog = json.load(f)
            entities = catalog.get("entities", {})
            with_aliases = sum(1 for e in entities.values() if e.get("aliases"))
            total_aliases = sum(len(e.get("aliases", [])) for e in entities.values())

            print("\nEntity Catalog:")
            print(f"  Total entities: {len(entities)}")
            print(f"  With aliases: {with_aliases}")
            print(f"  Total aliases: {total_aliases}")

            if with_aliases:
                print("\n  Entities with aliases:")
                for name, info in sorted(entities.items()):
                    aliases = info.get("aliases", [])
                    if aliases:
                        print(f"    {name} ({info.get('type', '?')}) → {aliases}")

    # Community info
    comm_dir = gold / "communities"
    if comm_dir.exists():
        levels = sorted(comm_dir.glob("level_*"))
        total_comms = 0
        print("\nCommunities:")
        for level_dir in levels:
            count = len(list(level_dir.glob("*.json")))
            total_comms += count
            print(f"  {level_dir.name}: {count} communities")
        print(f"  Total: {total_comms}")

    print("=" * 60)


if __name__ == "__main__":
    main()
