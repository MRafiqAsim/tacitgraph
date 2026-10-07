"""Generate comprehensive Gold layer statistics.

Reports on knowledge graph, communities, paths, embeddings, and entity catalog.

Usage:
    python scripts/gold_stats.py --gold data/gold_llm
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Gold layer statistics")
    parser.add_argument("--gold", required=True, help="Path to Gold layer")
    args = parser.parse_args()

    gold = Path(args.gold)

    print("=" * 70)
    print("GOLD LAYER STATISTICS")
    print("=" * 70)

    # ============================================================
    # KNOWLEDGE GRAPH
    # ============================================================
    nodes_file = gold / "knowledge_graph" / "nodes.json"
    edges_file = gold / "knowledge_graph" / "edges.json"

    if nodes_file.exists():
        with open(nodes_file) as f:
            nodes = json.load(f)

        node_types = Counter()
        top_entities = {}
        for node_id, node in nodes.items():
            ntype = node.get("node_type", node.get("type", "UNKNOWN"))
            node_types[ntype] += 1
            # Track top entities by mention count
            mention_count = node.get("mention_count", 0)
            if ntype not in ["CHUNK", "THREAD", "EMAIL"]:
                name = node.get("name", node_id)
                if ntype not in top_entities:
                    top_entities[ntype] = []
                top_entities[ntype].append((name, mention_count))

        print("\n--- Knowledge Graph ---")
        print(f"  Total nodes:                  {len(nodes)}")

        print("\n  Node types:")
        for ntype, count in node_types.most_common():
            print(f"    {ntype:25s} {count}")

        # Top entities per type — sorted by total mentions descending, show only main types
        print("\n--- Top Entities by Type (top 10 types) ---")
        type_total_mentions = {}
        for ntype, ents in top_entities.items():
            type_total_mentions[ntype] = sum(m for _, m in ents)

        sorted_types = sorted(
            type_total_mentions.keys(), key=lambda t: type_total_mentions[t], reverse=True
        )

        for ntype in sorted_types[:10]:
            entities = sorted(top_entities[ntype], key=lambda x: x[1], reverse=True)[:5]
            total = type_total_mentions[ntype]
            if entities:
                print(f"\n  {ntype} ({total} total mentions):")
                for name, mentions in entities:
                    print(f"    {name:40s} {mentions} mentions")
    else:
        print(f"\n  Knowledge graph not found at {nodes_file}")

    if edges_file.exists():
        with open(edges_file) as f:
            edges_raw = json.load(f)

        # Support both dict (keyed by edge_id) and list formats
        if isinstance(edges_raw, dict):
            edges = list(edges_raw.values())
        else:
            edges = edges_raw

        edge_types = Counter()
        for edge in edges:
            if isinstance(edge, dict):
                etype = edge.get("edge_type", edge.get("type", "UNKNOWN"))
            else:
                etype = "UNKNOWN"
            edge_types[etype] += 1

        print("\n--- Edges ---")
        print(f"  Total edges:                  {len(edges)}")
        print("\n  Edge types:")
        for etype, count in edge_types.most_common():
            print(f"    {etype:25s} {count}")

    # ============================================================
    # COMMUNITIES
    # ============================================================
    communities_dir = gold / "communities"
    if communities_dir.exists():
        print("\n--- Communities ---")
        total_communities = 0
        for level_dir in sorted(communities_dir.iterdir()):
            if level_dir.is_dir() and level_dir.name.startswith("level_"):
                level = level_dir.name
                comm_files = list(level_dir.glob("*.json"))
                total_communities += len(comm_files)

                # Sample community sizes
                sizes = []
                has_summary = 0
                for cf in comm_files:
                    try:
                        with open(cf) as f:
                            c = json.load(f)
                        sizes.append(len(c.get("node_ids", [])))
                        if c.get("summary", "").strip():
                            has_summary += 1
                    except Exception:
                        pass

                avg_size = sum(sizes) // len(sizes) if sizes else 0
                max_size = max(sizes) if sizes else 0
                min_size = min(sizes) if sizes else 0
                print(
                    f"  {level}: {len(comm_files)} communities (avg {avg_size} nodes, min {min_size}, max {max_size}, {has_summary} summarized)"
                )

        print(f"  Total communities:            {total_communities}")
    else:
        print("\n  Communities directory not found")

    # ============================================================
    # PATHS
    # ============================================================
    paths_dir = gold / "path_index"
    if paths_dir.exists():
        paths_file = paths_dir / "paths.json"
        if paths_file.exists():
            with open(paths_file) as f:
                paths = json.load(f)

            # Support both list and dict formats
            if isinstance(paths, dict):
                paths = list(paths.values())

            path_types = Counter()
            path_lengths = Counter()
            for p in paths:
                if not isinstance(p, dict):
                    continue
                src_type = p.get("source_entity", {}).get("type", "?")
                tgt_type = p.get("target_entity", {}).get("type", "?")
                path_types[f"{src_type} → {tgt_type}"] += 1
                path_lengths[len(p.get("path_nodes", []))] += 1

            print("\n--- Path Index ---")
            print(f"  Total paths:                  {len(paths)}")

            print("\n  Path type pairs:")
            for ptype, count in path_types.most_common(15):
                print(f"    {ptype:30s} {count}")

            print("\n  Path lengths (hops):")
            for length, count in sorted(path_lengths.items()):
                print(f"    {length} hops: {count}")
        else:
            # Check for individual path files
            path_files = list(paths_dir.glob("*.json"))
            print("\n--- Path Index ---")
            print(f"  Path files:                   {len(path_files)}")
    else:
        print("\n  Path index not found")

    # ============================================================
    # EMBEDDINGS
    # ============================================================
    emb_dir = gold / "embeddings"
    if emb_dir.exists():
        print("\n--- Embeddings ---")

        # Check for .npy + _ids.json format
        for ids_file in sorted(emb_dir.glob("*_ids.json")):
            prefix = ids_file.name.replace("_ids.json", "")
            emb_file = emb_dir / f"{prefix}_embeddings.npy"

            try:
                with open(ids_file) as f:
                    ids = json.load(f)
                if emb_file.exists():
                    import numpy as np

                    embeddings = np.load(emb_file)
                    dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
                    size_mb = emb_file.stat().st_size / (1024 * 1024)
                    print(f"  {prefix:25s} {len(ids)} vectors, {dim} dimensions, {size_mb:.1f} MB")
                else:
                    print(f"  {prefix:25s} {len(ids)} IDs (no .npy file)")
            except Exception:
                print(f"  {prefix:25s} (error reading)")

        # Also check .npz format
        for emb_file in sorted(emb_dir.glob("*.npz")):
            try:
                import numpy as np

                data = np.load(emb_file, allow_pickle=True)
                ids = data.get("ids", [])
                embeddings = data.get("embeddings", [])
                dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
                print(f"  {emb_file.name:25s} {len(ids)} vectors, {dim} dimensions")
            except Exception:
                size_mb = emb_file.stat().st_size / (1024 * 1024)
                print(f"  {emb_file.name:25s} {size_mb:.1f} MB")

        if not list(emb_dir.glob("*_ids.json")) and not list(emb_dir.glob("*.npz")):
            print("  No embedding files found")
    else:
        print("\n  Embeddings directory not found")

    # ============================================================
    # ENTITY CATALOG
    # ============================================================
    catalog_file = gold / "entity_catalog.json"
    if catalog_file.exists():
        with open(catalog_file) as f:
            catalog = json.load(f)

        entities = catalog.get("entities", {})
        catalog_types = Counter()
        has_aliases = 0
        for name, info in entities.items():
            catalog_types[info.get("type", "UNKNOWN")] += 1
            if info.get("aliases"):
                has_aliases += 1

        print("\n--- Entity Catalog ---")
        print(f"  Total catalog entries:        {len(entities)}")
        print(f"  Entries with aliases:         {has_aliases}")

        print("\n  Catalog entity types:")
        for ctype, count in catalog_types.most_common():
            print(f"    {ctype:25s} {count}")
    else:
        print("\n  Entity catalog not found")

    # ============================================================
    # GRAPH STATS FILE
    # ============================================================
    stats_file = gold / "gold_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        print("\n--- Processing Info ---")
        print(f"  Timestamp:                    {stats.get('timestamp', 'unknown')}")
        print(f"  Mode:                         {stats.get('mode', 'unknown')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
