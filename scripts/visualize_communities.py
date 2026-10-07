"""Visualize knowledge graph communities as interactive HTML.

Color-codes nodes by community membership. Each community gets a unique color.

Usage:
    python scripts/visualize_communities.py --gold data/gold_llm
    python scripts/visualize_communities.py --gold data/gold_llm --level 0 --max-nodes 500
    python scripts/visualize_communities.py --gold data/gold_llm --level 1 --community 5
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def generate_colors(n):
    """Generate n distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        r = int(128 + 127 * abs((hue * 6) % 2 - 1))
        g = int(128 + 127 * abs(((hue * 6) + 4) % 6 - 3) / 3)
        b = int(128 + 127 * abs(((hue * 6) + 2) % 6 - 3) / 3)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def main():
    parser = argparse.ArgumentParser(description="Visualize communities as interactive HTML")
    parser.add_argument("--gold", required=True, help="Path to Gold layer")
    parser.add_argument("--level", type=int, default=0, help="Community level (default: 0)")
    parser.add_argument("--max-nodes", type=int, default=0, help="Max nodes to display (0 = all)")
    parser.add_argument("--top", type=int, default=0, help="Show only top N largest communities (0 = all)")
    parser.add_argument("--community", type=int, help="Show only a specific community ID")
    parser.add_argument("--output", default="communities.html", help="Output HTML file")
    parser.add_argument("--show-labels", action="store_true", help="Show edge type labels (best with --top 10 or fewer)")
    args = parser.parse_args()

    try:
        from pyvis.network import Network
    except ImportError:
        print("Install pyvis: pip install pyvis")
        return

    gold = Path(args.gold)

    # Load nodes
    nodes_file = gold / "knowledge_graph" / "nodes.json"
    edges_file = gold / "knowledge_graph" / "edges.json"

    if not nodes_file.exists():
        print(f"Nodes not found: {nodes_file}")
        return

    with open(nodes_file) as f:
        nodes_data = json.load(f)

    with open(edges_file) as f:
        edges_raw = json.load(f)
    edges_data = list(edges_raw.values()) if isinstance(edges_raw, dict) else edges_raw

    # Load communities
    comm_dir = gold / "communities" / f"level_{args.level}"
    if not comm_dir.exists():
        print(f"Community level {args.level} not found at {comm_dir}")
        return

    # Build node → community mapping
    node_to_community = {}
    community_info = {}

    for cf in sorted(comm_dir.glob("*.json")):
        try:
            with open(cf) as f:
                c = json.load(f)
            comm_id = c.get("community_id", cf.stem)
            node_ids = c.get("node_ids", [])
            summary = c.get("summary", "")
            key_entities = c.get("key_entities", [])
            key_topics = c.get("key_topics", [])

            community_info[comm_id] = {
                "size": len(node_ids),
                "summary": summary[:200],
                "key_entities": key_entities[:5],
                "key_topics": key_topics[:5],
            }

            for nid in node_ids:
                node_to_community[nid] = comm_id
        except Exception:
            pass

    print(f"Level {args.level}: {len(community_info)} communities, {len(node_to_community)} nodes assigned")

    # Filter to top N communities by size
    if args.top > 0:
        sorted_comms = sorted(community_info.keys(), key=lambda c: community_info[c]["size"], reverse=True)
        top_comms = set(sorted_comms[:args.top])
        # Remove nodes not in top communities
        node_to_community = {nid: cid for nid, cid in node_to_community.items() if cid in top_comms}
        community_info = {cid: info for cid, info in community_info.items() if cid in top_comms}
        print(f"Filtered to top {args.top} communities ({sum(c['size'] for c in community_info.values())} nodes)")

    # Filter to specific community if requested
    if args.community is not None:
        target_comm = str(args.community)
        matching = [cid for cid in community_info if str(cid) == target_comm or target_comm in str(cid)]
        if not matching:
            print(f"Community {args.community} not found. Available: {list(community_info.keys())[:20]}")
            return
        target_comm = matching[0]
        allowed_nodes = {nid for nid, cid in node_to_community.items() if cid == target_comm}
        print(f"Showing community {target_comm}: {len(allowed_nodes)} nodes")
    else:
        allowed_nodes = None

    # Generate colors per community
    comm_ids = sorted(community_info.keys(), key=lambda c: community_info[c]["size"], reverse=True)
    colors = generate_colors(len(comm_ids))
    comm_colors = {cid: colors[i] for i, cid in enumerate(comm_ids)}

    # Filter nodes — skip CHUNK/THREAD, apply limits
    skip_types = {"CHUNK", "THREAD", "EMAIL"}
    entity_nodes = {}

    for node_id, node in nodes_data.items():
        ntype = node.get("node_type", node.get("type", ""))
        if ntype in skip_types:
            continue
        if allowed_nodes and node_id not in allowed_nodes:
            continue
        if node_id in node_to_community:
            entity_nodes[node_id] = node

    # Sort by mentions, limit
    sorted_nodes = sorted(entity_nodes.items(), key=lambda x: x[1].get("mention_count", 0), reverse=True)
    if args.max_nodes > 0 and len(sorted_nodes) > args.max_nodes:
        sorted_nodes = sorted_nodes[:args.max_nodes]
        print(f"Limited to top {args.max_nodes} nodes")

    if len(sorted_nodes) > 5000:
        print(f"WARNING: {len(sorted_nodes)} nodes — browser may be slow. Use --max-nodes to limit.")

    included_ids = {n[0] for n in sorted_nodes}

    # Build network
    net = Network(height="900px", width="100%", cdn_resources="remote", select_menu=True)

    for node_id, node in sorted_nodes:
        ntype = node.get("node_type", node.get("type", "UNKNOWN"))
        name = node.get("name", node_id)
        mentions = node.get("mention_count", 0)
        comm_id = node_to_community.get(node_id, "none")
        color = comm_colors.get(comm_id, "#BDC3C7")

        info = community_info.get(comm_id, {})
        topics = ", ".join(info.get("key_topics", [])[:3])

        net.add_node(
            node_id,
            label=name[:25],
            title=f"<b>{name}</b><br>Type: {ntype}<br>Mentions: {mentions}<br>Community: {comm_id}<br>Topics: {topics}",
            color=color,
            size=min(8 + mentions * 0.3, 50),
            group=str(comm_id),
        )

    # Add edges
    edge_count = 0
    for edge in edges_data:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        if src in included_ids and tgt in included_ids:
            etype = edge.get("edge_type", "")
            edge_kwargs = {"title": etype, "width": 0.5}
            if args.show_labels:
                edge_kwargs["label"] = etype
                edge_kwargs["font"] = {"size": 8, "color": "#666666"}
            net.add_edge(src, tgt, **edge_kwargs)
            edge_count += 1

    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=100)
    net.show_buttons(filter_=["physics"])
    net.save_graph(args.output)

    print(f"\nCommunity visualization saved: {args.output}")
    print(f"  Level: {args.level}")
    print(f"  Nodes: {len(included_ids)}")
    print(f"  Edges: {edge_count}")
    print(f"  Communities shown: {len(set(node_to_community.get(n, '') for n in included_ids))}")

    # Print top communities
    print(f"\n  Top 10 communities by size:")
    for cid in comm_ids[:10]:
        info = community_info[cid]
        topics = ", ".join(info.get("key_topics", [])[:3])
        print(f"    Community {cid}: {info['size']} nodes — {topics}")

    print(f"\nOpen in browser:")
    print(f"  Mac:   open {args.output}")
    print(f"  Linux: xdg-open {args.output}")


if __name__ == "__main__":
    main()
