"""Generate interactive knowledge graph visualization as HTML.

Opens in any browser with zoom, pan, search, hover, and physics simulation.

Usage:
    python scripts/visualize_graph.py --gold data/gold_llm
    python scripts/visualize_graph.py --gold data/gold_llm --output my_graph.html
    python scripts/visualize_graph.py --gold data/gold_llm --max-nodes 500
    python scripts/visualize_graph.py --gold data/gold_llm --type PERSON ORG GPE
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Visualize knowledge graph as interactive HTML")
    parser.add_argument("--gold", required=True, help="Path to Gold layer")
    parser.add_argument("--output", default="knowledge_graph.html", help="Output HTML file")
    parser.add_argument(
        "--max-nodes", type=int, default=0, help="Max nodes to display (0 = all, default: all)"
    )
    parser.add_argument(
        "--type", nargs="*", help="Filter to specific node types (e.g., PERSON ORG GPE)"
    )
    parser.add_argument(
        "--min-mentions", type=int, default=1, help="Min mention count to include (default: 1)"
    )
    parser.add_argument(
        "--show-labels", action="store_true", help="Show edge type labels (best with <500 nodes)"
    )
    args = parser.parse_args()

    try:
        from pyvis.network import Network
    except ImportError:
        print("Install pyvis: pip install pyvis")
        return

    gold = Path(args.gold)

    # Load graph
    nodes_file = gold / "knowledge_graph" / "nodes.json"
    edges_file = gold / "knowledge_graph" / "edges.json"

    if not nodes_file.exists():
        # Try graphml
        graphml_file = gold / "knowledge_graph" / "graph.graphml"
        if graphml_file.exists():
            import networkx as nx

            G = nx.read_graphml(str(graphml_file))
            print(f"Loaded GraphML: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        else:
            print(f"No graph found at {gold}/knowledge_graph/")
            return
    else:
        G = None

    colors = {
        "PERSON": "#FF6B6B",
        "ORG": "#4ECDC4",
        "GPE": "#45B7D1",
        "LOC": "#A3E4D7",
        "FACILITY": "#FFA07A",
        "PRODUCT": "#98D8C8",
        "PROCESS": "#F7DC6F",
        "PROJECT": "#BB8FCE",
        "DOCUMENT": "#85C1E9",
        "EQUIPMENT": "#F0B27A",
        "CONCEPT": "#AED6F1",
        "EVENT": "#F5B7B1",
        "LAW": "#D5DBDB",
        "NORP": "#E8DAEF",
        "MATERIAL": "#FADBD8",
    }

    # Build from JSON
    if G is None:
        with open(nodes_file) as f:
            nodes_data = json.load(f)
        with open(edges_file) as f:
            edges_raw = json.load(f)

        edges_data = list(edges_raw.values()) if isinstance(edges_raw, dict) else edges_raw

        # Filter to entity nodes (skip CHUNK, THREAD, EMAIL)
        skip_types = {"CHUNK", "THREAD", "EMAIL"}
        entity_nodes = {}

        for node_id, node in nodes_data.items():
            ntype = node.get("node_type", node.get("type", "UNKNOWN"))
            if ntype in skip_types:
                continue
            if args.type and ntype not in args.type:
                continue
            mentions = node.get("mention_count", 0)
            if mentions < args.min_mentions:
                continue
            entity_nodes[node_id] = node

        # Sort by mentions and limit
        sorted_nodes = sorted(
            entity_nodes.items(), key=lambda x: x[1].get("mention_count", 0), reverse=True
        )
        if args.max_nodes > 0 and len(sorted_nodes) > args.max_nodes:
            sorted_nodes = sorted_nodes[: args.max_nodes]
            print(f"Limited to top {args.max_nodes} nodes by mention count")

        if len(sorted_nodes) > 5000:
            print(
                f"WARNING: {len(sorted_nodes)} nodes — browser may be slow. Use --max-nodes to limit."
            )

        included_ids = {n[0] for n in sorted_nodes}

        # Build pyvis network
        net = Network(
            height="900px",
            width="100%",
            cdn_resources="remote",
            select_menu=True,
            filter_menu=True,
        )

        # Add nodes
        for node_id, node in sorted_nodes:
            ntype = node.get("node_type", node.get("type", "UNKNOWN"))
            name = node.get("name", node_id)
            mentions = node.get("mention_count", 0)

            net.add_node(
                node_id,
                label=name[:30],
                title=f"<b>{name}</b><br>Type: {ntype}<br>Mentions: {mentions}",
                color=colors.get(ntype, "#BDC3C7"),
                size=min(8 + mentions * 0.3, 50),
                group=ntype,
            )

        # Add edges
        edge_count = 0
        for edge in edges_data:
            if not isinstance(edge, dict):
                continue
            src = edge.get("source_id", "")
            tgt = edge.get("target_id", "")
            if src in included_ids and tgt in included_ids:
                etype = edge.get("edge_type", "RELATED_TO")
                weight = edge.get("weight", 1.0)
                edge_kwargs = {"title": etype, "width": max(0.5, weight * 2)}
                if args.show_labels:
                    edge_kwargs["label"] = etype
                    edge_kwargs["font"] = {"size": 8, "color": "#666666"}
                net.add_edge(src, tgt, **edge_kwargs)
                edge_count += 1

    else:
        # Build from NetworkX GraphML
        entity_nodes = [
            n
            for n in G.nodes()
            if G.nodes[n].get("node_type", "") not in ["CHUNK", "THREAD", "EMAIL"]
        ]

        if args.type:
            entity_nodes = [n for n in entity_nodes if G.nodes[n].get("node_type", "") in args.type]

        # Limit
        entity_nodes = sorted(entity_nodes, key=lambda n: G.degree(n), reverse=True)
        if args.max_nodes > 0:
            entity_nodes = entity_nodes[: args.max_nodes]
        included_ids = set(entity_nodes)

        net = Network(
            height="900px", width="100%", cdn_resources="remote", select_menu=True, filter_menu=True
        )

        for n in entity_nodes:
            ntype = G.nodes[n].get("node_type", "UNKNOWN")
            name = G.nodes[n].get("name", n)
            net.add_node(
                n,
                label=name[:30],
                title=f"<b>{name}</b><br>Type: {ntype}",
                color=colors.get(ntype, "#BDC3C7"),
                group=ntype,
            )

        edge_count = 0
        for e in G.edges():
            if e[0] in included_ids and e[1] in included_ids:
                net.add_edge(e[0], e[1])
                edge_count += 1

    # Configure physics
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=100)
    net.show_buttons(filter_=["physics"])

    # Save
    net.save_graph(args.output)

    node_count = len(included_ids) if G is None else len(entity_nodes)
    print(f"\nGraph visualization saved: {args.output}")
    print(f"  Nodes: {node_count}")
    print(f"  Edges: {edge_count}")
    print("\nOpen in browser:")
    print(f"  Mac:   open {args.output}")
    print(f"  Linux: xdg-open {args.output}")
    print(f"  Windows: start {args.output}")


if __name__ == "__main__":
    main()
