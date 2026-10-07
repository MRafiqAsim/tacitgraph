#!/usr/bin/env python3
"""
Visualize the Knowledge Graph structure from Gold layer.

Generates two outputs:
1. schema_diagram.png  — entity-relationship schema (node types + edge types)
2. sample_graph.png    — actual sample subgraph from your Gold data

Usage:
    python visualize_graph.py --gold ./data/gold_llm
    python visualize_graph.py --gold ./data/gold_llm --sample 100
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    print("ERROR: Install matplotlib and networkx first:")
    print("  pip install matplotlib networkx")
    exit(1)


# Colors per node type
NODE_COLORS = {
    "PERSON": "#4A90D9",
    "ORG": "#E8A838",
    "GPE": "#5CB85C",
    "PRODUCT": "#F39C12",
    "LAW": "#8E44AD",
    "CONCEPT": "#95A5A6",
    "DOCUMENT": "#9B59B6",
    "CHUNK": "#E74C3C",
    "THREAD": "#1ABC9C",
}
DEFAULT_COLOR = "#BDC3C7"


def draw_schema(output_path: str):
    """Draw the schema diagram — node types and edge types."""
    G = nx.DiGraph()

    node_types = [
        "PERSON",
        "ORG",
        "GPE",
        "PRODUCT",
        "LAW",
        "CONCEPT",
        "DOCUMENT",
        "CHUNK",
        "THREAD",
    ]
    for nt in node_types:
        G.add_node(nt)

    edges = [
        ("PERSON", "THREAD", "PARTICIPATED_IN"),
        ("PERSON", "ORG", "WORKS_FOR"),
        ("PERSON", "GPE", "RELATED_TO"),
        ("PERSON", "PERSON", "RELATED_TO"),
        ("ORG", "GPE", "LOCATED_IN"),
        ("ORG", "PRODUCT", "RELATED_TO"),
        ("ORG", "ORG", "RELATED_TO"),
        ("CHUNK", "THREAD", "PART_OF_THREAD"),
        ("CHUNK", "DOCUMENT", "HAS_ATTACHMENT"),
        ("PERSON", "CHUNK", "MENTIONED_IN"),
        ("ORG", "CHUNK", "MENTIONED_IN"),
        ("GPE", "CHUNK", "MENTIONED_IN"),
        ("CONCEPT", "CHUNK", "MENTIONED_IN"),
        ("LAW", "CHUNK", "MENTIONED_IN"),
    ]

    for src, tgt, label in edges:
        G.add_edge(src, tgt, label=label)

    _fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_title(
        "Knowledge Graph Schema — Node Types & Relationships",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    pos = {
        "PERSON": (-3, 1),
        "ORG": (-1, 2),
        "GPE": (1, 2),
        "PRODUCT": (3, 1),
        "LAW": (3, -1),
        "CONCEPT": (1, -2),
        "CHUNK": (-1, -2),
        "THREAD": (-3, -1),
        "DOCUMENT": (0, 0),
    }

    node_color_list = [NODE_COLORS.get(n, DEFAULT_COLOR) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list, node_size=2500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold", font_color="white")

    edge_labels = {(s, t): label for s, t, label in edges}
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#555555",
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        width=1.5,
        alpha=0.7,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=7,
        font_color="#333333",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
    )

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=t) for t, c in NODE_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, title="Node Types")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Schema diagram saved: {output_path}")


def draw_sample_graph(gold_path: str, output_path: str, sample_size: int = 80):
    """Draw a sample subgraph from actual Gold data."""
    gold = Path(gold_path)
    nodes_file = gold / "knowledge_graph" / "nodes.json"
    edges_file = gold / "knowledge_graph" / "edges.json"

    if not nodes_file.exists():
        print(f"ERROR: {nodes_file} not found. Run Gold indexing first.")
        return

    with open(nodes_file, encoding="utf-8") as f:
        raw = json.load(f)
        nodes = list(raw.values()) if isinstance(raw, dict) else raw
    with open(edges_file, encoding="utf-8") as f:
        raw = json.load(f)
        edges = list(raw.values()) if isinstance(raw, dict) else raw

    print(f"Full graph: {len(nodes)} nodes, {len(edges)} edges")

    # Sample: take top node types, exclude CHUNK nodes (too many)
    priority_types = ["PERSON", "ORG", "GPE", "PRODUCT", "CONCEPT", "THREAD"]
    sampled_nodes = []
    type_counts: defaultdict[str, int] = defaultdict(int)
    per_type_limit = max(5, sample_size // len(priority_types))

    for n in nodes:
        nt = n.get("node_type", "")
        if nt in priority_types and type_counts[nt] < per_type_limit:
            sampled_nodes.append(n)
            type_counts[nt] += 1

    sampled_ids = {n["node_id"] for n in sampled_nodes}

    # Only edges between sampled nodes
    sampled_edges = [
        e
        for e in edges
        if e.get("source_id") in sampled_ids
        and e.get("target_id") in sampled_ids
        and e.get("edge_type") != "MENTIONED_IN"  # skip for clarity
    ]

    print(f"Sample: {len(sampled_nodes)} nodes, {len(sampled_edges)} edges")

    G = nx.DiGraph()
    for n in sampled_nodes:
        G.add_node(n["node_id"], label=n.get("name", "")[:20], node_type=n.get("node_type", ""))
    for e in sampled_edges:
        G.add_edge(e["source_id"], e["target_id"], label=e.get("edge_type", ""))

    _fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_title(
        f"Knowledge Graph Sample — {len(sampled_nodes)} nodes, {len(sampled_edges)} edges\n(CHUNK and MENTIONED_IN edges hidden for clarity)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    try:
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    node_color_list = [
        NODE_COLORS.get(G.nodes[n].get("node_type", ""), DEFAULT_COLOR) for n in G.nodes()
    ]
    labels = {n: G.nodes[n].get("label", n[:10]) for n in G.nodes()}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list, node_size=800, alpha=0.85)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6, font_color="white")
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#888888",
        arrows=True,
        arrowsize=12,
        width=0.8,
        alpha=0.5,
        connectionstyle="arc3,rad=0.05",
    )

    # Type counts in subtitle
    type_summary = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))
    ax.set_xlabel(f"Node breakdown: {type_summary}", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=NODE_COLORS.get(t, DEFAULT_COLOR), label=t) for t in priority_types
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, title="Node Types")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample graph saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize knowledge graph structure")
    parser.add_argument(
        "--gold", "-g", type=str, default="./data/gold_llm", help="Path to Gold layer directory"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=80,
        help="Number of nodes to sample for sample graph (default: 80)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for diagrams (default: current dir)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    draw_schema(str(out / "kg_schema.png"))
    draw_sample_graph(args.gold, str(out / "kg_sample.png"), sample_size=args.sample)

    print(f"\nDone. Files saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
