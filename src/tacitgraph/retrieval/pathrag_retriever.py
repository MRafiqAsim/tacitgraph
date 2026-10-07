"""
PathRAG Retriever Integration

Integrates the official PathRAG algorithms with our pipeline.
Uses PathRAG's flow-based path pruning for query-time path finding.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class PathRAGConfig:
    """Configuration for PathRAG retrieval."""

    max_hops: int = 3  # Maximum path length (1-3 hops between entities)
    flow_threshold: float = 0.3  # Threshold for flow-based pruning
    flow_alpha: float = 0.8  # Alpha parameter for BFS weighted paths
    top_k_entities: int = 10  # Top-k entities from vector search
    top_k_paths: int = 15  # Maximum paths to return
    max_token_for_context: int = 4000


@dataclass
class PathResult:
    """Result from path-based retrieval."""

    path: list[str]  # Node IDs in path
    path_names: list[str]  # Node names
    path_types: list[str]  # Node types
    edges: list[dict[str, Any]]  # Edge data
    weight: float  # Path weight from flow-based pruning
    hop_count: int
    natural_language: str  # Path converted to text
    evidence_chunks: list[str]  # Source chunks


class PathRAGRetriever:
    """
    PathRAG retriever using flow-based path pruning.

    Implements the key algorithms from the PathRAG paper:
    1. DFS path finding (up to 3 hops)
    2. Flow-based pruning with BFS weighted paths
    3. Path-to-text conversion for LLM prompting
    """

    def __init__(self, gold_path: str, config: PathRAGConfig | None = None):
        self.gold_path = Path(gold_path)
        self.config = config or PathRAGConfig()

        # Lazy-loaded components
        self._graph = None
        self._nx_graph = None
        self._embeddings = None

        logger.info("PathRAGRetriever initialized")

    def _load_graph(self):
        """Load knowledge graph."""
        if self._graph is None:
            from tacitgraph.gold.graph_builder import GraphBuilder

            builder = GraphBuilder("", str(self.gold_path))
            self._graph = builder.load()
            logger.info(
                f"Loaded graph: {len(self._graph.nodes)} nodes, {len(self._graph.edges)} edges"
            )
        return self._graph

    def _build_nx_graph(self) -> nx.Graph:
        """
        Build entity-only NetworkX graph for PathRAG path finding.

        Per the PathRAG paper, the indexing graph contains entity nodes
        and their relationships — NOT chunk/thread nodes. We also exclude
        high-fan-out structural edges (MENTIONED_IN, PARTICIPATED_IN,
        HAS_ATTACHMENT, PART_OF_THREAD) which connect entities to chunks
        rather than to each other meaningfully.
        """
        if self._nx_graph is None:
            graph = self._load_graph()
            self._nx_graph = nx.Graph()

            from tacitgraph.entity_registry import get_entity_node_types, get_structural_edge_types

            ENTITY_TYPES = get_entity_node_types()
            EXCLUDE_EDGE_TYPES = get_structural_edge_types()

            entity_ids = {
                nid for nid, node in graph.nodes.items() if node.node_type in ENTITY_TYPES
            }

            # Add entity nodes only
            for node_id in entity_ids:
                node = graph.nodes[node_id]
                self._nx_graph.add_node(
                    node_id, name=node.name, type=node.node_type, source_chunks=node.source_chunks
                )

            # Add entity-to-entity edges only
            for edge_id, edge in graph.edges.items():
                if edge.edge_type in EXCLUDE_EDGE_TYPES:
                    continue
                if edge.source_id in entity_ids and edge.target_id in entity_ids:
                    self._nx_graph.add_edge(
                        edge.source_id,
                        edge.target_id,
                        edge_type=edge.edge_type,
                        weight=edge.weight,
                        description=edge.properties.get("description", ""),
                        keywords=edge.properties.get("keywords", ""),
                    )

            logger.info(
                f"Built entity-only PathRAG graph: "
                f"{self._nx_graph.number_of_nodes()} nodes, "
                f"{self._nx_graph.number_of_edges()} edges"
            )

        return self._nx_graph

    async def find_paths_between_entities(
        self, source_entities: list[str]
    ) -> tuple[dict, dict, list, list, list]:
        """
        Find paths between entities using DFS (PathRAG algorithm).

        Optimizations vs naive DFS:
        - Fan-out limit per node (max_neighbors) to prevent explosion on hub nodes
        - Neighbors sorted by edge weight (highest-weight = most relevant first)
        - Max paths per pair cap to stop early
        - Entity-only graph (no CHUNK/THREAD nodes)

        Args:
            source_entities: List of entity node IDs

        Returns:
            Tuple of (result_dict, path_stats, one_hop, two_hop, three_hop paths)
        """
        G = self._build_nx_graph()

        result = defaultdict(lambda: {"paths": [], "edges": set()})
        path_stats = {"1-hop": 0, "2-hop": 0, "3-hop": 0}
        one_hop_paths = []
        two_hop_paths = []
        three_hop_paths = []

        MAX_NEIGHBORS = 15  # Fan-out limit per node (prevents hub explosion)
        MAX_PATHS_PER_PAIR = 5  # Stop DFS early once enough paths found

        # Pre-compute sorted neighbor lists (by edge weight, descending)
        _neighbor_cache: dict[str, list[str]] = {}

        def _get_neighbors(node_id: str) -> list[str]:
            if node_id not in _neighbor_cache:
                neighbors = list(G.neighbors(node_id))
                # Sort by edge weight descending (most important connections first)
                neighbors.sort(
                    key=lambda n: G.edges[node_id, n].get("weight", 0.5),
                    reverse=True,
                )
                _neighbor_cache[node_id] = neighbors[:MAX_NEIGHBORS]
            return _neighbor_cache[node_id]

        def dfs(current: str, target: str, path: list[str], depth: int, pair_key):
            """DFS to find paths up to max_hops with fan-out limiting."""
            if depth > self.config.max_hops:
                return
            if current == target:
                result[pair_key]["paths"].append(list(path))
                for u, v in zip(path[:-1], path[1:]):
                    result[pair_key]["edges"].add(tuple(sorted((u, v))))
                if depth == 1:
                    path_stats["1-hop"] += 1
                    one_hop_paths.append(list(path))
                elif depth == 2:
                    path_stats["2-hop"] += 1
                    two_hop_paths.append(list(path))
                elif depth == 3:
                    path_stats["3-hop"] += 1
                    three_hop_paths.append(list(path))
                return

            if current not in G:
                return

            # Early stopping: enough paths for this pair
            if len(result[pair_key]["paths"]) >= MAX_PATHS_PER_PAIR:
                return

            path_set = set(path)  # O(1) membership check
            for neighbor in _get_neighbors(current):
                if neighbor not in path_set:
                    dfs(neighbor, target, path + [neighbor], depth + 1, pair_key)
                    # Re-check after recursion
                    if len(result[pair_key]["paths"]) >= MAX_PATHS_PER_PAIR:
                        return

        # Find paths between all entity pairs
        for node1 in source_entities:
            for node2 in source_entities:
                if node1 != node2:
                    pair_key = (node1, node2)
                    dfs(node1, node2, [node1], 0, pair_key)

        # Convert edges to lists
        for key in result:
            result[key]["edges"] = list(result[key]["edges"])

        logger.info(f"DFS paths: {path_stats}")
        return dict(result), path_stats, one_hop_paths, two_hop_paths, three_hop_paths

    def bfs_weighted_paths(
        self, paths: list[list[str]], source: str, target: str
    ) -> list[tuple[list[str], float]]:
        """
        Flow-based path pruning using BFS weighted paths.

        This is the key PathRAG algorithm that prunes redundant paths
        based on edge flow weights.
        """
        threshold = self.config.flow_threshold
        alpha = self.config.flow_alpha

        edge_weights = defaultdict(float)
        follow_dict = {}

        # Build follow dictionary from paths
        for p in paths:
            for i in range(len(p) - 1):
                current = p[i]
                next_node = p[i + 1]
                if current in follow_dict:
                    follow_dict[current].add(next_node)
                else:
                    follow_dict[current] = {next_node}

        if source not in follow_dict:
            return [(p, 1.0) for p in paths]  # Return all paths if no flow

        results = []

        # BFS to compute edge weights
        for neighbor in follow_dict[source]:
            edge_weights[(source, neighbor)] += 1 / len(follow_dict[source])

            if neighbor == target:
                results.append([source, neighbor])
                continue

            if edge_weights[(source, neighbor)] > threshold and neighbor in follow_dict:
                for second_neighbor in follow_dict[neighbor]:
                    weight = edge_weights[(source, neighbor)] * alpha / len(follow_dict[neighbor])
                    edge_weights[(neighbor, second_neighbor)] += weight

                    if second_neighbor == target:
                        results.append([source, neighbor, second_neighbor])
                        continue

                    if (
                        edge_weights[(neighbor, second_neighbor)] > threshold
                        and second_neighbor in follow_dict
                    ):
                        for third_neighbor in follow_dict[second_neighbor]:
                            weight = (
                                edge_weights[(neighbor, second_neighbor)]
                                * alpha
                                / len(follow_dict[second_neighbor])
                            )
                            edge_weights[(second_neighbor, third_neighbor)] += weight

                            if third_neighbor == target:
                                results.append([source, neighbor, second_neighbor, third_neighbor])

        # Calculate path weights
        path_weights = []
        for p in paths:
            path_weight = 0
            for i in range(len(p) - 1):
                edge = (p[i], p[i + 1])
                path_weight += edge_weights.get(edge, 0)
            path_weights.append(path_weight / (len(p) - 1) if len(p) > 1 else 0)

        return list(zip(paths, path_weights))

    def path_to_natural_language(self, path: list[str]) -> str:
        """
        Convert a path to natural language description.

        This is how PathRAG presents paths to the LLM.
        """
        G = self._build_nx_graph()
        graph = self._load_graph()

        parts = []

        for i, node_id in enumerate(path):
            node_data = G.nodes.get(node_id, {})
            node_name = node_data.get("name", node_id)
            node_type = node_data.get("type", "UNKNOWN")

            # Get node description from knowledge graph
            kg_node = graph.get_node(node_id)
            description = ""
            if kg_node and kg_node.properties:
                description = kg_node.properties.get("description", "")[:100]

            entity_desc = f"Entity '{node_name}' ({node_type})"
            if description:
                entity_desc += f": {description}"

            if i == 0:
                parts.append(entity_desc)
            else:
                # Get edge info
                prev_id = path[i - 1]
                edge_data = G.edges.get((prev_id, node_id), G.edges.get((node_id, prev_id), {}))
                edge_type = edge_data.get("edge_type", "related_to")
                keywords = edge_data.get("keywords", "")

                connection = f" --[{edge_type}"
                if keywords:
                    connection += f": {keywords}"
                connection += f"]--> {entity_desc}"
                parts.append(connection)

        return "".join(parts)

    async def retrieve(self, entity_ids: list[str], max_paths: int = None) -> list[PathResult]:
        """
        Main retrieval method using PathRAG algorithms.

        Args:
            entity_ids: List of entity node IDs to find paths between
            max_paths: Maximum paths to return

        Returns:
            List of PathResult objects
        """
        max_paths = max_paths or self.config.top_k_paths

        if len(entity_ids) == 0:
            return []

        # Single entity: 1-hop neighbor lookup
        if len(entity_ids) == 1:
            return self._retrieve_single_entity(entity_ids[0], max_paths)

        # Step 1: Find all paths using DFS
        result, path_stats, one_hop, two_hop, three_hop = await self.find_paths_between_entities(
            entity_ids
        )

        logger.info(f"Path stats: {path_stats}")

        # Step 2: Apply flow-based pruning
        all_results = []
        for node1 in entity_ids:
            for node2 in entity_ids:
                if node1 != node2 and (node1, node2) in result:
                    paths = result[(node1, node2)]["paths"]
                    weighted_paths = self.bfs_weighted_paths(paths, node1, node2)
                    all_results.extend(weighted_paths)

        # Sort by weight (descending)
        all_results = sorted(all_results, key=lambda x: x[1], reverse=True)

        # Deduplicate paths
        seen = set()
        unique_results = []
        for path, weight in all_results:
            sorted_path = tuple(sorted(path))
            if sorted_path not in seen:
                seen.add(sorted_path)
                unique_results.append((path, weight))

        # Take top-k
        unique_results = unique_results[: max_paths * 2]  # Get extra for re-ranking

        # Step 3: Convert to PathResult objects
        G = self._build_nx_graph()
        graph = self._load_graph()

        path_results = []
        for path, weight in unique_results:
            # Get node names and types
            path_names = []
            path_types = []
            # Collect evidence chunks, prioritizing chunks shared by multiple path entities
            chunks_per_node = []

            for node_id in path:
                node_data = G.nodes.get(node_id, {})
                path_names.append(node_data.get("name", node_id))
                path_types.append(node_data.get("type", "UNKNOWN"))

                kg_node = graph.get_node(node_id)
                if kg_node:
                    chunks_per_node.append(set(kg_node.source_chunks))

            # Score chunks by how many path entities mention them (intersection priority)
            chunk_counts = {}
            for chunk_set in chunks_per_node:
                for cid in chunk_set:
                    chunk_counts[cid] = chunk_counts.get(cid, 0) + 1

            # Sort: chunks mentioned by most path entities first
            ranked_chunks = sorted(chunk_counts.keys(), key=lambda c: chunk_counts[c], reverse=True)
            evidence_chunks = ranked_chunks

            # Get edges
            edges = []
            for i in range(len(path) - 1):
                edge_data = G.edges.get(
                    (path[i], path[i + 1]), G.edges.get((path[i + 1], path[i]), {})
                )
                edges.append(dict(edge_data))

            # Convert to natural language
            nl_description = self.path_to_natural_language(path)

            path_results.append(
                PathResult(
                    path=path,
                    path_names=path_names,
                    path_types=path_types,
                    edges=edges,
                    weight=weight,
                    hop_count=len(path) - 1,
                    natural_language=nl_description,
                    evidence_chunks=list(evidence_chunks)[:20],
                )
            )

        logger.info(f"Returning {len(path_results)} paths")
        return path_results

    def _retrieve_single_entity(self, entity_id: str, max_paths: int) -> list[PathResult]:
        """1-hop neighbor retrieval for single-entity queries."""
        G = self._build_nx_graph()
        graph = self._load_graph()

        if entity_id not in G:
            logger.warning(f"Entity {entity_id} not in graph")
            return []

        entity_data = G.nodes.get(entity_id, {})
        entity_name = entity_data.get("name", entity_id)
        entity_type = entity_data.get("type", "UNKNOWN")

        # Get all 1-hop neighbors with edge info
        path_results = []
        neighbors = list(G.neighbors(entity_id))

        # Sort by edge weight (most important connections first)
        neighbors.sort(
            key=lambda n: G.edges[entity_id, n].get("weight", 0.5),
            reverse=True,
        )

        for neighbor_id in neighbors[:max_paths]:
            neighbor_data = G.nodes.get(neighbor_id, {})
            neighbor_name = neighbor_data.get("name", neighbor_id)
            neighbor_type = neighbor_data.get("type", "UNKNOWN")

            edge_data = G.edges.get((entity_id, neighbor_id), {})
            edge_type = edge_data.get("edge_type", "RELATED_TO")
            edge_weight = edge_data.get("weight", 0.5)

            # Collect evidence chunks
            evidence_chunks = set()
            kg_node = graph.get_node(entity_id)
            if kg_node:
                evidence_chunks.update(kg_node.source_chunks)
            kg_neighbor = graph.get_node(neighbor_id)
            if kg_neighbor:
                evidence_chunks.update(kg_neighbor.source_chunks)

            nl = f"Entity '{entity_name}' ({entity_type}) --[{edge_type}]--> Entity '{neighbor_name}' ({neighbor_type})"

            path_results.append(
                PathResult(
                    path=[entity_id, neighbor_id],
                    path_names=[entity_name, neighbor_name],
                    path_types=[entity_type, neighbor_type],
                    edges=[dict(edge_data)],
                    weight=edge_weight,
                    hop_count=1,
                    natural_language=nl,
                    evidence_chunks=list(evidence_chunks)[:max_paths],
                )
            )

        logger.info(f"Single entity '{entity_name}': {len(path_results)} 1-hop neighbors")
        return path_results

    def find_entity_ids_by_name(self, names: list[str]) -> list[str]:
        """
        Find entity node IDs by name.

        Only matches entity nodes (not CHUNK/THREAD nodes) and builds
        the entity-only graph so we can prioritize high-degree nodes.
        """
        graph = self._load_graph()
        G = self._build_nx_graph()

        from tacitgraph.entity_registry import expand_entity_aliases, get_internal_node_types

        SKIP_TYPES = get_internal_node_types()
        entity_ids = []
        seen = set()

        # Expand each name to include all aliases
        expanded_names = []
        for name in names:
            expanded_names.extend(expand_entity_aliases(name))

        for name in expanded_names:
            name_lower = name.lower().strip()
            if not name_lower or name_lower in seen:
                continue
            seen.add(name_lower)

            best_id = None
            best_score = -1

            for node_id, node in graph.nodes.items():
                if node.node_type in SKIP_TYPES:
                    continue
                if node_id not in G:
                    continue

                node_name = node.name.replace("\n", " ").strip().lower()

                # Exact match (best)
                if node_name == name_lower:
                    score = 1000 + G.degree(node_id)
                # Query is substring of node name
                elif name_lower in node_name:
                    score = 100 + G.degree(node_id)
                # Node name is substring of query
                elif node_name in name_lower:
                    score = 50 + G.degree(node_id)
                else:
                    continue

                if score > best_score:
                    best_score = score
                    best_id = node_id

            if best_id and best_id not in entity_ids:
                entity_ids.append(best_id)

        return entity_ids


# Async wrapper for synchronous use
def retrieve_paths_sync(
    gold_path: str,
    entity_names: list[str],
    max_paths: int = 15,
    config: PathRAGConfig | None = None,
) -> list[PathResult]:
    """
    Synchronous wrapper for path retrieval.

    Args:
        gold_path: Path to Gold layer
        entity_names: List of entity names to find paths between
        max_paths: Maximum paths to return
        config: PathRAG configuration

    Returns:
        List of PathResult objects
    """
    retriever = PathRAGRetriever(gold_path, config)
    entity_ids = retriever.find_entity_ids_by_name(entity_names)

    if len(entity_ids) < 2:
        return []

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(retriever.retrieve(entity_ids, max_paths))
    finally:
        loop.close()
