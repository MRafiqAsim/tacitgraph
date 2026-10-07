"""
Path Indexer Module (PathRAG)

Pre-computes and indexes reasoning paths through the knowledge graph
for efficient PathRAG retrieval.
"""

import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
import heapq

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .graph_builder import KnowledgeGraph, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


@dataclass
class PathStep:
    """A single step in a reasoning path."""
    node_id: str
    node_name: str
    node_type: str
    edge_type: Optional[str] = None  # Edge leading to this node

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "node_type": self.node_type,
            "edge_type": self.edge_type
        }


@dataclass
class ReasoningPath:
    """A complete reasoning path between entities."""
    path_id: str
    source_entity: Dict[str, str]
    target_entity: Dict[str, str]
    steps: List[PathStep]
    path_length: int
    path_weight: float
    evidence_chunks: List[str] = field(default_factory=list)
    path_type: str = ""  # e.g., "PERSON→PROJECT", "EMAIL→DOCUMENT"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "steps": [s.to_dict() for s in self.steps],
            "path_length": self.path_length,
            "path_weight": self.path_weight,
            "evidence_chunks": self.evidence_chunks,
            "path_type": self.path_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningPath":
        steps = [PathStep(**s) for s in data.pop("steps", [])]
        return cls(**data, steps=steps)


@dataclass
class PathIndexConfig:
    """Configuration for path indexing."""
    max_path_length: int = 5  # Maximum hops
    min_path_weight: float = 0.3  # Minimum path weight to include
    max_paths_per_pair: int = 3  # Max paths between entity pair
    index_entity_types: Set[str] = field(default_factory=lambda: _path_index_types())
    exclude_edge_types: Set[str] = field(default_factory=lambda: _structural_edge_types())
    prioritize_edge_types: Set[str] = field(default_factory=lambda: _prioritized_edge_types())


def _path_index_types() -> Set[str]:
    from entity_registry import get_path_index_types
    return get_path_index_types()


def _structural_edge_types() -> Set[str]:
    from entity_registry import get_structural_edge_types
    return get_structural_edge_types()


def _prioritized_edge_types() -> Set[str]:
    from entity_registry import get_prioritized_edge_types
    return get_prioritized_edge_types()


# ======================================================================
# Module-level worker for multiprocessing (must be serializable)
# ======================================================================

def _worker_find_paths(args):
    """
    Worker function for parallel path finding.

    Args:
        args: tuple of (pairs_batch, G_undirected_data, G_directed_data,
              graph_nodes_data, config_dict)

    Returns:
        List of ReasoningPath dicts for this batch
    """
    (pairs_batch, G_undi_pickle, G_dir_pickle,
     nodes_data, max_path_length, min_path_weight, max_paths_per_pair) = args

    import pickle
    G_undirected = pickle.loads(G_undi_pickle)
    G_directed = pickle.loads(G_dir_pickle)

    results = []

    for source_id, target_id in pairs_batch:
        if source_id not in G_undirected or target_id not in G_undirected:
            continue

        try:
            paths_found = 0
            for path_nodes in nx.all_shortest_paths(
                G_undirected, source_id, target_id, weight="weight"
            ):
                if len(path_nodes) > max_path_length + 1:
                    continue
                if paths_found >= max_paths_per_pair:
                    break

                # Build reasoning path inline (avoid serializing self)
                rp = _worker_build_path(
                    G_directed, nodes_data, path_nodes, min_path_weight
                )
                if rp:
                    results.append(rp)
                    paths_found += 1

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return results


def _worker_build_path(G_directed, nodes_data, path_nodes, min_path_weight):
    """Build a ReasoningPath dict from a node sequence (worker-safe)."""
    if len(path_nodes) < 2:
        return None

    steps = []
    evidence_chunks = set()
    total_weight = 0.0

    for i, node_id in enumerate(path_nodes):
        node_info = nodes_data.get(node_id)
        if not node_info:
            continue

        edge_type = None
        if i > 0:
            prev_id = path_nodes[i - 1]
            if G_directed.has_edge(prev_id, node_id):
                edge_data = G_directed.edges[prev_id, node_id]
                edge_type = edge_data.get("edge_type")
                total_weight += edge_data.get("original_weight", 0.5)
            elif G_directed.has_edge(node_id, prev_id):
                edge_data = G_directed.edges[node_id, prev_id]
                edge_type = edge_data.get("edge_type")
                total_weight += edge_data.get("original_weight", 0.5)

        steps.append({
            "node_id": node_id,
            "node_name": node_info["name"],
            "node_type": node_info["type"],
            "edge_type": edge_type,
        })
        evidence_chunks.update(node_info.get("source_chunks", []))

    if len(steps) < 2:
        return None

    path_weight = total_weight / (len(steps) - 1) if len(steps) > 1 else 0
    if path_weight < min_path_weight:
        return None

    source_type = steps[0]["node_type"]
    target_type = steps[-1]["node_type"]
    path_id = f"path_{hash(tuple(s['node_id'] for s in steps)) & 0xFFFFFFFF:08x}"

    source_info = nodes_data.get(path_nodes[0], {})
    target_info = nodes_data.get(path_nodes[-1], {})

    return {
        "path_id": path_id,
        "source_entity": {"id": path_nodes[0], "name": source_info.get("name", ""), "type": source_type},
        "target_entity": {"id": path_nodes[-1], "name": target_info.get("name", ""), "type": target_type},
        "steps": steps,
        "path_length": len(steps) - 1,
        "path_weight": path_weight,
        "evidence_chunks": list(evidence_chunks)[:20],
        "path_type": f"{source_type}→{target_type}",
    }


class PathIndexer:
    """
    Indexes reasoning paths through the knowledge graph for PathRAG.

    Pre-computes paths between important entity pairs to enable
    efficient path-based retrieval at query time.
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        gold_path: str,
        config: Optional[PathIndexConfig] = None
    ):
        """
        Initialize the path indexer.

        Args:
            knowledge_graph: The knowledge graph to index
            gold_path: Path to Gold layer for output
            config: Indexing configuration
        """
        self.graph = knowledge_graph
        self.gold_path = Path(gold_path)
        self.config = config or PathIndexConfig()

        # Output directory
        self.paths_path = self.gold_path / "path_index"
        self.paths_path.mkdir(parents=True, exist_ok=True)

        # Path storage
        self.paths: List[ReasoningPath] = []
        self.paths_by_source: Dict[str, List[str]] = defaultdict(list)
        self.paths_by_target: Dict[str, List[str]] = defaultdict(list)
        self.paths_by_type: Dict[str, List[str]] = defaultdict(list)

        logger.info("PathIndexer initialized")

    def build_index(self, progress_callback=None, num_workers: int = 0) -> List[ReasoningPath]:
        """
        Build the path index by finding paths between entity pairs.

        Uses multiprocessing to distribute work across all CPU cores.

        Args:
            progress_callback: Optional progress callback
            num_workers: Number of parallel workers (0 = auto-detect CPU count)

        Returns:
            List of indexed ReasoningPaths
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for path indexing")

        if num_workers <= 0:
            num_workers = os.cpu_count() or 4

        logger.info("Building path index...")

        # Get indexable entities (filter by type)
        entities = [
            node for node in self.graph.nodes.values()
            if node.node_type in self.config.index_entity_types
        ]

        type_counts = Counter(n.node_type for n in entities)
        logger.info(f"Indexable entities: {len(entities)} — {dict(type_counts)}")

        # Build NetworkX graphs ONCE
        logger.info("Building weighted NetworkX graph...")
        G_directed = self._build_weighted_graph()
        logger.info(f"NetworkX graph: {G_directed.number_of_nodes()} nodes, {G_directed.number_of_edges()} edges")

        logger.info("Converting to undirected graph (once)...")
        G_undirected = G_directed.to_undirected()

        # Serialize graphs for workers
        import pickle
        G_undi_pickle = pickle.dumps(G_undirected)
        G_dir_pickle = pickle.dumps(G_directed)

        # Build lightweight node lookup for workers
        nodes_data = {}
        for node_id, node in self.graph.nodes.items():
            nodes_data[node_id] = {
                "name": node.name,
                "type": node.node_type,
                "source_chunks": list(node.source_chunks)[:20],
            }

        # Generate all entity pairs
        all_pairs = []
        for i, source_entity in enumerate(entities):
            for target_entity in entities[i + 1:]:
                all_pairs.append((source_entity.node_id, target_entity.node_id))

        total_pairs = len(all_pairs)
        logger.info(f"Total entity pairs: {total_pairs:,}")
        logger.info(f"Using {num_workers} worker processes")

        # Split into chunks for workers
        chunk_size = max(500, total_pairs // (num_workers * 10))
        batches = []
        for start in range(0, total_pairs, chunk_size):
            batch = all_pairs[start:start + chunk_size]
            batches.append((
                batch, G_undi_pickle, G_dir_pickle, nodes_data,
                self.config.max_path_length,
                self.config.min_path_weight,
                self.config.max_paths_per_pair,
            ))

        logger.info(f"Split into {len(batches)} batches of ~{chunk_size} pairs each")

        # Process in parallel
        start_time = time.time()
        completed_batches = 0
        completed_pairs = 0
        path_count = 0

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=num_workers) as pool:
            for batch_results in pool.imap_unordered(_worker_find_paths, batches):
                completed_batches += 1
                batch_pair_count = len(batches[0][0]) if batches else chunk_size
                completed_pairs += batch_pair_count

                # Convert result dicts to ReasoningPath objects
                for rp_dict in batch_results:
                    steps = [PathStep(**s) for s in rp_dict.pop("steps")]
                    rp = ReasoningPath(**rp_dict, steps=steps)
                    self.paths.append(rp)
                    self._index_path(rp)
                    path_count += 1

                # Log progress
                elapsed = time.time() - start_time
                rate = completed_pairs / elapsed if elapsed > 0 else 0
                remaining = total_pairs - min(completed_pairs, total_pairs)
                eta = remaining / rate if rate > 0 else 0
                pct = min(100, 100 * completed_pairs / total_pairs)
                logger.info(
                    f"Path indexing: batch {completed_batches}/{len(batches)} — "
                    f"~{pct:.1f}% — "
                    f"{path_count} paths found — "
                    f"{rate:.0f} pairs/sec — ETA: {eta/60:.1f}min"
                )

                if progress_callback:
                    progress_callback(min(completed_pairs, total_pairs), total_pairs)

        elapsed = time.time() - start_time
        logger.info(
            f"Indexed {len(self.paths)} paths between entities "
            f"in {elapsed/60:.1f} minutes ({num_workers} workers)"
        )
        return self.paths

    def _build_weighted_graph(self) -> "nx.DiGraph":
        """Build NetworkX graph with edge weights for path finding."""
        G = nx.DiGraph()

        # Add nodes
        for node_id, node in self.graph.nodes.items():
            G.add_node(node_id, **{
                "name": node.name,
                "type": node.node_type
            })

        # Add edges with weights
        for edge_id, edge in self.graph.edges.items():
            if edge.edge_type in self.config.exclude_edge_types:
                continue

            # Calculate edge weight (inverse for shortest path)
            weight = 1.0 - edge.weight + 0.1  # Ensure positive

            # Boost priority edge types
            if edge.edge_type in self.config.prioritize_edge_types:
                weight *= 0.5

            G.add_edge(
                edge.source_id,
                edge.target_id,
                edge_id=edge_id,
                edge_type=edge.edge_type,
                weight=weight,
                original_weight=edge.weight
            )

        return G

    def _find_paths(
        self,
        G: "nx.DiGraph",
        source_id: str,
        target_id: str,
        G_undirected: Optional["nx.Graph"] = None
    ) -> List[ReasoningPath]:
        """Find paths between two entities."""
        paths = []

        if source_id not in G or target_id not in G:
            return paths

        # Use pre-built undirected graph if provided, else convert once
        if G_undirected is None:
            G_undirected = G.to_undirected()

        try:
            # Find shortest paths
            for path_nodes in nx.all_shortest_paths(
                G_undirected,
                source_id,
                target_id,
                weight="weight"
            ):
                if len(path_nodes) > self.config.max_path_length + 1:
                    continue

                if len(paths) >= self.config.max_paths_per_pair:
                    break

                reasoning_path = self._build_reasoning_path(G, path_nodes)
                if reasoning_path and reasoning_path.path_weight >= self.config.min_path_weight:
                    paths.append(reasoning_path)

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        return paths

    def _build_reasoning_path(
        self,
        G: "nx.DiGraph",
        path_nodes: List[str]
    ) -> Optional[ReasoningPath]:
        """Build a ReasoningPath from a node sequence."""
        if len(path_nodes) < 2:
            return None

        steps = []
        evidence_chunks = set()
        total_weight = 0.0

        for i, node_id in enumerate(path_nodes):
            node = self.graph.get_node(node_id)
            if not node:
                continue

            # Get edge type from previous node
            edge_type = None
            if i > 0:
                prev_id = path_nodes[i - 1]
                # Check both directions
                if G.has_edge(prev_id, node_id):
                    edge_data = G.edges[prev_id, node_id]
                    edge_type = edge_data.get("edge_type")
                    total_weight += edge_data.get("original_weight", 0.5)
                elif G.has_edge(node_id, prev_id):
                    edge_data = G.edges[node_id, prev_id]
                    edge_type = edge_data.get("edge_type")
                    total_weight += edge_data.get("original_weight", 0.5)

            steps.append(PathStep(
                node_id=node_id,
                node_name=node.name,
                node_type=node.node_type,
                edge_type=edge_type
            ))

            # Collect evidence chunks
            evidence_chunks.update(node.source_chunks)

        if len(steps) < 2:
            return None

        # Calculate path weight
        path_weight = total_weight / (len(steps) - 1) if len(steps) > 1 else 0

        # Build path type
        source_type = steps[0].node_type
        target_type = steps[-1].node_type
        path_type = f"{source_type}→{target_type}"

        # Generate path ID
        path_id = f"path_{hash(tuple(s.node_id for s in steps)) & 0xFFFFFFFF:08x}"

        source_node = self.graph.get_node(path_nodes[0])
        target_node = self.graph.get_node(path_nodes[-1])

        return ReasoningPath(
            path_id=path_id,
            source_entity={
                "id": path_nodes[0],
                "name": source_node.name if source_node else "",
                "type": source_type
            },
            target_entity={
                "id": path_nodes[-1],
                "name": target_node.name if target_node else "",
                "type": target_type
            },
            steps=steps,
            path_length=len(steps) - 1,
            path_weight=path_weight,
            evidence_chunks=list(evidence_chunks)[:20],  # Limit chunks
            path_type=path_type
        )

    def _index_path(self, path: ReasoningPath):
        """Add path to indexes."""
        self.paths_by_source[path.source_entity["id"]].append(path.path_id)
        self.paths_by_target[path.target_entity["id"]].append(path.path_id)
        self.paths_by_type[path.path_type].append(path.path_id)

    def find_paths_from_entity(self, entity_id: str) -> List[ReasoningPath]:
        """Find all paths originating from an entity."""
        path_ids = self.paths_by_source.get(entity_id, [])
        return [p for p in self.paths if p.path_id in path_ids]

    def find_paths_to_entity(self, entity_id: str) -> List[ReasoningPath]:
        """Find all paths ending at an entity."""
        path_ids = self.paths_by_target.get(entity_id, [])
        return [p for p in self.paths if p.path_id in path_ids]

    def find_paths_between(
        self,
        source_id: str,
        target_id: str
    ) -> List[ReasoningPath]:
        """Find all indexed paths between two entities."""
        source_paths = set(self.paths_by_source.get(source_id, []))
        target_paths = set(self.paths_by_target.get(target_id, []))

        # Paths where source is source and target is target
        direct = source_paths & target_paths

        # Also check reverse
        source_as_target = set(self.paths_by_target.get(source_id, []))
        target_as_source = set(self.paths_by_source.get(target_id, []))
        reverse = source_as_target & target_as_source

        path_ids = direct | reverse
        return [p for p in self.paths if p.path_id in path_ids]

    def find_paths_on_demand(
        self,
        source_id: str,
        target_id: str,
        max_paths: int = 3
    ) -> List[ReasoningPath]:
        """
        Find paths between two entities ON-DEMAND (no pre-computation).

        This is the key PathRAG optimization - only compute paths for
        the specific entities mentioned in the query, not all pairs.
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for path finding")

        # Check if we have pre-computed paths first
        cached_paths = self.find_paths_between(source_id, target_id)
        if cached_paths:
            return cached_paths[:max_paths]

        # Build graph on-demand (cached after first call)
        if not hasattr(self, '_nx_graph') or self._nx_graph is None:
            self._nx_graph = self._build_weighted_graph()

        # Find paths just for these two entities
        return self._find_paths(self._nx_graph, source_id, target_id)[:max_paths]

    def find_paths_for_entities(
        self,
        entity_ids: List[str],
        max_paths_per_pair: int = 3
    ) -> List[ReasoningPath]:
        """
        Find paths between a list of entities ON-DEMAND.

        This is efficient because it only computes paths between
        the k entities from the query (usually 2-5), not all n² pairs.

        Complexity: O(k²) where k = len(entity_ids), typically k < 5
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for path finding")

        if len(entity_ids) < 2:
            return []

        # Build graph on-demand (cached after first call)
        if not hasattr(self, '_nx_graph') or self._nx_graph is None:
            self._nx_graph = self._build_weighted_graph()

        all_paths = []

        # Only compute paths between query entities (fast!)
        for i, source_id in enumerate(entity_ids):
            for target_id in entity_ids[i + 1:]:
                # Check cache first
                cached = self.find_paths_between(source_id, target_id)
                if cached:
                    all_paths.extend(cached[:max_paths_per_pair])
                else:
                    # Compute on-demand
                    paths = self._find_paths(self._nx_graph, source_id, target_id)
                    all_paths.extend(paths[:max_paths_per_pair])

        return all_paths

    def find_paths_by_type(self, path_type: str) -> List[ReasoningPath]:
        """Find all paths of a specific type (e.g., 'PERSON→PROJECT')."""
        path_ids = self.paths_by_type.get(path_type, [])
        return [p for p in self.paths if p.path_id in path_ids]

    def get_path_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed paths."""
        path_lengths = [p.path_length for p in self.paths]
        path_weights = [p.path_weight for p in self.paths]

        return {
            "total_paths": len(self.paths),
            "unique_sources": len(self.paths_by_source),
            "unique_targets": len(self.paths_by_target),
            "path_types": {t: len(ids) for t, ids in self.paths_by_type.items()},
            "avg_path_length": sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            "max_path_length": max(path_lengths) if path_lengths else 0,
            "avg_path_weight": sum(path_weights) / len(path_weights) if path_weights else 0
        }

    def save(self):
        """Save path index to disk."""
        # Save paths
        paths_data = [p.to_dict() for p in self.paths]
        paths_file = self.paths_path / "paths.json"
        with open(paths_file, 'w', encoding='utf-8') as f:
            json.dump(paths_data, f)

        # Save indexes
        indexes = {
            "by_source": dict(self.paths_by_source),
            "by_target": dict(self.paths_by_target),
            "by_type": dict(self.paths_by_type)
        }
        indexes_file = self.paths_path / "path_indexes.json"
        with open(indexes_file, 'w', encoding='utf-8') as f:
            json.dump(indexes, f, indent=2)

        # Save statistics
        stats = self.get_path_statistics()
        stats["indexed_at"] = datetime.now().isoformat()
        stats_file = self.paths_path / "path_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved {len(self.paths)} paths to {self.paths_path}")

    def load(self) -> List[ReasoningPath]:
        """Load path index from disk."""
        paths_file = self.paths_path / "paths.json"
        indexes_file = self.paths_path / "path_indexes.json"

        if not paths_file.exists():
            raise FileNotFoundError(f"Path index not found at {paths_file}")

        # Load paths
        with open(paths_file, 'r', encoding='utf-8') as f:
            paths_data = json.load(f)

        self.paths = [ReasoningPath.from_dict(p) for p in paths_data]

        # Load indexes
        if indexes_file.exists():
            with open(indexes_file, 'r', encoding='utf-8') as f:
                indexes = json.load(f)

            self.paths_by_source = defaultdict(list, indexes.get("by_source", {}))
            self.paths_by_target = defaultdict(list, indexes.get("by_target", {}))
            self.paths_by_type = defaultdict(list, indexes.get("by_type", {}))
        else:
            # Rebuild indexes
            for path in self.paths:
                self._index_path(path)

        logger.info(f"Loaded {len(self.paths)} paths")
        return self.paths
