"""
Community Detector Module (GraphRAG)

Implements hierarchical community detection using Leiden algorithm
and generates community summaries for GraphRAG retrieval.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import igraph as ig
    import leidenalg

    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False

from .graph_builder import GraphNode, KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """Represents a community in the graph."""

    community_id: str
    level: int
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)
    entity_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    summary: str = ""
    key_topics: list[str] = field(default_factory=list)
    key_entities: list[dict[str, str]] = field(default_factory=list)
    summary_embedding: list[float] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_id": self.community_id,
            "level": self.level,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "node_ids": self.node_ids,
            "source_chunk_ids": self.source_chunk_ids,
            "entity_count": self.entity_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "summary": self.summary,
            "key_topics": self.key_topics,
            "key_entities": self.key_entities,
            "summary_embedding": self.summary_embedding,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Community":
        return cls(**data)


@dataclass
class CommunityConfig:
    """Configuration for community detection."""

    resolution: float = 1.0  # Leiden resolution parameter
    num_levels: int = 3  # Number of hierarchy levels
    min_community_size: int = 3  # Minimum nodes per community
    max_summary_tokens: int = 500  # Max tokens for summary
    use_llm_summarization: bool = True  # Use LLM for summaries


class CommunityDetector:
    """
    Detects hierarchical communities in the knowledge graph using Leiden algorithm.
    Generates summaries for each community for GraphRAG retrieval.
    """

    def __init__(
        self, knowledge_graph: KnowledgeGraph, gold_path: str, config: CommunityConfig | None = None
    ):
        """
        Initialize the community detector.

        Args:
            knowledge_graph: The knowledge graph to process
            gold_path: Path to Gold layer for output
            config: Detection configuration
        """
        self.graph = knowledge_graph
        self.gold_path = Path(gold_path)
        self.config = config or CommunityConfig()

        # Output directory
        self.communities_path = self.gold_path / "communities"
        self.communities_path.mkdir(parents=True, exist_ok=True)

        # Store communities by level
        self.communities: dict[int, dict[str, Community]] = defaultdict(dict)

        # LLM client for summarization
        self.llm_client: Any = None  # OpenAI or AzureOpenAI client
        if self.config.use_llm_summarization:
            self._initialize_llm()

        logger.info("CommunityDetector initialized")

    def _initialize_llm(self):
        """Initialize LLM client for summarization."""
        import os

        try:
            # Try Azure OpenAI
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_key = os.getenv("AZURE_OPENAI_API_KEY")

            if azure_endpoint and azure_key:
                from openai import AzureOpenAI

                self.llm_client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                )
                self.llm_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
                logger.info("Using Azure OpenAI for community summarization")
                return

            # Try OpenAI direct
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI

                self.llm_client = OpenAI(api_key=openai_key)
                self.llm_model = "gpt-4o"
                logger.info("Using OpenAI for community summarization")
                return

            logger.warning("No LLM credentials found, using extractive summarization")

        except ImportError:
            logger.warning("openai package not installed, using extractive summarization")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")

    def detect_communities(self, progress_callback=None) -> dict[int, dict[str, Community]]:
        """
        Detect hierarchical communities in the graph.

        Uses Leiden algorithm at multiple resolutions for hierarchy.

        Args:
            progress_callback: Optional progress callback

        Returns:
            Dictionary mapping level -> {community_id -> Community}
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for community detection")

        logger.info("Starting community detection...")

        # Convert to NetworkX
        nx_graph = self.graph.to_networkx()

        # Filter to entity nodes only (exclude CHUNK nodes for community detection)
        entity_nodes = [
            n for n in nx_graph.nodes() if self.graph.nodes[n].node_type not in ["CHUNK", "THREAD"]
        ]
        entity_subgraph = nx_graph.subgraph(entity_nodes).copy()

        logger.info(f"Detecting communities on {len(entity_subgraph.nodes())} entity nodes")

        if HAS_LEIDEN:
            # Use Leiden algorithm
            self._detect_with_leiden(entity_subgraph, progress_callback)
        else:
            # Fallback to Louvain (built into networkx)
            self._detect_with_louvain(entity_subgraph, progress_callback)

        # Build hierarchy
        self._build_hierarchy()

        # Generate summaries
        self._generate_summaries(progress_callback)

        logger.info(
            f"Community detection complete: {sum(len(c) for c in self.communities.values())} communities"
        )
        return self.communities

    def _detect_with_leiden(self, nx_graph: "nx.Graph", progress_callback=None):
        """Detect communities using Leiden algorithm."""
        # Convert NetworkX to igraph
        ig_graph = ig.Graph.from_networkx(nx_graph)

        # Level 0 = finest (most communities), higher levels = coarser (fewer, larger communities)
        # Dividing by 4^i gives proper hierarchy: 1.0 → 0.25 → 0.0625
        resolutions = [self.config.resolution / (4**i) for i in range(self.config.num_levels)]

        for level, resolution in enumerate(resolutions):
            logger.info(f"Detecting level {level} communities (resolution={resolution:.2f})")

            # Run Leiden
            partition = leidenalg.find_partition(
                ig_graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution
            )

            # Create community objects
            for comm_idx, members in enumerate(partition):
                if len(members) < self.config.min_community_size:
                    continue

                comm_id = f"comm_l{level}_{comm_idx:04d}"
                node_ids = [ig_graph.vs[m]["_nx_name"] for m in members]

                community = Community(
                    community_id=comm_id,
                    level=level,
                    node_ids=node_ids,
                    entity_count=len(node_ids),
                    generated_at=datetime.now().isoformat(),
                )

                # Calculate density
                subgraph = nx_graph.subgraph(node_ids)
                community.edge_count = subgraph.number_of_edges()
                max_edges = len(node_ids) * (len(node_ids) - 1) / 2
                community.density = community.edge_count / max_edges if max_edges > 0 else 0

                # Extract key entities and source chunks
                community.key_entities = self._extract_key_entities(node_ids)
                community.source_chunk_ids = self._collect_source_chunks(node_ids)

                self.communities[level][comm_id] = community

            if progress_callback:
                progress_callback(level + 1, self.config.num_levels)

    def _detect_with_louvain(self, nx_graph: "nx.Graph", progress_callback=None):
        """Detect communities using Louvain algorithm (fallback)."""
        from networkx.algorithms.community import louvain_communities

        for level in range(self.config.num_levels):
            resolution = self.config.resolution / (4**level)
            logger.info(
                f"Detecting level {level} communities with Louvain (resolution={resolution:.2f})"
            )

            # Run Louvain
            communities_list = louvain_communities(nx_graph, resolution=resolution, seed=42)

            # Create community objects
            for comm_idx, members in enumerate(communities_list):
                if len(members) < self.config.min_community_size:
                    continue

                comm_id = f"comm_l{level}_{comm_idx:04d}"
                node_ids = list(members)

                community = Community(
                    community_id=comm_id,
                    level=level,
                    node_ids=node_ids,
                    entity_count=len(node_ids),
                    generated_at=datetime.now().isoformat(),
                )

                # Calculate density
                subgraph = nx_graph.subgraph(node_ids)
                community.edge_count = subgraph.number_of_edges()
                max_edges = len(node_ids) * (len(node_ids) - 1) / 2
                community.density = community.edge_count / max_edges if max_edges > 0 else 0

                # Extract key entities and source chunks
                community.key_entities = self._extract_key_entities(node_ids)
                community.source_chunk_ids = self._collect_source_chunks(node_ids)

                self.communities[level][comm_id] = community

            if progress_callback:
                progress_callback(level + 1, self.config.num_levels)

    def _collect_source_chunks(self, node_ids: list[str]) -> list[str]:
        """Aggregate source chunk IDs from all member nodes."""
        chunk_ids = set()
        for node_id in node_ids:
            node = self.graph.get_node(node_id)
            if node:
                chunk_ids.update(node.source_chunks)
        return sorted(chunk_ids)

    def _extract_key_entities(
        self, node_ids: list[str], max_entities: int = 10
    ) -> list[dict[str, str]]:
        """Extract key entities from a community."""
        entities = []

        for node_id in node_ids[:max_entities]:
            node = self.graph.get_node(node_id)
            if node:
                entities.append({"id": node_id, "name": node.name, "type": node.node_type})

        # Sort by mention count
        entities.sort(
            key=lambda x: self.graph.nodes.get(x["id"], GraphNode("", "", "")).mention_count,
            reverse=True,
        )

        return entities[:max_entities]

    def _build_hierarchy(self):
        """Build parent-child relationships between community levels."""
        levels = sorted(self.communities.keys())

        for i in range(len(levels) - 1):
            child_level = levels[i]
            parent_level = levels[i + 1]

            for child_comm in self.communities[child_level].values():
                child_nodes = set(child_comm.node_ids)

                # Find parent with most overlap
                best_parent = None
                best_overlap = 0

                for parent_comm in self.communities[parent_level].values():
                    parent_nodes = set(parent_comm.node_ids)
                    overlap = len(child_nodes & parent_nodes)

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = parent_comm.community_id

                if best_parent:
                    child_comm.parent_id = best_parent
                    self.communities[parent_level][best_parent].child_ids.append(
                        child_comm.community_id
                    )

    def _generate_summaries(self, progress_callback=None):
        """Generate summaries for all communities."""
        total_communities = sum(len(c) for c in self.communities.values())
        processed = 0

        for level in sorted(self.communities.keys()):
            for comm_id, community in self.communities[level].items():
                try:
                    if self.llm_client:
                        summary = self._generate_llm_summary(community)
                    else:
                        summary = self._generate_extractive_summary(community)

                    community.summary = summary
                    community.key_topics = self._extract_topics(community)

                except Exception as e:
                    logger.warning(f"Failed to generate summary for {comm_id}: {e}")
                    community.summary = self._generate_extractive_summary(community)

                processed += 1
                if progress_callback and processed % 10 == 0:
                    progress_callback(processed, total_communities)

    def _generate_llm_summary(self, community: Community) -> str:
        """Generate summary using LLM."""
        # Build context from entities
        entities_text = "\n".join(
            [f"- {e['name']} ({e['type']})" for e in community.key_entities[:20]]
        )

        # Get sample relationships
        relationships = []
        for node_id in community.node_ids[:10]:
            for edge in self.graph.get_edges_from(node_id):
                if edge.target_id in community.node_ids:
                    source = self.graph.get_node(edge.source_id)
                    target = self.graph.get_node(edge.target_id)
                    if source and target:
                        relationships.append(f"{source.name} --[{edge.edge_type}]--> {target.name}")

        relationships_text = "\n".join(relationships[:15])

        from tacitgraph.prompt_loader import format_prompt, get_prompt

        prompt = format_prompt(
            get_prompt("gold", "community_summary", "user_prompt"),
            entity_count=community.entity_count,
            entities_text=entities_text,
            relationships_text=relationships_text,
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt("gold", "community_summary", "system_prompt"),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=get_prompt("gold", "community_summary", "max_tokens", 200),
                temperature=get_prompt("gold", "community_summary", "temperature", 0.3),
            )
            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            return extract_llm_content(response, context="community summary")

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, network timeouts, etc. Falls back to extractive summary.
            logger.warning(f"LLM community summarization failed: {e}")
            return self._generate_extractive_summary(community)

    def _generate_extractive_summary(self, community: Community) -> str:
        """Generate simple extractive summary without LLM."""
        # Group entities by type
        by_type: dict[str, list[str]] = defaultdict(list)
        for entity in community.key_entities:
            by_type[entity["type"]].append(entity["name"])

        parts = []
        parts.append(f"Community with {community.entity_count} entities")

        if "PERSON" in by_type:
            parts.append(f"People: {', '.join(by_type['PERSON'][:5])}")

        if "ORG" in by_type:
            parts.append(f"Organizations: {', '.join(by_type['ORG'][:3])}")

        if "PROJECT" in by_type or "CONCEPT" in by_type:
            topics = by_type.get("PROJECT", []) + by_type.get("CONCEPT", [])
            parts.append(f"Topics: {', '.join(topics[:5])}")

        return ". ".join(parts) + "."

    def _extract_topics(self, community: Community) -> list[str]:
        """Extract key topics from community."""
        topics = []

        for entity in community.key_entities:
            if entity["type"] in ["PROJECT", "CONCEPT", "PROCESS"]:
                topics.append(entity["name"])

        return topics[:10]

    def save(self):
        """Save communities to disk."""
        for level, communities in self.communities.items():
            level_path = self.communities_path / f"level_{level}"
            level_path.mkdir(exist_ok=True)

            for comm_id, community in communities.items():
                comm_path = level_path / f"{comm_id}.json"
                with open(comm_path, "w", encoding="utf-8") as f:
                    json.dump(community.to_dict(), f, indent=2)

        # Save index
        index = {
            "levels": list(self.communities.keys()),
            "communities_per_level": {
                level: list(comms.keys()) for level, comms in self.communities.items()
            },
            "total_communities": sum(len(c) for c in self.communities.values()),
            "generated_at": datetime.now().isoformat(),
        }

        index_path = self.communities_path / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        logger.info(f"Saved {index['total_communities']} communities to {self.communities_path}")

    def load(self) -> dict[int, dict[str, Community]]:
        """Load communities from disk."""
        index_path = self.communities_path / "index.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Community index not found at {index_path}")

        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        self.communities = defaultdict(dict)

        for level in index["levels"]:
            level_path = self.communities_path / f"level_{level}"

            for comm_id in index["communities_per_level"][str(level)]:
                comm_path = level_path / f"{comm_id}.json"

                if comm_path.exists():
                    with open(comm_path, encoding="utf-8") as f:
                        data = json.load(f)
                    self.communities[level][comm_id] = Community.from_dict(data)

        logger.info(f"Loaded {sum(len(c) for c in self.communities.values())} communities")
        return self.communities

    def get_community(self, community_id: str) -> Community | None:
        """Get a specific community by ID."""
        for level_communities in self.communities.values():
            if community_id in level_communities:
                return level_communities[community_id]
        return None

    def get_communities_at_level(self, level: int) -> list[Community]:
        """Get all communities at a specific level."""
        return list(self.communities.get(level, {}).values())

    def find_community_for_entity(self, entity_id: str, level: int = 0) -> Community | None:
        """Find the community containing an entity at a given level."""
        for community in self.communities.get(level, {}).values():
            if entity_id in community.node_ids:
                return community
        return None
