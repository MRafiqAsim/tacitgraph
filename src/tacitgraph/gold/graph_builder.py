"""
Knowledge Graph Builder Module

Builds a unified knowledge graph from processed email chunks,
aggregating entities and relationships for PathRAG/GraphRAG retrieval.
"""

import hashlib
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

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""

    node_id: str
    name: str
    node_type: str  # PERSON, ORG, PROJECT, CONCEPT, EMAIL, CHUNK, DOCUMENT, THREAD
    properties: dict[str, Any] = field(default_factory=dict)
    mention_count: int = 1
    source_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type,
            "properties": self.properties,
            "mention_count": self.mention_count,
            "source_chunks": self.source_chunks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphNode":
        return cls(**data)


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""

    edge_id: str
    source_id: str
    target_id: str
    edge_type: str  # SENT_BY, WORKS_ON, MENTIONS, CO_OCCURS, HAS_ATTACHMENT, etc.
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    evidence_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "properties": self.properties,
            "evidence_chunks": self.evidence_chunks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphEdge":
        return cls(**data)


@dataclass
class KnowledgeGraph:
    """The complete knowledge graph structure."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Indexes for quick lookup
    nodes_by_type: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    edges_by_type: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    edges_by_source: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    edges_by_target: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        if node.node_id in self.nodes:
            # Merge with existing node
            existing = self.nodes[node.node_id]
            existing.mention_count += node.mention_count
            existing.source_chunks.extend(node.source_chunks)
            existing.properties.update(node.properties)
        else:
            self.nodes[node.node_id] = node
            self.nodes_by_type[node.node_type].add(node.node_id)

    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph."""
        if edge.edge_id in self.edges:
            # Merge with existing edge
            existing = self.edges[edge.edge_id]
            existing.weight += edge.weight
            existing.evidence_chunks.extend(edge.evidence_chunks)
            existing.properties.update(edge.properties)
        else:
            self.edges[edge.edge_id] = edge
            self.edges_by_type[edge.edge_type].add(edge.edge_id)
            self.edges_by_source[edge.source_id].add(edge.edge_id)
            self.edges_by_target[edge.target_id].add(edge.edge_id)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> list[GraphNode]:
        """Get all nodes of a specific type."""
        node_ids = self.nodes_by_type.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids]

    def get_edges_from(self, node_id: str) -> list[GraphEdge]:
        """Get all edges originating from a node."""
        edge_ids = self.edges_by_source.get(node_id, set())
        return [self.edges[eid] for eid in edge_ids]

    def get_edges_to(self, node_id: str) -> list[GraphEdge]:
        """Get all edges pointing to a node."""
        edge_ids = self.edges_by_target.get(node_id, set())
        return [self.edges[eid] for eid in edge_ids]

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get all neighbor node IDs."""
        neighbors = set()
        for edge in self.get_edges_from(node_id):
            neighbors.add(edge.target_id)
        for edge in self.get_edges_to(node_id):
            neighbors.add(edge.source_id)
        return list(neighbors)

    def to_networkx(self) -> "nx.DiGraph":
        """Convert to NetworkX graph."""
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for this operation")

        G = nx.DiGraph()

        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(
                node_id,
                name=node.name,
                node_type=node.node_type,
                mention_count=node.mention_count,
                **node.properties,
            )

        # Add edges
        for edge_id, edge in self.edges.items():
            G.add_edge(
                edge.source_id,
                edge.target_id,
                edge_id=edge_id,
                edge_type=edge.edge_type,
                weight=edge.weight,
                **edge.properties,
            )

        return G

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self.edges.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        """Create from dictionary."""
        kg = cls()
        kg.metadata = data.get("metadata", {})

        for nid, ndata in data.get("nodes", {}).items():
            node = GraphNode.from_dict(ndata)
            kg.add_node(node)

        for eid, edata in data.get("edges", {}).items():
            edge = GraphEdge.from_dict(edata)
            kg.add_edge(edge)

        return kg

    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": {t: len(ids) for t, ids in self.nodes_by_type.items()},
            "edges_by_type": {t: len(ids) for t, ids in self.edges_by_type.items()},
        }


class GraphBuilder:
    """
    Builds a knowledge graph from processed email chunks.

    Aggregates entities and relationships from Silver layer chunks
    and creates a unified graph structure for retrieval.
    """

    def __init__(self, silver_path: str, gold_path: str):
        """
        Initialize the graph builder.

        Args:
            silver_path: Path to Silver layer with processed chunks
            gold_path: Path to Gold layer for output
        """
        self.silver_path = Path(silver_path)
        self.gold_path = Path(gold_path)
        self.graph = KnowledgeGraph()

        # Create output directory
        self.kg_output_path = self.gold_path / "knowledge_graph"
        self.kg_output_path.mkdir(parents=True, exist_ok=True)

        # Track entity name normalization
        self.entity_name_map: dict[str, str] = {}  # normalized -> standard

        # Load entity catalog for alias resolution
        self._catalog_path = str(self.gold_path / "entity_catalog.json")
        try:
            from tacitgraph.entity_registry import load_catalog

            self._catalog = load_catalog(self._catalog_path)
        except Exception:
            self._catalog = {"entities": {}, "metadata": {}}

        # Load identity registry for PERSON entity resolution
        self._identity_name_map: dict[str, str] = {}  # alias → canonical name
        self._load_identity_registry()

        logger.info(f"GraphBuilder initialized: {silver_path} -> {gold_path}")

    def _load_identity_registry(self):
        """Load identity registry and build a name → canonical name map for PERSON dedup.

        When the same alias appears in multiple identities, the identity with
        the highest email_count wins (most likely the correct person).
        """
        for reg_path in [
            self.silver_path.parent / "identity_registry.json",
            self.silver_path / "identity_registry.json",
            Path("data") / "identity_registry.json",
        ]:
            if reg_path.exists():
                try:
                    import json as _json

                    with open(reg_path) as f:
                        reg_data = _json.load(f)
                    identities = (
                        reg_data if isinstance(reg_data, list) else reg_data.get("identities", [])
                    )

                    # Track alias → (canonical, email_count) to resolve conflicts
                    alias_candidates: dict[str, list] = {}

                    for identity in identities:
                        canonical = identity.get("canonical_name", "")
                        email_count = identity.get("email_count", 0)
                        # Strip email from canonical: "Doe Jane (jane.doe@example.com)" → "Doe Jane"
                        if "(" in canonical:
                            canonical = canonical[: canonical.index("(")].strip()
                        if not canonical:
                            continue
                        # If canonical is an email address, try to extract name from aliases
                        if "@" in canonical:
                            name_aliases = [
                                a.strip("'\"").strip()
                                for a in identity.get("aliases", [])
                                if "@" not in a
                                and not a.startswith("=?")
                                and len(a.strip("'\"")) > 2
                            ]
                            if name_aliases:
                                canonical = name_aliases[0]
                            else:
                                continue

                        all_names = [canonical]
                        for alias in identity.get("aliases", []):
                            clean = alias.strip("'\"").strip()
                            if clean.startswith("=?") or "@" in clean or len(clean) < 2:
                                continue
                            all_names.append(clean)

                        for name in all_names:
                            key = name.lower()
                            if key not in alias_candidates:
                                alias_candidates[key] = []
                            alias_candidates[key].append((canonical, email_count))

                    # Pick the canonical with highest email_count for each alias
                    for key, candidates in alias_candidates.items():
                        best = max(candidates, key=lambda x: x[1])
                        self._identity_name_map[key] = best[0]

                    logger.info(
                        f"Identity registry loaded: {len(self._identity_name_map)} name mappings from {reg_path}"
                    )
                    return
                except Exception as e:
                    logger.warning(f"Failed to load identity registry from {reg_path}: {e}")
        logger.info("No identity registry found — PERSON entities will not be deduplicated")

    def _resolve_person_name(self, name: str) -> str:
        """Resolve a person name via identity registry. Skip bare first names."""
        if not self._identity_name_map or not name:
            return name
        if " " not in name and "." not in name:
            return name  # bare first name — ambiguous
        canonical = self._identity_name_map.get(name.lower())
        if canonical and canonical != name:
            return canonical
        return name

    def _generate_node_id(self, name: str, node_type: str) -> str:
        """Generate a unique node ID."""
        normalized = self._normalize_entity_name(name)
        content = f"{node_type}:{normalized}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _generate_edge_id(self, source_id: str, target_id: str, edge_type: str) -> str:
        """Generate a unique edge ID."""
        content = f"{source_id}:{edge_type}:{target_id}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        # Basic normalization
        normalized = name.lower().strip()
        # Remove common prefixes/suffixes
        for prefix in ["the ", "a ", "an "]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        return normalized

    def build_from_chunks(self, progress_callback=None) -> KnowledgeGraph:
        """
        Build knowledge graph from all Silver layer chunks.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            The constructed KnowledgeGraph
        """
        logger.info("Building knowledge graph from Silver chunks...")

        # Collect all chunk files
        chunk_files = []
        for pattern in [
            "not_personal/email_chunks/*.json",
            "not_personal/attachment_chunks/*.json",
            "not_personal/document_chunks/*.json",
        ]:
            chunk_files.extend(self.silver_path.glob(pattern))

        total_chunks = len(chunk_files)
        logger.info(f"Found {total_chunks} chunk files to process")

        # Process each chunk
        for i, chunk_file in enumerate(chunk_files):
            try:
                self._process_chunk_file(chunk_file)
            except Exception as e:
                logger.warning(f"Error processing chunk {chunk_file} (file={chunk_file.name}): {e}")

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, total_chunks)

        # Add thread-level nodes and relationships
        self._build_thread_relationships()

        # Normalize edge weights
        self._normalize_weights()

        # Update metadata
        self.graph.metadata = {
            "built_at": datetime.now().isoformat(),
            "source_path": str(self.silver_path),
            "total_chunks_processed": total_chunks,
            **self.graph.stats(),
        }

        logger.info(f"Knowledge graph built: {self.graph.stats()}")
        return self.graph

    def _process_chunk_file(self, chunk_file: Path):
        """Process a single chunk file and extract graph elements."""
        with open(chunk_file, encoding="utf-8") as f:
            chunk_data = json.load(f)

        chunk_id = chunk_data.get("chunk_id", chunk_file.stem)
        thread_id = chunk_data.get("thread_id", "")
        thread_subject = chunk_data.get("thread_subject", "")
        logger.debug(f"Processing chunk {chunk_id} (file={chunk_file.name}, thread={thread_id})")

        # Create chunk node
        chunk_node = GraphNode(
            node_id=chunk_id,
            name=f"Chunk: {chunk_id}",
            node_type="CHUNK",
            properties={
                "thread_id": thread_id,
                "thread_subject": thread_subject,
                "token_count": chunk_data.get("token_count", 0),
                "has_attachments": chunk_data.get("has_attachments", False),
                "email_position": chunk_data.get("email_position", ""),
            },
            source_chunks=[chunk_id],
        )
        self.graph.add_node(chunk_node)

        # Process KG entities
        for entity in chunk_data.get("kg_entities", []):
            entity_node = self._create_entity_node(entity, chunk_id)
            if entity_node:
                self.graph.add_node(entity_node)

                # Create MENTIONED_IN edge
                edge = GraphEdge(
                    edge_id=self._generate_edge_id(entity_node.node_id, chunk_id, "MENTIONED_IN"),
                    source_id=entity_node.node_id,
                    target_id=chunk_id,
                    edge_type="MENTIONED_IN",
                    weight=entity.get("confidence", 0.8),
                    evidence_chunks=[chunk_id],
                )
                self.graph.add_edge(edge)

        # Process KG relationships
        for rel in chunk_data.get("kg_relationships", []):
            self._create_relationship_edges(rel, chunk_id)

        # Process email sender/recipient structure
        sender_name = chunk_data.get("email_sender", "")
        sender_email = chunk_data.get("email_sender_address", "")
        recipients_to = chunk_data.get("email_recipients_to", [])
        recipients_cc = chunk_data.get("email_recipients_cc", [])
        sent_date = (chunk_data.get("sent_timestamp") or "")[:10]  # YYYY-MM-DD

        if sender_name:
            # Resolve sender name via identity registry
            resolved_sender = self._resolve_person_name(sender_name)
            sender_node = GraphNode(
                node_id=self._generate_node_id(resolved_sender, "PERSON"),
                name=resolved_sender,
                node_type="PERSON",
                properties={"email": sender_email} if sender_email else {},
                source_chunks=[chunk_id],
            )
            self.graph.add_node(sender_node)

            # Sender → Chunk (SENT edge)
            edge = GraphEdge(
                edge_id=self._generate_edge_id(sender_node.node_id, chunk_id, "SENT"),
                source_id=sender_node.node_id,
                target_id=chunk_id,
                edge_type="SENT",
                properties={"date": sent_date} if sent_date else {},
                evidence_chunks=[chunk_id],
            )
            self.graph.add_edge(edge)

            # Sender → each Recipient (SENT_TO edge)
            for recipient in recipients_to + recipients_cc:
                r_name = recipient.get("name", "")
                if not r_name:
                    continue
                # Resolve recipient name via identity registry
                resolved_r = self._resolve_person_name(r_name)
                if resolved_r == resolved_sender:
                    continue
                r_node = GraphNode(
                    node_id=self._generate_node_id(resolved_r, "PERSON"),
                    name=resolved_r,
                    node_type="PERSON",
                    properties={"email": recipient.get("email", "")},
                    source_chunks=[chunk_id],
                )
                self.graph.add_node(r_node)

                sent_to_edge = GraphEdge(
                    edge_id=self._generate_edge_id(sender_node.node_id, r_node.node_id, "SENT_TO"),
                    source_id=sender_node.node_id,
                    target_id=r_node.node_id,
                    edge_type="SENT_TO",
                    properties={"date": sent_date} if sent_date else {},
                    evidence_chunks=[chunk_id],
                )
                self.graph.add_edge(sent_to_edge)

        # Fallback: use thread_participants for chunks without sender info (e.g., attachments)
        if not sender_name:
            for participant in chunk_data.get("thread_participants", []):
                if participant:
                    resolved_p = self._resolve_person_name(participant)
                    participant_node = GraphNode(
                        node_id=self._generate_node_id(resolved_p, "PERSON"),
                        name=resolved_p,
                        node_type="PERSON",
                        source_chunks=[chunk_id],
                    )
                    self.graph.add_node(participant_node)

                    edge = GraphEdge(
                        edge_id=self._generate_edge_id(
                            participant_node.node_id, chunk_id, "PARTICIPATED_IN"
                        ),
                        source_id=participant_node.node_id,
                        target_id=chunk_id,
                        edge_type="PARTICIPATED_IN",
                        evidence_chunks=[chunk_id],
                    )
                    self.graph.add_edge(edge)

        # Process attachments
        for att_filename in chunk_data.get("attachment_filenames", []):
            if att_filename:
                att_node = GraphNode(
                    node_id=self._generate_node_id(att_filename, "DOCUMENT"),
                    name=att_filename,
                    node_type="DOCUMENT",
                    properties={"filename": att_filename},
                    source_chunks=[chunk_id],
                )
                self.graph.add_node(att_node)

                # Link to chunk
                edge = GraphEdge(
                    edge_id=self._generate_edge_id(chunk_id, att_node.node_id, "HAS_ATTACHMENT"),
                    source_id=chunk_id,
                    target_id=att_node.node_id,
                    edge_type="HAS_ATTACHMENT",
                    evidence_chunks=[chunk_id],
                )
                self.graph.add_edge(edge)

    def _create_entity_node(self, entity: dict[str, Any], chunk_id: str) -> GraphNode | None:
        """Create a graph node from an entity."""
        from tacitgraph.entity_registry import get_standard_type, resolve_entity_name

        name = (entity.get("entity") or entity.get("text", "")).strip()
        if not name or len(name) < 2:
            return None

        # Resolve aliases via catalog (e.g., "ARC" → "Acme Research Centre")
        name = resolve_entity_name(name)

        # For PERSON entities, resolve via identity registry
        # This merges "Jane Doe", "Doe Jane", "Ms. Doe" etc.
        # Skip short first-name-only matches (e.g., "Jane") — too ambiguous
        entity_type_raw = entity.get("type", "CONCEPT")
        if get_standard_type(entity_type_raw) == "PERSON" and self._identity_name_map:
            if " " in name or "." in name:  # require surname or initial, skip bare first names
                canonical = self._identity_name_map.get(name.lower())
                if canonical and canonical != name:
                    logger.debug(f"PERSON resolved: '{name}' → '{canonical}' (identity registry)")
                    name = canonical

        # Prefer catalog type over LLM type (catalog is built from many chunks)
        entity_type = get_standard_type(entity.get("type", "CONCEPT"))
        catalog_entities = self._catalog.get("entities", {})
        catalog_entry = catalog_entities.get(name)
        if not catalog_entry:
            # Try case-insensitive match
            for cat_name, cat_info in catalog_entities.items():
                if cat_name.lower() == name.lower():
                    catalog_entry = cat_info
                    break
        if catalog_entry and catalog_entry.get("type"):
            catalog_type = catalog_entry["type"]
            if catalog_type != entity_type and catalog_entry.get("mention_count", 0) > 5:
                logger.debug(
                    f"Entity '{name}' type overridden: {entity_type} → {catalog_type} (catalog, {catalog_entry['mention_count']} mentions)"
                )
                entity_type = catalog_type
        node_id = self._generate_node_id(name, entity_type)

        # Update catalog with this entity and any LLM-discovered aliases
        self._update_catalog(name, entity_type, entity.get("aliases", []), chunk_id)

        return GraphNode(
            node_id=node_id,
            name=name,
            node_type=entity_type,
            properties={
                "confidence": entity.get("confidence", 0.8),
                "source": entity.get("source", "unknown"),
                "is_pii": entity.get("is_pii", False),
            },
            source_chunks=[chunk_id],
        )

    def _create_relationship_edges(self, rel: dict[str, Any], chunk_id: str):
        """Create edges from a relationship."""
        source_name = rel.get("source", "").strip()
        target_name = rel.get("target", "").strip()
        rel_type = (
            (rel.get("relationship") or rel.get("description", "RELATED_TO"))
            .upper()
            .replace(" ", "_")
        )

        if not source_name or not target_name:
            return

        from tacitgraph.entity_registry import (
            get_standard_type,
            normalize_relationship_type,
            should_flip_relationship,
        )

        # Flip source/target if relationship implies reversed direction
        # e.g., "Acme EMPLOYS Jane" → "Jane WORKS_AT Acme"
        if should_flip_relationship(rel_type):
            source_name, target_name = target_name, source_name
            source_type_raw, target_type_raw = (
                rel.get("target_type", "CONCEPT"),
                rel.get("source_type", "CONCEPT"),
            )
        else:
            source_type_raw = rel.get("source_type", "CONCEPT")
            target_type_raw = rel.get("target_type", "CONCEPT")

        rel_type = normalize_relationship_type(rel_type)

        source_type = get_standard_type(source_type_raw)
        target_type = get_standard_type(target_type_raw)

        source_id = self._generate_node_id(source_name, source_type)
        target_id = self._generate_node_id(target_name, target_type)

        # Ensure source and target nodes exist
        if source_id not in self.graph.nodes:
            self.graph.add_node(
                GraphNode(
                    node_id=source_id,
                    name=source_name,
                    node_type=source_type,
                    source_chunks=[chunk_id],
                )
            )

        if target_id not in self.graph.nodes:
            self.graph.add_node(
                GraphNode(
                    node_id=target_id,
                    name=target_name,
                    node_type=target_type,
                    source_chunks=[chunk_id],
                )
            )

        # Create edge
        edge = GraphEdge(
            edge_id=self._generate_edge_id(source_id, target_id, rel_type),
            source_id=source_id,
            target_id=target_id,
            edge_type=rel_type,
            weight=rel.get("confidence", 0.7),
            properties={"evidence": rel.get("evidence", "")},
            evidence_chunks=[chunk_id],
        )
        self.graph.add_edge(edge)

    def _build_thread_relationships(self):
        """Build relationships between chunks in the same thread."""
        # Group chunks by thread
        chunks_by_thread: dict[str, list[str]] = defaultdict(list)

        for node_id, node in self.graph.nodes.items():
            if node.node_type == "CHUNK":
                thread_id = node.properties.get("thread_id", "")
                if thread_id:
                    chunks_by_thread[thread_id].append(node_id)

        # Create SAME_THREAD edges between chunks
        for thread_id, chunk_ids in chunks_by_thread.items():
            if len(chunk_ids) < 2:
                continue

            # Create thread node
            thread_node = GraphNode(
                node_id=self._generate_node_id(thread_id, "THREAD"),
                name=thread_id,
                node_type="THREAD",
                properties={"chunk_count": len(chunk_ids)},
                source_chunks=chunk_ids,
            )
            self.graph.add_node(thread_node)

            # Link chunks to thread
            for chunk_id in chunk_ids:
                edge = GraphEdge(
                    edge_id=self._generate_edge_id(chunk_id, thread_node.node_id, "PART_OF_THREAD"),
                    source_id=chunk_id,
                    target_id=thread_node.node_id,
                    edge_type="PART_OF_THREAD",
                    evidence_chunks=[chunk_id],
                )
                self.graph.add_edge(edge)

    def _update_catalog(self, name: str, entity_type: str, aliases: list[str], chunk_id: str):
        """Update entity catalog with discovered entity and aliases."""
        entities = self._catalog.setdefault("entities", {})

        if name in entities:
            entry = entities[name]
            entry["mention_count"] = entry.get("mention_count", 0) + 1
            existing_aliases = set(entry.get("aliases", []))
            for alias in aliases:
                if alias and alias != name:
                    existing_aliases.add(alias)
            entry["aliases"] = sorted(existing_aliases)
        else:
            entities[name] = {
                "standard_name": name,
                "type": entity_type,
                "aliases": sorted(set(a for a in aliases if a and a != name)),
                "mention_count": 1,
                "first_seen_chunk": chunk_id,
            }

    def _normalize_weights(self):
        """Normalize edge weights to 0-1 range."""
        if not self.graph.edges:
            return

        max_weight = max(e.weight for e in self.graph.edges.values())
        if max_weight > 0:
            for edge in self.graph.edges.values():
                edge.weight = edge.weight / max_weight

    def save(self):
        """Save the knowledge graph to disk."""
        # Save as JSON
        nodes_path = self.kg_output_path / "nodes.json"
        edges_path = self.kg_output_path / "edges.json"
        stats_path = self.kg_output_path / "graph_stats.json"

        # Save nodes
        nodes_data = {nid: n.to_dict() for nid, n in self.graph.nodes.items()}
        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, indent=2)

        # Save edges
        edges_data = {eid: e.to_dict() for eid, e in self.graph.edges.items()}
        with open(edges_path, "w", encoding="utf-8") as f:
            json.dump(edges_data, f, indent=2)

        # Save stats
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.graph.metadata, f, indent=2)

        # Save entity catalog
        from tacitgraph.entity_registry import save_catalog

        self._catalog["metadata"] = {
            "built_at": datetime.now().isoformat(),
            "total_entities": len(self._catalog.get("entities", {})),
            "total_aliases": sum(
                len(e.get("aliases", [])) for e in self._catalog.get("entities", {}).values()
            ),
        }
        save_catalog(self._catalog, self._catalog_path)

        # Save as GraphML (if networkx available)
        if HAS_NETWORKX:
            try:
                G = self.graph.to_networkx()
                graphml_path = self.kg_output_path / "graph.graphml"
                nx.write_graphml(G, str(graphml_path))
                logger.info(f"Saved GraphML to {graphml_path}")
            except Exception as e:
                logger.warning(f"Failed to save GraphML: {e}")

        logger.info(f"Knowledge graph saved to {self.kg_output_path}")

    def load(self) -> KnowledgeGraph:
        """Load knowledge graph from disk."""
        nodes_path = self.kg_output_path / "nodes.json"
        edges_path = self.kg_output_path / "edges.json"

        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError(f"Knowledge graph not found at {self.kg_output_path}")

        with open(nodes_path, encoding="utf-8") as f:
            nodes_data = json.load(f)

        with open(edges_path, encoding="utf-8") as f:
            edges_data = json.load(f)

        self.graph = KnowledgeGraph()
        for nid, ndata in nodes_data.items():
            self.graph.add_node(GraphNode.from_dict(ndata))

        for eid, edata in edges_data.items():
            self.graph.add_edge(GraphEdge.from_dict(edata))

        logger.info(f"Loaded knowledge graph: {self.graph.stats()}")
        return self.graph
