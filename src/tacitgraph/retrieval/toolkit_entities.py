"""Entity matching and lookup: maps query keywords to knowledge-graph entities."""

import json
import logging
from datetime import datetime
from typing import Any

from tacitgraph.retrieval.toolkit_data import ToolkitData
from tacitgraph.retrieval.toolkit_models import ToolResult

logger = logging.getLogger(__name__)


class ToolkitEntitySearch(ToolkitData):
    """Entity matching and lookup: maps query keywords to knowledge-graph entities."""

    def node_retrieval(self, keywords: list[str], top_n: int = 10) -> list[tuple[str, str]]:
        """
        PathRAG Node Retrieval: dense vector matching of keywords against entity embeddings.

        Per the PathRAG paper (Stage 1):
        1. Encode keywords using the same embedding model
        2. Cosine similarity against pre-computed entity embeddings
        3. Return top-N entity names

        Args:
            keywords: Extracted keywords from query
            top_n: Maximum number of entities to return

        Returns:
            List of (entity_id, entity_name) tuples
        """
        entity_ids, entity_embeddings = self._load_entity_embeddings()
        if entity_embeddings is None or len(entity_ids) == 0:
            return []

        try:
            import numpy as np

            # Get or create embedding generator for query encoding
            if self._embedding_generator is None:
                from tacitgraph.gold.embedding_generator import EmbeddingGenerator

                self._embedding_generator = EmbeddingGenerator(str(self.gold_path), mode=self.mode)

            generator = self._embedding_generator

            # Embed all keywords in a batch
            keyword_embeddings = generator.embed_batch(keywords)
            valid_embeddings = [e for e in keyword_embeddings if e is not None]
            if not valid_embeddings:
                return []

            keyword_matrix = np.array(valid_embeddings)

            # Normalize for cosine similarity
            keyword_norms = np.linalg.norm(keyword_matrix, axis=1, keepdims=True)
            keyword_norms[keyword_norms == 0] = 1
            keyword_matrix_norm = keyword_matrix / keyword_norms

            entity_norms = np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
            entity_norms[entity_norms == 0] = 1
            entity_matrix_norm = entity_embeddings / entity_norms

            # Compute similarity: each keyword against all entities
            # Shape: (num_keywords, num_entities)
            similarity_matrix = keyword_matrix_norm @ entity_matrix_norm.T

            # For each entity, take the max similarity across all keywords
            max_similarities = similarity_matrix.max(axis=0)

            # Get more candidates than needed to account for dedup filtering
            top_indices = np.argsort(max_similarities)[::-1][: top_n * 3]

            # Collect candidates with semantic deduplication.
            # Alias expansion can produce 40+ variants of one concept (e.g. "migration")
            # that dominate all top-N slots, leaving no room for other query entities
            # (e.g. BER, Berlin Office). Skip entities whose embedding is >0.85 similar
            # to an already-selected entity to force diversity.
            matched = []
            matched_embeddings: list[Any] = []
            seen_ids = set()
            seen_names = set()

            for idx in top_indices:
                if max_similarities[idx] < 0.2:
                    break
                entity_id = entity_ids[idx]
                if entity_id in seen_ids:
                    continue
                name = self._entity_id_to_name.get(entity_id, entity_id)
                name_lower = name.lower()
                if name_lower in seen_names:
                    continue

                # Semantic dedup: skip if too similar to an already-selected entity
                if matched_embeddings:
                    candidate_vec = entity_matrix_norm[idx]
                    existing = np.array(matched_embeddings)
                    sims = existing @ candidate_vec
                    if sims.max() > 0.70:
                        continue

                seen_ids.add(entity_id)
                seen_names.add(name_lower)
                matched.append((entity_id, name))
                matched_embeddings.append(entity_matrix_norm[idx])
                if len(matched) >= top_n:
                    break

            logger.info(
                f"Node retrieval: {len(keywords)} keywords → {len(matched)} entities "
                f"(top score: {max_similarities[top_indices[0]]:.3f})"
            )
            return matched

        except Exception as e:
            logger.error(f"Node retrieval failed: {e}")
            return []

    def list_entities(self, entity_type: str) -> ToolResult:
        """List all entities of a given type from the knowledge graph."""
        start_time = datetime.now()

        try:
            # Load nodes directly from JSON
            nodes_file = self.gold_path / "knowledge_graph" / "nodes.json"
            if not nodes_file.exists():
                return ToolResult(
                    tool_name="list_entities",
                    success=False,
                    data=[],
                    message="Knowledge graph nodes not found",
                )

            with open(nodes_file, encoding="utf-8") as f:
                nodes = json.load(f)

            entity_type_upper = entity_type.upper()
            entities = []

            for node_id, node_data in nodes.items():
                node_type = node_data.get("node_type", node_data.get("type", ""))
                if node_type == entity_type_upper:
                    name = node_data.get("name", node_id)
                    mention_count = node_data.get("mention_count", 0)
                    entities.append(
                        {
                            "name": name,
                            "type": entity_type_upper,
                            "connections": mention_count,
                        }
                    )

            # Sort by mention count (most mentioned first)
            entities.sort(key=lambda x: x["connections"], reverse=True)

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="list_entities",
                success=True,
                data=entities,
                message=f"Found {len(entities)} entities of type {entity_type_upper}",
                execution_time=execution_time,
            )

        except Exception as e:
            return ToolResult(tool_name="list_entities", success=False, data=[], message=str(e))

    def list_entity_types(self) -> ToolResult:
        """Discover what entity types exist in the knowledge graph."""
        start_time = datetime.now()

        try:
            nodes_file = self.gold_path / "knowledge_graph" / "nodes.json"
            if not nodes_file.exists():
                return ToolResult(
                    tool_name="list_entity_types",
                    success=False,
                    data=[],
                    message="Knowledge graph nodes not found",
                )

            with open(nodes_file, encoding="utf-8") as f:
                nodes = json.load(f)

            # Count entities per type
            type_counts: dict[str, int] = {}
            for _node_id, node_data in nodes.items():
                node_type = node_data.get("node_type", node_data.get("type", "UNKNOWN"))
                type_counts[node_type] = type_counts.get(node_type, 0) + 1

            # Sort by count descending
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            results = [{"type": t, "count": c} for t, c in sorted_types]

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="list_entity_types",
                success=True,
                data=results,
                message=f"Found {len(results)} entity types: "
                + ", ".join(f"{t}({c})" for t, c in sorted_types),
                execution_time=execution_time,
            )

        except Exception as e:
            return ToolResult(tool_name="list_entity_types", success=False, data=[], message=str(e))

    def entity_lookup(self, entity_name: str) -> ToolResult:
        """Look up entity details and relationships."""
        start_time = datetime.now()

        try:
            graph = self._load_graph()

            # Find matching entity
            matching_nodes = []
            for _node_id, node in graph.nodes.items():
                if entity_name.lower() in node.name.lower():
                    matching_nodes.append(node)

            if not matching_nodes:
                return ToolResult(
                    tool_name="entity_lookup",
                    success=False,
                    data=None,
                    message=f"Entity '{entity_name}' not found",
                )

            # Get details for best match
            node = matching_nodes[0]

            # Get relationships
            outgoing = graph.get_edges_from(node.node_id)
            incoming = graph.get_edges_to(node.node_id)

            relationships = []
            for edge in outgoing[:10]:
                target = graph.get_node(edge.target_id)
                if target:
                    relationships.append(
                        {
                            "direction": "outgoing",
                            "type": edge.edge_type,
                            "target": target.name,
                            "target_type": target.node_type,
                        }
                    )

            for edge in incoming[:10]:
                source = graph.get_node(edge.source_id)
                if source:
                    relationships.append(
                        {
                            "direction": "incoming",
                            "type": edge.edge_type,
                            "source": source.name,
                            "source_type": source.node_type,
                        }
                    )

            result = {
                "node_id": node.node_id,
                "name": node.name,
                "type": node.node_type,
                "mention_count": node.mention_count,
                "properties": node.properties,
                "relationships": relationships,
                "source_chunks": node.source_chunks[:5],
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="entity_lookup",
                success=True,
                data=result,
                message=f"Found entity: {node.name}",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Entity lookup failed: {e}")
            return ToolResult(tool_name="entity_lookup", success=False, data=None, message=str(e))
