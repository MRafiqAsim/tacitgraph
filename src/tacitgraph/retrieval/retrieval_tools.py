"""
Retrieval Tools Module

Defines tools available to the ReAct agent for knowledge retrieval.
Each tool wraps a specific retrieval strategy (PathRAG, GraphRAG, Vector, etc.)
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    success: bool
    data: Any
    message: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "execution_time": self.execution_time,
        }


@dataclass
class Tool:
    """Definition of a tool available to the ReAct agent."""

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]  # param_name -> {type, description, required}
    function: Callable

    def to_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function schema."""
        properties = {}
        required = []

        for param_name, param_info in self.parameters.items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if param_info.get("required", False):
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }


class RetrievalToolkit:
    """
    Collection of retrieval tools for the ReAct agent.

    Provides unified access to:
    - PathRAG: Path-based reasoning retrieval
    - GraphRAG: Community-based retrieval
    - Vector Search: Similarity-based retrieval
    - Entity/Chunk lookup
    - Temporal filtering
    """

    def __init__(
        self,
        gold_path: str,
        silver_path: str | None = None,
        mode: str = "llm",
    ):
        """
        Initialize the retrieval toolkit.

        Args:
            gold_path: Path to Gold layer with graph and indexes
            silver_path: Path to Silver layer with chunks (optional)
            mode: Processing mode — "local" uses local models
        """
        self.gold_path = Path(gold_path)
        self.silver_path = Path(silver_path) if silver_path else None
        self.mode = mode

        # Lazy-loaded components
        self._graph = None
        self._path_indexer = None
        self._community_detector = None
        self._embedding_generator = None
        self._chunk_embeddings = None
        self._chunk_ids = None
        self._entity_embeddings = None
        self._entity_ids = None
        self._llm_client = None
        self._llm_model = None

        # Register tools
        self.tools: dict[str, Tool] = {}
        self._register_tools()

        logger.info("RetrievalToolkit initialized")

    def _register_tools(self):
        """Register all available tools."""

        # PathRAG Search
        self.tools["pathrag_search"] = Tool(
            name="pathrag_search",
            description="Find reasoning paths between entities in the knowledge graph. "
            "Use this for multi-hop queries that connect different concepts.",
            parameters={
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entity names to find paths between (resolved to graph IDs internally)",
                    "required": True,
                },
                "max_paths": {
                    "type": "integer",
                    "description": "Maximum number of paths to return (default: 5)",
                    "required": False,
                },
            },
            function=self.pathrag_search,
        )

        # GraphRAG Global Search (map-reduce over all communities, uses LLM)
        self.tools["global_search"] = Tool(
            name="global_search",
            description="GraphRAG Global Search: map-reduce over ALL community summaries using LLM. "
            "Best for aggregate/broad questions like 'what projects are discussed?', "
            "'list all topics', 'summarize the main themes', 'what teams are involved?'. "
            "This scans the entire knowledge graph — use it when no single entity or chunk "
            "can answer the question. Returns a synthesized answer with scored key points.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The broad/aggregate search query",
                    "required": True,
                },
                "level": {
                    "type": "integer",
                    "description": "Community hierarchy level (0=fine, higher=coarse). Default: 0",
                    "required": False,
                },
            },
            function=self._global_search_tool,
        )

        # GraphRAG Local Search (entity-centric context building, uses LLM)
        self.tools["local_search"] = Tool(
            name="local_search",
            description="GraphRAG Local Search: find entities related to query via embeddings, "
            "then expand to connected relationships, community reports, and source text. "
            "Best for entity-specific questions like 'what did PERSON_001 work on?', "
            "'what issues were reported about JIRA?', 'tell me about Project Atlas'. "
            "Returns a detailed answer grounded in entity context.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The entity-specific search query",
                    "required": True,
                }
            },
            function=self._local_search_tool,
        )

        # Vector Search
        self.tools["vector_search"] = Tool(
            name="vector_search",
            description="Find text chunks most similar to a query using semantic similarity. "
            "Use this for finding specific information or evidence.",
            parameters={
                "query": {"type": "string", "description": "The search query", "required": True},
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default: 10)",
                    "required": False,
                },
                "filter_thread": {
                    "type": "string",
                    "description": "Filter to specific thread ID (optional)",
                    "required": False,
                },
            },
            function=self.vector_search,
        )

        # Entity Lookup
        self.tools["entity_lookup"] = Tool(
            name="entity_lookup",
            description="Get details about a specific entity including its relationships. "
            "Use this to understand an entity's role in the knowledge graph.",
            parameters={
                "entity_name": {
                    "type": "string",
                    "description": "Name of the entity to look up",
                    "required": True,
                }
            },
            function=self.entity_lookup,
        )

        # List Entities by Type
        self.tools["list_entities"] = Tool(
            name="list_entities",
            description="List all entities of a given type from the knowledge graph. "
            "Use this for aggregate questions like 'list all projects', 'who are the people', "
            "'what organizations are mentioned'. "
            "Valid types: PERSON, ORG, PRODUCT, GPE, DOCUMENT, EVENT, LOC, FAC, WORK_OF_ART.",
            parameters={
                "entity_type": {
                    "type": "string",
                    "description": "The entity type to list (e.g., PRODUCT, PERSON, ORG)",
                    "required": True,
                }
            },
            function=self.list_entities,
        )

        # List Entity Types
        self.tools["list_entity_types"] = Tool(
            name="list_entity_types",
            description="Discover what entity types exist in the knowledge graph and how many of each. "
            "Use this FIRST for aggregate/listing queries to understand what data is available "
            "before searching for specific types.",
            parameters={},
            function=self.list_entity_types,
        )

        # Get Chunk Context
        self.tools["get_chunk_context"] = Tool(
            name="get_chunk_context",
            description="Get the full context around a chunk including thread and attachments. "
            "Use this when you need more context about a specific piece of evidence.",
            parameters={
                "chunk_id": {
                    "type": "string",
                    "description": "The chunk ID to get context for",
                    "required": True,
                }
            },
            function=self.get_chunk_context,
        )

        # Temporal Filter
        self.tools["temporal_filter"] = Tool(
            name="temporal_filter",
            description="Filter chunks by date range. "
            "Use this when the query involves specific time periods.",
            parameters={
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of chunk IDs to filter",
                    "required": True,
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format",
                    "required": False,
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format",
                    "required": False,
                },
            },
            function=self.temporal_filter,
        )

        # get_attachment_content removed — attachment content is searchable via vector_search

    def _get_llm_client(self):
        """Lazy-initialize LLM client for global/local search tools."""
        if self._llm_client is None:
            import httpx

            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_key = os.getenv("AZURE_OPENAI_API_KEY")
            if azure_endpoint and azure_key:
                from openai import AzureOpenAI

                self._llm_client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    max_retries=2,
                )
                self._llm_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        return self._llm_client, self._llm_model

    def _global_search_tool(self, query: str, level: int = 0) -> ToolResult:
        """Wrapper for global_search that auto-provides LLM client."""
        client, model = self._get_llm_client()
        if not client:
            return ToolResult(
                tool_name="global_search",
                success=False,
                data={},
                message="LLM client not available",
            )
        return self.global_search(query, llm_client=client, model=model, level=level)

    def _local_search_tool(self, query: str) -> ToolResult:
        """Wrapper for local_search that auto-provides LLM client."""
        client, model = self._get_llm_client()
        if not client:
            return ToolResult(
                tool_name="local_search", success=False, data={}, message="LLM client not available"
            )
        return self.local_search(query, llm_client=client, model=model)

    def _load_graph(self):
        """Lazy load the knowledge graph."""
        if self._graph is None:
            from tacitgraph.gold.graph_builder import GraphBuilder

            builder = GraphBuilder("", str(self.gold_path))
            self._graph = builder.load()
        return self._graph

    def _load_path_indexer(self):
        """Lazy load the path indexer."""
        if self._path_indexer is None:
            from tacitgraph.gold.path_indexer import PathIndexer

            self._path_indexer = PathIndexer(self._load_graph(), str(self.gold_path))
            try:
                self._path_indexer.load()
                logger.info("Loaded pre-computed path index")
            except FileNotFoundError:
                # Don't build full index - use on-demand path finding instead
                logger.info("No pre-computed path index, will use on-demand path finding")
        return self._path_indexer

    def _load_embeddings(self):
        """Lazy load chunk embeddings."""
        if self._chunk_embeddings is None:
            from tacitgraph.gold.embedding_generator import EmbeddingGenerator

            generator = EmbeddingGenerator(str(self.gold_path), mode=self.mode)
            self._embedding_generator = generator
            try:
                self._chunk_ids, self._chunk_embeddings = generator.load_embeddings("chunks")
            except FileNotFoundError:
                logger.warning("Chunk embeddings not found")
                self._chunk_ids = []
                self._chunk_embeddings = None
        return self._chunk_ids, self._chunk_embeddings

    def _load_entity_embeddings(self):
        """Lazy load entity embeddings and ID-to-name mapping."""
        if self._entity_embeddings is None:
            from tacitgraph.gold.embedding_generator import EmbeddingGenerator

            generator = EmbeddingGenerator(str(self.gold_path), mode=self.mode)
            try:
                self._entity_ids, self._entity_embeddings = generator.load_embeddings("entities")
                # Build ID → name mapping from graph nodes
                self._entity_id_to_name = {}
                nodes_file = self.gold_path / "knowledge_graph" / "nodes.json"
                if nodes_file.exists():
                    with open(nodes_file, encoding="utf-8") as f:
                        nodes = json.load(f)
                    for node_id, node_data in nodes.items():
                        self._entity_id_to_name[node_id] = node_data.get("name", node_id)
                logger.info(f"Loaded {len(self._entity_ids)} entity embeddings for node retrieval")
            except FileNotFoundError:
                logger.warning("Entity embeddings not found — falling back to string matching")
                self._entity_ids = []
                self._entity_embeddings = None
                self._entity_id_to_name = {}
        return self._entity_ids, self._entity_embeddings

    def node_retrieval(self, keywords: list[str], top_n: int = 10) -> list[str]:
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
            List of entity names (graph node names)
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
            matched_embeddings = []
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

    def pathrag_search(
        self,
        entities: list[str] | None = None,
        entity_ids: list[str] | None = None,
        entity_names: list[str] | None = None,
        max_paths: int = 5,
    ) -> ToolResult:
        """
        Find paths between entities using PathRAG algorithms.

        Accepts either:
        - entity_ids: Direct graph node IDs (from node_retrieval, preferred)
        - entities: Entity names (from ReAct agent, resolved via embedding matching)

        Args:
            entities: Entity names (resolved to IDs internally, used by ReAct)
            entity_ids: Graph node IDs (from node_retrieval, no re-matching needed)
            entity_names: Optional names for logging
            max_paths: Maximum paths to return
        """
        start_time = datetime.now()

        try:
            import asyncio

            from .pathrag_retriever import PathRAGConfig, PathRAGRetriever

            config = PathRAGConfig(
                max_hops=3, flow_threshold=0.05, flow_alpha=0.8, top_k_paths=max_paths
            )
            retriever = PathRAGRetriever(str(self.gold_path), config)

            # Resolve entity IDs if only names provided (ReAct agent path)
            if entity_ids is None and entities:
                # Use embedding matching to resolve names → IDs
                matched = self.node_retrieval(entities, top_n=len(entities))
                if matched:
                    entity_ids = [m[0] for m in matched]
                    entity_names = [m[1] for m in matched]
                else:
                    # Fallback to string matching
                    entity_ids = retriever.find_entity_ids_by_name(entities)
                    entity_names = entities

            if not entity_ids:
                return ToolResult(
                    tool_name="pathrag_search",
                    success=False,
                    data=[],
                    message="Could not find any matching entities in the graph",
                )

            logger.info(f"PathRAG search: {len(entity_ids)} entities {entity_names or entity_ids}")

            # Run async retrieval — handle both standalone and notebook (nested loop) contexts
            try:
                loop = asyncio.get_running_loop()
                # Already inside an event loop (Jupyter/Synapse) — use nest_asyncio
                import nest_asyncio

                nest_asyncio.apply()
                path_results = loop.run_until_complete(retriever.retrieve(entity_ids, max_paths))
            except RuntimeError:
                # No running loop — create one
                loop = asyncio.new_event_loop()
                try:
                    path_results = loop.run_until_complete(
                        retriever.retrieve(entity_ids, max_paths)
                    )
                finally:
                    loop.close()

            # Format results
            results = []
            for pr in path_results:
                results.append(
                    {
                        "path_id": f"path_{hash(tuple(pr.path)) & 0xFFFFFFFF:08x}",
                        "description": pr.natural_language,
                        "path": pr.path_names,
                        "path_types": pr.path_types,
                        "hop_count": pr.hop_count,
                        "weight": pr.weight,
                        "evidence_chunks": pr.evidence_chunks[:5],
                    }
                )

            # Semantic re-ranking: score paths by relevance to query entities
            if results and len(results) > 1:
                try:
                    generator = self._embedding_generator
                    if generator is None:
                        from tacitgraph.gold.embedding_generator import (
                            EmbeddingConfig,
                            EmbeddingGenerator,
                        )

                        self._embedding_generator = EmbeddingGenerator(
                            str(self.gold_path), EmbeddingConfig(), mode=self.mode
                        )
                        generator = self._embedding_generator

                    query_text = " ".join(entity_names or entities or entity_ids)
                    query_emb = generator.embed_text(query_text)
                    if query_emb:
                        import numpy as np

                        query_vec = np.array(query_emb)
                        query_norm = query_vec / np.linalg.norm(query_vec)

                        for r in results:
                            path_emb = generator.embed_text(r["description"])
                            if path_emb:
                                path_vec = np.array(path_emb)
                                path_norm = path_vec / np.linalg.norm(path_vec)
                                semantic_score = float(np.dot(query_norm, path_norm))
                                # Combine flow weight and semantic score
                                r["semantic_score"] = semantic_score
                                r["combined_score"] = r["weight"] * 0.4 + semantic_score * 0.6
                            else:
                                r["semantic_score"] = 0.0
                                r["combined_score"] = r["weight"]

                        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
                        logger.info(
                            f"PathRAG semantic re-ranking: top path score {results[0].get('combined_score', 0):.3f}"
                        )
                except Exception as e:
                    logger.warning(f"PathRAG semantic re-ranking failed, using flow weights: {e}")

            # Trim to max_paths after re-ranking
            results = results[:max_paths]

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="pathrag_search",
                success=True,
                data=results,
                message=f"Found {len(results)} paths (flow + semantic scoring)",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"PathRAG search failed: {e}")
            import traceback

            traceback.print_exc()
            return ToolResult(tool_name="pathrag_search", success=False, data=[], message=str(e))

    def vector_search(
        self, query: str, top_k: int = 10, filter_thread: str | None = None
    ) -> ToolResult:
        """Semantic similarity search over chunks."""
        start_time = datetime.now()

        try:
            chunk_ids, chunk_embeddings = self._load_embeddings()
            generator = self._embedding_generator

            if chunk_embeddings is None or len(chunk_ids) == 0:
                return ToolResult(
                    tool_name="vector_search",
                    success=False,
                    data=[],
                    message="Chunk embeddings not available",
                )

            # Search
            results = generator.similarity_search(
                query,
                chunk_embeddings,
                chunk_ids,
                top_k=top_k * 2,  # Get more for filtering
            )

            # Load chunk details (deduplicate text vs summary matches)
            detailed_results = []
            seen_chunks = set()
            for chunk_id, score in results:
                # Strip _sum suffix — summary embeddings map to original chunk
                load_id = chunk_id.removesuffix("_sum")
                if load_id in seen_chunks:
                    continue
                seen_chunks.add(load_id)

                chunk_data = self._load_chunk(load_id)
                if chunk_data:
                    detailed_results.append(
                        {
                            "chunk_id": load_id,
                            "similarity_score": score,
                            "text": chunk_data.get("text_english")
                            or chunk_data.get("text_anonymized", ""),
                            "text_english": chunk_data.get("text_english", ""),
                            "summary": chunk_data.get("summary", ""),
                            "thread_id": chunk_data.get("thread_id"),
                            "thread_subject": chunk_data.get("thread_subject"),
                            "source_type": chunk_data.get("source_type", "email"),
                            "source_attachment_filename": chunk_data.get(
                                "source_attachment_filename", ""
                            ),
                            "has_attachments": chunk_data.get("has_attachments", False),
                            "email_sender": chunk_data.get("email_sender", ""),
                            "sent_timestamp": chunk_data.get("sent_timestamp", ""),
                            "received_timestamp": chunk_data.get("received_timestamp", ""),
                        }
                    )

                if len(detailed_results) >= top_k:
                    break

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="vector_search",
                success=True,
                data=detailed_results,
                message=f"Found {len(detailed_results)} similar chunks",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return ToolResult(tool_name="vector_search", success=False, data=[], message=str(e))

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
            type_counts = {}
            for node_id, node_data in nodes.items():
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
            for node_id, node in graph.nodes.items():
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

    def get_chunk_context(self, chunk_id: str) -> ToolResult:
        """Get full context for a chunk."""
        start_time = datetime.now()

        try:
            chunk_data = self._load_chunk(chunk_id)

            if not chunk_data:
                return ToolResult(
                    tool_name="get_chunk_context",
                    success=False,
                    data=None,
                    message=f"Chunk '{chunk_id}' not found",
                )

            result = {
                "chunk_id": chunk_id,
                "text": chunk_data.get("text_english")
                or chunk_data.get("text_anonymized", chunk_data.get("text_original", "")),
                "text_english": chunk_data.get("text_english", ""),
                "summary": chunk_data.get("summary", ""),
                "thread_id": chunk_data.get("thread_id"),
                "thread_subject": chunk_data.get("thread_subject"),
                "email_position": chunk_data.get("email_position"),
                "thread_participants": chunk_data.get("thread_participants", []),
                "has_attachments": chunk_data.get("has_attachments", False),
                "attachment_filenames": chunk_data.get("attachment_filenames", []),
                "source_type": chunk_data.get("source_type", "email"),
                "source_attachment_filename": chunk_data.get("source_attachment_filename", ""),
                "email_sender": chunk_data.get("email_sender", ""),
                "sent_timestamp": chunk_data.get("sent_timestamp", ""),
                "kg_entities": chunk_data.get("kg_entities", []),
                "kg_relationships": chunk_data.get("kg_relationships", []),
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="get_chunk_context",
                success=True,
                data=result,
                message="Chunk context retrieved",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Get chunk context failed: {e}")
            return ToolResult(
                tool_name="get_chunk_context", success=False, data=None, message=str(e)
            )

    def temporal_filter(
        self,
        chunk_ids: list[str] = None,
        start_date: str | None = None,
        end_date: str | None = None,
        date_range: str | None = None,
        **kwargs,
    ) -> ToolResult:
        """Filter chunks by date range."""
        start_time = datetime.now()

        # Handle LLM passing date_range instead of start_date/end_date
        if date_range and not start_date and not end_date:
            from .date_filter import extract_date_range

            dr = extract_date_range(date_range)
            if dr:
                start_date = dr.start
                end_date = dr.end

        chunk_ids = chunk_ids or []

        try:
            filtered = []

            for chunk_id in chunk_ids:
                chunk_data = self._load_chunk(chunk_id)
                if not chunk_data:
                    continue

                # Use received_timestamp (preferred) or sent_timestamp from Bronze
                chunk_date = (
                    chunk_data.get("received_timestamp", "") or chunk_data.get("sent_timestamp", "")
                )[:10]

                include = True
                if start_date and chunk_date < start_date:
                    include = False
                if end_date and chunk_date > end_date:
                    include = False

                if include:
                    filtered.append(chunk_id)

            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="temporal_filter",
                success=True,
                data=filtered,
                message=f"Filtered to {len(filtered)} chunks in date range",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Temporal filter failed: {e}")
            return ToolResult(tool_name="temporal_filter", success=False, data=[], message=str(e))

    def route_graphrag_query(self, query: str) -> str:
        """Route query to global or local search based on query type."""
        import re

        query_lower = query.lower()

        # Global indicators: aggregate, summary, overview questions
        global_patterns = [
            r"(what|list|all|every|which)\b.*\b(topics?|themes?|projects?|discussed|mentioned|about|names?|types?)",
            r"(summarize|overview|main|key)\b.*\b(topics?|themes?|activities|discussions)",
            r"how many\b",
            r"(most common|frequently|overall|general|across|throughout)",
        ]

        # Local indicators: specific entity or relationship questions
        local_patterns = [
            r"(who|what did|what is|what was|what are|what were|tell me about|details?|describe)\b",
            r"(relationship|connection|between|involved in|work on|responsible)",
            r"(when did|when was|when were|when is|where|how did|how is|how was)\b",
            r"\b(advice|recommended|suggested|proposed)\b",
        ]

        global_score = sum(1 for p in global_patterns if re.search(p, query_lower))
        local_score = sum(1 for p in local_patterns if re.search(p, query_lower))

        route = "global" if global_score > local_score else "local"
        logger.info(f"GraphRAG query route: {route} (global={global_score}, local={local_score})")
        return route

    def global_search(
        self,
        query: str,
        llm_client,
        model: str = None,
        level: int = 0,
        max_chunks_per_community: int = 3,
    ) -> ToolResult:
        """
        GraphRAG Global Search — map-reduce over community summaries.

        Map phase: each community summary → LLM → rated key points.
        Reduce phase: top points → LLM → final synthesized answer.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tacitgraph.prompt_loader import get_prompt

        model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        start_time = datetime.now()

        try:
            # Load all communities at the chosen level
            communities_path = self.gold_path / "communities" / f"level_{level}"
            if not communities_path.exists():
                return ToolResult(
                    tool_name="global_search",
                    success=False,
                    data={},
                    message=f"Community level {level} not found",
                )

            communities = []
            for comm_file in sorted(communities_path.glob("*.json")):
                with open(comm_file, encoding="utf-8") as f:
                    communities.append(json.load(f))

            if not communities:
                return ToolResult(
                    tool_name="global_search",
                    success=False,
                    data={},
                    message="No communities found",
                )

            logger.info(f"Global search: {len(communities)} communities at level {level}")

            # Check if embedding-based search is available and enabled
            use_embedding_search = (
                os.getenv("GRAPH_RAG_GLOBAL_SEARCH_MODE", "embedding").lower() == "embedding"
            )

            if use_embedding_search:
                return self._global_search_embedding(
                    query, communities, llm_client, model, start_time
                )

            # MAP PHASE: process each community (legacy map-reduce)
            def map_community(comm):
                """Send community context to LLM, get rated points."""
                summary = comm.get("summary", "")
                entities_text = ", ".join(e["name"] for e in comm.get("key_entities", [])[:15])

                # Load sample source chunks for richer context
                source_chunks = comm.get("source_chunk_ids", [])
                source_text_parts = []
                for cid in source_chunks[:max_chunks_per_community]:
                    chunk_data = self._load_chunk(cid)
                    if chunk_data:
                        text = (
                            chunk_data.get("text_english")
                            or chunk_data.get("text_anonymized")
                            or chunk_data.get("summary", "")
                        )
                        if text:
                            source_text_parts.append(text)

                source_text = (
                    "\n---\n".join(source_text_parts)
                    if source_text_parts
                    else "(no source text available)"
                )

                try:
                    response = llm_client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": get_prompt(
                                    "retrieval",
                                    "graphrag_global_map",
                                    "system_prompt",
                                    "Extract key points relevant to the query. Return JSON array of {point, score} objects.",
                                ),
                            },
                            {
                                "role": "user",
                                "content": get_prompt(
                                    "retrieval",
                                    "graphrag_global_map",
                                    "user_prompt",
                                    "Query: {query}\nContext: {community_context}",
                                )
                                .replace("{query}", query)
                                .replace("{community_context}", summary)
                                .replace("{entities}", entities_text)
                                .replace("{source_text}", source_text),
                            },
                        ],
                        temperature=get_prompt("retrieval", "graphrag_global_map", "temperature"),
                        max_tokens=get_prompt("retrieval", "graphrag_global_map", "max_tokens"),
                    )

                    # extract_llm_content raises LLMContentError for content_filter / length
                    from tacitgraph.llm_response import extract_llm_content

                    content = extract_llm_content(response, context="global search map")
                    # Parse JSON array from response; strip markdown code fences if present
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                    points = json.loads(content)
                    if not isinstance(points, list):
                        points = []

                    # Attach community metadata
                    for p in points:
                        p["community_id"] = comm.get("community_id", "")
                        p["source_chunk_ids"] = source_chunks[:5]

                    return points

                except Exception as e:
                    logger.warning(f"Map failed for community {comm.get('community_id')}: {e}")
                    return []

            # Run map phase with thread pool (5 concurrent LLM calls)
            all_points = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(map_community, c): c for c in communities}
                for future in as_completed(futures):
                    points = future.result()
                    all_points.extend(points)

            logger.info(f"Map phase: {len(all_points)} points from {len(communities)} communities")

            if not all_points:
                return ToolResult(
                    tool_name="global_search",
                    success=True,
                    data={
                        "answer": "No relevant information found across communities.",
                        "points": [],
                        "source_chunk_ids": [],
                    },
                    message="No relevant points found",
                )

            # Filter and sort points by score
            all_points = [p for p in all_points if p.get("score", 0) >= 30]
            all_points.sort(key=lambda x: x.get("score", 0), reverse=True)
            top_points = all_points[:40]

            # Collect source chunk IDs from top-scoring communities
            source_chunk_ids = []
            source_communities = set()
            for p in top_points:
                source_communities.add(p.get("community_id", ""))
                source_chunk_ids.extend(p.get("source_chunk_ids", []))
            source_chunk_ids = list(dict.fromkeys(source_chunk_ids))[:20]  # deduplicate, cap at 20

            # REDUCE PHASE: synthesize final answer
            points_text = "\n".join(
                f"- [{p.get('score', 0)}] {p.get('point', '')}" for p in top_points
            )

            reduce_response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt(
                            "retrieval",
                            "graphrag_global_reduce",
                            "system_prompt",
                            "Synthesize the key points into a comprehensive answer.",
                        ),
                    },
                    {
                        "role": "user",
                        "content": get_prompt(
                            "retrieval",
                            "graphrag_global_reduce",
                            "user_prompt",
                            "Query: {query}\nPoints: {points}",
                        )
                        .replace("{query}", query)
                        .replace("{points}", points_text),
                    },
                ],
                temperature=get_prompt("retrieval", "graphrag_global_reduce", "temperature"),
                max_tokens=get_prompt("retrieval", "graphrag_global_reduce", "max_tokens"),
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            answer = extract_llm_content(reduce_response, context="global search reduce")
            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="global_search",
                success=True,
                data={
                    "answer": answer,
                    "points": [{"point": p["point"], "score": p["score"]} for p in top_points[:10]],
                    "source_chunk_ids": source_chunk_ids,
                    "source_communities": list(source_communities),
                },
                message=f"Global search: {len(top_points)} points from {len(source_communities)} communities",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Global search failed: {e}")
            return ToolResult(tool_name="global_search", success=False, data={}, message=str(e))

    def _global_search_embedding(self, query, communities, llm_client, model, start_time):
        """Embedding-based global search — find top communities by similarity, load source chunks."""
        from tacitgraph.prompt_loader import get_prompt

        try:
            # Load or build community summary embeddings
            if not hasattr(self, "_community_ids") or self._community_ids is None:
                if self._embedding_generator is None:
                    from tacitgraph.gold.embedding_generator import (
                        EmbeddingConfig,
                        EmbeddingGenerator,
                    )

                    self._embedding_generator = EmbeddingGenerator(
                        str(self.gold_path), EmbeddingConfig(), mode=self.mode
                    )

                try:
                    self._community_ids, self._community_embeddings = (
                        self._embedding_generator.load_embeddings("community_summaries")
                    )
                    logger.info(f"Loaded {len(self._community_ids)} community summary embeddings")
                except FileNotFoundError:
                    # Embed on-the-fly if not pre-computed
                    summaries = [
                        {"id": c.get("community_id", ""), "summary": c.get("summary", "")}
                        for c in communities
                        if c.get("summary")
                    ]
                    if summaries:
                        self._community_ids, self._community_embeddings = (
                            self._embedding_generator.embed_summaries(summaries)
                        )
                        self._embedding_generator.save_embeddings(
                            self._community_ids, self._community_embeddings, "community_summaries"
                        )
                        logger.info(
                            f"Embedded {len(self._community_ids)} community summaries on-the-fly"
                        )
                    else:
                        self._community_ids, self._community_embeddings = [], None

            if self._community_embeddings is None or len(self._community_ids) == 0:
                return ToolResult(
                    tool_name="global_search",
                    success=False,
                    data={},
                    message="No community embeddings available",
                )

            # Find top matching communities
            top_matches = self._embedding_generator.similarity_search(
                query, self._community_embeddings, self._community_ids, top_k=10
            )

            # Build community ID → data lookup
            comm_lookup = {c.get("community_id", ""): c for c in communities}

            # Collect source chunks from top matching communities
            source_chunk_ids = []
            source_communities = []
            for comm_id, score in top_matches:
                comm = comm_lookup.get(comm_id)
                if comm:
                    source_communities.append(comm_id)
                    source_chunk_ids.extend(comm.get("source_chunk_ids", [])[:5])

            source_chunk_ids = list(dict.fromkeys(source_chunk_ids))[:20]

            # Load source chunks for answer generation
            source_text_parts = []
            for cid in source_chunk_ids:
                chunk_data = self._load_chunk(cid)
                if chunk_data:
                    text = chunk_data.get("text_english") or chunk_data.get("text_anonymized", "")
                    if text:
                        source_text_parts.append(text)

            if not source_text_parts:
                return ToolResult(
                    tool_name="global_search",
                    success=True,
                    data={
                        "answer": "No relevant source evidence found.",
                        "points": [],
                        "source_chunk_ids": [],
                    },
                    message="No source chunks from matched communities",
                )

            # Single LLM call to generate answer from source chunks
            context = "\n\n---\n\n".join(source_text_parts[:10])
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt(
                            "retrieval", "graphrag_global_reduce", "system_prompt"
                        ),
                    },
                    {
                        "role": "user",
                        "content": get_prompt("retrieval", "graphrag_global_reduce", "user_prompt")
                        .replace("{query}", query)
                        .replace("{points}", context),
                    },
                ],
                temperature=0,
                max_tokens=get_prompt("retrieval", "graphrag_global_reduce", "max_tokens"),
            )

            from tacitgraph.llm_response import extract_llm_content

            answer = extract_llm_content(response, context="global search embedding")
            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="global_search",
                success=True,
                data={
                    "answer": answer,
                    "points": [
                        {
                            "point": f"Community {cid} (score: {score:.3f})",
                            "score": int(score * 100),
                        }
                        for cid, score in top_matches[:5]
                    ],
                    "source_chunk_ids": source_chunk_ids,
                    "source_communities": source_communities,
                },
                message=f"Global search (embedding): top {len(source_communities)} communities, {len(source_chunk_ids)} source chunks",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Embedding-based global search failed: {e}")
            return ToolResult(tool_name="global_search", success=False, data={}, message=str(e))

    def local_search(
        self,
        query: str,
        llm_client,
        model: str = None,
        top_entities: int = 10,
        max_relationships: int = 20,
    ) -> ToolResult:
        """
        GraphRAG Local Search — entity-centric context building.

        1. Find entities semantically related to query (via entity embeddings)
        2. Expand: connected entities, relationships, community reports, source chunks
        3. Build context and generate answer via LLM
        """
        from tacitgraph.prompt_loader import get_prompt

        model = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        start_time = datetime.now()

        try:
            graph = self._load_graph()

            # Step 1: Find seed entities via per-keyword embedding matching
            # Use LLM keyword extraction when available, fall back to word splitting
            try:
                from .hybrid_retriever import HybridRetriever

                temp = HybridRetriever.__new__(HybridRetriever)
                temp.mode = self.mode
                temp.llm_client = llm_client
                if self.mode == "llm" and llm_client:
                    keywords = temp._extract_keywords_llm(query)
                else:
                    keywords = query.split()
            except Exception:
                keywords = query.split()

            # Per-keyword matching: each keyword gets its own entity slot
            matched = []
            seen_ids = set()
            seen_names = set()
            for kw in keywords:
                kw_matches = self.node_retrieval([kw], top_n=3)
                for eid, ename in kw_matches:
                    if eid not in seen_ids and ename.lower() not in seen_names:
                        matched.append((eid, ename))
                        seen_ids.add(eid)
                        seen_names.add(ename.lower())
                        break
            # Fill remaining slots from all keywords
            if len(matched) < top_entities:
                all_matched = self.node_retrieval(keywords, top_n=top_entities * 2)
                for eid, ename in all_matched:
                    if eid not in seen_ids and ename.lower() not in seen_names:
                        matched.append((eid, ename))
                        seen_ids.add(eid)
                        seen_names.add(ename.lower())
                        if len(matched) >= top_entities:
                            break
            matched = matched[:top_entities]

            if not matched:
                return ToolResult(
                    tool_name="local_search",
                    success=False,
                    data={},
                    message="No matching entities found",
                )

            # node_retrieval returns (id, name) tuples
            entity_ids = [m[0] for m in matched]
            entity_names = [m[1] for m in matched]

            if not entity_ids:
                return ToolResult(
                    tool_name="local_search",
                    success=False,
                    data={},
                    message=f"Found {len(entity_names)} entities by name but none matched graph node IDs",
                )

            logger.info(
                f"Local search: {len(entity_ids)} seed entities (from {len(entity_names)} name matches)"
            )

            # Step 2: Build context from entities
            entities_text_parts = []
            relationships_text_parts = []

            # Collect source chunks with intersection ranking (same as PathRAG)
            chunk_counts = {}
            for node_id in entity_ids:
                node = graph.get_node(node_id)
                if not node:
                    continue

                # Entity details
                desc = node.properties.get("description", "")[:100] if node.properties else ""
                entities_text_parts.append(
                    f"- {node.name} ({node.node_type}){': ' + desc if desc else ''}"
                )
                for cid in node.source_chunks:
                    chunk_counts[cid] = chunk_counts.get(cid, 0) + 1

            # Chunks mentioned by most seed entities first
            source_chunk_ids = sorted(
                chunk_counts.keys(), key=lambda c: chunk_counts[c], reverse=True
            )[:20]

            # Connected relationships
            for node_id in entity_ids:
                node = graph.get_node(node_id)
                if not node:
                    continue
                rel_count = 0
                for edge_id, edge in graph.edges.items():
                    if rel_count >= max_relationships:
                        break
                    if edge.source_id == node_id or edge.target_id == node_id:
                        other_id = edge.target_id if edge.source_id == node_id else edge.source_id
                        other_node = graph.get_node(other_id)
                        other_name = other_node.name if other_node else other_id
                        edge_desc = (
                            edge.properties.get("description", "") if edge.properties else ""
                        )
                        relationships_text_parts.append(
                            f"- {node.name} --[{edge.edge_type}]--> {other_name}"
                            + (f" ({edge_desc})" if edge_desc else "")
                        )
                        rel_count += 1

            # Step 3: Load community reports for seed entities
            community_reports = []
            entity_to_comm = self._build_entity_to_community_index()
            seen_comms = set()
            for node_id in entity_ids:
                comm_id = entity_to_comm.get(node_id)
                if comm_id and comm_id not in seen_comms:
                    seen_comms.add(comm_id)
                    # Load community summary
                    for level_dir in sorted(self.gold_path.glob("communities/level_*")):
                        comm_file = level_dir / f"{comm_id}.json"
                        if comm_file.exists():
                            with open(comm_file, encoding="utf-8") as f:
                                comm_data = json.load(f)
                            community_reports.append(comm_data.get("summary", ""))
                            break

            # Step 4: Load source text from chunks
            source_text_parts = []
            for cid in list(source_chunk_ids)[:10]:
                chunk_data = self._load_chunk(cid)
                if chunk_data:
                    text = (
                        chunk_data.get("text_english")
                        or chunk_data.get("text_anonymized")
                        or chunk_data.get("summary", "")
                    )
                    if text:
                        source_text_parts.append(text)

            # Step 5: Build context and call LLM
            entities_text = "\n".join(entities_text_parts[:20]) or "(none)"
            relationships_text = "\n".join(relationships_text_parts[:30]) or "(none)"
            community_text = "\n\n".join(community_reports[:5]) or "(none)"
            source_text = "\n---\n".join(source_text_parts[:8]) or "(none)"

            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt(
                            "retrieval",
                            "graphrag_local",
                            "system_prompt",
                            "Answer using the provided knowledge graph context.",
                        ),
                    },
                    {
                        "role": "user",
                        "content": get_prompt(
                            "retrieval",
                            "graphrag_local",
                            "user_prompt",
                            "Query: {query}\nContext: {entities}",
                        )
                        .replace("{query}", query)
                        .replace("{entities}", entities_text)
                        .replace("{relationships}", relationships_text)
                        .replace("{community_reports}", community_text)
                        .replace("{source_text}", source_text),
                    },
                ],
                temperature=get_prompt("retrieval", "graphrag_local", "temperature"),
                max_tokens=get_prompt("retrieval", "graphrag_local", "max_tokens"),
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            answer = extract_llm_content(response, context="local search")
            execution_time = (datetime.now() - start_time).total_seconds()

            return ToolResult(
                tool_name="local_search",
                success=True,
                data={
                    "answer": answer,
                    "entities": entities_text,
                    "relationships": relationships_text,
                    "community_reports": community_text,
                    "source_text": source_text,
                    "source_chunk_ids": list(source_chunk_ids)[:20],
                    "entity_count": len(entity_ids),
                    "relationship_count": len(relationships_text_parts),
                    "community_count": len(seen_comms),
                },
                message=f"Local search: {len(entity_ids)} entities, {len(relationships_text_parts)} relationships",
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Local search failed: {e}")
            return ToolResult(tool_name="local_search", success=False, data={}, message=str(e))

    def _build_entity_to_community_index(self, level: int = 0) -> dict[str, str]:
        """Build reverse mapping from node_id to community_id. Cached."""
        if not hasattr(self, "_entity_to_community"):
            self._entity_to_community = {}
            communities_path = self.gold_path / "communities" / f"level_{level}"
            if communities_path.exists():
                for comm_file in communities_path.glob("*.json"):
                    with open(comm_file, encoding="utf-8") as f:
                        comm = json.load(f)
                    comm_id = comm.get("community_id", comm_file.stem)
                    for node_id in comm.get("node_ids", []):
                        self._entity_to_community[node_id] = comm_id
            logger.info(f"Entity-to-community index: {len(self._entity_to_community)} mappings")
        return self._entity_to_community

    def _load_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Load chunk data from Silver layer."""
        if not self.silver_path:
            return None

        for pattern in [
            "not_personal/email_chunks",
            "not_personal/attachment_chunks",
            "not_personal/document_chunks",
        ]:
            chunk_path = self.silver_path / pattern / f"{chunk_id}.json"
            if chunk_path.exists():
                with open(chunk_path, encoding="utf-8") as f:
                    return json.load(f)

        # Glob fallback
        for chunk_file in self.silver_path.glob(f"**/{chunk_id}.json"):
            with open(chunk_file, encoding="utf-8") as f:
                return json.load(f)

        return None

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """List all available tool names."""
        return list(self.tools.keys())

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function schemas for all tools."""
        return [tool.to_schema() for tool in self.tools.values()]

    def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with arguments."""
        tool = self.tools.get(name)
        if not tool:
            return ToolResult(
                tool_name=name, success=False, data=None, message=f"Tool '{name}' not found"
            )

        return tool.function(**kwargs)
