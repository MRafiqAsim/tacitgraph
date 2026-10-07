"""
Retrieval Tools Module

Defines tools available to the ReAct agent for knowledge retrieval.
Each tool wraps a specific retrieval strategy (PathRAG, GraphRAG, Vector, etc.)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from tacitgraph.retrieval.toolkit_graphrag import ToolkitGraphRAG
from tacitgraph.retrieval.toolkit_models import Tool, ToolResult

logger = logging.getLogger(__name__)


class RetrievalToolkit(ToolkitGraphRAG):
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
        chunk_ids: list[str] | None = None,
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
