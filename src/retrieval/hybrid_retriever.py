"""
Hybrid Retriever Module

Combines multiple retrieval strategies (PathRAG, GraphRAG, Vector, ReAct)
for comprehensive and accurate question answering.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from prompt_loader import get_prompt, get_section, format_prompt

from .retrieval_tools import RetrievalToolkit, ToolResult
from .react_retriever import ReActRetriever, ReActResult
from .date_filter import extract_date_range, filter_chunks_by_date, DateRange

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    VECTOR = "vector"
    PATHRAG = "pathrag"
    GRAPHRAG = "graphrag"
    HYBRID = "hybrid"
    REACT = "react"


@dataclass
class RetrievalResult:
    """Result from any retrieval strategy."""
    query: str
    answer: str
    chunks: List[Dict[str, Any]]
    strategy: str
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    is_grounded: bool = True
    missing_info: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "chunks": self.chunks,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "sources": self.sources,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "is_grounded": self.is_grounded,
            "missing_info": self.missing_info,
        }


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""
    # Strategy weights for fusion
    vector_weight: float = 0.3
    pathrag_weight: float = 0.4
    graphrag_weight: float = 0.3

    # Retrieval parameters
    top_k_per_strategy: int = 10
    final_top_k: int = 20
    min_confidence: float = 0.3

    # GraphRAG search mode: "auto" (query-based routing), "global", or "local"
    graphrag_search_type: str = ""

    # Answer generation
    use_llm_answer: bool = True
    answer_model: str = ""  # resolved from AZURE_OPENAI_DEPLOYMENT env var
    max_context_chunks: int = 20

    def __post_init__(self):
        if not self.answer_model:
            self.answer_model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        if not self.graphrag_search_type:
            self.graphrag_search_type = os.getenv("GRAPHRAG_SEARCH_TYPE", "auto")


class HybridRetriever:
    """
    Hybrid retriever that intelligently combines multiple strategies.

    Supports:
    1. Vector Search: Fast semantic similarity
    2. PathRAG: Multi-hop reasoning through entity paths
    3. GraphRAG: Community-based context retrieval
    4. Hybrid: Weighted fusion of all strategies
    5. ReAct: Autonomous agent with tool use
    """

    def __init__(
        self,
        gold_path: str,
        silver_path: Optional[str] = None,
        config: Optional[HybridConfig] = None,
        mode: str = "llm",
    ):
        """
        Initialize the hybrid retriever.

        Args:
            gold_path: Path to Gold layer
            silver_path: Path to Silver layer
            config: Retrieval configuration
            mode: Processing mode — "local" uses local models, "llm" uses Azure OpenAI
        """
        self.gold_path = Path(gold_path)
        self.silver_path = Path(silver_path) if silver_path else None
        self.config = config or HybridConfig()
        self.mode = mode

        # Initialize toolkit
        self.toolkit = RetrievalToolkit(str(gold_path), str(silver_path) if silver_path else None, mode=mode)

        # Initialize ReAct retriever — shares toolkit with other strategies
        self.react_retriever = ReActRetriever(str(gold_path), str(silver_path) if silver_path else None, mode=mode, toolkit=self.toolkit)

        # Load entity catalog for alias resolution
        catalog_path = self.gold_path / "entity_catalog.json"
        if catalog_path.exists():
            from entity_registry import load_catalog
            load_catalog(str(catalog_path))
            logger.info(f"Loaded entity catalog from {catalog_path}")

        # LLM client for answer generation (always try — even local mode uses LLM for answers)
        self.llm_client = None
        self._initialize_llm()

        logger.info(f"HybridRetriever initialized (mode={mode})")

    def _initialize_llm(self):
        """Initialize LLM for answer generation."""
        try:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_key = os.getenv("AZURE_OPENAI_API_KEY")

            if azure_endpoint and azure_key:
                from openai import AzureOpenAI
                self.llm_client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                )
                # answer_model already resolved from env in HybridConfig.__post_init__
                return

            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=openai_key)
                return

        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        conversation_history: str = "",
    ) -> RetrievalResult:
        """
        Retrieve relevant information for a query.

        Args:
            query: The user's question
            strategy: Which retrieval strategy to use
            conversation_history: Formatted conversation history for answer generation

        Returns:
            RetrievalResult with answer and supporting chunks
        """
        start_time = datetime.now()

        # Store context for use during this call
        self._conversation_history = conversation_history
        self._date_range = extract_date_range(query)
        if self._date_range:
            logger.info(f"Temporal filter detected: {self._date_range}")

        if strategy == RetrievalStrategy.VECTOR:
            result = self._vector_retrieve(query)
        elif strategy == RetrievalStrategy.PATHRAG:
            result = self._pathrag_retrieve(query)
        elif strategy == RetrievalStrategy.GRAPHRAG:
            result = self._graphrag_retrieve(query)
        elif strategy == RetrievalStrategy.REACT:
            result = self._react_retrieve(query)
        else:  # HYBRID
            result = self._hybrid_retrieve(query)

        self._conversation_history = ""
        self._date_range = None
        result.execution_time = (datetime.now() - start_time).total_seconds()
        return result

    def _vector_retrieve(self, query: str) -> RetrievalResult:
        """Pure vector similarity retrieval."""
        tool_result = self.toolkit.vector_search(query, top_k=self.config.top_k_per_strategy)

        chunks = tool_result.data if tool_result.success else []

        # Thread expansion: fetch sibling chunks from same threads
        scored_chunks = list(chunks) if chunks else []
        if chunks:
            all_chunks = self._expand_by_thread(chunks)

            # Inherit parent thread's score for expanded chunks
            thread_best_score = {}
            for c in scored_chunks:
                tid = c.get("thread_id")
                score = c.get("similarity_score", 0)
                if tid and score > thread_best_score.get(tid, 0):
                    thread_best_score[tid] = score
            for c in all_chunks:
                if c.get("_expanded") and c.get("similarity_score", 0) == 0:
                    tid = c.get("thread_id")
                    if tid in thread_best_score:
                        c["similarity_score"] = thread_best_score[tid] * 0.9

            all_chunks.sort(key=lambda c: c.get("similarity_score", 0), reverse=True)
        else:
            all_chunks = []

        all_chunks = self._apply_temporal_filter(all_chunks)
        final = all_chunks[:self.config.final_top_k]

        # Generate answer if configured
        answer = ""
        is_grounded = True
        missing_info = None
        gen_tokens = 0
        if self.config.use_llm_answer and final:
            answer, is_grounded, missing_info, gen_tokens = self._generate_answer(query, final[:self.config.max_context_chunks])

        return RetrievalResult(
            query=query,
            answer=answer,
            chunks=final,
            strategy="vector",
            confidence=self._calculate_confidence(final),
            sources=[c.get("chunk_id", "") for c in final],
            metadata={"tool_message": tool_result.message, "total_tokens": gen_tokens},
            is_grounded=is_grounded,
            missing_info=missing_info,
        )

    def _pathrag_retrieve(self, query: str) -> RetrievalResult:
        """PathRAG: Entity path-based retrieval."""
        # Extract entities from query — returns (IDs, names) directly
        entity_ids, entity_names = self._extract_query_entities(query)

        if not entity_ids:
            return self._vector_retrieve(query)

        # Find paths using IDs directly — no re-matching needed
        path_result = self.toolkit.pathrag_search(
            entity_ids=entity_ids,
            entity_names=entity_names,
            max_paths=self.config.top_k_per_strategy,
        )

        if not path_result.success or not path_result.data:
            return self._vector_retrieve(query)

        # Collect evidence chunks from paths
        chunk_ids = set()
        for path in path_result.data:
            chunk_ids.update(path.get("evidence_chunks", []))

        # Load chunk details
        all_chunks = []
        for chunk_id in chunk_ids:
            chunk_result = self.toolkit.get_chunk_context(chunk_id)
            if chunk_result.success and chunk_result.data:
                all_chunks.append(chunk_result.data)

        # Rank evidence chunks by semantic similarity to the query
        if all_chunks and len(all_chunks) > self.config.top_k_per_strategy:
            try:
                generator = self.toolkit._embedding_generator
                if generator is None:
                    from gold.embedding_generator import EmbeddingGenerator, EmbeddingConfig
                    self.toolkit._embedding_generator = EmbeddingGenerator(
                        str(self.gold_path), EmbeddingConfig(), mode=self.mode
                    )
                    generator = self.toolkit._embedding_generator

                import numpy as np
                query_emb = generator.embed_text(query)
                if query_emb:
                    query_vec = np.array(query_emb)
                    query_norm = query_vec / np.linalg.norm(query_vec)

                    for chunk in all_chunks:
                        chunk_text = chunk.get("text", "")
                        chunk_emb = generator.embed_text(chunk_text)
                        if chunk_emb:
                            chunk_vec = np.array(chunk_emb)
                            chunk_norm = chunk_vec / np.linalg.norm(chunk_vec)
                            chunk["similarity_score"] = float(np.dot(query_norm, chunk_norm))
                        else:
                            chunk["similarity_score"] = 0.0

                    all_chunks.sort(key=lambda c: c.get("similarity_score", 0), reverse=True)
                    logger.info(
                        f"PathRAG chunk re-ranking: {len(all_chunks)} chunks, "
                        f"top score {all_chunks[0].get('similarity_score', 0):.3f}"
                    )
            except Exception as e:
                logger.warning(f"PathRAG chunk re-ranking failed: {e}")

        chunks = all_chunks[:self.config.top_k_per_strategy]
        chunks = self._apply_temporal_filter(chunks)

        # Generate answer
        answer = ""
        is_grounded = True
        missing_info = None
        gen_tokens = 0
        if self.config.use_llm_answer and chunks:
            # Include path information in context
            path_context = "\n".join([
                f"Path: {p.get('description', '')}"
                for p in path_result.data[:3]
            ])
            answer, is_grounded, missing_info, gen_tokens = self._generate_answer(
                query,
                chunks[:self.config.max_context_chunks],
                extra_context=f"Reasoning paths found:\n{path_context}"
            )

        return RetrievalResult(
            query=query,
            answer=answer,
            chunks=chunks,
            strategy="pathrag",
            confidence=self._calculate_confidence(chunks),
            sources=[c.get("chunk_id", "") for c in chunks],
            metadata={
                "paths_found": len(path_result.data),
                "entities_used": entity_names,
                "total_tokens": gen_tokens,
            },
            is_grounded=is_grounded,
            missing_info=missing_info,
        )

    def _detect_aggregate_type(self, query: str) -> Optional[List[str]]:
        """Detect if query asks for a list of entities and return matching entity types.

        Returns a list of entity types to try (primary + related fallbacks).
        For example, 'projects' → ['PRODUCT', 'DOCUMENT', 'CONCEPT'] because
        emails may label work items differently than the user's terminology.
        """
        import re
        query_lower = query.lower()

        # Map user terms to primary + fallback entity types
        type_patterns = {
            r"(project|initiative|task|work item|topic|subject)": ["PRODUCT", "DOCUMENT", "CONCEPT"],
            r"(product|system|tool|software|application|platform)": ["PRODUCT"],
            r"(people|person|employee|team member|participant|who)": ["PERSON"],
            r"(organization|company|department|team|vendor|supplier|client)": ["ORG"],
            r"(event|meeting|milestone|deadline|conference)": ["EVENT"],
            r"(document|report|file|attachment|specification)": ["DOCUMENT"],
            r"(location|city|country|office|place|region)": ["GPE"],
        }

        # Check for listing/aggregate intent
        if not re.search(r"(list|all|every|what\b.*\b(are|names|types)|provide|show|give me|how many|which|discussed|mentioned)", query_lower):
            return None

        for pattern, entity_types in type_patterns.items():
            if re.search(pattern, query_lower):
                return entity_types

        return None

    def _graphrag_retrieve(self, query: str) -> RetrievalResult:
        """GraphRAG retrieval using Global Search (map-reduce) or Local Search.

        Raw evidence from GraphRAG is passed through the unified _generate_answer()
        prompt so that all strategies are compared on equal footing.
        """
        import time
        start = time.time()

        if not self.llm_client:
            logger.warning("GraphRAG requires LLM — no LLM available")
            return RetrievalResult(
                query=query, answer="GraphRAG requires an LLM connection.",
                chunks=[], strategy="graphrag", confidence=0.0,
            )

        # Route: global (aggregate/broad) vs local (specific entity)
        if self.config.graphrag_search_type == "auto":
            search_type = self.toolkit.route_graphrag_query(query)
        else:
            search_type = self.config.graphrag_search_type
            logger.info(f"GraphRAG search type forced to: {search_type}")
        model = self.config.answer_model

        if search_type == "global":
            result = self.toolkit.global_search(
                query,
                llm_client=self.llm_client,
                model=model,
                level=0,
            )
        else:
            result = self.toolkit.local_search(
                query,
                llm_client=self.llm_client,
                model=model,
            )

        # If first search failed or returned no evidence, try the other type
        has_evidence = result.success and (
            result.data.get("source_chunk_ids")
            or result.data.get("points")
            or result.data.get("entities")
            or result.data.get("source_text")
        )
        if not has_evidence:
            alt_type = "local" if search_type == "global" else "global"
            logger.info(f"GraphRAG {search_type} found no evidence — trying {alt_type}")
            if alt_type == "global":
                result = self.toolkit.global_search(query, llm_client=self.llm_client, model=model, level=0)
            else:
                result = self.toolkit.local_search(query, llm_client=self.llm_client, model=model)
            search_type = alt_type

        # Load source chunks for citation
        source_chunk_ids = result.data.get("source_chunk_ids", [])

        chunks = []
        for chunk_id in source_chunk_ids[:self.config.top_k_per_strategy]:
            chunk_result = self.toolkit.get_chunk_context(chunk_id)
            if chunk_result.success and chunk_result.data:
                chunks.append(chunk_result.data)

        chunks = self._apply_temporal_filter(chunks)

        # No community summaries or LLM-generated context passed to answer generation.
        # Only actual source chunks are used — keeps answers precise and citable.
        extra_context = ""
        points = result.data.get("points", [])

        # If GraphRAG found no evidence, return a clear message
        if not chunks and not extra_context:
            logger.info("GraphRAG found no evidence for this query")
            execution_time = time.time() - start
            return RetrievalResult(
                query=query,
                answer="GraphRAG could not find relevant information for this query in the knowledge graph communities. Try rephrasing or use a different strategy like Vector or Hybrid.",
                chunks=[], strategy=f"graphrag_{search_type}",
                confidence=0.0, sources=[],
                metadata={"search_type": search_type, "total_tokens": 0},
                execution_time=execution_time,
            )

        # Generate answer through the unified prompt
        answer = ""
        is_grounded = True
        missing_info = None
        gen_tokens = 0
        if self.config.use_llm_answer and (chunks or extra_context):
            answer, is_grounded, missing_info, gen_tokens = self._generate_answer(
                query,
                chunks[:self.config.max_context_chunks],
                extra_context=extra_context,
            )

        execution_time = time.time() - start

        return RetrievalResult(
            query=query,
            answer=answer,
            chunks=chunks,
            strategy=f"graphrag_{search_type}",
            confidence=self._calculate_confidence(chunks),
            sources=[c.get("chunk_id", "") for c in chunks],
            metadata={
                "search_type": search_type,
                "communities_used": result.data.get("source_communities", []),
                "points_count": len(points),
                "entity_count": result.data.get("entity_count", 0),
                "tool_message": result.message,
                "total_tokens": gen_tokens,
            },
            is_grounded=is_grounded,
            missing_info=missing_info,
            execution_time=execution_time,
        )

    def _hybrid_retrieve(self, query: str) -> RetrievalResult:
        """Hybrid: Weighted fusion of all strategies."""
        # Run all strategies
        vector_result = self._vector_retrieve(query)
        pathrag_result = self._pathrag_retrieve(query)
        graphrag_result = self._graphrag_retrieve(query)

        # Collect and score chunks
        chunk_scores: Dict[str, Tuple[Dict, float]] = {}

        # Add vector chunks
        for i, chunk in enumerate(vector_result.chunks):
            chunk_id = chunk.get("chunk_id", f"vec_{i}")
            score = self.config.vector_weight * (1.0 - i * 0.1)  # Decay by position
            if chunk.get("similarity_score"):
                score *= chunk["similarity_score"]
            chunk_scores[chunk_id] = (chunk, chunk_scores.get(chunk_id, (chunk, 0))[1] + score)

        # Add PathRAG chunks
        for i, chunk in enumerate(pathrag_result.chunks):
            chunk_id = chunk.get("chunk_id", f"path_{i}")
            score = self.config.pathrag_weight * (1.0 - i * 0.1)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] = (chunk, chunk_scores[chunk_id][1] + score)
            else:
                chunk_scores[chunk_id] = (chunk, score)

        # Add GraphRAG chunks
        for i, chunk in enumerate(graphrag_result.chunks):
            chunk_id = chunk.get("chunk_id", f"graph_{i}")
            score = self.config.graphrag_weight * (1.0 - i * 0.1)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] = (chunk, chunk_scores[chunk_id][1] + score)
            else:
                chunk_scores[chunk_id] = (chunk, score)

        # Sort by combined score
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top-k
        final_chunks = [chunk for chunk, score in sorted_chunks[:self.config.final_top_k]]

        # Thread expansion: fetch sibling chunks (email body + attachments) from same threads
        all_chunks_for_llm = self._expand_by_thread(final_chunks)
        all_chunks_for_llm = self._apply_temporal_filter(all_chunks_for_llm)

        # Inherit parent thread's score for expanded chunks so they sort meaningfully.
        thread_best_score = {}
        for c in final_chunks:
            tid = c.get("thread_id")
            score = c.get("similarity_score", 0)
            if tid and score > thread_best_score.get(tid, 0):
                thread_best_score[tid] = score
        for c in all_chunks_for_llm:
            if c.get("_expanded") and c.get("similarity_score", 0) == 0:
                tid = c.get("thread_id")
                if tid in thread_best_score:
                    c["similarity_score"] = thread_best_score[tid] * 0.9

        all_chunks_for_llm.sort(key=lambda c: c.get("similarity_score", 0), reverse=True)
        final_chunks = all_chunks_for_llm[:self.config.final_top_k]

        # Generate answer
        answer = ""
        is_grounded = True
        missing_info = None
        gen_tokens = 0
        if self.config.use_llm_answer and final_chunks:
            # Include context from all strategies
            extra_context = ""

            # For aggregate queries, include entity listing + thread subjects
            aggregate_type = self._detect_aggregate_type(query)
            if aggregate_type:
                for etype in aggregate_type:
                    entity_list_result = self.toolkit.list_entities(etype)
                    if entity_list_result.success and entity_list_result.data:
                        entity_names = [e["name"] for e in entity_list_result.data]
                        extra_context += (
                            f"{etype} entities ({len(entity_names)} total):\n"
                            + "\n".join(f"- {name}" for name in entity_names)
                            + "\n\n"
                        )
                # Thread subjects are deliberately not injected: they bias the LLM
                # toward subject-line keyword matches instead of the scored results.

            if pathrag_result.metadata.get("paths_found", 0) > 0:
                extra_context += f"Found {pathrag_result.metadata['paths_found']} reasoning paths.\n"
            # Community summaries removed from answer context — only source chunks used

            answer, is_grounded, missing_info, gen_tokens = self._generate_answer(
                query,
                final_chunks[:self.config.max_context_chunks],
                extra_context=extra_context
            )

        return RetrievalResult(
            query=query,
            answer=answer,
            chunks=final_chunks,
            strategy="hybrid",
            confidence=self._calculate_confidence(final_chunks),
            sources=[c.get("chunk_id", "") for c in final_chunks],
            metadata={
                "vector_chunks": len(vector_result.chunks),
                "pathrag_chunks": len(pathrag_result.chunks),
                "graphrag_chunks": len(graphrag_result.chunks),
                "fusion_scores": {k: v[1] for k, v in list(chunk_scores.items())[:10]},
                "total_tokens": gen_tokens,
            },
            is_grounded=is_grounded,
            missing_info=missing_info,
        )

    def _react_retrieve(self, query: str) -> RetrievalResult:
        """ReAct: Autonomous agent retrieval.

        The ReAct agent handles tool selection and evidence gathering.
        The final answer is generated through the unified _generate_answer()
        prompt so that all strategies are compared on equal footing.
        """
        react_result = self.react_retriever.query(query)

        # Convert ReAct sources to chunks
        chunks = []
        seen_chunk_ids = set()
        for source in react_result.sources:
            chunk_id = source.get("chunk_id", "")
            if source.get("type") == "chunk" and chunk_id and chunk_id not in seen_chunk_ids:
                chunk_result = self.toolkit.get_chunk_context(chunk_id)
                if chunk_result.success and chunk_result.data:
                    chunk_data = chunk_result.data
                    chunk_data["similarity_score"] = source.get("similarity_score", 0.5)
                    chunks.append(chunk_data)
                    seen_chunk_ids.add(chunk_id)

        # Fallback: if the agent produced an answer but no chunk sources
        # (e.g. used only list_entities/graphrag), run a vector search for evidence
        if react_result.answer and not chunks:
            logger.info("ReAct produced answer but no chunk sources — running fallback vector search")
            vector_result = self.toolkit.vector_search(query, top_k=10)
            if vector_result.success and vector_result.data:
                for item in vector_result.data:
                    cid = item.get("chunk_id", "")
                    if cid and cid not in seen_chunk_ids:
                        chunk_result = self.toolkit.get_chunk_context(cid)
                        if chunk_result.success and chunk_result.data:
                            chunk_data = chunk_result.data
                            chunk_data["similarity_score"] = item.get("similarity_score", 0.0)
                            chunks.append(chunk_data)
                            seen_chunk_ids.add(cid)

        # Compute real similarity scores for chunks that only have default 0.5
        # (from local_search/graphrag) so they rank fairly against vector_search chunks
        unscored = [c for c in chunks if c.get("similarity_score", 0) == 0.5]
        if unscored:
            try:
                generator = self.toolkit._embedding_generator
                if generator is None:
                    from gold.embedding_generator import EmbeddingGenerator, EmbeddingConfig
                    self.toolkit._embedding_generator = EmbeddingGenerator(
                        str(self.gold_path), EmbeddingConfig(), mode=self.mode
                    )
                    generator = self.toolkit._embedding_generator

                import numpy as np
                query_emb = generator.embed_text(query)
                if query_emb:
                    query_vec = np.array(query_emb)
                    query_norm = query_vec / np.linalg.norm(query_vec)
                    for chunk in unscored:
                        chunk_text = chunk.get("text", "")
                        chunk_emb = generator.embed_text(chunk_text)
                        if chunk_emb:
                            chunk_vec = np.array(chunk_emb)
                            chunk_norm = chunk_vec / np.linalg.norm(chunk_vec)
                            chunk["similarity_score"] = float(np.dot(query_norm, chunk_norm))
            except Exception as e:
                logger.warning(f"ReAct chunk scoring failed: {e}")

        chunks.sort(key=lambda c: c.get("similarity_score", 0), reverse=True)
        chunks = self._apply_temporal_filter(chunks)

        # No ReAct observations as extra context — answer generated from source chunks only.
        # Observations may contain LLM-generated summaries that could bias the answer.
        extra_context = ""

        # Generate answer through the unified prompt
        answer = ""
        is_grounded = True
        missing_info = None
        gen_tokens = 0
        if self.config.use_llm_answer and chunks:
            answer, is_grounded, missing_info, gen_tokens = self._generate_answer(
                query,
                chunks[:self.config.max_context_chunks],
            )
        elif not chunks:
            tools_tried = [s.action for s in react_result.steps if s.action]
            if tools_tried:
                tools_str = ", ".join(dict.fromkeys(tools_tried))  # unique, ordered
                answer = (
                    f"I searched using {tools_str} but could not find relevant "
                    f"source evidence to answer this question. The topic may not be "
                    f"covered in the indexed emails, or it may appear under a different name."
                )
            else:
                answer = "I could not find relevant source evidence to answer this question."
            is_grounded = False

        return RetrievalResult(
            query=query,
            answer=answer,
            chunks=chunks,
            strategy="react",
            confidence=self._calculate_confidence(chunks),
            sources=[s.get("chunk_id", s.get("community_id", s.get("path_id", "")))
                    for s in react_result.sources],
            metadata={
                "steps": len(react_result.steps),
                "total_tokens": react_result.total_tokens + gen_tokens,
                "reasoning_trace": [s.to_dict() for s in react_result.steps]
            },
            is_grounded=is_grounded,
            missing_info=missing_info,
        )

    def _get_all_thread_subjects(self) -> List[str]:
        """Get all unique thread subjects from silver layer.

        NOTE: Currently unused. Injecting all subjects as LLM context caused
        the model to surface wrong emails based on subject-line keyword matches
        rather than relying on scored retrieval results.
        """
        if not self.silver_path:
            return []

        subjects = set()
        silver = Path(self.silver_path)
        for folder in ["not_personal/email_chunks",
                       "not_personal/attachment_chunks",
                       "not_personal/document_chunks"]:
            folder_path = silver / folder
            if not folder_path.exists():
                continue
            for f in folder_path.glob("*.json"):
                try:
                    with open(f, 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                    subj = data.get("thread_subject", "")
                    if subj:
                        subjects.add(subj)
                except Exception:
                    pass
        return list(subjects)

    def _extract_query_entities(self, query: str, top_n: int = 10) -> Tuple[List[str], List[str]]:
        """
        PathRAG Node Retrieval (Stage 1 per paper).

        Per-keyword matching: each original keyword gets its own best entity match,
        guaranteeing diverse representation. Then remaining slots are filled from
        expanded aliases.

        Returns:
            Tuple of (entity_ids, entity_names)
        """
        # Extract original keywords (before expansion)
        if self.mode == "llm":
            original_keywords = self._extract_keywords_llm(query)
        else:
            original_keywords = self._extract_keywords_local(query)

        if not original_keywords:
            return [], []

        logger.info(f"Original keywords: {original_keywords}")

        # Per-keyword matching: find best entity for EACH keyword individually
        # This guarantees "Lisbon" gets a slot even when "migration" has higher scores
        matched = []
        seen_ids = set()
        seen_names = set()

        for kw in original_keywords:
            # Match this single keyword
            kw_matches = self.toolkit.node_retrieval([kw], top_n=3)
            for entity_id, name in kw_matches:
                if entity_id not in seen_ids and name.lower() not in seen_names:
                    matched.append((entity_id, name))
                    seen_ids.add(entity_id)
                    seen_names.add(name.lower())
                    logger.info(f"  '{kw}' → {name}")
                    break  # one entity per keyword

        logger.info(f"Per-keyword matches: {len(matched)} entities from {len(original_keywords)} keywords")

        # Fill remaining slots from expanded aliases
        if len(matched) < top_n:
            from entity_registry import expand_entity_aliases
            expanded = []
            for kw in original_keywords:
                expanded.extend(expand_entity_aliases(kw))
            all_keywords = list(set(expanded))

            expanded_matches = self.toolkit.node_retrieval(all_keywords, top_n=top_n * 2)
            for entity_id, name in expanded_matches:
                if entity_id not in seen_ids and name.lower() not in seen_names:
                    matched.append((entity_id, name))
                    seen_ids.add(entity_id)
                    seen_names.add(name.lower())
                    if len(matched) >= top_n:
                        break

        matched = matched[:top_n]
        if matched:
            entity_ids = [m[0] for m in matched]
            entity_names = [m[1] for m in matched]
            logger.info(f"PathRAG node retrieval: {len(entity_ids)} entities: {entity_names}")
            return entity_ids, entity_names

        return [], []



    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query with alias expansion.

        LLM mode: GPT-4o extracts entity names and keywords.
        Local mode: spaCy NER + content-word heuristics.
        Both modes: resolve aliases via entity catalog.

        Note: _extract_query_entities uses _extract_keywords_llm/_extract_keywords_local
        directly for two-phase matching. This method is used by other callers that
        need the full expanded keyword list.
        """
        if self.mode == "llm":
            keywords = self._extract_keywords_llm(query)
        else:
            keywords = self._extract_keywords_local(query)

        from entity_registry import expand_entity_aliases
        logger.info(f"Keywords before expansion: {keywords}")
        expanded = []
        for kw in keywords:
            expanded.extend(expand_entity_aliases(kw))
        result = list(set(expanded))
        logger.info(f"Keywords after expansion: {result}")
        return result

    def _extract_keywords_llm(self, query: str) -> List[str]:
        """Extract keywords using GPT-4o."""
        import json as _json

        try:
            from entity_registry import get_all_entity_types
            entity_types = list(get_all_entity_types())
            prompt_template = get_prompt("retrieval", "query_keyword_extraction", "user_prompt")
            system_prompt = format_prompt(
                get_prompt("retrieval", "query_keyword_extraction", "system_prompt"),
                entity_types=entity_types,
            )
            user_prompt = format_prompt(prompt_template, query=query)

            from llm_response import extract_llm_content
            if not self.llm_client:
                return self._extract_keywords_local(query)
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

            response = self.llm_client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=200,
            )

            content = extract_llm_content(response, context="keyword extraction")

            import re
            if content.startswith("```"):
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    content = match.group()

            keywords = _json.loads(content)
            if isinstance(keywords, list):
                return [str(k) for k in keywords if k]
        except Exception as e:
            logger.warning(f"LLM keyword extraction failed, falling back to local: {e}")

        return self._extract_keywords_local(query)

    def _extract_keywords_local(self, query: str) -> List[str]:
        """Extract keywords using spaCy NER + content word heuristics."""
        import re

        keywords = []

        # 1. Quoted phrases (highest priority — explicit user intent)
        quoted = re.findall(r'"([^"]+)"', query)
        keywords.extend(quoted)

        # 2. spaCy NER extraction (local, no LLM needed)
        try:
            import spacy
            if not hasattr(self, '_nlp'):
                self._nlp = spacy.load("en_core_web_sm")
            doc = self._nlp(query)
            for ent in doc.ents:
                keywords.append(ent.text)
        except Exception:
            pass  # spaCy not available, fall through to heuristics

        # 3. Content words (capitalized, acronyms, meaningful terms)
        stop_words = {
            'the', 'a', 'an', 'what', 'who', 'where', 'when', 'how', 'why', 'which',
            'did', 'does', 'do', 'is', 'are', 'was', 'were', 'and', 'or', 'but',
            'about', 'tell', 'me', 'can', 'you', 'find', 'show', 'get', 'give',
            'provide', 'list', 'all', 'every', 'any', 'some',
            'project', 'projects', 'email', 'emails', 'thread', 'threads',
            'information', 'details', 'data', 'discussed', 'mentioned',
            'people', 'person', 'things', 'stuff', 'work', 'used',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'by',
        }

        words = query.split()
        content_words = []
        for word in words:
            clean = re.sub(r'[?.!,;:]$', '', word)
            if clean.lower() in stop_words or len(clean) < 2:
                continue
            content_words.append(clean)
            # Capitalized words and acronyms as separate keywords
            if clean.isupper() and len(clean) >= 2:
                keywords.append(clean)
            elif clean[0:1].isupper() and len(clean) > 1:
                keywords.append(clean)

        # 4. Bigrams/trigrams from content words
        if len(content_words) >= 2:
            for i in range(len(content_words)):
                for j in range(i + 2, min(i + 4, len(content_words) + 1)):
                    keywords.append(" ".join(content_words[i:j]))

        # Add individual content words
        keywords.extend(content_words)

        return list(set(keywords))

    def _match_graph_entities_by_name(self, candidates: List[str]) -> List[str]:
        """Fallback: match candidate strings against known graph entity names."""
        if not self.gold_path:
            return candidates

        # Load graph entity names (cached)
        if not hasattr(self, '_graph_entity_names'):
            self._graph_entity_names = {}
            graph_file = Path(self.gold_path) / "knowledge_graph" / "nodes.json"
            if graph_file.exists():
                try:
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        graph_data = json.load(f)
                    for node_id, node in graph_data.items():
                        name = node.get("name", node_id)
                        node_type = node.get("node_type", node.get("type", ""))
                        if name and node_type not in ("CHUNK", "THREAD", ""):
                            self._graph_entity_names[name.lower()] = name
                except Exception:
                    pass

        if not self._graph_entity_names:
            return candidates

        matched = []
        for candidate in candidates:
            candidate_lower = candidate.lower()
            if candidate_lower in self._graph_entity_names:
                matched.append(self._graph_entity_names[candidate_lower])
                continue
            for graph_lower, graph_name in self._graph_entity_names.items():
                if candidate_lower in graph_lower or graph_lower in candidate_lower:
                    matched.append(graph_name)

        return list(set(matched))

    def _expand_by_thread(
        self,
        chunks: List[Dict[str, Any]],
        max_siblings: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Expand retrieval results by fetching sibling chunks and summaries
        from the same thread.

        When a chunk is retrieved, also pull in:
        1. Sibling email/attachment chunks from the same thread
        2. Thread summary (if available)
        3. Attachment summaries (if available)

        This gives the LLM full context regardless of which entry point was hit.
        """
        if not self.silver_path:
            return chunks

        existing_ids = {c.get("chunk_id") for c in chunks}

        # Rank threads by their best chunk score, expand only top 3
        thread_best_score = {}
        for chunk in chunks:
            tid = chunk.get("thread_id")
            score = chunk.get("similarity_score", 0)
            if tid and score > thread_best_score.get(tid, 0):
                thread_best_score[tid] = score

        if not thread_best_score:
            return chunks

        # Only expand the top 3 most relevant threads
        sorted_threads = sorted(thread_best_score.items(), key=lambda x: x[1], reverse=True)
        thread_ids = {tid for tid, _ in sorted_threads[:3]}

        # 1. Search Silver layer for sibling chunks
        search_dirs = [
            self.silver_path / "not_personal" / "email_chunks",
            self.silver_path / "not_personal" / "attachment_chunks",
            self.silver_path / "not_personal" / "document_chunks",
        ]

        sibling_chunks = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for chunk_file in search_dir.glob("*.json"):
                if len(sibling_chunks) >= max_siblings:
                    break
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                    cid = chunk_data.get("chunk_id")
                    tid = chunk_data.get("thread_id")
                    if tid in thread_ids and cid not in existing_ids:
                        sibling_chunks.append({
                            "chunk_id": cid,
                            "text": chunk_data.get("summary") or chunk_data.get("text_english") or chunk_data.get("text_anonymized", ""),
                            "text_english": chunk_data.get("text_english", ""),
                            "summary": chunk_data.get("summary", ""),
                            "thread_id": tid,
                            "thread_subject": chunk_data.get("thread_subject"),
                            "source_type": chunk_data.get("source_type", "email"),
                            "source_attachment_filename": chunk_data.get("source_attachment_filename", ""),
                            "has_attachments": chunk_data.get("has_attachments", False),
                            "email_sender": chunk_data.get("email_sender", ""),
                            "sent_timestamp": chunk_data.get("sent_timestamp", ""),
                            "similarity_score": 0.0,
                            "_expanded": True,
                        })
                        existing_ids.add(cid)
                except Exception:
                    continue

        # 2. Load thread summaries for matched threads
        summary_context = []
        thread_summaries_dir = self.silver_path / "not_personal" / "thread_summaries"
        if thread_summaries_dir.exists():
            for summary_file in thread_summaries_dir.glob("*.json"):
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                    if summary_data.get("thread_id") in thread_ids:
                        summary_text = summary_data.get("summary", "")
                        if summary_text:
                            summary_context.append({
                                "chunk_id": f"summary_{summary_data['thread_id']}",
                                "text": f"[Thread Summary] {summary_data.get('subject', '')}: {summary_text}",
                                "thread_id": summary_data["thread_id"],
                                "source_type": "thread_summary",
                                "similarity_score": 0.0,
                                "_expanded": True,
                            })
                            # 3. Follow cross-references to attachment summaries
                            for att_id in summary_data.get("attachment_ids", []):
                                att_summary = self._load_attachment_summary(att_id)
                                if att_summary:
                                    summary_context.append({
                                        "chunk_id": f"att_summary_{att_id}",
                                        "text": f"[Attachment: {att_summary.get('filename', '')}] {att_summary.get('summary', '')}",
                                        "thread_id": summary_data["thread_id"],
                                        "source_type": "attachment_summary",
                                        "similarity_score": 0.0,
                                        "_expanded": True,
                                    })
                except Exception:
                    continue

        # Per-email summaries are not added: sibling chunks + thread summary suffice.

        expanded_count = len(sibling_chunks) + len(summary_context)
        if expanded_count:
            logger.info(
                f"Thread expansion: added {len(sibling_chunks)} sibling chunks, "
                f"{len(summary_context)} summaries from {len(thread_ids)} threads"
            )

        return list(chunks) + sibling_chunks + summary_context

    def _load_attachment_summary(self, attachment_id: str) -> Optional[Dict[str, Any]]:
        """Load an attachment summary by ID from Silver layer."""
        if not self.silver_path:
            return None
        summary_dir = self.silver_path / "not_personal" / "attachment_summaries"
        if not summary_dir.exists():
            return None
        # Try exact match and sanitized match
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in attachment_id)[:100]
        for candidate in (f"{attachment_id}.json", f"{safe_id}.json"):
            path = summary_dir / candidate
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    pass
        return None

    def _apply_temporal_filter(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter chunks by date range if a temporal query was detected."""
        date_range = getattr(self, "_date_range", None)
        if not date_range or not date_range.is_set:
            return chunks
        before = len(chunks)
        filtered = filter_chunks_by_date(chunks, date_range)
        if len(filtered) < before:
            logger.info(f"Temporal filter: {before} → {len(filtered)} chunks ({date_range})")
        return filtered

    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence as average cosine similarity of retrieved chunks.
        Excludes expanded/sibling chunks (score=0) that weren't directly matched."""
        if not chunks:
            return 0.0

        scores = [c.get("similarity_score", 0) for c in chunks if c.get("similarity_score", 0) > 0]
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        extra_context: str = ""
    ) -> Tuple[str, bool, Optional[str], int]:
        """
        Generate answer using LLM with grounding check.
        Falls back to extractive summary in local mode.

        Returns:
            (answer, is_grounded, missing_info, total_tokens)
        """
        if not chunks and not extra_context:
            return "", True, None, 0

        # Local mode: use BART summarizer or extractive fallback
        if not self.llm_client:
            answer, is_grounded, missing_info = self._generate_local_answer(query, chunks, extra_context)
            return answer, is_grounded, missing_info, 0


        # Build context from chunks — label email vs attachment for LLM clarity
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", chunk.get("summary") or chunk.get("text_english") or chunk.get("text_anonymized", ""))
            chunk_id = chunk.get("chunk_id", "unknown")
            thread = chunk.get("thread_subject", "")
            source_type = chunk.get("source_type", "email")
            # Include email date so LLM can reference it in answers
            recv = chunk.get("received_timestamp", "")
            sent = chunk.get("sent_timestamp", "")
            if recv:
                date_str = f" | Received: {recv[:10]}"
            elif sent:
                date_str = f" | Sent: {sent[:10]}"
            else:
                date_str = ""

            sender = chunk.get("email_sender", "")
            sender_str = f" | From: {sender}" if sender else ""

            if source_type == "attachment":
                filename = chunk.get("source_attachment_filename", "unknown")
                label = f"[Source {i}] [{chunk_id}] (Thread: {thread}{date_str}{sender_str} | Attachment: {filename})"
            else:
                label = f"[Source {i}] [{chunk_id}] (Thread: {thread}{date_str}{sender_str} | Email body)"

            context_parts.append(f"{label}\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        if extra_context:
            context = f"{extra_context}\n\n---\n\nEvidence:\n{context}"

        # Inject entity alias hints so LLM connects query terms with chunk content
        from entity_registry import expand_entity_aliases
        alias_hints = []
        for word in query.split():
            clean = word.strip("?,.:;!()\"'")
            if len(clean) >= 2:
                aliases = expand_entity_aliases(clean)
                if len(aliases) > 1:
                    alias_hints.append(f'"{clean}" is also known as: {", ".join(a for a in aliases if a != clean)}')
        if alias_hints:
            context = "Entity aliases (treat these as the same entity):\n" + "\n".join(alias_hints) + "\n\n---\n\n" + context

        # Format prompts from config/prompts.json
        system_prompt = get_prompt("retrieval", "generation", "system_prompt")
        user_prompt = format_prompt(
            get_prompt("retrieval", "generation", "user_prompt"),
            context=context,
            question=query,
        )

        # Build messages — include conversation history if available
        messages = [{"role": "system", "content": system_prompt}]
        conv_history = getattr(self, "_conversation_history", "")
        if conv_history:
            messages.append({"role": "user", "content": f"Conversation so far:\n{conv_history}"})
            messages.append({"role": "assistant", "content": "I have the conversation context. Please provide the new question and evidence."})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.answer_model,
                messages=messages,
                temperature=get_prompt("retrieval", "generation", "temperature"),
                max_tokens=get_prompt("retrieval", "generation", "max_tokens"),
            )
            # extract_llm_content raises LLMContentError for content_filter / length
            from llm_response import extract_llm_content, LLMContentError
            answer = extract_llm_content(response, context="answer generation")
            total_tokens = getattr(response.usage, 'total_tokens', 0) if response.usage else 0
        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, network timeouts, etc.
            logger.error(f"Answer generation failed: {e}")
            return "", True, None, 0

        # Grounding check
        is_grounded = True
        missing_info: Optional[str] = None

        missing_indicators = get_prompt("retrieval", "generation", "missing_indicators", [
            "don't have enough information",
            "not in the context",
            "cannot find",
            "no information",
            "not mentioned",
        ])
        for indicator in missing_indicators:
            if indicator.lower() in answer.lower():
                is_grounded = False
                missing_info = "Required information not found in knowledge base"
                break

        return answer, is_grounded, missing_info, total_tokens

    def _check_grounding(self, answer: str) -> Tuple[bool, Optional[str]]:
        """Check if an answer indicates missing information."""
        from prompt_loader import get_prompt
        missing_indicators = get_prompt("retrieval", "generation", "missing_indicators", [
            "don't have enough information",
            "not in the context",
            "cannot find",
            "no information",
            "not mentioned",
        ])
        for indicator in missing_indicators:
            if indicator.lower() in answer.lower():
                return False, "Required information not found in knowledge base"
        return True, None

    def _generate_local_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        extra_context: str = ""
    ) -> Tuple[str, bool, Optional[str]]:
        """Generate answer locally using BART summarizer or extractive fallback."""
        # Combine chunk texts
        texts = []
        for chunk in chunks[:5]:
            text = chunk.get("text", chunk.get("summary") or chunk.get("text_english") or chunk.get("text_anonymized", ""))
            thread = chunk.get("thread_subject", "")
            if thread:
                texts.append(f"[{thread}] {text}")
            else:
                texts.append(text)

        combined = "\n\n".join(texts)

        if extra_context:
            combined = f"{extra_context}\n\n{combined}"

        # Try BART summarizer
        try:
            from silver.local_summarizer import summarize_text
            summary = summarize_text(combined, max_length=200, min_length=50)
            if summary:
                return summary, True, None
        except Exception as e:
            logger.debug(f"BART summarization failed: {e}")

        # Extractive fallback: return first few chunk texts
        answer_parts = []
        for chunk in chunks[:3]:
            text = chunk.get("text", chunk.get("summary") or chunk.get("text_english") or chunk.get("text_anonymized", ""))
            thread = chunk.get("thread_subject", "")
            if thread:
                answer_parts.append(f"**{thread}**: {text}")
            else:
                answer_parts.append(text)

        return "\n\n".join(answer_parts), True, None

    def compare_strategies(self, query: str) -> Dict[str, RetrievalResult]:
        """
        Run all strategies and compare results.

        Useful for evaluation and analysis.
        """
        results = {}

        for strategy in RetrievalStrategy:
            if strategy != RetrievalStrategy.HYBRID:  # Hybrid includes others
                result = self.retrieve(query, strategy)
                results[strategy.value] = result

        return results
