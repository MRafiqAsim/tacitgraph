"""
Hybrid Retriever Module

Combines multiple retrieval strategies (PathRAG, GraphRAG, Vector, ReAct)
for comprehensive and accurate question answering.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from tacitgraph.retrieval.answer_generation import AnswerGeneration

from .date_filter import extract_date_range
from .react_retriever import ReActRetriever
from .retrieval_tools import RetrievalToolkit

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
    chunks: list[dict[str, Any]]
    strategy: str
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    is_grounded: bool = True
    missing_info: str | None = None

    def to_dict(self) -> dict[str, Any]:
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


class HybridRetriever(AnswerGeneration):
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
        silver_path: str | None = None,
        config: HybridConfig | None = None,
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
        self.toolkit = RetrievalToolkit(
            str(gold_path), str(silver_path) if silver_path else None, mode=mode
        )

        # Initialize ReAct retriever — shares toolkit with other strategies
        self.react_retriever = ReActRetriever(
            str(gold_path),
            str(silver_path) if silver_path else None,
            mode=mode,
            toolkit=self.toolkit,
        )

        # Load entity catalog for alias resolution
        catalog_path = self.gold_path / "entity_catalog.json"
        if catalog_path.exists():
            from tacitgraph.entity_registry import load_catalog

            load_catalog(str(catalog_path))
            logger.info(f"Loaded entity catalog from {catalog_path}")

        # LLM client for answer generation (always try — even local mode uses LLM for answers)
        self.llm_client = None
        self._initialize_llm()

        logger.info(f"HybridRetriever initialized (mode={mode})")

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
        final = all_chunks[: self.config.final_top_k]

        # Generate answer if configured
        answer = ""
        is_grounded = True
        missing_info = None
        gen_tokens = 0
        if self.config.use_llm_answer and final:
            answer, is_grounded, missing_info, gen_tokens = self._generate_answer(
                query, final[: self.config.max_context_chunks]
            )

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
                    from tacitgraph.gold.embedding_generator import (
                        EmbeddingConfig,
                        EmbeddingGenerator,
                    )

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

        chunks = all_chunks[: self.config.top_k_per_strategy]
        chunks = self._apply_temporal_filter(chunks)

        # Generate answer
        answer = ""
        is_grounded = True
        missing_info = None
        gen_tokens = 0
        if self.config.use_llm_answer and chunks:
            # Include path information in context
            path_context = "\n".join(
                [f"Path: {p.get('description', '')}" for p in path_result.data[:3]]
            )
            answer, is_grounded, missing_info, gen_tokens = self._generate_answer(
                query,
                chunks[: self.config.max_context_chunks],
                extra_context=f"Reasoning paths found:\n{path_context}",
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
                query=query,
                answer="GraphRAG requires an LLM connection.",
                chunks=[],
                strategy="graphrag",
                confidence=0.0,
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
                result = self.toolkit.global_search(
                    query, llm_client=self.llm_client, model=model, level=0
                )
            else:
                result = self.toolkit.local_search(query, llm_client=self.llm_client, model=model)
            search_type = alt_type

        # Load source chunks for citation
        source_chunk_ids = result.data.get("source_chunk_ids", [])

        chunks = []
        for chunk_id in source_chunk_ids[: self.config.top_k_per_strategy]:
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
                chunks=[],
                strategy=f"graphrag_{search_type}",
                confidence=0.0,
                sources=[],
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
                chunks[: self.config.max_context_chunks],
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
        chunk_scores: dict[str, tuple[dict, float]] = {}

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
        sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x[1], reverse=True)

        # Take top-k
        final_chunks = [chunk for chunk, score in sorted_chunks[: self.config.final_top_k]]

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
        final_chunks = all_chunks_for_llm[: self.config.final_top_k]

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
                extra_context += (
                    f"Found {pathrag_result.metadata['paths_found']} reasoning paths.\n"
                )
            # Community summaries removed from answer context — only source chunks used

            answer, is_grounded, missing_info, gen_tokens = self._generate_answer(
                query, final_chunks[: self.config.max_context_chunks], extra_context=extra_context
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
            logger.info(
                "ReAct produced answer but no chunk sources — running fallback vector search"
            )
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
                    from tacitgraph.gold.embedding_generator import (
                        EmbeddingConfig,
                        EmbeddingGenerator,
                    )

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

        # Generate answer through the unified prompt
        answer = ""
        is_grounded = True
        missing_info = None
        gen_tokens = 0
        if self.config.use_llm_answer and chunks:
            answer, is_grounded, missing_info, gen_tokens = self._generate_answer(
                query,
                chunks[: self.config.max_context_chunks],
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
            sources=[
                s.get("chunk_id", s.get("community_id", s.get("path_id", "")))
                for s in react_result.sources
            ],
            metadata={
                "steps": len(react_result.steps),
                "total_tokens": react_result.total_tokens + gen_tokens,
                "reasoning_trace": [s.to_dict() for s in react_result.steps],
            },
            is_grounded=is_grounded,
            missing_info=missing_info,
        )

    def compare_strategies(self, query: str) -> dict[str, RetrievalResult]:
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
