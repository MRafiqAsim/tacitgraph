"""GraphRAG search: query routing, global search over community summaries and local search around entities."""

import json
import logging
import os
from datetime import datetime

from tacitgraph.llm_client import chat_model
from tacitgraph.retrieval.toolkit_entities import ToolkitEntitySearch
from tacitgraph.retrieval.toolkit_models import ToolResult

logger = logging.getLogger(__name__)


class ToolkitGraphRAG(ToolkitEntitySearch):
    """GraphRAG search: query routing, global search over community summaries and local search around entities."""

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
        model: str | None = None,
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

        model = model or chat_model()
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
                        str(self.gold_path), EmbeddingConfig(), mode=self.mode, match_index=True
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
            for comm_id, _score in top_matches:
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
        model: str | None = None,
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

        model = model or chat_model()
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
            chunk_counts: dict[str, int] = {}
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
                for _edge_id, edge in graph.edges.items():
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
