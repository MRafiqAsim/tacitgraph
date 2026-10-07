"""
Cosmos DB Adapter for fast querying.

Two APIs:
  - Gremlin API: Knowledge graph (nodes, edges, paths) — graph traversals
  - NoSQL API: Chunks, communities, thread summaries — document lookups

Usage:
    cosmos = CosmosAdapter.from_env()

    # Graph queries (Gremlin)
    node = cosmos.get_node("person_001_org")
    neighbors = cosmos.get_neighbors("person_001_org", direction="both", edge_type="RELATED_TO")
    paths = cosmos.find_paths("person_001_org", "org_003_org", max_hops=3)

    # Document queries (NoSQL)
    chunk = cosmos.get_chunk("abc123")
    chunks = cosmos.get_chunks_by_thread("thread_42")
    community = cosmos.get_community("comm_0_1")
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class CosmosAdapter:
    """
    Unified Cosmos DB client for graph and document queries.

    Gremlin API → knowledge graph traversals
    NoSQL API → chunk/community/summary lookups
    """

    def __init__(
        self,
        gremlin_endpoint: str | None = None,
        gremlin_key: str | None = None,
        gremlin_database: str = "email-kg",
        gremlin_graph: str = "knowledge-graph",
        nosql_endpoint: str | None = None,
        nosql_key: str | None = None,
        nosql_database: str = "email-kg",
    ):
        self.gremlin_endpoint = gremlin_endpoint
        self.gremlin_key = gremlin_key
        self.gremlin_database = gremlin_database
        self.gremlin_graph = gremlin_graph
        self.nosql_endpoint = nosql_endpoint
        self.nosql_key = nosql_key
        self.nosql_database = nosql_database

        self._gremlin_client = None
        self._nosql_client = None
        self._nosql_db = None

        # Container references (lazy initialized)
        self._containers: dict[str, Any] = {}

    @classmethod
    def from_env(cls) -> "CosmosAdapter":
        """Create from environment variables."""
        return cls(
            gremlin_endpoint=os.getenv("COSMOS_GREMLIN_ENDPOINT"),
            gremlin_key=os.getenv("COSMOS_GREMLIN_KEY"),
            gremlin_database=os.getenv("COSMOS_DATABASE", "email-kg"),
            gremlin_graph=os.getenv("COSMOS_GRAPH", "knowledge-graph"),
            nosql_endpoint=os.getenv("COSMOS_NOSQL_ENDPOINT"),
            nosql_key=os.getenv("COSMOS_NOSQL_KEY"),
            nosql_database=os.getenv("COSMOS_DATABASE", "email-kg"),
        )

    @property
    def is_configured(self) -> bool:
        """Check if Cosmos DB is configured."""
        return bool(self.gremlin_endpoint or self.nosql_endpoint)

    # =========================================================================
    # Gremlin Client (Knowledge Graph)
    # =========================================================================

    def _get_gremlin_client(self):
        """Lazy initialize Gremlin client."""
        if self._gremlin_client is None:
            from gremlin_python.driver import client as gremlin_client
            from gremlin_python.driver import serializer

            self._gremlin_client = gremlin_client.Client(
                url=self.gremlin_endpoint,
                traversal_source="g",
                username=f"/dbs/{self.gremlin_database}/colls/{self.gremlin_graph}",
                password=self.gremlin_key,
                message_serializer=serializer.GraphSONSerializersV2d0(),
            )
            logger.info(f"Gremlin client connected: {self.gremlin_endpoint}")
        return self._gremlin_client

    def _gremlin_query(self, query: str) -> list[Any]:
        """Execute a Gremlin query and return results."""
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            pass
        client = self._get_gremlin_client()
        result_set = client.submitAsync(query).result()
        return result_set.all().result()

    def _gremlin_query_fire_and_forget(self, query: str) -> None:
        """Submit a Gremlin query without waiting for full result parsing.
        Used for bulk writes where we only care about success/failure."""
        client = self._get_gremlin_client()
        future = client.submitAsync(query)
        # Wait for completion but don't parse results
        future.result()

    @staticmethod
    def _safe_gremlin_str(value: str) -> str:
        """Escape a string for safe use in Gremlin queries.
        Replaces backslashes with forward slashes, escapes single quotes."""
        return value.replace("\\", "/").replace("'", "\\'")

    # =========================================================================
    # NoSQL Client (Documents)
    # =========================================================================

    def _get_nosql_db(self):
        """Lazy initialize NoSQL database client."""
        if self._nosql_db is None:
            if not self.nosql_endpoint or not self.nosql_key:
                logger.debug("Cosmos NoSQL not configured — using AI Search or local files")
                return None
            from azure.cosmos import CosmosClient

            try:
                client = CosmosClient(self.nosql_endpoint, credential=self.nosql_key)
                self._nosql_db = client.get_database_client(self.nosql_database)
                logger.info(f"NoSQL client connected: {self.nosql_endpoint}")
            except Exception as e:
                logger.warning(
                    f"Cosmos NoSQL connection failed: {e} — falling back to AI Search or local files"
                )
                return None
        return self._nosql_db

    def _get_container(self, name: str):
        """Get or create a container reference."""
        if name not in self._containers:
            db = self._get_nosql_db()
            if db is None:
                return None
            self._containers[name] = db.get_container_client(name)
        return self._containers[name]

    # =========================================================================
    # WRITE — Graph (called from Gold layer during indexing)
    # =========================================================================

    def _build_node_query(
        self, node_id: str, name: str, node_type: str, properties: dict | None = None
    ) -> str:
        """Build Gremlin upsert query for a node."""
        props = properties or {}
        safe = self._safe_gremlin_str
        prop_str = ""
        for k, v in props.items():
            if isinstance(v, (int, float)):
                prop_str += f".property('{k}', {v})"
            elif isinstance(v, str):
                prop_str += f".property('{k}', '{safe(v)}')"
            elif isinstance(v, list):
                clean_list = [safe(item) if isinstance(item, str) else item for item in v]
                safe_v = json.dumps(clean_list, ensure_ascii=False).replace("'", "\\'")
                prop_str += f".property('{k}', '{safe_v}')"

        safe_name = safe(name)
        return (
            f"g.V('{node_id}').fold().coalesce("
            f"unfold().property('name', '{safe_name}'){prop_str},"
            f"addV('{node_type}').property('id', '{node_id}').property('node_type', '{node_type}').property('name', '{safe_name}'){prop_str}"
            f")"
        )

    def _build_edge_query(
        self, source_id: str, target_id: str, edge_type: str, properties: dict | None = None
    ) -> str:
        """Build Gremlin upsert query for an edge."""
        props = properties or {}
        safe = self._safe_gremlin_str
        prop_str = ""
        for k, v in props.items():
            if isinstance(v, (int, float)):
                prop_str += f".property('{k}', {v})"
            elif isinstance(v, str):
                prop_str += f".property('{k}', '{safe(v)}')"

        return (
            f"g.V('{source_id}').as('s')"
            f".V('{target_id}').as('t')"
            f".select('s').coalesce("
            f"outE('{edge_type}').where(inV().hasId('{target_id}')),"
            f"addE('{edge_type}').to('t')"
            f"){prop_str}"
        )

    def upsert_node(
        self, node_id: str, name: str, node_type: str, properties: dict | None = None
    ) -> None:
        """Add or update a graph node."""
        query = self._build_node_query(node_id, name, node_type, properties)
        self._gremlin_query(query)

    def upsert_edge(
        self, source_id: str, target_id: str, edge_type: str, properties: dict | None = None
    ) -> None:
        """Add or update a graph edge."""
        query = self._build_edge_query(source_id, target_id, edge_type, properties)
        self._gremlin_query(query)

    def upsert_path(
        self,
        path_id: str,
        source_id: str,
        target_id: str,
        path_nodes: list[str],
        path_edges: list[str],
        description: str = "",
        weight: float = 1.0,
    ) -> None:
        """Store a PathRAG path as a Gremlin edge with metadata."""
        safe = self._safe_gremlin_str
        query = (
            f"g.V('{source_id}').as('s')"
            f".V('{target_id}').as('t')"
            f".select('s').addE('PATHRAG_PATH').to('t')"
            f".property('path_id', '{path_id}')"
            f".property('path_nodes', '{safe(json.dumps(path_nodes))}')"
            f".property('path_edges', '{safe(json.dumps(path_edges))}')"
            f".property('description', '{safe(description)}')"
            f".property('weight', {weight})"
        )
        self._gremlin_query(query)

    def bulk_upsert_nodes(
        self, nodes: list[dict], batch_size: int = 50, max_workers: int = 10
    ) -> int:
        """Bulk upsert nodes with concurrent execution. Returns count."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        count = 0
        failed = 0
        threading.Lock()

        def _upsert_one_node(node):
            try:
                query = self._build_node_query(
                    node_id=node["node_id"],
                    name=node["name"],
                    node_type=node["node_type"],
                    properties=node.get("properties", {}),
                )
                self._gremlin_query(query)
                return True
            except Exception as e:
                logger.warning(f"Failed to upsert node {node.get('node_id')}: {e}")
                return False

        total = len(nodes)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i in range(0, total, batch_size * max_workers):
                batch = nodes[i : i + batch_size * max_workers]
                futures = {pool.submit(_upsert_one_node, n): n for n in batch}
                for future in as_completed(futures):
                    if future.result():
                        count += 1
                    else:
                        failed += 1
                logger.info(f"  Nodes: {count + failed}/{total} ({count} ok, {failed} failed)")

        return count

    def bulk_upsert_edges(
        self, edges: list[dict], batch_size: int = 50, max_workers: int = 10
    ) -> int:
        """Bulk upsert edges with concurrent execution. Returns count."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        count = 0
        failed = 0

        def _upsert_one_edge(edge):
            try:
                query = self._build_edge_query(
                    source_id=edge["source_id"],
                    target_id=edge["target_id"],
                    edge_type=edge["edge_type"],
                    properties=edge.get("properties", {}),
                )
                self._gremlin_query(query)
                return True
            except Exception as e:
                logger.warning(f"Failed to upsert edge: {e}")
                return False

        total = len(edges)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i in range(0, total, batch_size * max_workers):
                batch = edges[i : i + batch_size * max_workers]
                futures = {pool.submit(_upsert_one_edge, e): e for e in batch}
                for future in as_completed(futures):
                    if future.result():
                        count += 1
                    else:
                        failed += 1
                logger.info(f"  Edges: {count + failed}/{total} ({count} ok, {failed} failed)")

        return count

    def drop_all_vertices(self, batch_size: int = 500) -> int:
        """Drop all vertices in batches. Edges are deleted automatically.
        Returns total vertices dropped."""
        total_dropped = 0
        while True:
            result = self._gremlin_query("g.V().count()")
            remaining = result[0] if result else 0
            if remaining == 0:
                break
            self._gremlin_query(f"g.V().limit({batch_size}).drop()")
            total_dropped += min(batch_size, remaining)
            logger.info(f"  Dropped batch — ~{remaining} remaining")
        logger.info(f"Graph cleared — {total_dropped} vertices dropped")
        return total_dropped

    # =========================================================================
    # WRITE — NoSQL Documents (called from Gold/Silver layer)
    # =========================================================================

    def upsert_chunk(self, chunk: dict) -> None:
        """Store a Silver chunk in NoSQL."""
        container = self._get_container("chunks")
        chunk["id"] = chunk.get("chunk_id", chunk.get("id"))
        chunk["partitionKey"] = chunk.get("thread_id", "unknown")
        container.upsert_item(chunk)

    def upsert_community(self, community: dict) -> None:
        """Store a community summary in NoSQL."""
        container = self._get_container("communities")
        community["id"] = community.get("community_id", community.get("id"))
        community["partitionKey"] = str(community.get("level", 0))
        container.upsert_item(community)

    def upsert_thread_summary(self, summary: dict) -> None:
        """Store a thread summary in NoSQL."""
        container = self._get_container("thread_summaries")
        summary["id"] = summary.get("thread_id", summary.get("id"))
        summary["partitionKey"] = summary.get("thread_id", "unknown")
        container.upsert_item(summary)

    def bulk_upsert_documents(
        self,
        container_name: str,
        docs: list[dict],
        partition_key_field: str = "partitionKey",
        batch_size: int = 100,
    ) -> int:
        """Bulk upsert documents to a NoSQL container."""
        container = self._get_container(container_name)
        count = 0
        for doc in docs:
            try:
                container.upsert_item(doc)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert doc {doc.get('id')}: {e}")
        logger.info(f"Upserted {count}/{len(docs)} to {container_name}")
        return count

    # =========================================================================
    # READ — Graph Queries (used by retrieval at query time)
    # =========================================================================

    def get_node(self, node_id: str) -> dict | None:
        """Get a single node by ID."""
        results = self._gremlin_query(f"g.V('{node_id}').valueMap(true)")
        if results:
            return self._parse_gremlin_node(results[0])
        return None

    def search_nodes(
        self, name_substring: str, node_type: str | None = None, limit: int = 10
    ) -> list[dict]:
        """Search nodes by name substring."""
        safe_name = name_substring.lower().replace("'", "\\'")
        if node_type:
            query = (
                f"g.V().has('node_type', '{node_type}')"
                f".filter{{it.get().value('name').toLowerCase().contains('{safe_name}')}}"
                f".limit({limit}).valueMap(true)"
            )
        else:
            query = (
                f"g.V().filter{{it.get().value('name').toLowerCase().contains('{safe_name}')}}"
                f".limit({limit}).valueMap(true)"
            )
        results = self._gremlin_query(query)
        return [self._parse_gremlin_node(r) for r in results]

    def get_neighbors(
        self, node_id: str, direction: str = "both", edge_type: str | None = None, limit: int = 20
    ) -> list[dict]:
        """
        Get neighboring nodes.

        Args:
            node_id: Source node
            direction: "out", "in", or "both"
            edge_type: Filter by edge type (optional)
            limit: Max neighbors to return
        """
        dir_fn = {"out": "out", "in": "in", "both": "both"}[direction]
        edge_filter = f"('{edge_type}')" if edge_type else "()"
        query = f"g.V('{node_id}').{dir_fn}{edge_filter}.limit({limit}).valueMap(true)"
        results = self._gremlin_query(query)
        return [self._parse_gremlin_node(r) for r in results]

    def get_edges_for_node(
        self, node_id: str, direction: str = "both", limit: int = 50
    ) -> list[dict]:
        """Get edges connected to a node with full details."""
        edges = []

        if direction in ("out", "both"):
            query = (
                f"g.V('{node_id}').outE().limit({limit})"
                f".project('edge_type', 'target_id', 'target_name', 'properties')"
                f".by(label()).by(inV().id()).by(inV().values('name')).by(valueMap())"
            )
            results = self._gremlin_query(query)
            for r in results:
                edges.append(
                    {
                        "direction": "outgoing",
                        "edge_type": r.get("edge_type", ""),
                        "target_id": r.get("target_id", ""),
                        "target_name": r.get("target_name", ""),
                        "properties": r.get("properties", {}),
                    }
                )

        if direction in ("in", "both"):
            query = (
                f"g.V('{node_id}').inE().limit({limit})"
                f".project('edge_type', 'source_id', 'source_name', 'properties')"
                f".by(label()).by(outV().id()).by(outV().values('name')).by(valueMap())"
            )
            results = self._gremlin_query(query)
            for r in results:
                edges.append(
                    {
                        "direction": "incoming",
                        "edge_type": r.get("edge_type", ""),
                        "source_id": r.get("source_id", ""),
                        "source_name": r.get("source_name", ""),
                        "properties": r.get("properties", {}),
                    }
                )

        return edges

    def find_paths(self, source_id: str, target_id: str, max_hops: int = 3) -> list[dict]:
        """Find paths between two nodes using Gremlin traversal."""
        query = (
            f"g.V('{source_id}').repeat(both().simplePath())"
            f".until(hasId('{target_id}').or().loops().is({max_hops}))"
            f".hasId('{target_id}').path().by(valueMap(true))"
            f".limit(10)"
        )
        results = self._gremlin_query(query)
        paths = []
        for path_result in results:
            nodes = [self._parse_gremlin_node(n) for n in path_result.get("objects", path_result)]
            paths.append({"nodes": nodes, "hop_count": len(nodes) - 1})
        return paths

    def get_pathrag_paths(self, source_id: str, target_id: str) -> list[dict]:
        """Get pre-computed PathRAG paths between two entities."""
        query = (
            f"g.V('{source_id}').outE('PATHRAG_PATH')"
            f".where(inV().hasId('{target_id}'))"
            f".valueMap(true)"
        )
        results = self._gremlin_query(query)
        paths = []
        for r in results:
            paths.append(
                {
                    "path_id": self._first(r.get("path_id")),
                    "path_nodes": json.loads(self._first(r.get("path_nodes", "[]"))),
                    "path_edges": json.loads(self._first(r.get("path_edges", "[]"))),
                    "description": self._first(r.get("description", "")),
                    "weight": self._first(r.get("weight", 1.0)),
                }
            )
        return paths

    def get_node_count(self, node_type: str | None = None) -> int:
        """Count nodes, optionally by type."""
        query = f"g.V().has('node_type', '{node_type}').count()" if node_type else "g.V().count()"
        results = self._gremlin_query(query)
        return results[0] if results else 0

    def get_edge_count(self) -> int:
        """Count all edges."""
        results = self._gremlin_query("g.E().count()")
        return results[0] if results else 0

    def list_node_types(self) -> list[str]:
        """Get all distinct node types."""
        results = self._gremlin_query("g.V().values('node_type').dedup()")
        return results

    # =========================================================================
    # READ — NoSQL Queries (used by retrieval at query time)
    # =========================================================================

    def get_chunk(self, chunk_id: str) -> dict | None:
        """Get a chunk by ID."""
        container = self._get_container("chunks")
        if container is None:
            return None
        try:
            # Cross-partition query since we may not know the thread_id
            query = f"SELECT * FROM c WHERE c.chunk_id = '{chunk_id}'"
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            return items[0] if items else None
        except Exception as e:
            logger.warning(f"Failed to get chunk {chunk_id}: {e}")
            return None

    def get_chunks_by_thread(self, thread_id: str) -> list[dict]:
        """Get all chunks for a thread (fast — uses partition key)."""
        container = self._get_container("chunks")
        if container is None:
            return []
        query = "SELECT * FROM c WHERE c.thread_id = @thread_id"
        items = list(
            container.query_items(
                query=query,
                parameters=[{"name": "@thread_id", "value": thread_id}],
                partition_key=thread_id,
            )
        )
        return items

    def get_chunks_by_entity(self, entity_name: str, limit: int = 20) -> list[dict]:
        """Get chunks that mention a specific entity."""
        container = self._get_container("chunks")
        if container is None:
            return []
        safe_name = entity_name.replace("'", "''")
        query = (
            f"SELECT TOP {limit} * FROM c "
            f"WHERE ARRAY_CONTAINS(c.kg_entities, {{'text': '{safe_name}'}}, true)"
        )
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        return items

    def get_community(self, community_id: str, level: int = 0) -> dict | None:
        """Get a community by ID."""
        container = self._get_container("communities")
        if container is None:
            return None
        try:
            return container.read_item(item=community_id, partition_key=str(level))
        except Exception:
            return None

    def get_communities_by_level(self, level: int = 0) -> list[dict]:
        """Get all communities at a specific resolution level."""
        container = self._get_container("communities")
        if container is None:
            return []
        query = "SELECT * FROM c WHERE c.level = @level"
        items = list(
            container.query_items(
                query=query,
                parameters=[{"name": "@level", "value": level}],
                partition_key=str(level),
            )
        )
        return items

    def get_communities_for_entity(self, node_id: str) -> list[dict]:
        """Get communities that contain a specific entity."""
        container = self._get_container("communities")
        query = f"SELECT * FROM c WHERE ARRAY_CONTAINS(c.node_ids, '{node_id}')"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        return items

    def get_thread_summary(self, thread_id: str) -> dict | None:
        """Get a thread summary."""
        container = self._get_container("thread_summaries")
        if container is None:
            return None
        try:
            return container.read_item(item=thread_id, partition_key=thread_id)
        except Exception:
            return None

    def search_chunks_by_text(self, keyword: str, limit: int = 20) -> list[dict]:
        """Full-text search across chunks (basic CONTAINS query)."""
        container = self._get_container("chunks")
        if container is None:
            return []
        safe_kw = keyword.replace("'", "''")
        query = (
            f"SELECT TOP {limit} * FROM c "
            f"WHERE CONTAINS(LOWER(c.summary), LOWER('{safe_kw}')) "
            f"OR CONTAINS(LOWER(c.text_english), LOWER('{safe_kw}'))"
        )
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        return items

    # =========================================================================
    # Graph Statistics
    # =========================================================================

    def get_graph_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "node_count": self.get_node_count(),
            "edge_count": self.get_edge_count(),
            "node_types": self.list_node_types(),
        }

    # =========================================================================
    # Database Setup (run once)
    # =========================================================================

    def create_nosql_containers(self) -> None:
        """Create NoSQL containers with appropriate partition keys."""
        db = self._get_nosql_db()
        if db is None:
            logger.warning("Cosmos NoSQL not available — skipping container creation")
            return

        containers = [
            {"id": "chunks", "partition_key": "/thread_id", "indexing": "consistent"},
            {"id": "communities", "partition_key": "/level", "indexing": "consistent"},
            {"id": "thread_summaries", "partition_key": "/thread_id", "indexing": "consistent"},
        ]

        for spec in containers:
            try:
                from azure.cosmos import PartitionKey

                db.create_container_if_not_exists(
                    id=spec["id"],
                    partition_key=PartitionKey(path=spec["partition_key"]),
                )
                logger.info(f"Container '{spec['id']}' ready (partition: {spec['partition_key']})")
            except Exception as e:
                logger.warning(f"Container '{spec['id']}' setup: {e}")

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _parse_gremlin_node(raw: dict) -> dict:
        """Parse Gremlin valueMap result into flat dict."""
        parsed = {}
        for key, value in raw.items():
            if key in ("id", "T.id"):
                parsed["node_id"] = value
            elif key in ("label", "T.label"):
                parsed["label"] = value
            elif isinstance(value, list) and len(value) == 1:
                parsed[key] = value[0]
            else:
                parsed[key] = value
        return parsed

    @staticmethod
    def _first(value):
        """Extract first element if list, else return as-is."""
        if isinstance(value, list) and value:
            return value[0]
        return value

    def close(self):
        """Close connections."""
        if self._gremlin_client:
            self._gremlin_client.close()
            self._gremlin_client = None
