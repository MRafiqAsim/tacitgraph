"""Loading of Gold-layer artifacts (graph, embeddings, chunks) and the LLM client."""

import json
import logging
from typing import Any

from tacitgraph.llm_client import chat_model, create_chat_client
from tacitgraph.retrieval.lexical_index import BM25Index
from tacitgraph.retrieval.toolkit_state import ToolkitState

logger = logging.getLogger(__name__)

CHUNK_DIRS = (
    "not_personal/email_chunks",
    "not_personal/attachment_chunks",
    "not_personal/document_chunks",
)


class ToolkitData(ToolkitState):
    """Loading of Gold-layer artifacts (graph, embeddings, chunks) and the LLM client."""

    def _get_llm_client(self):
        """Lazy-initialize LLM client for global/local search tools."""
        if self._llm_client is None:
            self._llm_client = create_chat_client()
            self._llm_model = chat_model()
        return self._llm_client, self._llm_model

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

            generator = EmbeddingGenerator(str(self.gold_path), mode=self.mode, match_index=True)
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

            generator = EmbeddingGenerator(str(self.gold_path), mode=self.mode, match_index=True)
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

    def _load_lexical_index(self) -> BM25Index:
        """Lazy-build a BM25 index over the embedded Silver chunks."""
        if self._lexical_index is None:
            chunk_ids, _ = self._load_embeddings()
            wanted = {cid.removesuffix("_sum") for cid in chunk_ids or []}
            docs: list[tuple[str, str]] = []
            if self.silver_path:
                for pattern in CHUNK_DIRS:
                    for chunk_file in sorted((self.silver_path / pattern).glob("*.json")):
                        if chunk_file.stem not in wanted:
                            continue
                        with open(chunk_file, encoding="utf-8") as f:
                            chunk = json.load(f)
                        fields = (
                            chunk.get("thread_subject"),
                            chunk.get("email_sender"),
                            chunk.get("source_attachment_filename"),
                            chunk.get("text_english") or chunk.get("text_anonymized"),
                            chunk.get("summary"),
                        )
                        docs.append((chunk_file.stem, " ".join(f for f in fields if f)))
            self._lexical_index = BM25Index(docs)
            logger.info(f"Keyword index built over {len(docs)} chunks")
        return self._lexical_index

    def _load_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Load chunk data from Silver layer."""
        if not self.silver_path:
            return None

        for pattern in CHUNK_DIRS:
            chunk_path = self.silver_path / pattern / f"{chunk_id}.json"
            if chunk_path.exists():
                with open(chunk_path, encoding="utf-8") as f:
                    return json.load(f)

        # Glob fallback
        for chunk_file in self.silver_path.glob(f"**/{chunk_id}.json"):
            with open(chunk_file, encoding="utf-8") as f:
                return json.load(f)

        return None

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
