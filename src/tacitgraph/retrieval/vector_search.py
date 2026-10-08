"""
Vector Search Module
====================
Azure AI Search integration with hybrid search capabilities.

Features:
- HNSW vector index (3072 dimensions for text-embedding-3-large)
- Hybrid search (vector + keyword)
- Semantic ranking
- Filtering by metadata
- Batch indexing with retry logic

"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result."""

    chunk_id: str
    content: str
    score: float
    reranker_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    highlights: str | None = None


@dataclass
class SearchResponse:
    """Search response with multiple results."""

    results: list[SearchResult]
    query: str
    total_count: int
    search_time_ms: float
    search_type: str  # vector, keyword, hybrid


@dataclass
class VectorSearchConfig:
    """Configuration for Azure AI Search."""

    # Index settings
    index_name: str = os.environ.get("AZURE_SEARCH_INDEX_NAME", "email-chunks")
    vector_dimensions: int = 3072  # text-embedding-3-large

    # HNSW parameters
    hnsw_m: int = 4  # Bi-directional links per node
    hnsw_ef_construction: int = 400  # Size of dynamic candidate list
    hnsw_ef_search: int = 500  # Size of dynamic candidate list for search

    # Search settings
    top_k: int = 10
    min_score: float = 0.0
    use_semantic_ranker: bool = True

    # Hybrid search weights
    vector_weight: float = 0.5
    keyword_weight: float = 0.5


class AzureSearchIndexer:
    """
    Azure AI Search index management and document indexing.

    Usage:
        indexer = AzureSearchIndexer(endpoint, api_key)
        indexer.create_index()
        indexer.index_documents(chunks)
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        embedding_endpoint: str,
        embedding_key: str,
        config: VectorSearchConfig | None = None,
    ):
        """
        Initialize Azure Search indexer.

        Args:
            endpoint: Azure AI Search endpoint
            api_key: Azure AI Search API key
            embedding_endpoint: Azure OpenAI endpoint for embeddings
            embedding_key: Azure OpenAI API key
            config: Search configuration
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.config = config or VectorSearchConfig()

        # Initialize clients
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import SearchIndexClient

        self.credential = AzureKeyCredential(api_key)
        self.index_client = SearchIndexClient(endpoint, self.credential)

        # Initialize embeddings using OpenAI SDK directly
        from openai import AzureOpenAI

        self._oai_client = AzureOpenAI(
            azure_endpoint=embedding_endpoint,
            api_key=embedding_key,
            api_version="2024-12-01-preview",
        )
        self._emb_deployment = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        )

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using Azure OpenAI."""
        response = self._oai_client.embeddings.create(input=text[:8000], model=self._emb_deployment)
        return response.data[0].embedding

    def create_index(self) -> bool:
        """
        Create or update the search index with vector configuration.

        Returns:
            True if successful
        """
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            HnswParameters,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            VectorSearch,
            VectorSearchProfile,
        )

        # Define fields
        fields = [
            # Key field
            SearchField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            # Content fields
            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="en.microsoft",
            ),
            SearchField(
                name="summary",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            # Vector field
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.config.vector_dimensions,
                vector_search_profile_name="hnsw-profile",
            ),
            # Metadata fields
            SearchField(
                name="parent_document_id",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="parent_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="detected_language",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="source_file",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SearchField(
                name="token_count",
                type=SearchFieldDataType.Int32,
                filterable=True,
            ),
            SearchField(
                name="entities",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
            ),
            # Thread & attachment relationship fields
            SearchField(
                name="thread_id",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="source_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchField(
                name="source_attachment_filename",
                type=SearchFieldDataType.String,
                filterable=True,
                searchable=True,
            ),
            SearchField(
                name="has_attachments",
                type=SearchFieldDataType.Boolean,
                filterable=True,
            ),
            SearchField(
                name="indexed_at",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
        ]

        # Vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-algo",
                    parameters=HnswParameters(
                        m=self.config.hnsw_m,
                        ef_construction=self.config.hnsw_ef_construction,
                        ef_search=self.config.hnsw_ef_search,
                        metric="cosine",
                    ),
                ),
            ],
            profiles=[
                VectorSearchProfile(
                    name="hnsw-profile",
                    algorithm_configuration_name="hnsw-algo",
                ),
            ],
        )

        # Semantic configuration
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
                title_fields=[SemanticField(field_name="summary")],
            ),
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])

        # Create index
        index = SearchIndex(
            name=self.config.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        try:
            self.index_client.create_or_update_index(index)
            logger.info(f"Created/updated index: {self.config.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def delete_index(self) -> bool:
        """Delete the search index."""
        try:
            self.index_client.delete_index(self.config.index_name)
            logger.info(f"Deleted index: {self.config.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_embedding_with_retry(self, text: str) -> list[float]:
        """Generate embedding with retry."""
        response = self._oai_client.embeddings.create(input=text[:8000], model=self._emb_deployment)
        return response.data[0].embedding

    def index_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 100,
        generate_embeddings: bool = True,
    ) -> tuple[int, int]:
        """
        Index documents to Azure AI Search.

        Args:
            documents: List of document dictionaries
            batch_size: Number of documents per batch
            generate_embeddings: Whether to generate embeddings

        Returns:
            Tuple of (success_count, error_count)
        """
        from azure.search.documents import SearchClient

        search_client = SearchClient(self.endpoint, self.config.index_name, self.credential)

        success_count = 0
        error_count = 0

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Prepare documents
            prepared_docs = []
            for doc in batch:
                try:
                    # Generate embedding if needed
                    if generate_embeddings and "content_vector" not in doc:
                        content = doc.get("content", "")
                        if content:
                            doc["content_vector"] = self._generate_embedding(content[:8000])

                    # Add indexed timestamp
                    doc["indexed_at"] = datetime.utcnow().isoformat() + "Z"

                    prepared_docs.append(doc)

                except Exception as e:
                    logger.warning(f"Failed to prepare doc {doc.get('chunk_id')}: {e}")
                    error_count += 1

            # Upload batch
            try:
                result = search_client.upload_documents(prepared_docs)
                success_count += sum(1 for r in result if r.succeeded)
                error_count += sum(1 for r in result if not r.succeeded)
            except Exception as e:
                logger.error(f"Batch upload failed: {e}")
                error_count += len(prepared_docs)

            logger.info(
                f"Indexed batch {i // batch_size + 1}, success: {success_count}, errors: {error_count}"
            )

        return success_count, error_count

    def get_index_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        try:
            index = self.index_client.get_index(self.config.index_name)
            return {
                "name": index.name,
                "fields": len(index.fields),
                # Note: Document count requires separate API call
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}


class HybridSearcher:
    """
    Hybrid search combining vector and keyword search.

    Usage:
        searcher = HybridSearcher(endpoint, api_key, embedding_endpoint, embedding_key)
        results = searcher.search("What is the project timeline?")
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        embedding_endpoint: str,
        embedding_key: str,
        config: VectorSearchConfig | None = None,
    ):
        """Initialize hybrid searcher."""
        self.endpoint = endpoint
        self.api_key = api_key
        self.config = config or VectorSearchConfig()

        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self.search_client = SearchClient(
            endpoint, self.config.index_name, AzureKeyCredential(api_key)
        )

        # Initialize embeddings using OpenAI SDK directly (no langchain dependency)
        from openai import AzureOpenAI

        self._oai_client = AzureOpenAI(
            azure_endpoint=embedding_endpoint,
            api_key=embedding_key,
            api_version="2024-12-01-preview",
        )
        self._emb_deployment = os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        )

    def _embed_query(self, text: str) -> list[float]:
        """Embed a query using Azure OpenAI."""
        response = self._oai_client.embeddings.create(input=text, model=self._emb_deployment)
        return response.data[0].embedding

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: str | None = None,
        search_type: str = "hybrid",  # vector, keyword, hybrid
    ) -> SearchResponse:
        """
        Execute search query.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: OData filter expression
            search_type: Type of search (vector, keyword, hybrid)

        Returns:
            SearchResponse with results
        """
        import time

        from azure.search.documents.models import VectorizedQuery

        start_time = time.time()
        top_k = top_k or self.config.top_k

        # Build search parameters
        search_kwargs: dict[str, Any] = {
            "top": top_k,
            "include_total_count": True,
            "select": [
                "chunk_id",
                "content",
                "summary",
                "parent_document_id",
                "parent_type",
                "detected_language",
                "source_file",
                "chunk_index",
                "entities",
                "thread_id",
                "source_type",
                "source_attachment_filename",
                "has_attachments",
            ],
        }

        if filters:
            search_kwargs["filter"] = filters

        # Configure search type
        if search_type == "vector":
            # Pure vector search
            query_vector = self._embed_query(query)
            search_kwargs["vector_queries"] = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="content_vector",
                )
            ]
            search_kwargs["search_text"] = None

        elif search_type == "keyword":
            # Pure keyword search
            search_kwargs["search_text"] = query
            search_kwargs["query_type"] = "simple"

        else:  # hybrid
            # Hybrid search (vector + keyword)
            query_vector = self._embed_query(query)
            search_kwargs["vector_queries"] = [
                VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="content_vector",
                )
            ]
            search_kwargs["search_text"] = query

            # Enable semantic ranking if configured
            if self.config.use_semantic_ranker:
                search_kwargs["query_type"] = "semantic"
                search_kwargs["semantic_configuration_name"] = "semantic-config"

        # Execute search
        response = self.search_client.search(**search_kwargs)

        # Process results
        results = []
        for result in response:
            search_result = SearchResult(
                chunk_id=result["chunk_id"],
                content=result.get("content", ""),
                score=result["@search.score"],
                reranker_score=result.get("@search.reranker_score"),
                metadata={
                    "parent_document_id": result.get("parent_document_id"),
                    "parent_type": result.get("parent_type"),
                    "detected_language": result.get("detected_language"),
                    "source_file": result.get("source_file"),
                    "chunk_index": result.get("chunk_index"),
                    "entities": result.get("entities", []),
                    "summary": result.get("summary"),
                    "thread_id": result.get("thread_id"),
                    "source_type": result.get("source_type"),
                    "source_attachment_filename": result.get("source_attachment_filename"),
                    "has_attachments": result.get("has_attachments", False),
                },
                highlights=result.get("@search.highlights"),
            )
            results.append(search_result)

        search_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results,
            query=query,
            total_count=response.get_count() or len(results),
            search_time_ms=search_time_ms,
            search_type=search_type,
        )

    def vector_search(self, query: str, top_k: int | None = None) -> SearchResponse:
        """Execute pure vector search."""
        return self.search(query, top_k, search_type="vector")

    def keyword_search(self, query: str, top_k: int | None = None) -> SearchResponse:
        """Execute pure keyword search."""
        return self.search(query, top_k, search_type="keyword")

    def hybrid_search(self, query: str, top_k: int | None = None) -> SearchResponse:
        """Execute hybrid search."""
        return self.search(query, top_k, search_type="hybrid")


# Export
__all__ = [
    "AzureSearchIndexer",
    "HybridSearcher",
    "SearchResponse",
    "SearchResult",
    "VectorSearchConfig",
]
