"""
Embedding Generator Module

Generates vector embeddings for chunks, entities, and community summaries
for hybrid retrieval combining graph-based and vector-based approaches.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    # Model settings
    model: str = "text-embedding-3-small"  # OpenAI model
    local_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model (384 dims)
    dimensions: int = 1536  # Embedding dimensions (overridden by local model)
    batch_size: int = 100  # Batch size for API calls

    # Azure settings (optional)
    azure_deployment: Optional[str] = None

    # Chunking for long text
    max_tokens_per_text: int = 8000

    # Cost tracking (per 1M tokens)
    cost_per_1m_tokens: float = 0.02

    def __post_init__(self):
        env_model = os.environ.get("LOCAL_EMBEDDING_MODEL")
        if env_model:
            self.local_model = env_model


class EmbeddingGenerator:
    """
    Generates embeddings for various content types using OpenAI API.

    Supports embedding:
    - Text chunks (from Silver layer)
    - Entity names + context
    - Community summaries (from GraphRAG)
    - Path descriptions (for PathRAG)
    """

    def __init__(
        self,
        gold_path: str,
        config: Optional[EmbeddingConfig] = None,
        mode: str = "llm",
    ):
        """
        Initialize the embedding generator.

        Args:
            gold_path: Path to Gold layer for output
            config: Embedding configuration
            mode: Processing mode — "local" uses sentence-transformers, "llm"/"hybrid" uses OpenAI
        """
        self.gold_path = Path(gold_path)
        self.config = config or EmbeddingConfig()
        self.mode = mode

        # Output directory
        self.embeddings_path = self.gold_path / "embeddings"
        self.embeddings_path.mkdir(parents=True, exist_ok=True)

        # Initialize client based on mode
        self.client = None
        self.local_model = None
        self.use_azure = False

        if mode == "local":
            self._initialize_local_model()
        else:
            self._initialize_client()

        # Cache for embeddings
        self.embedding_cache: Dict[str, List[float]] = {}

    def _initialize_local_model(self):
        """Initialize sentence-transformers model for local embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.local_model
            logger.info(f"Loading local embedding model ({model_name})...")
            try:
                self.local_model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                self.local_model = SentenceTransformer(model_name)
            self.config.dimensions = self.local_model.get_sentence_embedding_dimension()
            logger.info(f"Local embedding model loaded ({self.config.dimensions} dims)")
        except ImportError:
            logger.error("sentence-transformers not installed. pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")

    def _initialize_client(self):
        """Initialize OpenAI client for embeddings."""
        try:
            # Try Azure OpenAI
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_key = os.getenv("AZURE_OPENAI_API_KEY")

            if azure_endpoint and azure_key:
                from openai import AzureOpenAI
                self.client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                )
                self.use_azure = True
                self.config.azure_deployment = os.getenv(
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                    "text-embedding-3-small"
                )
                logger.info("Using Azure OpenAI for embeddings")
                return

            # Try OpenAI direct
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_key)
                self.use_azure = False
                logger.info("Using OpenAI for embeddings")
                return

            logger.warning("No OpenAI credentials found - embedding generation will fail")

        except ImportError:
            logger.error("openai package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if failed
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Local mode: sentence-transformers
        if self.local_model is not None:
            try:
                embedding = self.local_model.encode(text, show_progress_bar=False).tolist()
                self.embedding_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                logger.error(f"Local embedding failed: {e}")
                return None

        # API mode: OpenAI/Azure
        if not self.client:
            logger.error("No embedding client initialized")
            return None

        try:
            if len(text) > self.config.max_tokens_per_text * 4:
                text = text[:self.config.max_tokens_per_text * 4]

            model = self.config.azure_deployment if self.use_azure else self.config.model
            response = self.client.embeddings.create(model=model, input=text)
            embedding = response.data[0].embedding
            self.embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failed)
        """
        # Local mode: sentence-transformers batch encoding
        if self.local_model is not None:
            try:
                embeddings = self.local_model.encode(texts, show_progress_bar=True, batch_size=64)
                results = []
                for i, emb in enumerate(embeddings):
                    emb_list = emb.tolist()
                    self.embedding_cache[self._get_cache_key(texts[i])] = emb_list
                    results.append(emb_list)
                return results
            except Exception as e:
                logger.error(f"Local batch embedding failed: {e}")
                return [None] * len(texts)

        # API mode: OpenAI/Azure
        if not self.client:
            return [None] * len(texts)

        results = []
        model = self.config.azure_deployment if self.use_azure else self.config.model

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            batch_results = []
            texts_to_embed = []
            text_indices = []

            for j, text in enumerate(batch):
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    batch_results.append(self.embedding_cache[cache_key])
                else:
                    batch_results.append(None)
                    texts_to_embed.append(text[:self.config.max_tokens_per_text * 4])
                    text_indices.append(j)

            if texts_to_embed:
                try:
                    response = self.client.embeddings.create(
                        model=model,
                        input=texts_to_embed
                    )

                    for k, embedding_data in enumerate(response.data):
                        original_idx = text_indices[k]
                        embedding = embedding_data.embedding
                        batch_results[original_idx] = embedding
                        cache_key = self._get_cache_key(batch[original_idx])
                        self.embedding_cache[cache_key] = embedding

                except Exception as e:
                    logger.error(f"Batch embedding failed: {e}")

            results.extend(batch_results)

        return results

    def embed_chunks(
        self,
        silver_path: str,
        use_anonymized: bool = True,
        progress_callback=None
    ) -> Tuple[List[str], Any]:
        """
        Generate embeddings for all Silver layer chunks.

        Args:
            silver_path: Path to Silver layer
            use_anonymized: Whether to embed anonymized text
            progress_callback: Optional progress callback

        Returns:
            Tuple of (chunk_ids, embeddings_array)
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for embedding storage")

        silver_dir = Path(silver_path)
        chunk_files = []

        for pattern in ["not_personal/email_chunks/*.json",
                       "not_personal/attachment_chunks/*.json",
                       "not_personal/document_chunks/*.json"]:
            chunk_files.extend(silver_dir.glob(pattern))

        logger.info(f"Embedding {len(chunk_files)} chunks")

        chunk_ids = []
        texts = []

        # Collect texts
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)

                chunk_id = chunk_data.get("chunk_id", chunk_file.stem)
                subject = chunk_data.get("thread_subject", "")
                subject_prefix = f"Subject: {subject}. " if subject else ""

                # Embed text_english (full content with technical details)
                text_english = chunk_data.get("text_english") or chunk_data.get("text_anonymized" if use_anonymized else "text_original", "")
                if text_english:
                    chunk_ids.append(chunk_id)
                    texts.append(f"{subject_prefix}{text_english}")

                # Also embed summary if available (semantic meaning)
                summary = chunk_data.get("summary", "")
                if summary and summary != text_english:
                    chunk_ids.append(f"{chunk_id}_sum")
                    texts.append(f"{subject_prefix}{summary}")

            except Exception as e:
                logger.warning(f"Failed to read chunk {chunk_file} (file={chunk_file.name}): {e}")

        # Generate embeddings
        embeddings = self.embed_batch(texts)

        # Filter successful embeddings
        valid_ids = []
        valid_embeddings = []

        for chunk_id, embedding in zip(chunk_ids, embeddings):
            if embedding is not None:
                valid_ids.append(chunk_id)
                valid_embeddings.append(embedding)

        # Convert to numpy array
        embeddings_array = np.array(valid_embeddings)

        logger.info(f"Generated {len(valid_ids)} chunk embeddings")
        return valid_ids, embeddings_array

    def embed_entities(
        self,
        entities: List[Dict[str, Any]],
        include_context: bool = True,
        progress_callback=None
    ) -> Tuple[List[str], Any]:
        """
        Generate embeddings for entities.

        Args:
            entities: List of entity dictionaries with 'id', 'name', 'type'
            include_context: Whether to include entity type in text
            progress_callback: Optional progress callback

        Returns:
            Tuple of (entity_ids, embeddings_array)
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for embedding storage")

        logger.info(f"Embedding {len(entities)} entities")

        entity_ids = []
        texts = []

        for entity in entities:
            entity_id = entity.get("id", entity.get("node_id", ""))
            name = entity.get("name", "")
            entity_type = entity.get("type", entity.get("node_type", ""))

            if not name:
                continue

            # Build text representation
            if include_context:
                text = f"{entity_type}: {name}"
            else:
                text = name

            entity_ids.append(entity_id)
            texts.append(text)

        # Generate embeddings
        embeddings = self.embed_batch(texts)

        # Filter successful
        valid_ids = []
        valid_embeddings = []

        for entity_id, embedding in zip(entity_ids, embeddings):
            if embedding is not None:
                valid_ids.append(entity_id)
                valid_embeddings.append(embedding)

        embeddings_array = np.array(valid_embeddings)

        logger.info(f"Generated {len(valid_ids)} entity embeddings")
        return valid_ids, embeddings_array

    def embed_summaries(
        self,
        summaries: List[Dict[str, Any]],
        progress_callback=None
    ) -> Tuple[List[str], Any]:
        """
        Generate embeddings for community summaries.

        Args:
            summaries: List of summary dictionaries with 'id', 'summary'
            progress_callback: Optional progress callback

        Returns:
            Tuple of (summary_ids, embeddings_array)
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for embedding storage")

        logger.info(f"Embedding {len(summaries)} summaries")

        summary_ids = []
        texts = []

        for summary in summaries:
            summary_id = summary.get("id", summary.get("community_id", ""))
            text = summary.get("summary", "")

            if not text:
                continue

            summary_ids.append(summary_id)
            texts.append(text)

        # Generate embeddings
        embeddings = self.embed_batch(texts)

        # Filter successful
        valid_ids = []
        valid_embeddings = []

        for summary_id, embedding in zip(summary_ids, embeddings):
            if embedding is not None:
                valid_ids.append(summary_id)
                valid_embeddings.append(embedding)

        embeddings_array = np.array(valid_embeddings)

        logger.info(f"Generated {len(valid_ids)} summary embeddings")
        return valid_ids, embeddings_array

    def save_embeddings(
        self,
        ids: List[str],
        embeddings: Any,
        name: str
    ):
        """
        Save embeddings to disk.

        Args:
            ids: List of IDs corresponding to embeddings
            embeddings: Numpy array of embeddings
            name: Name for the embedding set (e.g., 'chunks', 'entities')
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required")

        # Save embeddings as numpy file
        embeddings_file = self.embeddings_path / f"{name}_embeddings.npy"
        np.save(str(embeddings_file), embeddings)

        # Save IDs
        ids_file = self.embeddings_path / f"{name}_ids.json"
        with open(ids_file, 'w', encoding='utf-8') as f:
            json.dump(ids, f)

        # Save config
        config_file = self.embeddings_path / f"{name}_config.json"
        config_data = {
            "model": self.config.model,
            "dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "count": len(ids),
            "generated_at": datetime.now().isoformat()
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved {len(ids)} {name} embeddings to {embeddings_file}")

    def load_embeddings(self, name: str) -> Tuple[List[str], Any]:
        """
        Load embeddings from disk.

        Args:
            name: Name of the embedding set

        Returns:
            Tuple of (ids, embeddings_array)
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required")

        embeddings_file = self.embeddings_path / f"{name}_embeddings.npy"
        ids_file = self.embeddings_path / f"{name}_ids.json"

        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")

        embeddings = np.load(str(embeddings_file))

        with open(ids_file, 'r', encoding='utf-8') as f:
            ids = json.load(f)

        logger.info(f"Loaded {len(ids)} {name} embeddings")
        return ids, embeddings

    def similarity_search(
        self,
        query: str,
        embeddings: Any,
        ids: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most similar items to a query.

        Args:
            query: Query text
            embeddings: Embeddings array to search
            ids: IDs corresponding to embeddings
            top_k: Number of results to return

        Returns:
            List of (id, similarity_score) tuples
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required")

        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return []

        query_vec = np.array(query_embedding)

        # Compute cosine similarity
        # Normalize
        query_norm = query_vec / np.linalg.norm(query_vec)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        similarities = np.dot(embeddings_norm, query_norm)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    def is_available(self) -> bool:
        """Check if embedding generation is available."""
        return self.client is not None or self.local_model is not None
