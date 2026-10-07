"""
Knowledge Graph Relationship Extractor Module

Extracts relationships between entities for PathRAG knowledge graph.
Supports multiple strategies for benchmarking:
- CooccurrenceExtractor: Fast, rule-based (entities in same sentence)
- LLMRelationshipExtractor: LLM-based (more accurate, extracts semantics)
- HybridRelationshipExtractor: Combines both approaches

PathRAG Relationship Format:
    (source_entity, target_entity, description, keywords, weight)

Usage:
    extractor = LLMRelationshipExtractor(api_key="...")
    relationships = extractor.extract(text, entities, language="en")
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from .kg_entity_extractor import KGEntity

logger = logging.getLogger(__name__)


@dataclass
class KGRelationship:
    """A relationship between two entities for PathRAG knowledge graph"""

    source: str  # Source entity text
    target: str  # Target entity text
    source_type: str  # Source entity type (PathRAG type)
    target_type: str  # Target entity type (PathRAG type)
    relationship: str  # Relationship type (e.g., "located in", "works at")
    weight: float = 1.0  # Relationship strength (0-1)
    source_method: str = "cooccurrence"  # Extraction method for benchmarking

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "relationship": self.relationship,
            "weight": self.weight,
            "extraction_method": self.source_method,
        }

    def to_pathrag_format(self) -> dict[str, Any]:
        """Convert to PathRAG's expected format"""
        return {
            "src_id": self.source,
            "tgt_id": self.target,
            "relationship": self.relationship,
            "weight": self.weight,
        }


class RelationshipExtractor(ABC):
    """
    Abstract base class for relationship extractors.

    Implement this interface to create new extraction strategies
    for benchmarking different approaches.
    """

    @abstractmethod
    def extract(
        self, text: str, entities: list[KGEntity], language: str = "en"
    ) -> list[KGRelationship]:
        """
        Extract relationships between entities in text.

        Args:
            text: Source text
            entities: List of entities already extracted from text
            language: Language code

        Returns:
            List of KGRelationship objects
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor name for logging/benchmarking"""
        pass


class CooccurrenceRelationshipExtractor(RelationshipExtractor):
    """
    Rule-based relationship extractor using co-occurrence.

    Fast extraction based on entities appearing in the same sentence.
    Good for initial relationship discovery, but lacks semantic understanding.
    """

    def __init__(
        self,
        window_size: int = 1,  # Number of sentences to consider
        min_weight: float = 0.3,
    ):
        """
        Initialize co-occurrence extractor.

        Args:
            window_size: Number of sentences for co-occurrence window
            min_weight: Minimum weight threshold for relationships
        """
        self.window_size = window_size
        self.min_weight = min_weight

    def extract(
        self, text: str, entities: list[KGEntity], language: str = "en"
    ) -> list[KGRelationship]:
        """Extract relationships based on co-occurrence in sentences"""
        relationships: list[KGRelationship] = []

        if len(entities) < 2:
            return relationships

        # Split text into sentences
        sentences = self._split_sentences(text)

        # Find which entities appear in which sentences
        entity_sentences: dict[str, set[int]] = {}
        for entity in entities:
            entity_key = entity.entity.lower()
            entity_sentences[entity_key] = set()

            for i, sentence in enumerate(sentences):
                if entity.entity.lower() in sentence.lower():
                    # Add this sentence and neighbors within window
                    for j in range(
                        max(0, i - self.window_size), min(len(sentences), i + self.window_size + 1)
                    ):
                        entity_sentences[entity_key].add(j)

        # Find co-occurring entity pairs
        seen_pairs: set[tuple[str, ...]] = set()
        {e.entity.lower(): e for e in entities}

        for e1 in entities:
            for e2 in entities:
                if e1.entity == e2.entity:
                    continue

                # Normalize pair order to avoid duplicates
                pair = tuple(sorted([e1.entity.lower(), e2.entity.lower()]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Check co-occurrence
                e1_sentences = entity_sentences.get(e1.entity.lower(), set())
                e2_sentences = entity_sentences.get(e2.entity.lower(), set())
                overlap = e1_sentences & e2_sentences

                if overlap:
                    # Calculate weight based on co-occurrence frequency
                    weight = min(1.0, len(overlap) * 0.3)

                    if weight >= self.min_weight:
                        relationships.append(
                            KGRelationship(
                                source=e1.entity,
                                target=e2.entity,
                                source_type=e1.entity_type,
                                target_type=e2.entity_type,
                                relationship="RELATED_TO",
                                weight=weight,
                                source_method="cooccurrence",
                            )
                        )

        return relationships

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _generate_keywords(self, e1: KGEntity, e2: KGEntity) -> list[str]:
        """Generate relationship keywords based on entity types"""
        keywords = []

        type_pair = {e1.entity_type, e2.entity_type}

        if "person" in type_pair and "organization" in type_pair:
            keywords.extend(["employment", "affiliation"])
        elif "person" in type_pair and "geo" in type_pair:
            keywords.extend(["location", "based_in"])
        elif "organization" in type_pair and "geo" in type_pair:
            keywords.extend(["headquarters", "operates_in"])
        elif "person" in type_pair and "event" in type_pair:
            keywords.extend(["participation", "involvement"])
        elif "organization" in type_pair and "product" in type_pair:
            keywords.extend(["produces", "develops"])
        else:
            keywords.append("related_to")

        return keywords

    @property
    def name(self) -> str:
        return "CooccurrenceRelationshipExtractor"


class LLMRelationshipExtractor(RelationshipExtractor):
    """
    LLM-based relationship extractor.

    Uses OpenAI/Azure OpenAI for semantic relationship extraction.
    More accurate but slower and requires API key.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_entities_per_call: int = 20,
        # Azure OpenAI settings
        use_azure: bool = False,
        azure_endpoint: str | None = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment: str | None = None,
    ):
        """
        Initialize LLM relationship extractor.

        Args:
            api_key: OpenAI or Azure OpenAI API key
            model: Model to use (for OpenAI) or deployment name (for Azure)
            max_entities_per_call: Maximum entities to process per API call
            use_azure: Whether to use Azure OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure API version
            azure_deployment: Azure deployment name (defaults to model)
        """
        self.api_key = api_key
        self.model = model
        self.max_entities_per_call = max_entities_per_call
        self.use_azure = use_azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment or model

    def _get_client(self):
        """Get the appropriate OpenAI client"""
        if self.use_azure:
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
                timeout=httpx.Timeout(120.0, connect=10.0),
                max_retries=2,
            )
        else:
            from openai import OpenAI

            return OpenAI(api_key=self.api_key)

    def extract(
        self, text: str, entities: list[KGEntity], language: str = "en"
    ) -> list[KGRelationship]:
        """Extract relationships using LLM"""
        if not self.api_key or len(entities) < 2:
            return []

        relationships = []

        try:
            import json

            client = self._get_client()

            # Prepare entity list for prompt
            entity_list = "\n".join(
                [f"- {e.entity} ({e.entity_type})" for e in entities[: self.max_entities_per_call]]
            )

            from tacitgraph.prompt_loader import format_prompt, get_prompt

            prompt = format_prompt(
                get_prompt("silver", "kg_relationship_extraction", "user_prompt"),
                text=text,
                entity_list=entity_list,
            )

            # Use deployment name for Azure, model name for OpenAI
            # Build system prompt with dynamic relationship types from config
            from tacitgraph.entity_registry import get_relationship_types_for_prompt

            system_prompt = format_prompt(
                get_prompt("silver", "kg_relationship_extraction", "system_prompt"),
                relationship_types=get_relationship_types_for_prompt(),
            )

            model_name = self.azure_deployment if self.use_azure else self.model
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=16384,
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            content = extract_llm_content(response, context="relationship extraction")

            # Strip markdown code fences that GPT-4o sometimes wraps around JSON
            if content.startswith("```"):
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    content = match.group()

            # Parse JSON response
            if content.startswith("["):
                raw_relationships = json.loads(content)

                # Create entity lookup for type information
                entity_map = {e.entity.lower(): e for e in entities}

                for rel in raw_relationships:
                    source_entity = entity_map.get(rel.get("source", "").lower())
                    target_entity = entity_map.get(rel.get("target", "").lower())

                    if source_entity and target_entity:
                        relationships.append(
                            KGRelationship(
                                source=rel.get("source", ""),
                                target=rel.get("target", ""),
                                source_type=source_entity.entity_type,
                                target_type=target_entity.entity_type,
                                relationship=rel.get("relationship", "RELATED_TO"),
                                weight=float(rel.get("weight", 0.5)),
                                source_method="llm",
                            )
                        )

        except Exception as e:
            logger.error(f"LLM relationship extraction error: {e}")

        return relationships

    @property
    def name(self) -> str:
        return f"LLMRelationshipExtractor({self.model})"


class HybridRelationshipExtractor(RelationshipExtractor):
    """
    Hybrid relationship extractor combining co-occurrence and LLM.

    Uses co-occurrence for fast initial extraction, then LLM to
    enhance descriptions and verify relationships.
    """

    def __init__(
        self,
        cooccurrence_extractor: CooccurrenceRelationshipExtractor,
        llm_extractor: LLMRelationshipExtractor | None = None,
        use_llm_for_descriptions: bool = True,
    ):
        """
        Initialize hybrid extractor.

        Args:
            cooccurrence_extractor: Primary fast extractor
            llm_extractor: Optional LLM extractor for enhancement
            use_llm_for_descriptions: Use LLM to improve descriptions
        """
        self.cooccurrence_extractor = cooccurrence_extractor
        self.llm_extractor = llm_extractor
        self.use_llm_for_descriptions = use_llm_for_descriptions

    def extract(
        self, text: str, entities: list[KGEntity], language: str = "en"
    ) -> list[KGRelationship]:
        """Extract using co-occurrence, optionally enhance with LLM"""
        # Primary extraction with co-occurrence
        relationships = self.cooccurrence_extractor.extract(text, entities, language)

        # Optional LLM enhancement
        if self.llm_extractor and self.use_llm_for_descriptions:
            llm_relationships = self.llm_extractor.extract(text, entities, language)

            # Merge LLM relationships, preferring LLM descriptions
            cooc_map = {(r.source.lower(), r.target.lower()): r for r in relationships}

            for llm_rel in llm_relationships:
                key = (llm_rel.source.lower(), llm_rel.target.lower())
                reverse_key = (llm_rel.target.lower(), llm_rel.source.lower())

                if key in cooc_map:
                    existing = cooc_map[key]
                    existing.relationship = llm_rel.relationship
                    existing.weight = max(existing.weight, llm_rel.weight)
                    existing.source_method = "hybrid"
                elif reverse_key in cooc_map:
                    existing = cooc_map[reverse_key]
                    existing.relationship = llm_rel.relationship
                    existing.weight = max(existing.weight, llm_rel.weight)
                    existing.source_method = "hybrid"
                else:
                    # New relationship from LLM
                    llm_rel.source_method = "llm"
                    relationships.append(llm_rel)

        return relationships

    @property
    def name(self) -> str:
        return "HybridRelationshipExtractor"


# Factory function for easy creation
def create_relationship_extractor(
    strategy: str = "cooccurrence",
    openai_api_key: str | None = None,
    # Azure OpenAI settings
    use_azure: bool = False,
    azure_endpoint: str | None = None,
    azure_api_version: str = "2024-12-01-preview",
    azure_deployment: str | None = None,
    **kwargs,
) -> RelationshipExtractor:
    """
    Factory function to create relationship extractor.

    Args:
        strategy: "cooccurrence", "llm", or "hybrid"
        openai_api_key: Required for LLM strategies
        use_azure: Whether to use Azure OpenAI
        azure_endpoint: Azure OpenAI endpoint URL
        azure_api_version: Azure API version
        azure_deployment: Azure deployment name

    Returns:
        Configured RelationshipExtractor instance
    """
    if strategy == "cooccurrence":
        return CooccurrenceRelationshipExtractor(**kwargs)

    elif strategy == "llm":
        if not openai_api_key:
            raise ValueError("openai_api_key required for LLM strategy")
        return LLMRelationshipExtractor(
            api_key=openai_api_key,
            use_azure=use_azure,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
            **kwargs,
        )

    elif strategy == "hybrid":
        cooc_ext = CooccurrenceRelationshipExtractor()
        llm_ext = None
        if openai_api_key:
            llm_ext = LLMRelationshipExtractor(
                api_key=openai_api_key,
                use_azure=use_azure,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version,
                azure_deployment=azure_deployment,
            )
        return HybridRelationshipExtractor(
            cooccurrence_extractor=cooc_ext, llm_extractor=llm_ext, **kwargs
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
