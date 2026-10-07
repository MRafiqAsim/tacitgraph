"""
Unified Processor

Combines PII detection, anonymization, and summarization based on configuration.
Supports three modes: LOCAL, LLM, and HYBRID.

| Feature        | LOCAL                        | LLM              | HYBRID                           |
|----------------|------------------------------|-------------------|----------------------------------|
| PII Detection  | Hardened Presidio+spaCy+regex| Azure GPT-4o     | Local first → LLM verify (<0.8) |
| Identity       | Registry lookup              | Registry lookup   | Registry lookup                  |
| Anonymization  | Local (replace)              | Local (replace)   | Local (replace)                  |
| Summarization  | Extractive (first sentences) | Azure GPT-4o     | Azure GPT-4o                     |
| KG Extraction  | spaCy                        | spaCy + LLM      | spaCy                            |
| Cost           | $0                           | Highest           | Medium                           |
"""

import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig, ProcessingMode, get_config
from .pii_detector import PIIDetector, PIIEntity, PIIType

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of unified text processing"""
    original_text: str
    anonymized_text: str
    summary: Optional[str]
    pii_entities: List[PIIEntity]
    key_entities: List[str]
    key_topics: List[str]
    processing_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "anonymized_text": self.anonymized_text,
            "summary": self.summary,
            "pii_entities": [e.to_dict() for e in self.pii_entities],
            "pii_count": len(self.pii_entities),
            "key_entities": self.key_entities,
            "key_topics": self.key_topics,
            "processing_mode": self.processing_mode,
            "metadata": self.metadata,
        }


class UnifiedProcessor:
    """
    Unified text processor that handles PII detection, anonymization, and summarization.

    Supports three modes based on configuration:
    - LOCAL: Hardened Presidio+spaCy+regex, extractive summaries, $0 cost
    - LLM: Azure GPT-4o for PII detection and summarization
    - HYBRID: Local detection first, LLM verifies low-confidence (<0.8), LLM summarization
    """

    def __init__(self, config: Optional[PipelineConfig] = None, identity_registry=None):
        """
        Initialize the unified processor.

        Args:
            config: Pipeline configuration (uses global config if not provided)
            identity_registry: Optional IdentityRegistry for consistent pseudonyms
        """
        self.config = config or get_config()
        self.identity_registry = identity_registry
        self._local_detector = None
        self._local_anonymizer = None
        self._openai_detector = None
        self._openai_anonymizer = None
        self._openai_summarizer = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize processing components based on mode"""
        mode = self.config.mode

        # Map OPENAI mode to LLM behavior
        if mode in [ProcessingMode.LOCAL, ProcessingMode.HYBRID]:
            self._init_local_components()

        if mode in [ProcessingMode.OPENAI, ProcessingMode.HYBRID]:
            self._init_openai_components()

    def _init_local_components(self):
        """Initialize local processing components"""
        try:
            from .pii_detector import PIIDetector
            from .anonymizer import Anonymizer

            self._local_detector = PIIDetector(
                confidence_threshold=self.config.pii.confidence_threshold,
                use_presidio=self.config.pii.use_presidio,
                use_spacy=self.config.pii.use_spacy,
                use_regex=self.config.pii.use_regex,
                identity_registry=self.identity_registry,
            )
            self._local_anonymizer = Anonymizer(
                identity_registry=self.identity_registry,
            )
            logger.info("Local processing components initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize local components: {e}")
            if self.config.mode == ProcessingMode.LOCAL:
                raise

    def _init_openai_components(self):
        """Initialize OpenAI/Azure OpenAI processing components"""
        try:
            from .openai_pii_detector import OpenAIPIIDetector, OpenAIAnonymizer
            from .openai_summarizer import OpenAISummarizer

            api_key = self.config.openai.api_key

            # Check for Azure OpenAI config
            use_azure = hasattr(self.config, 'azure_openai') and self.config.azure_openai
            azure_endpoint = getattr(self.config, 'azure_openai', {})
            azure_kwargs = {}

            if use_azure and isinstance(azure_endpoint, dict):
                azure_kwargs = {
                    "use_azure": True,
                    "azure_endpoint": azure_endpoint.get("endpoint"),
                    "azure_api_version": azure_endpoint.get("api_version", "2024-12-01-preview"),
                    "azure_deployment": azure_endpoint.get("deployment"),
                }
                api_key = api_key or azure_endpoint.get("api_key")

            if not api_key:
                import os
                api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError("API key not configured. Set AZURE_OPENAI_API_KEY or OPENAI_API_KEY.")

            model = self.config.openai.model

            self._openai_detector = OpenAIPIIDetector(
                api_key=api_key,
                model=model,
                confidence_threshold=self.config.pii.confidence_threshold,
                identity_registry=self.identity_registry,
                **azure_kwargs,
            )
            self._openai_anonymizer = OpenAIAnonymizer(
                api_key=api_key,
                model=model,
                generate_synthetic=self.config.anonymization.generate_synthetic,
                **azure_kwargs,
            )
            self._openai_summarizer = OpenAISummarizer(
                api_key=api_key,
                model=model,
                max_summary_length=self.config.summarization.max_summary_length,
            )
            logger.info("OpenAI/Azure processing components initialized")

        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI components: {e}")
            raise

    def process(
        self,
        text: str,
        language: str = "en",
        include_summary: bool = True
    ) -> ProcessingResult:
        """
        Process text: detect PII, anonymize, and optionally summarize.

        Args:
            text: Text to process
            language: Text language
            include_summary: Whether to generate summary

        Returns:
            ProcessingResult with all outputs
        """
        if not text or not text.strip():
            return ProcessingResult(
                original_text=text,
                anonymized_text=text,
                summary=None,
                pii_entities=[],
                key_entities=[],
                key_topics=[],
                processing_mode=self.config.mode.value,
            )

        mode = self.config.mode

        if mode == ProcessingMode.OPENAI:
            return self._process_llm(text, language, include_summary)
        elif mode == ProcessingMode.LOCAL:
            return self._process_local(text, language, include_summary)
        else:  # HYBRID
            return self._process_hybrid(text, language, include_summary)

    def _process_llm(
        self,
        text: str,
        language: str,
        include_summary: bool
    ) -> ProcessingResult:
        """Process using LLM (Azure GPT-4o) for PII detection + summarization"""
        # Detect PII with LLM
        pii_entities = self._openai_detector.detect(text, language)

        # Anonymize locally (all modes use local replacement)
        anonymized = self._apply_local_anonymization(text, pii_entities, language)

        # Summarize with LLM
        summary = None
        key_entities = []
        key_topics = []

        if include_summary and self.config.summarization.enabled:
            summary_result = self._openai_summarizer.summarize(
                anonymized,
                style=self.config.summarization.summary_style,
            )
            summary = summary_result.summary
            key_entities = summary_result.key_entities
            key_topics = summary_result.key_topics

        return ProcessingResult(
            original_text=text,
            anonymized_text=anonymized,
            summary=summary,
            pii_entities=pii_entities,
            key_entities=key_entities,
            key_topics=key_topics,
            processing_mode="llm",
        )

    def _process_local(
        self,
        text: str,
        language: str,
        include_summary: bool
    ) -> ProcessingResult:
        """Process using local models only (Presidio+spaCy+regex)"""
        # Detect PII locally
        pii_entities = self._local_detector.detect(text, language)

        # Anonymize locally
        anonymized = self._apply_local_anonymization(text, pii_entities, language)

        # Extractive summary (no LLM cost)
        summary = None
        if include_summary:
            summary = self._generate_extractive_summary(anonymized)

        return ProcessingResult(
            original_text=text,
            anonymized_text=anonymized,
            summary=summary,
            pii_entities=pii_entities,
            key_entities=[],
            key_topics=[],
            processing_mode="local",
        )

    def _process_hybrid(
        self,
        text: str,
        language: str,
        include_summary: bool
    ) -> ProcessingResult:
        """Process using hybrid approach: local first, LLM verifies low-confidence entities"""
        # Step 1: Local detection
        local_entities = self._local_detector.detect(text, language)

        # Step 2: Separate high and low confidence entities
        high_conf_entities = []
        low_conf_entities = []
        threshold = self.config.pii.hybrid_confidence_threshold

        for entity in local_entities:
            if entity.confidence >= threshold:
                high_conf_entities.append(entity)
            else:
                low_conf_entities.append(entity)

        # Step 3: Use LLM for verification if there are low-confidence entities
        final_entities = high_conf_entities.copy()

        if low_conf_entities or self._should_verify_with_llm(text, local_entities):
            openai_entities = self._openai_detector.detect(text, language)

            # Merge: keep LLM entities that don't overlap with high-conf local
            for oe in openai_entities:
                overlap = False
                for le in high_conf_entities:
                    if self._entities_overlap(oe, le):
                        overlap = True
                        break
                if not overlap:
                    final_entities.append(oe)

        # Deduplicate
        final_entities = self._deduplicate_entities(final_entities)

        # Anonymize locally (all modes use local replacement)
        anonymized = self._apply_local_anonymization(text, final_entities, language)

        # Summarize with LLM
        summary = None
        key_entities = []
        key_topics = []

        if include_summary and self.config.summarization.enabled:
            summary_result = self._openai_summarizer.summarize(
                anonymized,
                style=self.config.summarization.summary_style,
            )
            summary = summary_result.summary
            key_entities = summary_result.key_entities
            key_topics = summary_result.key_topics

        return ProcessingResult(
            original_text=text,
            anonymized_text=anonymized,
            summary=summary,
            pii_entities=final_entities,
            key_entities=key_entities,
            key_topics=key_topics,
            processing_mode="hybrid",
            metadata={
                "local_entities_count": len(local_entities),
                "high_conf_count": len(high_conf_entities),
                "llm_verified": True,
            }
        )

    def _apply_local_anonymization(
        self,
        text: str,
        entities: List[PIIEntity],
        language: str,
    ) -> str:
        """Apply local anonymization using the Anonymizer (with identity registry)."""
        if not entities:
            return text

        from .anonymizer import Anonymizer, AnonymizationStrategy

        # Use the pre-initialized anonymizer if available
        anonymizer = self._local_anonymizer
        if not anonymizer:
            anonymizer = Anonymizer(identity_registry=self.identity_registry)

        strategy_map = {
            "replace": AnonymizationStrategy.REPLACE,
            "hash": AnonymizationStrategy.HASH,
            "mask": AnonymizationStrategy.MASK,
            "redact": AnonymizationStrategy.REDACT,
        }
        strategy = strategy_map.get(
            self.config.anonymization.default_strategy,
            AnonymizationStrategy.REPLACE
        )

        result = anonymizer.anonymize(
            text=text,
            language=language,
            strategy=strategy,
            entities=entities,
        )
        return result.anonymized_text

    def _generate_extractive_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary (first N sentences, no LLM)."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out very short fragments and header lines
        meaningful = [
            s for s in sentences
            if len(s) > 20 and not s.startswith(('[', '---', 'From:', 'To:', 'Date:', 'Subject:'))
        ]
        if not meaningful:
            return text[:300] + "..." if len(text) > 300 else text
        return " ".join(meaningful[:max_sentences])

    def _should_verify_with_llm(
        self,
        text: str,
        local_entities: List[PIIEntity]
    ) -> bool:
        """Determine if LLM verification is needed"""
        words = text.split()
        if len(words) > 50 and len(local_entities) == 0:
            return True

        # Check for potential missed person names
        potential_names = re.findall(r'(?<=[a-z]\s)[A-Z][a-z]+', text)
        person_entities = [e for e in local_entities if e.pii_type == PIIType.PERSON]
        if len(potential_names) > len(person_entities) * 2:
            return True

        return False

    def _entities_overlap(self, e1: PIIEntity, e2: PIIEntity) -> bool:
        """Check if two entities overlap"""
        return not (e1.end <= e2.start or e2.end <= e1.start)

    def _deduplicate_entities(
        self,
        entities: List[PIIEntity]
    ) -> List[PIIEntity]:
        """Remove duplicate entities, keeping highest confidence"""
        if not entities:
            return []

        sorted_entities = sorted(
            entities,
            key=lambda e: (e.start, -e.confidence)
        )

        result = []
        for entity in sorted_entities:
            overlap = False
            for existing in result:
                if self._entities_overlap(entity, existing):
                    overlap = True
                    break
            if not overlap:
                result.append(entity)

        return sorted(result, key=lambda e: e.start)

    def process_batch(
        self,
        texts: List[str],
        language: str = "en",
        include_summary: bool = True
    ) -> List[ProcessingResult]:
        """
        Process multiple texts.

        Args:
            texts: List of texts to process
            language: Text language
            include_summary: Whether to generate summaries

        Returns:
            List of ProcessingResult objects
        """
        return [
            self.process(text, language, include_summary)
            for text in texts
        ]


# Convenience function
def process_text(
    text: str,
    language: str = "en",
    mode: str = "local",
    include_summary: bool = True
) -> ProcessingResult:
    """
    Process text with specified mode.

    Args:
        text: Text to process
        language: Text language
        mode: Processing mode ("local", "llm", "hybrid")
        include_summary: Whether to generate summary

    Returns:
        ProcessingResult
    """
    from config import init_config

    # Map "llm" to "openai" for backward compatibility with ProcessingMode enum
    mode_map = {"llm": "openai", "local": "local", "hybrid": "hybrid"}
    config = init_config(mode=mode_map.get(mode, mode))
    processor = UnifiedProcessor(config)
    return processor.process(text, language, include_summary)
