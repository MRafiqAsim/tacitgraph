"""
Anonymizer

Anonymizes detected PII in text using various strategies:
- Replacement with placeholders
- Hashing
- Masking
- Redaction
"""

import hashlib
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from .pii_detector import PIIDetector, PIIEntity, PIIType

logger = logging.getLogger(__name__)


class AnonymizationStrategy(Enum):
    """Anonymization strategies"""
    REPLACE = "replace"        # Replace with typed placeholder: [PERSON_1]
    HASH = "hash"              # Replace with hash
    MASK = "mask"              # Replace with asterisks: J*** S****
    REDACT = "redact"          # Replace with [REDACTED]
    ENCRYPT = "encrypt"        # Reversible encryption (requires key)


@dataclass
class AnonymizationResult:
    """Result of anonymization"""

    # Original and anonymized text
    original_text: str
    anonymized_text: str

    # Detected and anonymized entities
    entities: List[PIIEntity]
    entity_count: int

    # Mapping for potential de-anonymization
    mapping: Dict[str, str] = field(default_factory=dict)

    # Metadata
    strategy: AnonymizationStrategy = AnonymizationStrategy.REPLACE
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anonymized_text": self.anonymized_text,
            "entity_count": self.entity_count,
            "entities": [e.to_dict() for e in self.entities],
            "strategy": self.strategy.value,
            "language": self.language,
        }


class Anonymizer:
    """
    Anonymize PII in text.

    Supports multiple anonymization strategies and maintains
    consistent replacements across documents. Optionally uses
    an IdentityRegistry for stable pseudonyms.
    """

    def __init__(
        self,
        detector: Optional[PIIDetector] = None,
        strategy: AnonymizationStrategy = AnonymizationStrategy.REPLACE,
        hash_length: int = 8,
        consistent_replacement: bool = True,
        identity_registry=None
    ):
        """
        Initialize the anonymizer.

        Args:
            detector: PII detector instance
            strategy: Default anonymization strategy
            hash_length: Length of hash for HASH strategy
            consistent_replacement: Use same replacement for same PII value
            identity_registry: Optional IdentityRegistry for stable PERSON pseudonyms
        """
        self.detector = detector or PIIDetector()
        self.strategy = strategy
        self.hash_length = hash_length
        self.consistent_replacement = consistent_replacement
        self.identity_registry = identity_registry

        # Counters for consistent replacement
        self._type_counters: Dict[PIIType, int] = {}
        self._value_mapping: Dict[str, str] = {}

    def anonymize(
        self,
        text: str,
        language: str = "en",
        strategy: Optional[AnonymizationStrategy] = None,
        entities: Optional[List[PIIEntity]] = None
    ) -> AnonymizationResult:
        """
        Anonymize PII in text.

        Args:
            text: Text to anonymize
            language: Text language
            strategy: Override default strategy
            entities: Pre-detected entities (skip detection if provided)

        Returns:
            AnonymizationResult with anonymized text
        """
        if not text:
            return AnonymizationResult(
                original_text="",
                anonymized_text="",
                entities=[],
                entity_count=0,
                strategy=strategy or self.strategy,
                language=language
            )

        # Detect PII if not provided
        if entities is None:
            entities = self.detector.detect(text, language)

        if not entities:
            return AnonymizationResult(
                original_text=text,
                anonymized_text=text,
                entities=[],
                entity_count=0,
                strategy=strategy or self.strategy,
                language=language
            )

        # Use specified or default strategy
        active_strategy = strategy or self.strategy

        # Generate replacements
        mapping = {}
        for entity in entities:
            replacement = self._generate_replacement(entity, active_strategy)
            entity.replacement = replacement
            mapping[entity.text] = replacement

        # Apply replacements (from end to start to preserve positions)
        anonymized = text
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            anonymized = (
                anonymized[:entity.start] +
                entity.replacement +
                anonymized[entity.end:]
            )

        return AnonymizationResult(
            original_text=text,
            anonymized_text=anonymized,
            entities=entities,
            entity_count=len(entities),
            mapping=mapping,
            strategy=active_strategy,
            language=language
        )

    def anonymize_batch(
        self,
        texts: List[str],
        language: str = "en"
    ) -> List[AnonymizationResult]:
        """
        Anonymize multiple texts with consistent replacements.

        Args:
            texts: List of texts to anonymize
            language: Text language

        Returns:
            List of AnonymizationResult
        """
        results = []

        for text in texts:
            result = self.anonymize(text, language)
            results.append(result)

        return results

    def _generate_replacement(
        self,
        entity: PIIEntity,
        strategy: AnonymizationStrategy
    ) -> str:
        """Generate replacement text for an entity"""

        # Check for consistent replacement
        if self.consistent_replacement and entity.text in self._value_mapping:
            return self._value_mapping[entity.text]

        # Generate replacement based on strategy
        if strategy == AnonymizationStrategy.REPLACE:
            replacement = self._generate_placeholder(entity)

        elif strategy == AnonymizationStrategy.HASH:
            replacement = self._generate_hash(entity)

        elif strategy == AnonymizationStrategy.MASK:
            replacement = self._generate_mask(entity)

        elif strategy == AnonymizationStrategy.REDACT:
            replacement = "[REDACTED]"

        elif strategy == AnonymizationStrategy.ENCRYPT:
            replacement = self._generate_encrypted(entity)

        else:
            replacement = f"[{entity.pii_type.value}]"

        # Store for consistent replacement
        if self.consistent_replacement:
            self._value_mapping[entity.text] = replacement

        return replacement

    def _generate_placeholder(self, entity: PIIEntity) -> str:
        """Generate typed placeholder: [PERSON_001], [EMAIL_2], etc.

        For PERSON and EMAIL entities, uses the IdentityRegistry
        to generate stable pseudonyms linked to real identities.
        """
        pii_type = entity.pii_type

        # Try identity registry for PERSON entities
        if self.identity_registry and pii_type == PIIType.PERSON:
            pseudonym = self.identity_registry.get_pseudonym(entity.text)
            if pseudonym:
                return f"[{pseudonym}]"

        # Try identity registry for EMAIL entities → link to same person
        if self.identity_registry and pii_type == PIIType.EMAIL:
            pseudonym = self.identity_registry.get_pseudonym(entity.text)
            if pseudonym:
                return f"[{pseudonym}_EMAIL]"

        # Fallback: auto-increment counter
        if pii_type not in self._type_counters:
            self._type_counters[pii_type] = 0

        self._type_counters[pii_type] += 1
        counter = self._type_counters[pii_type]

        return f"[{pii_type.value}_{counter}]"

    def _generate_hash(self, entity: PIIEntity) -> str:
        """Generate hash replacement"""
        hash_value = hashlib.sha256(entity.text.encode()).hexdigest()
        short_hash = hash_value[:self.hash_length].upper()
        return f"[{entity.pii_type.value}:{short_hash}]"

    def _generate_mask(self, entity: PIIEntity) -> str:
        """Generate masked version preserving length"""
        text = entity.text

        if entity.pii_type == PIIType.EMAIL:
            # Mask email: j***@c***.com
            parts = text.split('@')
            if len(parts) == 2:
                local = parts[0][0] + '*' * (len(parts[0]) - 1)
                domain_parts = parts[1].split('.')
                if len(domain_parts) >= 2:
                    domain = domain_parts[0][0] + '*' * (len(domain_parts[0]) - 1)
                    return f"{local}@{domain}.{domain_parts[-1]}"

        elif entity.pii_type == PIIType.PHONE:
            # Mask phone: +32 *** *** **89
            digits = re.findall(r'\d', text)
            if len(digits) >= 4:
                masked = re.sub(r'\d', '*', text)
                # Keep last 2 digits
                for d in digits[-2:]:
                    masked = masked[::-1].replace('*', d[::-1], 1)[::-1]
                return masked

        elif entity.pii_type == PIIType.CREDIT_CARD:
            # Mask card: **** **** **** 1234
            digits = re.findall(r'\d', text)
            if len(digits) >= 4:
                return '**** **** **** ' + ''.join(digits[-4:])

        elif entity.pii_type == PIIType.PERSON:
            # Mask name: J*** S***
            words = text.split()
            masked_words = []
            for word in words:
                if len(word) > 1:
                    masked_words.append(word[0] + '*' * (len(word) - 1))
                else:
                    masked_words.append('*')
            return ' '.join(masked_words)

        # Default: keep first char, mask rest
        if len(text) > 1:
            return text[0] + '*' * (len(text) - 1)
        return '*'

    def _generate_encrypted(self, entity: PIIEntity) -> str:
        """Generate encrypted replacement (simple for demo)"""
        # In production, use proper encryption with key management
        import base64
        encoded = base64.b64encode(entity.text.encode()).decode()
        return f"[ENC:{encoded}]"

    def reset_counters(self) -> None:
        """Reset replacement counters (for new document batch)"""
        self._type_counters = {}
        self._value_mapping = {}

    def get_mapping(self) -> Dict[str, str]:
        """Get the current value-to-replacement mapping"""
        return self._value_mapping.copy()


class PIIAnonymizationPipeline:
    """
    Complete PII anonymization pipeline for document processing.
    """

    def __init__(
        self,
        languages: List[str] = None,
        strategy: AnonymizationStrategy = AnonymizationStrategy.REPLACE,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the pipeline.

        Args:
            languages: Supported languages
            strategy: Anonymization strategy
            confidence_threshold: Minimum confidence for detection
        """
        self.languages = languages or ["en", "nl"]

        self.detector = PIIDetector(
            languages=self.languages,
            confidence_threshold=confidence_threshold
        )

        self.anonymizer = Anonymizer(
            detector=self.detector,
            strategy=strategy,
            consistent_replacement=True
        )

    def process(
        self,
        text: str,
        language: str = "en"
    ) -> AnonymizationResult:
        """
        Process text through the anonymization pipeline.

        Args:
            text: Text to process
            language: Text language

        Returns:
            AnonymizationResult
        """
        return self.anonymizer.anonymize(text, language)

    def process_batch(
        self,
        texts: List[str],
        languages: Optional[List[str]] = None
    ) -> List[AnonymizationResult]:
        """
        Process multiple texts.

        Args:
            texts: List of texts
            languages: Optional list of languages (one per text)

        Returns:
            List of AnonymizationResult
        """
        if languages is None:
            languages = ["en"] * len(texts)

        results = []
        for text, lang in zip(texts, languages):
            result = self.process(text, lang)
            results.append(result)

        return results

    def reset(self) -> None:
        """Reset the pipeline for a new document set"""
        self.anonymizer.reset_counters()


# Convenience function
def anonymize_text(
    text: str,
    language: str = "en",
    strategy: str = "replace"
) -> AnonymizationResult:
    """
    Anonymize PII in text.

    Args:
        text: Text to anonymize
        language: Text language
        strategy: Anonymization strategy (replace, hash, mask, redact)

    Returns:
        AnonymizationResult
    """
    strategy_map = {
        "replace": AnonymizationStrategy.REPLACE,
        "hash": AnonymizationStrategy.HASH,
        "mask": AnonymizationStrategy.MASK,
        "redact": AnonymizationStrategy.REDACT,
    }

    pipeline = PIIAnonymizationPipeline(strategy=strategy_map.get(strategy, AnonymizationStrategy.REPLACE))
    return pipeline.process(text, language)
