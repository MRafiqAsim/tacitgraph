"""
Thread-Aware Silver Processor

Processes email threads with semantic context preservation:
1. Groups emails into conversation threads
2. Concatenates thread emails chronologically
3. Chunks with thread boundaries respected
4. Generates thread summaries for high-level retrieval
5. Maintains consistent anonymization across thread
"""

import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tacitgraph.bronze.attachment_processor import AttachmentProcessor
from tacitgraph.bronze.thread_grouper import EmailThread, ThreadGrouper
from tacitgraph.silver.anonymizer import AnonymizationStrategy, Anonymizer
from tacitgraph.silver.attachment_classifier import AttachmentClassifier
from tacitgraph.silver.chunker import SemanticChunker
from tacitgraph.silver.email_sensitivity_classifier import (
    EmailSensitivityClassifier,
    LLMSensitivityClassifier,
    SensitivityResult,
)
from tacitgraph.silver.email_text_cleaner import clean_email_text
from tacitgraph.silver.kg_entity_extractor import (
    KGEntity,
    KGEntityExtractor,
    create_kg_extractor,
)
from tacitgraph.silver.language_detector import LanguageDetector
from tacitgraph.silver.pii_detector import PIIDetector
from tacitgraph.silver.relationship_extractor import (
    RelationshipExtractor,
    create_relationship_extractor,
)

logger = logging.getLogger(__name__)


@dataclass
class ThreadChunk:
    """A chunk from a thread with context"""

    chunk_id: str
    thread_id: str
    chunk_index: int

    # Content
    text_original: str
    text_anonymized: str

    # Thread context
    thread_subject: str
    thread_participants: list[str]
    thread_email_count: int
    email_position: str  # e.g., "1/5", "3/5"

    # Content (with defaults)
    text_english: str = ""  # English-normalized text for retrieval (translation if non-English)
    summary: str = ""  # Intent-only summary (Phase 2 anonymization — no sensitive details)
    token_count: int = 0

    # PII (for anonymization)
    pii_entities: list[dict[str, Any]] = field(default_factory=list)
    pii_count: int = 0

    # Knowledge Graph Entities (for PathRAG)
    # These are non-PII entities useful for building knowledge graphs
    kg_entities: list[dict[str, Any]] = field(default_factory=list)

    # Knowledge Graph Relationships (for PathRAG)
    # Relationships between entities: (source, target, description, keywords, weight)
    kg_relationships: list[dict[str, Any]] = field(default_factory=list)

    # Attachment information
    has_attachments: bool = False
    attachment_count: int = 0
    attachment_filenames: list[str] = field(default_factory=list)

    # Bronze lineage — record_id(s) of the source email(s) in Bronze layer
    source_email_ids: list[str] = field(default_factory=list)

    # Source tracking
    source_type: str = "email"  # "email" | "attachment"
    source_attachment_filename: str = ""  # set when source_type == "attachment"
    attachment_classification: str = ""  # "knowledge" | "transactional" | ""
    classification_confidence: float = 0.0  # 0.0–1.0 from Bronze classifier

    # Sensitivity
    anonymization_skipped: bool = False  # True if thread was classified as not_personal

    # Email sender/recipient (from Bronze headers)
    email_sender: str = ""  # sender name
    email_sender_address: str = ""  # sender email
    email_recipients_to: list[dict[str, str]] = field(
        default_factory=list
    )  # [{"name": ..., "email": ...}]
    email_recipients_cc: list[dict[str, str]] = field(
        default_factory=list
    )  # [{"name": ..., "email": ...}]

    # Temporal — from Bronze email document_metadata
    sent_timestamp: str = ""  # ISO format, e.g. "2013-05-03T10:32:09"
    received_timestamp: str = ""  # ISO format, e.g. "2013-05-03T10:32:12.664372"

    # Metadata
    language: str = "en"
    processing_mode: str = "local"
    processing_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "thread_id": self.thread_id,
            "chunk_index": self.chunk_index,
            "text_original": self.text_original,
            "text_anonymized": self.text_anonymized,
            "text_english": self.text_english,
            "summary": self.summary,
            "token_count": self.token_count,
            "thread_subject": self.thread_subject,
            "thread_participants": self.thread_participants,
            "thread_email_count": self.thread_email_count,
            "email_position": self.email_position,
            "pii_entities": self.pii_entities,
            "pii_count": self.pii_count,
            "kg_entities": self.kg_entities,
            "kg_relationships": self.kg_relationships,
            "has_attachments": self.has_attachments,
            "attachment_count": self.attachment_count,
            "attachment_filenames": self.attachment_filenames,
            "source_email_ids": self.source_email_ids,
            "source_type": self.source_type,
            "source_attachment_filename": self.source_attachment_filename,
            "attachment_classification": self.attachment_classification,
            "classification_confidence": self.classification_confidence,
            "anonymization_skipped": self.anonymization_skipped,
            "email_sender": self.email_sender,
            "email_sender_address": self.email_sender_address,
            "email_recipients_to": self.email_recipients_to,
            "email_recipients_cc": self.email_recipients_cc,
            "sent_timestamp": self.sent_timestamp,
            "received_timestamp": self.received_timestamp,
            "language": self.language,
            "processing_mode": self.processing_mode,
            "processing_time": self.processing_time.isoformat(),
        }


@dataclass
class ThreadSummary:
    """Summary of an email thread for high-level retrieval"""

    thread_id: str
    subject: str
    participants: list[str]
    email_count: int
    date_range: str
    summary: str
    key_topics: list[str]
    chunk_ids: list[str]
    attachment_ids: list[str] = field(default_factory=list)
    source_email_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "participants": self.participants,
            "email_count": self.email_count,
            "date_range": self.date_range,
            "summary": self.summary,
            "key_topics": self.key_topics,
            "chunk_ids": self.chunk_ids,
            "attachment_ids": self.attachment_ids,
            "source_email_ids": self.source_email_ids,
        }


@dataclass
class AttachmentSummary:
    """Summary of an email attachment for retrieval context"""

    attachment_id: str
    thread_id: str
    filename: str
    summary: str
    chunk_ids: list[str]
    classification: str = "knowledge"
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attachment_id": self.attachment_id,
            "thread_id": self.thread_id,
            "filename": self.filename,
            "summary": self.summary,
            "chunk_ids": self.chunk_ids,
            "classification": self.classification,
            "token_count": self.token_count,
        }


@dataclass
class EmailSummary:
    """Summary of an individual email for fine-grained retrieval"""

    email_id: str  # Bronze record_id
    thread_id: str
    subject: str
    sender: str
    date: str
    summary: str
    chunk_ids: list[str]
    attachment_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "email_id": self.email_id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "sender": self.sender,
            "date": self.date,
            "summary": self.summary,
            "chunk_ids": self.chunk_ids,
            "attachment_ids": self.attachment_ids,
        }


class ThreadAwareProcessor:
    """
    Process email threads with semantic context preservation.

    Pipeline:
    1. Load emails from Bronze layer
    2. Group into conversation threads
    3. Concatenate thread emails (chronological)
    4. Detect language
    5. Chunk with thread boundaries
    6. Detect and anonymize PII (consistent across thread)
    7. Generate thread summaries
    8. Save to Silver layer
    """

    def __init__(
        self,
        bronze_path: str,
        silver_path: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
        anonymization_strategy: AnonymizationStrategy = AnonymizationStrategy.REPLACE,
        confidence_threshold: float = 0.5,
        generate_summaries: bool = True,
        openai_api_key: str | None = None,
        kg_extractor_strategy: str = "spacy",
        kg_extractor: KGEntityExtractor | None = None,
        relationship_extractor_strategy: str = "cooccurrence",
        relationship_extractor: RelationshipExtractor | None = None,
        extract_relationships: bool = True,
        # Azure OpenAI settings
        use_azure: bool = False,
        azure_endpoint: str | None = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment: str | None = None,
        # Attachment processing
        process_attachments: bool = True,
        include_attachment_text: bool = True,
        # Processing mode and identity
        processing_mode: str = "local",
        identity_registry=None,
    ):
        """
        Initialize thread-aware processor.

        Args:
            bronze_path: Path to Bronze layer
            silver_path: Path to Silver layer output
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            anonymization_strategy: Strategy for PII anonymization
            confidence_threshold: Minimum confidence for PII detection
            generate_summaries: Whether to generate thread summaries
            openai_api_key: OpenAI/Azure API key for LLM features
            kg_extractor_strategy: Strategy for KG extraction ("spacy", "llm", "hybrid")
            kg_extractor: Optional custom KG extractor (overrides strategy)
            relationship_extractor_strategy: Strategy for relationship extraction
                                             ("cooccurrence", "llm", "hybrid")
            relationship_extractor: Optional custom relationship extractor
            extract_relationships: Whether to extract relationships (default: True)
            use_azure: Whether to use Azure OpenAI instead of OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure API version
            azure_deployment: Azure deployment name
            process_attachments: Whether to process email attachments
            include_attachment_text: Include attachment text in chunks for KG extraction
            processing_mode: Pipeline mode ("local", "llm", "hybrid")
            identity_registry: Optional IdentityRegistry for consistent pseudonyms
        """
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.generate_summaries = generate_summaries
        self.openai_api_key = openai_api_key
        self.processing_mode = processing_mode
        self.identity_registry = identity_registry

        # Initialize components
        self.thread_grouper = ThreadGrouper()
        self.chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.language_detector = LanguageDetector()

        # Initialize PII detector based on processing mode
        if processing_mode == "llm":
            from tacitgraph.silver.openai_pii_detector import OpenAIPIIDetector

            self.pii_detector = OpenAIPIIDetector(
                api_key=openai_api_key,
                confidence_threshold=confidence_threshold,
                identity_registry=identity_registry,
                use_azure=use_azure,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version,
                azure_deployment=azure_deployment,
            )
            logger.info("PII detection mode: LLM (OpenAI)")
        else:
            self.pii_detector = PIIDetector(
                confidence_threshold=confidence_threshold,
                identity_registry=identity_registry,
            )
            logger.info(f"PII detection mode: {processing_mode} (Presidio)")

        self.anonymizer = Anonymizer(
            detector=self.pii_detector,
            strategy=anonymization_strategy,
            consistent_replacement=True,  # Important: same entity = same placeholder
            identity_registry=identity_registry,
        )

        # Attachment processing — env flag overrides constructor arg
        env_process_att = os.environ.get("PROCESS_ATTACHMENTS", "").lower()
        if env_process_att in ("false", "0", "no"):
            self.process_attachments = False
            logger.info("Attachment processing DISABLED (PROCESS_ATTACHMENTS=false)")
        elif env_process_att in ("true", "1", "yes"):
            self.process_attachments = process_attachments  # respect constructor arg
        else:
            self.process_attachments = process_attachments
        self.include_attachment_text = include_attachment_text

        self.attachment_processor = None

        if self.process_attachments:
            try:
                self.attachment_processor = AttachmentProcessor(
                    bronze_path=str(self.bronze_path),
                    extract_tables=True,
                )
                # Attachment classifier for Silver-layer classification
                self.attachment_classifier = AttachmentClassifier(bronze_path=str(self.bronze_path))
                logger.info("AttachmentProcessor initialized (classification in Silver)")
            except Exception as e:
                logger.warning(f"Failed to initialize AttachmentProcessor: {e}")
                self.process_attachments = False

        # Store Azure settings
        self.use_azure = use_azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment

        # Initialize Vision OCR for scanned/image PDFs (LLM and hybrid modes only)
        self.vision_extractor = None
        if processing_mode in ("llm", "hybrid") and openai_api_key:
            try:
                from tacitgraph.silver.openai_vision_extractor import (
                    OpenAIVisionExtractor,
                    VisionConfig,
                )

                vision_config = VisionConfig(
                    azure_endpoint=azure_endpoint if use_azure else None,
                    azure_api_key=openai_api_key if use_azure else None,
                    azure_api_version=azure_api_version,
                    azure_deployment=azure_deployment or "gpt-4o",
                    openai_api_key=openai_api_key if not use_azure else None,
                )
                self.vision_extractor = OpenAIVisionExtractor(config=vision_config)
                if self.vision_extractor.is_available():
                    logger.info("Vision OCR enabled for scanned/image PDFs")
                else:
                    self.vision_extractor = None
                    logger.warning("Vision OCR client not available — scanned PDFs will be skipped")
            except Exception as e:
                logger.warning(f"Failed to initialize Vision OCR: {e}")

        # Initialize sensitivity classifier
        # Classification now runs entirely in Silver (moved from Bronze)
        # - local mode: regex-based EmailSensitivityClassifier
        # - llm mode: GPT-4o LLMSensitivityClassifier
        # - hybrid mode: regex-based (LLM fallback if available)
        self.sensitivity_classifier = None
        if processing_mode in ("llm",) and openai_api_key:
            try:
                self.sensitivity_classifier = LLMSensitivityClassifier(
                    api_key=openai_api_key,
                    use_azure=use_azure,
                    azure_endpoint=azure_endpoint,
                    azure_api_version=azure_api_version,
                    azure_deployment=azure_deployment,
                )
                logger.info("Using LLM sensitivity classifier (GPT-4o)")
            except Exception as e:
                logger.warning(
                    f"Failed to init LLM sensitivity classifier, falling back to regex: {e}"
                )
                self.sensitivity_classifier = EmailSensitivityClassifier()
        else:
            # local and hybrid modes: regex-based classifier
            self.sensitivity_classifier = EmailSensitivityClassifier()
            logger.info("Using regex-based sensitivity classifier")

        # Initialize KG extractor (modular for benchmarking)
        if kg_extractor:
            self.kg_extractor = kg_extractor
        else:
            self.kg_extractor = create_kg_extractor(
                strategy=kg_extractor_strategy,
                languages=["en", "nl"],
                openai_api_key=openai_api_key,
                use_azure=use_azure,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version,
                azure_deployment=azure_deployment,
            )
        logger.info(f"Using KG extractor: {self.kg_extractor.name}")

        # Initialize relationship extractor (modular for benchmarking)
        self.extract_relationships = extract_relationships
        if extract_relationships:
            if relationship_extractor:
                self.relationship_extractor = relationship_extractor
            else:
                self.relationship_extractor = create_relationship_extractor(
                    strategy=relationship_extractor_strategy,
                    openai_api_key=openai_api_key,
                    use_azure=use_azure,
                    azure_endpoint=azure_endpoint,
                    azure_api_version=azure_api_version,
                    azure_deployment=azure_deployment,
                )
            logger.info(f"Using relationship extractor: {self.relationship_extractor.name}")
        else:
            self.relationship_extractor = None

        # Create directories
        self._create_directories()

        # Statistics
        self.stats = {
            "threads_processed": 0,
            "single_emails": 0,
            "multi_email_threads": 0,
            "chunks_created": 0,
            "summaries_generated": 0,
            "pii_detected": 0,
            "kg_entities_extracted": 0,
            "kg_relationships_extracted": 0,
            "attachments_processed": 0,
            "attachments_with_text": 0,
            "attachments_skipped_non_knowledge": 0,
            "attachment_chunks_created": 0,
            "attachment_summaries_generated": 0,
            "vision_ocr_extracted": 0,
            "threads_not_personal": 0,
            "threads_skipped_personal": 0,
            "emails_not_personal": 0,
            "emails_skipped_personal": 0,
            "emails_skipped_empty": 0,
            "email_summaries_generated": 0,
            "errors": 0,
            "content_filter_blocks": 0,
            "llm_failures": [],
            "start_time": None,
            "end_time": None,
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean text for embedding-optimized downstream use.

        Preserves all semantic content and context while removing noise
        that degrades embedding quality:
        - Email metadata headers (From:, Date:, Subject: lines already in chunk metadata)
        - Thread/email separator lines (--- Email 1/2 ---)
        - Quoted-reply markers (>)
        - Redundant whitespace and control characters
        - Email signatures and disclaimers
        - Forwarded message boilerplate

        Keeps: all substantive content, paragraph structure (as single newlines),
        entity names, technical terms, decisions, facts.
        """
        import re

        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines (will handle spacing later)
            if not stripped:
                continue

            # Skip thread/email metadata headers (already in chunk metadata fields)
            if re.match(r"^\[THREAD:", stripped, re.IGNORECASE):
                continue
            if re.match(r"^\[Participants:", stripped, re.IGNORECASE):
                continue
            if re.match(r"^\[Emails:\s*\d+\]", stripped, re.IGNORECASE):
                continue
            if re.match(r"^---\s*Email\s+\d+/\d+\s*---", stripped):
                continue
            if re.match(r"^---\s*Forwarded\s*---", stripped, re.IGNORECASE):
                continue

            # Skip email header lines (From:, Date:, Subject:, To:, Cc:, Sent:)
            # but NOT lines where these words appear mid-sentence
            if re.match(r"^(From|Date|Sent|To|Cc|Bcc|Subject):\s", stripped):
                continue

            # Remove quoted-reply markers but keep the content
            stripped = re.sub(r"^>+\s*", "", stripped)

            # Skip device/app boilerplate and disclaimers (keep regards/thanks for context)
            if re.match(
                r"^(sent from my|get outlook|disclaimer|confidential|this email)",
                stripped,
                re.IGNORECASE,
            ):
                continue
            if re.match(r"^[-_=]{5,}$", stripped):
                continue

            # Skip lines that are just a person's name (likely signature, 1-3 words, all title case)
            if (
                re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s*$", stripped)
                and len(stripped) < 40
            ):
                # Only skip if it looks like a standalone name (not part of content)
                if len(stripped.split()) <= 3:
                    continue

            # Normalize tabs to spaces
            stripped = stripped.replace("\t", " ")

            # Collapse multiple spaces
            stripped = re.sub(r" {2,}", " ", stripped)

            if stripped:
                cleaned_lines.append(stripped)

        # Join with single space — flat text is best for embedding models
        # Embedding models don't benefit from newlines; dense text = better vectors
        result = " ".join(cleaned_lines)

        # Final cleanup: collapse any remaining multiple spaces
        result = re.sub(r" {2,}", " ", result)

        return result.strip()

    def _extract_kg_entities(
        self, text: str, language: str, chunk_id: str = ""
    ) -> tuple[list[dict[str, Any]], list[KGEntity], str, str]:
        """
        Extract knowledge graph entities using modular extractor.

        Delegates to configured KG extractor (spaCy, LLM, or hybrid).
        In LLM mode, also returns text_english and detected source_language
        from the same API call (combined KG extraction + translation).

        Returns tuple of (entity_dicts, raw_entities, text_english, detected_language).
        For spaCy/local mode, text_english and detected_language are empty strings
        (caller falls back to using anonymized_text as-is).
        """
        try:
            entities = self.kg_extractor.extract(text, language)
        except Exception as e:
            error_str = str(e)
            is_content_filter = "content_filter" in error_str
            if is_content_filter:
                self.stats["content_filter_blocks"] += 1
            self.stats["llm_failures"].append(
                {
                    "chunk_id": chunk_id,
                    "stage": "entity_extraction",
                    "reason": "content_filter" if is_content_filter else "error",
                    "message": error_str[:200],
                }
            )
            logger.error(f"KG entity extraction failed for chunk_id={chunk_id}: {e}")
            return [], [], "", ""

        # Deduplicate entities by (name, type) — keep first occurrence
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.entity.lower(), e.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        entity_dicts = [e.to_dict() for e in unique_entities]
        self.stats["kg_entities_extracted"] += len(entity_dicts)

        # LLM extractor returns text_english and source_language from the same call
        text_english = getattr(self.kg_extractor, "last_text_english", "") or ""
        detected_lang = getattr(self.kg_extractor, "last_source_language", "") or ""

        return entity_dicts, unique_entities, text_english, detected_lang

    def _get_person_pseudonym(self, name: str) -> str:
        """
        Get a pseudonym for a person name.

        Lookup order:
        1. Identity registry — known email senders/recipients (exact + fuzzy)
        2. Anonymizer's value mapping — consistent within processing run
        3. Anonymizer pass — generates new PERSON_N and caches it

        Known people (registry) get stable pseudonyms tied to their email.
        NER-only names get consistent pseudonyms via the anonymizer's
        _value_mapping (same text → same pseudonym within a run).
        """
        # 1. Identity registry (email-based, stable across runs)
        if self.identity_registry:
            pseudonym = self.identity_registry.get_pseudonym(name)
            if pseudonym:
                return pseudonym

        # 2. Anonymizer's existing mapping (consistent within run)
        mapped = self.anonymizer._value_mapping.get(name, "")
        if mapped:
            return mapped.strip("[]")

        # 3. Run anonymizer — detects as PERSON PII, generates & caches pseudonym
        anon_result = self.anonymizer.anonymize(name, "en")
        if anon_result.anonymized_text != name:
            return anon_result.anonymized_text.strip("[]")

        # 4. Anonymizer didn't detect it as PII — force a consistent pseudonym
        if not hasattr(self, "_ner_person_counter"):
            self._ner_person_counter = 900  # high offset to avoid collision
        self._ner_person_counter += 1
        pseudonym = f"PERSON_{self._ner_person_counter:03d}"
        self.anonymizer._value_mapping[name] = f"[{pseudonym}]"
        return pseudonym

    def _anonymize_kg_entities(self, entity_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Replace PERSON entity text with pseudonyms in KG entity dicts.

        All PERSON entities are anonymized regardless of is_pii flag,
        since person names in the knowledge graph are always PII.
        """
        anonymized = []
        for e in entity_dicts:
            e_copy = dict(e)
            if e_copy.get("type") == "PERSON":
                name = e_copy.get("entity", e_copy.get("text", ""))
                pseudonym = self._get_person_pseudonym(name)
                e_copy["entity"] = pseudonym
                e_copy["original_entity"] = name
            anonymized.append(e_copy)
        return anonymized

    def _anonymize_kg_relationships(self, rel_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Replace PERSON source/target names with pseudonyms in KG relationship dicts.
        """
        anonymized = []
        for r in rel_dicts:
            r_copy = dict(r)
            for role in ("source", "target"):
                name = r_copy.get(role, "")
                role_type = r_copy.get(f"{role}_type", "")
                if role_type in ("PERSON", "person") and name:
                    r_copy[role] = self._get_person_pseudonym(name)
            anonymized.append(r_copy)
        return anonymized

    def _anonymize_participants(self, participants: list[str]) -> list[str]:
        """
        Anonymize a list of participant names using identity registry.
        """
        result = []
        for p in participants:
            pseudonym = None
            if self.identity_registry:
                pseudonym = self.identity_registry.get_pseudonym(p)
            if pseudonym:
                result.append(pseudonym)
            else:
                # Fall back to full anonymizer (handles emails, etc.)
                anon = self.anonymizer.anonymize(p, "en")
                result.append(anon.anonymized_text)
        return result

    def _classify_email(self, email: dict[str, Any]) -> SensitivityResult:
        """Classify a single email as personal or not_personal."""
        return self.sensitivity_classifier.classify(email)

    def _extract_kg_relationships(
        self,
        text: str,
        entities: list[KGEntity],
        language: str,
        chunk_id: str = "",
    ) -> list[dict[str, Any]]:
        """
        Extract relationships between entities using modular extractor.

        Delegates to configured relationship extractor (cooccurrence, LLM, hybrid).
        Returns list of relationship dicts for PathRAG knowledge graph.
        """
        if not self.extract_relationships or not self.relationship_extractor:
            return []

        if len(entities) < 2:
            return []

        try:
            relationships = self.relationship_extractor.extract(text, entities, language)
        except Exception as e:
            error_str = str(e)
            is_content_filter = "content_filter" in error_str
            if is_content_filter:
                self.stats["content_filter_blocks"] += 1
            self.stats["llm_failures"].append(
                {
                    "chunk_id": chunk_id,
                    "stage": "relationship_extraction",
                    "reason": "content_filter" if is_content_filter else "error",
                    "message": error_str[:200],
                }
            )
            logger.error(f"KG relationship extraction failed for chunk_id={chunk_id}: {e}")
            return []

        relationship_dicts = [r.to_dict() for r in relationships]
        self.stats["kg_relationships_extracted"] += len(relationship_dicts)
        return relationship_dicts

    def _save_attachment_chunk(self, chunk: ThreadChunk) -> None:
        """Save attachment chunk to Silver layer not_personal/attachment_chunks/"""
        chunk_dir = self.silver_path / "not_personal" / "attachment_chunks"
        chunk_file = chunk_dir / f"{chunk.chunk_id}.json"
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        logger.debug(
            f"    Saved attachment chunk: {chunk.chunk_id} (file={chunk.source_attachment_filename}, thread={chunk.thread_id})"
        )

    def _process_attachments_separately(
        self,
        attachment_contents: list,
        thread_id: str,
        subject: str,
        participants: list[str],
        email_count: int,
        language: str,
        should_anonymize: bool = True,
        source_email_ids: list[str] | None = None,
        sent_timestamp: str = "",
        received_timestamp: str = "",
    ) -> tuple[list[ThreadChunk], list[str]]:
        """
        Process attachments separately from email body text.

        Each attachment is classified, chunked, anonymized, summarized,
        and stored in attachment_chunks/ and attachment_summaries/.

        Returns:
            Tuple of (all_chunks, attachment_ids) for cross-referencing.
        """
        chunks = []
        attachment_ids = []

        for att_content in attachment_contents:
            # 1. Vision OCR: scanned PDF with empty text → extract via GPT-4o Vision
            #    (Thread is already classified as not_personal — we're only here for work threads)
            is_pdf = att_content.doc_type == "pdf"
            has_no_text = not att_content.text.strip()
            if is_pdf:
                logger.info(
                    f"PDF attachment: '{att_content.filename}' | "
                    f"text_len={len(att_content.text)} | empty={has_no_text} | "
                    f"vision_available={self.vision_extractor is not None}"
                )
            if is_pdf and has_no_text and self.vision_extractor:
                file_path = None
                if self.attachment_processor:
                    file_path = self.attachment_processor.find_attachment_file(
                        att_content.attachment_id, att_content.filename
                    )
                if file_path:
                    logger.info(
                        f"Scanned PDF detected: '{att_content.filename}' — running Vision OCR | {file_path}"
                    )
                    vision_result = self.vision_extractor.extract(
                        file_path=file_path,
                        attachment_id=att_content.attachment_id,
                    )
                    if vision_result.success and vision_result.extracted_text.strip():
                        att_content.text = vision_result.extracted_text
                        att_content.extraction_success = True
                        self.stats["vision_ocr_extracted"] += 1
                        logger.info(
                            f"Vision OCR extracted {len(vision_result.extracted_text)} chars from '{att_content.filename}'"
                        )
                    else:
                        logger.warning(
                            f"Vision OCR failed for '{att_content.filename}': {vision_result.error_message}"
                        )
                else:
                    logger.warning(
                        f"Scanned PDF '{att_content.filename}' — file not found on disk, cannot run Vision OCR"
                    )

            # 2. Skip data files (CSV, Excel) — not useful for knowledge retrieval
            skip_types = {".csv", ".xlsx", ".xls", ".xlsm", ".xlsb"}
            ext = Path(att_content.filename).suffix.lower() if att_content.filename else ""
            if ext in skip_types:
                logger.info(f"Skipping data file '{att_content.filename}' ({ext})")
                continue

            # 3. Skip if no text (extraction failed or scanned PDF without Vision)
            if not att_content.extraction_success or not att_content.text.strip():
                continue

            # 4. Skip garbled/binary text (failed DOC parsing, corrupt extractions)
            # Check ratio of actual readable words in sample:
            # - Latin words: 2+ consecutive a-zA-Z
            # - CJK text: sequences of CJK unified ideographs
            import re as _re

            sample = att_content.text[:2000]
            sample_len = max(len(sample), 1)
            latin_words = _re.findall(r"[a-zA-Z]{2,}", sample)
            cjk_chars = _re.findall(r"[\u4e00-\u9fff]+", sample)
            readable_chars = sum(len(w) for w in latin_words) + sum(len(c) for c in cjk_chars)
            word_char_ratio = readable_chars / sample_len
            if word_char_ratio < 0.15:
                logger.info(
                    f"Skipping garbled text '{att_content.filename}' (readable ratio: {word_char_ratio:.1%})"
                )
                print(
                    f"    Skipping garbled attachment: '{att_content.filename}' (readable ratio: {word_char_ratio:.1%})"
                )
                continue

            # 4. Knowledge/transactional filtering is currently disabled:
            #    every readable attachment is treated as knowledge.
            classification = "knowledge"

            self.stats["attachments_with_text"] += 1
            logger.info(f"  Attachment: '{att_content.filename}' ({len(att_content.text)} chars)")

            text = att_content.text

            # Chunk the attachment text
            text_chunks = self.chunker.chunk(
                text=text,
                doc_id=att_content.attachment_id,
                metadata={"filename": att_content.filename},
            )

            logger.info(f"    Chunked into {len(text_chunks)} chunks")

            # Track chunks per attachment for summary generation
            att_chunk_objects = []

            consecutive_filter_blocks = 0
            skip_llm_for_attachment = False

            for ci, chunk in enumerate(text_chunks):
                if ci % 10 == 0:
                    logger.info(
                        f"    Chunk {ci + 1}/{len(text_chunks)} of '{att_content.filename}'"
                    )

                if should_anonymize:
                    anon_result = self.anonymizer.anonymize(chunk.text, language)
                    self.stats["pii_detected"] += anon_result.entity_count
                    anonymized_text = anon_result.anonymized_text
                    pii_entities = [e.to_dict() for e in anon_result.entities]
                    pii_count = anon_result.entity_count
                else:
                    anonymized_text = self._clean_text(chunk.text)
                    pii_entities = []
                    pii_count = 0

                att_chunk_id = self._safe_filename(
                    f"att_{att_content.attachment_id}_{chunk.chunk_index}"
                )

                # Circuit breaker: skip LLM if 3+ consecutive content filter blocks
                if skip_llm_for_attachment:
                    kg_entity_dicts, kg_entities_raw, llm_text_english, detected_lang = (
                        [],
                        [],
                        "",
                        "",
                    )
                    text_english = anonymized_text
                    kg_relationships = []
                else:
                    prev_filter_count = self.stats.get("content_filter_blocks", 0)

                    # Extract KG entities — in LLM mode, also returns text_english
                    kg_entity_dicts, kg_entities_raw, llm_text_english, detected_lang = (
                        self._extract_kg_entities(anonymized_text, language, chunk_id=att_chunk_id)
                    )

                    if llm_text_english:
                        text_english = llm_text_english
                        language = detected_lang or language
                    else:
                        text_english = anonymized_text
                    kg_relationships = self._extract_kg_relationships(
                        text_english, kg_entities_raw, language, chunk_id=att_chunk_id
                    )

                    # Track consecutive content filter blocks
                    new_filter_count = self.stats.get("content_filter_blocks", 0)
                    if new_filter_count > prev_filter_count:
                        consecutive_filter_blocks += 1
                        if consecutive_filter_blocks >= 3:
                            skip_llm_for_attachment = True
                            logger.warning(
                                f"Content filter blocked 3 consecutive chunks for '{att_content.filename}' "
                                f"— skipping LLM for remaining {len(text_chunks) - ci - 1} chunks"
                            )
                    else:
                        consecutive_filter_blocks = 0

                if should_anonymize:
                    final_kg_entities = self._anonymize_kg_entities(kg_entity_dicts)
                    final_kg_rels = self._anonymize_kg_relationships(kg_relationships)
                    final_participants = self._anonymize_participants(participants)
                else:
                    final_kg_entities = kg_entity_dicts
                    final_kg_rels = kg_relationships
                    final_participants = participants

                # Precise lineage: use attachment's parent email_id if available,
                # otherwise fall back to the thread-level email IDs
                att_email_id = getattr(att_content, "email_id", "")
                att_source_ids = [att_email_id] if att_email_id else (source_email_ids or [])

                thread_chunk = ThreadChunk(
                    chunk_id=att_chunk_id,
                    thread_id=thread_id,
                    chunk_index=chunk.chunk_index,
                    text_original=chunk.text,
                    text_anonymized=anonymized_text,
                    text_english=text_english,
                    token_count=chunk.token_count,
                    thread_subject=subject,
                    thread_participants=final_participants,
                    thread_email_count=email_count,
                    email_position="attachment",
                    pii_entities=pii_entities,
                    pii_count=pii_count,
                    kg_entities=final_kg_entities,
                    kg_relationships=final_kg_rels,
                    has_attachments=True,
                    attachment_count=1,
                    attachment_filenames=[att_content.filename],
                    source_email_ids=att_source_ids,
                    source_type="attachment",
                    source_attachment_filename=att_content.filename,
                    attachment_classification=classification,
                    classification_confidence=getattr(
                        att_content, "classification_confidence", 0.0
                    ),
                    anonymization_skipped=not should_anonymize,
                    sent_timestamp=sent_timestamp,
                    received_timestamp=received_timestamp,
                    language=language,
                    processing_mode=self.processing_mode,
                )

                chunks.append(thread_chunk)
                att_chunk_objects.append(thread_chunk)
                self._save_attachment_chunk(thread_chunk)
                self.stats["attachment_chunks_created"] += 1
                self.stats["chunks_created"] += 1

            # Generate per-attachment summary (saved as separate AttachmentSummary file)
            if att_chunk_objects:
                att_summary = self._generate_attachment_summary(
                    attachment_id=att_content.attachment_id,
                    filename=att_content.filename,
                    thread_id=thread_id,
                    chunks=att_chunk_objects,
                    classification=classification,
                    language=language,
                )
                if att_summary:
                    attachment_ids.append(att_content.attachment_id)

        return chunks, attachment_ids

    def _create_directories(self) -> None:
        """Create Silver layer directory structure.

        Layout:
            silver/
            ├── not_personal/         ← processed (chunked, KG extracted, summarized)
            │   ├── email_chunks/     ← per-email chunks (both single and thread emails)
            │   ├── attachment_chunks/← per-attachment chunks (knowledge/transactional)
            │   ├── thread_summaries/ ← full-thread outcome summaries
            │   └── attachment_summaries/
            ├── personal/             ← raw Bronze data, skipped
            └── metadata/
        """
        tech = self.silver_path / "not_personal"
        directories = [
            tech / "email_chunks",
            tech / "thread_summaries",
            tech / "attachment_chunks",
            tech / "attachment_summaries",
            self.silver_path / "personal",
            self.silver_path / "metadata",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self) -> set:
        """Load set of already-processed thread IDs from checkpoint file."""
        checkpoint_file = self.silver_path / "checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return set()

    def _save_checkpoint(self, processed_ids: set) -> None:
        """Save processed thread IDs to checkpoint file."""
        checkpoint_file = self.silver_path / "checkpoint.json"
        self.silver_path.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(list(processed_ids), f)

    def process(
        self,
        progress_callback: Callable[[int, str], None] | None = None,
        max_threads: int | None = None,
        resume: bool = False,
    ) -> dict[str, int]:
        """
        Process Bronze layer emails into thread-aware Silver layer.

        Args:
            progress_callback: Optional callback(count, message)
            max_threads: Maximum number of threads to process
            resume: Skip threads already recorded in checkpoint.json

        Returns:
            Processing statistics
        """
        self.stats["start_time"] = datetime.now().isoformat()

        # Step 1: Group emails into threads
        logger.info("Step 1: Grouping emails into threads...")
        threads = self.thread_grouper.group_from_bronze(str(self.bronze_path))

        if not threads:
            logger.warning("No emails found to process")
            return self.stats

        if max_threads:
            threads = threads[:max_threads]
            logger.info(f"Limited to {len(threads)} threads (--limit {max_threads})")

        logger.info(f"Found {len(threads)} threads")

        # Load checkpoint for resume mode
        processed_ids: set = set()
        if resume:
            processed_ids = self._load_checkpoint()
            skipped = sum(1 for t in threads if t.conversation_id in processed_ids)
            logger.info(
                f"Resume mode: {len(processed_ids)} threads in checkpoint, {skipped} will be skipped"
            )

        # Step 2: Process each thread
        logger.info("Step 2: Processing threads...")

        for i, thread in enumerate(threads):
            if resume and thread.conversation_id in processed_ids:
                logger.debug(
                    f"[{i + 1}/{len(threads)}] Skipping (already processed): '{thread.subject[:60]}'"
                )
                continue

            try:
                logger.info(
                    f"[{i + 1}/{len(threads)}] Processing: '{thread.subject[:60]}' ({thread.email_count} emails)"
                )
                if thread.is_thread:
                    self._process_thread(thread)
                    self.stats["multi_email_threads"] += 1
                else:
                    self._process_single_email(thread)
                    self.stats["single_emails"] += 1

                self.stats["threads_processed"] += 1
                processed_ids.add(thread.conversation_id)

                # Save checkpoint every 10 threads (always, not just in resume mode)
                if self.stats["threads_processed"] % 10 == 0:
                    self._save_checkpoint(processed_ids)

                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback(i + 1, f"Processed {i + 1}/{len(threads)} threads")

            except Exception as e:
                current = getattr(self, "_current_email_id", "unknown")
                logger.error(
                    f"Error processing thread {thread.conversation_id} "
                    f"('{thread.subject[:60]}') at email_id={current}: {e}"
                )
                self.stats["errors"] += 1

        # Save final checkpoint (always, so next run can resume)
        self._save_checkpoint(processed_ids)

        # Step 3: Save metadata
        self.stats["end_time"] = datetime.now().isoformat()
        self._save_metadata()

        # Step 4: Process standalone documents from bronze/documents/
        docs_dir = self.bronze_path / "documents"
        if docs_dir.exists():
            doc_files = list(docs_dir.rglob("*.json"))
            if doc_files:
                logger.info(f"Step 4: Processing {len(doc_files)} standalone documents...")
                self._process_standalone_documents(doc_files)

        logger.info(f"Processing complete: {self.stats}")
        return self.stats

    def _process_standalone_documents(self, doc_files: list) -> None:
        """Process standalone documents (PDF, DOCX, etc.) from bronze/documents/.

        Each document is chunked, entity-extracted, and saved as document_chunks.
        No thread grouping or classification — all documents are treated as knowledge.
        """
        for i, doc_file in enumerate(doc_files):
            try:
                with open(doc_file, encoding="utf-8") as f:
                    doc_data = json.load(f)

                doc_id = doc_data.get("doc_id", doc_file.stem)
                text = doc_data.get("text", "")
                filename = doc_data.get("source_path", doc_data.get("filename", doc_file.name))
                doc_data.get("doc_type", "document")

                if not text or not text.strip() or len(text.strip()) < 50:
                    continue

                logger.info(f"  [{i + 1}/{len(doc_files)}] Document: {filename[:60]}")

                # Clean text
                cleaned = clean_email_text(text)

                # Detect language
                lang_result = self.language_detector.detect(cleaned)
                language = lang_result.language

                # Chunk
                text_chunks = self.chunker.chunk(
                    text=cleaned, doc_id=doc_id, metadata={"filename": filename}
                )

                if not text_chunks:
                    continue

                # Entity extraction + translation
                extract_per_chunk = len(cleaned) > 6000
                if not extract_per_chunk:
                    kg_entity_dicts, kg_entities_raw, llm_text_english, _detected_lang = (
                        self._extract_kg_entities(cleaned, language, chunk_id=doc_id)
                    )
                    text_english = llm_text_english if llm_text_english else cleaned
                    kg_relationships = self._extract_kg_relationships(
                        text_english, kg_entities_raw, language, chunk_id=doc_id
                    )
                else:
                    kg_entity_dicts = None
                    kg_relationships = None
                    text_english = cleaned

                # Create and save chunks
                for chunk in text_chunks:
                    chunk_id = self._safe_filename(f"doc_{doc_id}_{chunk.chunk_index}")

                    chunk_cleaned = clean_email_text(chunk.text)

                    # Per-chunk extraction for large documents
                    if extract_per_chunk:
                        chunk_entities, chunk_entities_raw, chunk_english, _chunk_lang = (
                            self._extract_kg_entities(chunk_cleaned, language, chunk_id=chunk_id)
                        )
                        if not chunk_english:
                            chunk_english = chunk_cleaned
                        chunk_rels = self._extract_kg_relationships(
                            chunk_english, chunk_entities_raw, language, chunk_id=chunk_id
                        )
                    else:
                        chunk_entities = kg_entity_dicts
                        chunk_rels = kg_relationships
                        chunk_english = text_english if len(text_chunks) == 1 else chunk_cleaned

                    thread_chunk = ThreadChunk(
                        chunk_id=chunk_id,
                        thread_id=f"doc:{doc_id}",
                        chunk_index=chunk.chunk_index,
                        text_original=chunk.text,
                        text_anonymized=chunk_cleaned,
                        text_english=chunk_english,
                        token_count=chunk.token_count,
                        thread_subject=filename,
                        thread_participants=[],
                        thread_email_count=0,
                        email_position="",
                        pii_entities=[],
                        pii_count=0,
                        kg_entities=chunk_entities or [],
                        kg_relationships=chunk_rels or [],
                        has_attachments=False,
                        attachment_count=0,
                        attachment_filenames=[],
                        source_email_ids=[doc_id],
                        source_type="document",
                        anonymization_skipped=True,
                        language=language,
                        processing_mode=self.processing_mode,
                    )

                    # Generate summary
                    if self.generate_summaries:
                        summary_doc = {"email_body_text": chunk.text, "record_id": doc_id}
                        summary_text = self._summarize_email(summary_doc, language)
                        if summary_text:
                            thread_chunk.summary = summary_text

                    self._save_individual_chunk(thread_chunk)
                    self.stats["chunks_created"] += 1

                self.stats.setdefault("documents_processed", 0)
                self.stats["documents_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing document {doc_file.name}: {e}")
                self.stats["errors"] += 1

    def _process_email_segments(
        self,
        email_text: str,
        email_record_id: str,
        thread: EmailThread,
        email: dict[str, Any],
        email_idx: int,
        total_emails: int,
        has_attachments: bool,
        attachment_count: int,
        attachment_filenames: list[str],
        language: str,
        bronze_index: dict | None = None,
    ) -> list[ThreadChunk]:
        """
        Process an email body by splitting quoted replies, extracting entities
        at the segment level (not chunk level), and chunking for retrieval.

        Flow:
        1. Split email into primary content + quoted replies
        2. Match quoted replies against Bronze index (skip matched, keep orphans)
        3. For each segment (primary + orphans):
           a. Extract entities from full segment text (one LLM call)
           b. Chunk segment if > chunk_size tokens
           c. All chunks inherit segment-level entities
           d. Each chunk gets its own summary
        """
        from tacitgraph.bronze.reply_splitter import match_quoted_replies, split_replies

        all_chunks = []
        email_headers = email.get("email_headers", {})
        parent_subject = email_headers.get("subject", thread.subject)

        # Step 1: Split quoted replies
        segments = split_replies(email_text)

        # Step 2: Match against Bronze index
        if bronze_index and len(segments) > 1:
            segments = match_quoted_replies(segments, bronze_index, parent_subject=parent_subject)
            matched = sum(1 for s in segments if s.status == "matched")
            orphan = sum(1 for s in segments if s.status == "orphan")
            if matched or orphan:
                logger.info(
                    f"    Reply split: {len(segments)} segments ({matched} matched Bronze, {orphan} orphan)"
                )

        # Step 3: Process each segment (primary + orphans)
        seg_counter = 0
        for seg in segments:
            if seg.status == "matched":
                continue  # Will be processed from its own Bronze record

            seg_text = seg.text.strip()
            if not seg_text or len(seg_text) < 20:
                continue

            # Determine sender/date for this segment
            if seg.is_primary:
                seg_sender = email_headers.get("sender", "")
                seg_sender_addr = email_headers.get("sender_email", "")
                seg_recipients_to = email_headers.get("recipients_to", [])
                seg_recipients_cc = email_headers.get("recipients_cc", [])
                seg_sent = self._get_email_timestamps(email)[0]
                seg_received = self._get_email_timestamps(email)[1]
                seg_position = f"{email_idx + 1}/{total_emails}"
            else:
                seg_sender = seg.parsed_sender
                seg_sender_addr = ""
                seg_recipients_to = []
                seg_recipients_cc = []
                seg_sent = seg.parsed_date
                seg_received = ""
                seg_position = f"{email_idx + 1}/{total_emails}:q{seg_counter}"

            # Prepend email metadata for better entity extraction (full names, facility refs)
            subject = thread.subject
            prefix_parts = []
            if subject:
                prefix_parts.append(f"Subject: {subject}")
            if seg_sender:
                prefix_parts.append(f"From: {seg_sender}")
            if seg_recipients_to:
                recip_names = [
                    r.get("name", "") if isinstance(r, dict) else str(r) for r in seg_recipients_to
                ]
                recip_names = [n for n in recip_names if n]
                if recip_names:
                    prefix_parts.append(f"To: {', '.join(recip_names)}")
            prefix = "\n".join(prefix_parts)
            full_text = f"{prefix}\n\n{seg_text}" if prefix else seg_text
            cleaned_text = self._clean_text(full_text)

            # Detect language for this segment
            lang_result = self.language_detector.detect(seg_text)
            seg_lang = lang_result.language

            # Chunk the segment text
            text_chunks = self.chunker.chunk(
                text=seg_text,
                doc_id=f"{email_record_id}_s{seg_counter}",
                metadata={"subject": subject},
            )

            # Extract entities: from full segment if small enough, per-chunk if large
            seg_chunk_id = self._safe_filename(f"{email_record_id}_s{seg_counter}")
            extract_per_chunk = len(cleaned_text) > 6000

            if not extract_per_chunk:
                # Small segment: extract once, share across chunks
                kg_entity_dicts, kg_entities_raw, llm_text_english, detected_lang = (
                    self._extract_kg_entities(cleaned_text, seg_lang, chunk_id=seg_chunk_id)
                )

                if llm_text_english:
                    text_english = llm_text_english
                    seg_lang = detected_lang or seg_lang
                else:
                    text_english = cleaned_text

                kg_relationships = self._extract_kg_relationships(
                    text_english, kg_entities_raw, seg_lang, chunk_id=seg_chunk_id
                )
            else:
                logger.info(
                    f"    Segment {seg_counter} too large ({len(cleaned_text)} chars), extracting per chunk"
                )
                kg_entity_dicts = None  # Will extract per chunk below
                kg_relationships = None
                text_english = cleaned_text

            # Create chunks
            seg_chunks = []
            consecutive_filter_blocks_seg = 0
            skip_llm_for_segment = False

            for chunk in text_chunks:
                chunk_id = self._safe_filename(
                    f"{email_record_id}_s{seg_counter}_{chunk.chunk_index}"
                )

                chunk_text_with_subject = (
                    f"Subject: {subject}. {chunk.text}" if subject else chunk.text
                )
                chunk_cleaned = self._clean_text(chunk_text_with_subject)

                # Per-chunk extraction for large segments
                if extract_per_chunk:
                    if skip_llm_for_segment:
                        chunk_entities, chunk_entities_raw = [], []
                        chunk_english = chunk_cleaned
                        chunk_rels = []
                    else:
                        prev_filter = self.stats.get("content_filter_blocks", 0)
                        chunk_entities, chunk_entities_raw, chunk_llm_english, _chunk_lang = (
                            self._extract_kg_entities(chunk_cleaned, seg_lang, chunk_id=chunk_id)
                        )
                        chunk_english = chunk_llm_english or chunk_cleaned
                        chunk_rels = self._extract_kg_relationships(
                            chunk_english, chunk_entities_raw, seg_lang, chunk_id=chunk_id
                        )
                        # Circuit breaker: skip LLM after 3 consecutive content filter blocks
                        if self.stats.get("content_filter_blocks", 0) > prev_filter:
                            consecutive_filter_blocks_seg += 1
                            if consecutive_filter_blocks_seg >= 3:
                                skip_llm_for_segment = True
                                logger.warning(
                                    f"Content filter blocked 3 consecutive chunks for segment — "
                                    f"skipping LLM for remaining {len(text_chunks) - text_chunks.index(chunk) - 1} chunks"
                                )
                        else:
                            consecutive_filter_blocks_seg = 0
                else:
                    # Small segment: inherit segment-level entities
                    chunk_entities = kg_entity_dicts
                    chunk_rels = kg_relationships
                    if llm_text_english and len(text_chunks) == 1:
                        chunk_english = text_english
                    else:
                        chunk_english = chunk_cleaned

                thread_chunk = ThreadChunk(
                    chunk_id=chunk_id,
                    thread_id=thread.conversation_id,
                    chunk_index=chunk.chunk_index,
                    text_original=chunk.text,
                    text_anonymized=chunk_cleaned,
                    text_english=chunk_english,
                    token_count=chunk.token_count,
                    thread_subject=subject,
                    thread_participants=thread.participants,
                    thread_email_count=total_emails,
                    email_position=seg_position,
                    pii_entities=[],
                    pii_count=0,
                    kg_entities=chunk_entities,
                    kg_relationships=chunk_rels,
                    has_attachments=has_attachments if seg.is_primary else False,
                    attachment_count=attachment_count if seg.is_primary else 0,
                    attachment_filenames=attachment_filenames if seg.is_primary else [],
                    source_email_ids=[email_record_id] if email_record_id != "unknown" else [],
                    source_type="email",
                    anonymization_skipped=True,
                    email_sender=seg_sender,
                    email_sender_address=seg_sender_addr,
                    email_recipients_to=seg_recipients_to,
                    email_recipients_cc=seg_recipients_cc,
                    sent_timestamp=seg_sent,
                    received_timestamp=seg_received,
                    language=seg_lang,
                    processing_mode=self.processing_mode,
                )

                seg_chunks.append(thread_chunk)
                all_chunks.append(thread_chunk)
                self._save_individual_chunk(thread_chunk)
                self.stats["chunks_created"] += 1

            # Per-segment summary — each segment gets its own summary
            if self.generate_summaries and seg_chunks:
                summary_email = dict(email)
                summary_email["email_body_text"] = seg_text
                summary_text = self._summarize_email(summary_email, seg_lang)
                if summary_text:
                    self.stats["email_summaries_generated"] += 1
                    for chunk in seg_chunks:
                        chunk.summary = summary_text
                        self._save_individual_chunk(chunk)

            seg_counter += 1

        return all_chunks

    def _process_thread(self, thread: EmailThread) -> list[ThreadChunk]:
        """Process a multi-email thread — each email chunked individually.

        Each email is classified individually. Personal emails are skipped.
        Each work email is chunked, summarized, and saved to email_chunks/.
        A thread summary is generated across all work emails for full-thread context.
        """
        chunks = []

        # Classify each email individually — keep only not_personal ones with content
        work_emails = []
        for email in thread.emails:
            # Skip emails with no body text AND no attachments
            body = email.get("email_body_text", "")
            has_att = email.get("document_metadata", {}).get("has_attachments", False)
            if (not body or not body.strip()) and not has_att:
                self.stats["emails_skipped_empty"] += 1
                continue

            result = self.sensitivity_classifier.classify(email)
            subj = email.get("email_headers", {}).get("subject", "")[:40]
            logger.debug(
                f"Sensitivity: '{subj}' → {result.classification} ({result.confidence:.2f})"
            )

            if result.classification == "personal":
                self.stats["emails_skipped_personal"] += 1
                logger.info(f"Email '{subj}' classified as personal — skipping")
            else:
                work_emails.append(email)
                self.stats["emails_not_personal"] += 1

        if not work_emails:
            self.stats["threads_skipped_personal"] += 1
            self._save_personal(thread)
            logger.info(f"Thread '{thread.subject[:50]}' — all emails personal — skipping")
            return chunks

        self.stats["threads_not_personal"] += 1

        # Build a filtered thread with only work emails (for thread summary later)
        filtered_thread = EmailThread(
            conversation_id=thread.conversation_id,
            subject=thread.subject,
            emails=work_emails,
            participants=thread.participants,
        )

        # Build Bronze index for reply matching (one-time per processing run)
        if not hasattr(self, "_bronze_index"):
            self._bronze_index = None  # Built lazily on first use

        # Process each email individually — split replies, extract KG, summarize
        all_attachment_contents = []
        language = "en"

        for email_idx, email in enumerate(work_emails):
            email_record_id = email.get("record_id", "unknown")
            self._current_email_id = email_record_id
            email_text = email.get("email_body_text", "")
            email_text = clean_email_text(email_text)
            email_meta = email.get("document_metadata", {})

            # Collect attachments for this email
            attachment_filenames = []
            email_attachment_contents = []
            if (
                self.process_attachments
                and self.attachment_processor
                and email_meta.get("has_attachments")
            ) and email_record_id != "unknown":
                raw_contents = self.attachment_processor.get_email_attachment_content(
                    email_record_id
                )
                for att_content in raw_contents:
                    self.stats["attachments_processed"] += 1
                    attachment_filenames.append(att_content.filename)
                    email_attachment_contents.append(att_content)
                    all_attachment_contents.append(att_content)

            has_attachments = len(attachment_filenames) > 0 or email_meta.get(
                "has_attachments", False
            )
            attachment_count = len(attachment_filenames) or email_meta.get("attachment_count", 0)

            if not email_text.strip():
                continue

            logger.info(f"  Email {email_idx + 1}/{len(work_emails)}: {len(email_text)} chars")

            email_chunks = self._process_email_segments(
                email_text=email_text,
                email_record_id=email_record_id,
                thread=filtered_thread,
                email=email,
                email_idx=email_idx,
                total_emails=len(work_emails),
                has_attachments=has_attachments,
                attachment_count=attachment_count,
                attachment_filenames=attachment_filenames,
                language=language,
                bronze_index=self._bronze_index,
            )
            chunks.extend(email_chunks)

        # Process attachments separately → attachment_chunks/ + attachment_summaries/
        attachment_ids = []
        if all_attachment_contents:
            # Use earliest email timestamps for attachments
            all_ts = [self._get_email_timestamps(e) for e in work_emails]
            sent_times = [t[0] for t in all_ts if t[0]]
            recv_times = [t[1] for t in all_ts if t[1]]
            att_chunks, attachment_ids = self._process_attachments_separately(
                attachment_contents=all_attachment_contents,
                thread_id=filtered_thread.conversation_id,
                subject=filtered_thread.subject,
                participants=filtered_thread.participants,
                email_count=filtered_thread.email_count,
                language=language,
                should_anonymize=False,
                source_email_ids=[
                    e.get("record_id", "") for e in work_emails if e.get("record_id")
                ],
                sent_timestamp=min(sent_times) if sent_times else "",
                received_timestamp=min(recv_times) if recv_times else "",
            )
            chunks.extend(att_chunks)

        # Generate thread summary — captures full conversation outcome across all emails
        all_email_chunks = [c for c in chunks if c.source_type == "email"]
        if self.generate_summaries and all_email_chunks:
            logger.info(
                f"  Generating thread summary for '{filtered_thread.subject[:50]}' ({len(all_email_chunks)} chunks)"
            )
            self._generate_thread_summary(
                filtered_thread, all_email_chunks, language, attachment_ids=attachment_ids
            )

        return chunks

    def _process_single_email(self, thread: EmailThread) -> list[ThreadChunk]:
        """Process a single email (attachments processed separately)"""
        chunks = []
        email = thread.emails[0] if thread.emails else None

        if not email:
            return chunks

        # Skip emails with no body text AND no attachments — nothing to process
        body = email.get("email_body_text", "")
        has_att = email.get("document_metadata", {}).get("has_attachments", False)
        if (not body or not body.strip()) and not has_att:
            self.stats["emails_skipped_empty"] += 1
            return chunks

        # Classify this email — if personal, skip processing entirely
        result = self._classify_email(email)
        if result.classification == "personal":
            self.stats["emails_skipped_personal"] += 1
            self.stats["threads_skipped_personal"] += 1
            self._save_personal(thread)
            logger.info(
                f"Email '{thread.subject[:50]}' classified as personal — skipping processing"
            )
            return chunks

        self.stats["emails_not_personal"] += 1
        self.stats["threads_not_personal"] += 1

        # Get email body text only (no attachments)
        email_text, _ = self._format_single_email(email, include_attachments=False)
        email_text = clean_email_text(email_text)
        # Check raw body — formatted text may have headers even when body is empty
        raw_body = email.get("email_body_text", "")
        has_body = bool(raw_body and raw_body.strip())

        # Collect attachment contents separately
        attachment_filenames = []
        attachment_contents = []

        email_meta = email.get("document_metadata", {})
        if (
            self.process_attachments
            and self.attachment_processor
            and email_meta.get("has_attachments")
        ):
            email_id = email.get("record_id", "")
            if email_id:
                raw_contents = self.attachment_processor.get_email_attachment_content(email_id)
                for att_content in raw_contents:
                    self.stats["attachments_processed"] += 1
                    attachment_filenames.append(att_content.filename)
                    attachment_contents.append(att_content)

        has_attachments = len(attachment_filenames) > 0 or email_meta.get("has_attachments", False)
        attachment_count = len(attachment_filenames) or email_meta.get("attachment_count", 0)

        if not has_body and not attachment_contents:
            return chunks

        # Detect language
        language = "en"
        if has_body:
            lang_result = self.language_detector.detect(email_text)
            language = lang_result.language

        email_record_id = email.get("record_id", "unknown")

        # Build Bronze index lazily
        if not hasattr(self, "_bronze_index"):
            self._bronze_index = None

        # Process email body if it has content
        if has_body:
            logger.info(f"  Email body: {len(email_text)} chars")
            email_chunks = self._process_email_segments(
                email_text=email_text,
                email_record_id=email_record_id,
                thread=thread,
                email=email,
                email_idx=0,
                total_emails=1,
                has_attachments=has_attachments,
                attachment_count=attachment_count,
                attachment_filenames=attachment_filenames,
                language=language,
                bronze_index=self._bronze_index,
            )
            chunks.extend(email_chunks)
        else:
            logger.info("  Email body empty, processing attachments only")

        # Process attachments separately → attachment_chunks/ + attachment_summaries/
        if attachment_contents:
            att_chunks, _attachment_ids = self._process_attachments_separately(
                attachment_contents=attachment_contents,
                thread_id=thread.conversation_id,
                subject=thread.subject,
                participants=thread.participants,
                email_count=1,
                language=language,
                should_anonymize=False,  # Not personal email: no anonymization
                source_email_ids=[email_record_id] if email_record_id != "unknown" else [],
                sent_timestamp=self._get_email_timestamps(email)[0],
                received_timestamp=self._get_email_timestamps(email)[1],
            )
            chunks.extend(att_chunks)

        # Summaries are generated per-segment inside _process_email_segments

        return chunks

    def _format_single_email(
        self, email: dict[str, Any], include_attachments: bool = True
    ) -> tuple[str, list[str]]:
        """
        Format a single email for processing, including attachment content.

        Args:
            email: Email data dictionary
            include_attachments: Whether to include attachment text

        Returns:
            Tuple of (formatted_text, attachment_filenames)
        """
        parts = []
        attachment_filenames = []

        headers = email.get("email_headers", {})
        meta = email.get("document_metadata", {})

        if headers.get("subject"):
            parts.append(f"Subject: {headers['subject']}")
        if headers.get("sender"):
            parts.append(f"From: {headers['sender']}")
        if meta.get("sent_time"):
            parts.append(f"Date: {meta['sent_time']}")

        parts.append("")

        if email.get("email_body_text"):
            parts.append(email["email_body_text"])

        # Process attachments if enabled
        if (
            include_attachments
            and self.include_attachment_text
            and self.attachment_processor
            and meta.get("has_attachments")
        ):
            email_id = email.get("record_id", "")
            if email_id:
                attachment_contents = self.attachment_processor.get_email_attachment_content(
                    email_id
                )

                for att_content in attachment_contents:
                    self.stats["attachments_processed"] += 1
                    attachment_filenames.append(att_content.filename)

                    if att_content.extraction_success and att_content.text.strip():
                        self.stats["attachments_with_text"] += 1
                        parts.append("")
                        parts.append(f"--- Attachment: {att_content.filename} ---")
                        parts.append(att_content.text)
                        parts.append("--- End Attachment ---")

        return "\n".join(parts), attachment_filenames

    def _generate_thread_summary(
        self,
        thread: EmailThread,
        chunks: list[ThreadChunk],
        language: str,
        attachment_ids: list[str] | None = None,
    ) -> ThreadSummary | None:
        """Generate a summary for the thread with attachment cross-references."""
        # Build date range string
        date_range = ""
        if thread.start_date and thread.end_date:
            if thread.start_date.date() == thread.end_date.date():
                date_range = thread.start_date.strftime("%Y-%m-%d")
            else:
                date_range = f"{thread.start_date.strftime('%Y-%m-%d')} to {thread.end_date.strftime('%Y-%m-%d')}"

        # Extract key topics (simple: from subject + common words)
        key_topics = self._extract_key_topics(thread, chunks)

        # Generate summary — LLM only in llm/hybrid mode
        if self.processing_mode in ("llm", "hybrid") and self.openai_api_key:
            summary_text = self._generate_llm_summary(thread, chunks)
        else:
            summary_text = self._generate_simple_summary(thread, chunks)

        summary = ThreadSummary(
            thread_id=thread.conversation_id,
            subject=thread.subject,
            participants=thread.participants,
            email_count=thread.email_count,
            date_range=date_range,
            summary=summary_text,
            key_topics=key_topics,
            chunk_ids=[c.chunk_id for c in chunks],
            attachment_ids=attachment_ids or [],
            source_email_ids=[e.get("record_id", "") for e in thread.emails if e.get("record_id")],
        )

        self._save_thread_summary(summary)
        self.stats["summaries_generated"] += 1

        return summary

    def _summarize_email(self, email: dict[str, Any], language: str) -> str | None:
        """Generate a work-focused summary for a single email.

        Strips personal/social chatter (congratulations, personal news, etc.)
        but keeps all professional content: names, roles, decisions, actions.
        Returns summary text only — no file is written.
        """
        body_text = email.get("email_body_text", "")
        if not body_text or not body_text.strip():
            return None

        body_text = clean_email_text(body_text)
        if not body_text.strip():
            return None

        if self.processing_mode in ("llm", "hybrid") and self.openai_api_key:
            return self._generate_llm_email_summary(email, body_text)
        else:
            return self._generate_simple_email_summary(email, body_text)

    def _generate_email_summary(
        self,
        email: dict[str, Any],
        thread: EmailThread,
        language: str,
        attachment_ids: list[str] | None = None,
    ) -> EmailSummary | None:
        """Generate a per-email summary and save to email_summaries/."""
        email_id = email.get("record_id", "")
        if not email_id:
            return None

        # Get email text
        body_text = email.get("email_body_text", "")
        if not body_text or not body_text.strip():
            return None

        # Clean the text
        body_text = clean_email_text(body_text)
        body_text = self._clean_text(body_text)

        if not body_text.strip():
            return None

        # Get sender and date
        headers = email.get("email_headers", {})
        sender = headers.get("sender", "") or headers.get("sender_email", "")
        doc_meta = email.get("document_metadata", {})
        date = doc_meta.get("delivery_time", "") or doc_meta.get("creation_time", "")

        # Generate summary text
        if self.processing_mode in ("llm", "hybrid") and self.openai_api_key:
            summary_text = self._generate_llm_email_summary(email, body_text)
        else:
            summary_text = self._generate_simple_email_summary(email, body_text)

        # Find chunk_ids that belong to this email
        chunk_ids = []
        # For single emails, chunk_id starts with record_id
        # For thread emails, we can't map precisely — leave empty
        if thread.email_count == 1:
            chunk_dir = self.silver_path / "not_personal" / "email_chunks"
            if chunk_dir.exists():
                for f in chunk_dir.glob(f"{email_id}_*.json"):
                    chunk_ids.append(f.stem)

        summary = EmailSummary(
            email_id=email_id,
            thread_id=thread.conversation_id,
            subject=headers.get("subject", thread.subject),
            sender=sender,
            date=str(date),
            summary=summary_text,
            chunk_ids=chunk_ids,
            attachment_ids=attachment_ids or [],
        )

        self._save_email_summary(summary)
        self.stats.setdefault("email_summaries_generated", 0)
        self.stats["email_summaries_generated"] += 1

        return summary

    def _generate_simple_email_summary(self, email: dict[str, Any], text: str) -> str:
        """Generate email summary using local BART model."""
        from tacitgraph.silver.local_summarizer import summarize_thread

        headers = email.get("email_headers", {})
        subject = headers.get("subject", "No Subject")
        sender = headers.get("sender", "")

        return summarize_thread(
            subject=subject,
            participants=[sender] if sender else [],
            email_count=1,
            text=text,
        )

    def _generate_llm_email_summary(self, email: dict[str, Any], text: str) -> str:
        """Generate intent-only email summary using LLM.

        Phase 2 anonymization: summaries strip irrelevant details and sensitive
        content, capturing only the intent — not specifics.
        """
        try:
            import httpx

            if self.use_azure:
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=self.openai_api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_api_version,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    max_retries=2,
                )
                model = self.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            else:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_api_key)
                model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

            headers = email.get("email_headers", {})
            subject = headers.get("subject", "")

            from tacitgraph.prompt_loader import get_prompt

            system_prompt = get_prompt(
                "silver",
                "email_summary",
                "system_prompt",
                "Summarize the intent of this email. Remove all sensitive content and specific details. Capture only what was discussed, decided, and needed.",
            )
            user_prompt_template = get_prompt(
                "silver",
                "email_summary",
                "user_prompt",
                "Summarize the intent of this email. Remove all sensitive content, names, and specific details.\n\nSubject: {subject}\n\n{text}\n\nIntent summary:",
            )
            user_prompt = user_prompt_template.format(subject=subject, text=text[:4000])

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=get_prompt("silver", "email_summary", "temperature", 0.3),
                max_tokens=get_prompt("silver", "email_summary", "max_tokens", 300),
            )

            from tacitgraph.llm_response import LLMContentError, extract_llm_content

            try:
                return extract_llm_content(response, context="email summary")
            except LLMContentError as lce:
                if "length" in str(lce):
                    # Truncated summary — use partial content instead of falling back
                    choice = response.choices[0] if response.choices else None
                    content = getattr(getattr(choice, "message", None), "content", None)
                    if content and len(content) > 20:
                        logger.warning(
                            f"Email summary truncated, using partial ({len(content)} chars)"
                        )
                        return content
                raise

        except Exception as e:
            email_id = email.get("record_id", "unknown")
            subject = email.get("email_headers", {}).get("subject", "")
            error_str = str(e)
            is_content_filter = "content_filter" in error_str
            if is_content_filter:
                self.stats["content_filter_blocks"] += 1
            self.stats["llm_failures"].append(
                {
                    "chunk_id": email_id,
                    "stage": "email_summary",
                    "reason": "content_filter" if is_content_filter else "error",
                    "message": error_str[:200],
                }
            )
            logger.warning(
                f"LLM email summary failed for email_id={email_id} subject='{subject[:80]}': {e}"
            )
            return self._generate_simple_email_summary(email, text)

    def _save_email_summary(self, summary: EmailSummary) -> None:
        """Save email summary to Silver layer."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in summary.email_id)[:100]
        summary_file = self.silver_path / "not_personal" / "email_summaries" / f"{safe_id}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _generate_attachment_summary(
        self,
        attachment_id: str,
        filename: str,
        thread_id: str,
        chunks: list[ThreadChunk],
        classification: str,
        language: str,
    ) -> AttachmentSummary | None:
        """Generate and save a per-attachment summary."""
        if not chunks:
            return None

        total_tokens = sum(c.token_count for c in chunks)
        chunk_ids = [c.chunk_id for c in chunks]

        # Generate summary text — LLM only in llm/hybrid mode
        if self.processing_mode in ("llm", "hybrid") and self.openai_api_key:
            summary_text = self._generate_llm_attachment_summary(filename, chunks)
        else:
            summary_text = self._generate_simple_attachment_summary(filename, chunks)

        att_summary = AttachmentSummary(
            attachment_id=attachment_id,
            thread_id=thread_id,
            filename=filename,
            summary=summary_text,
            chunk_ids=chunk_ids,
            classification=classification,
            token_count=total_tokens,
        )

        self._save_attachment_summary(att_summary)
        self.stats["attachment_summaries_generated"] += 1

        return att_summary

    def _generate_simple_attachment_summary(
        self,
        filename: str,
        chunks: list[ThreadChunk],
    ) -> str:
        """Generate an attachment summary using local BART model."""
        from tacitgraph.silver.local_summarizer import summarize_attachment

        combined_text = "\n\n".join(c.text_anonymized for c in chunks)
        return summarize_attachment(filename, len(chunks), combined_text)

    def _generate_llm_attachment_summary(
        self,
        filename: str,
        chunks: list[ThreadChunk],
    ) -> str:
        """Generate an LLM-based attachment summary scaled to document size."""
        try:
            if self.use_azure:
                import httpx
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=self.openai_api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_api_version,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    max_retries=2,
                )
                model = self.azure_deployment or "gpt-4o"
            else:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_api_key)
                model = "gpt-4o"

            # Scale context window with document size
            # Small docs (1-2 chunks): 4000 chars, large docs (10+): up to 12000
            max_chars = min(12000, 4000 + len(chunks) * 1000)
            combined_text = "\n\n".join([c.text_anonymized for c in chunks])[:max_chars]

            from tacitgraph.prompt_loader import format_prompt, get_prompt

            user_template = get_prompt(
                "silver",
                "attachment_summary",
                "user_prompt",
                "Summarize this attachment:\nFilename: {filename}\nChunks: {chunk_count}\n\nContent:\n{content}\n\nSummary:",
            )
            user_prompt = format_prompt(
                user_template,
                filename=filename,
                chunk_count=str(len(chunks)),
                content=combined_text,
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt(
                            "silver",
                            "attachment_summary",
                            "system_prompt",
                            "Summarize this document attachment proportional to its length.",
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=get_prompt("silver", "attachment_summary", "temperature", 0.3),
                max_tokens=get_prompt("silver", "attachment_summary", "max_tokens", 1000),
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            return extract_llm_content(response, context="attachment summary")

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, network timeouts, etc. Falls back to extractive summary.
            logger.warning(f"LLM attachment summary failed for '{filename}': {e}")
            return self._generate_simple_attachment_summary(filename, chunks)

    def _save_attachment_summary(self, summary: AttachmentSummary) -> None:
        """Save attachment summary to Silver layer."""
        summary_dir = self.silver_path / "not_personal" / "attachment_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in summary.attachment_id)[
            :100
        ]
        summary_file = summary_dir / f"{safe_id}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _extract_key_topics(self, thread: EmailThread, chunks: list[ThreadChunk]) -> list[str]:
        """Extract key topics from thread"""
        # Simple extraction: words from subject
        topics = []
        subject_words = thread.subject.lower().split()
        stopwords = {"re", "fw", "fwd", "the", "a", "an", "and", "or", "is", "are", "was", "were"}

        for word in subject_words:
            word = word.strip(":-.,!?()[]")
            if len(word) > 2 and word not in stopwords:
                topics.append(word)

        return topics[:5]

    def _generate_simple_summary(self, thread: EmailThread, chunks: list[ThreadChunk]) -> str:
        """Generate a thread summary using local BART model."""
        if not chunks:
            return "No content available"

        from tacitgraph.silver.local_summarizer import summarize_thread

        combined_text = "\n\n".join(c.text_anonymized for c in chunks)
        return summarize_thread(
            subject=thread.subject,
            participants=thread.participants,
            email_count=thread.email_count,
            text=combined_text,
        )

    def _generate_llm_summary(self, thread: EmailThread, chunks: list[ThreadChunk]) -> str:
        """Generate summary using LLM (supports both OpenAI and Azure OpenAI)"""
        try:
            if self.use_azure:
                import httpx
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=self.openai_api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_api_version,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    max_retries=2,
                )
                model = self.azure_deployment or "gpt-4o"
            else:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_api_key)
                model = "gpt-4o"

            # Combine anonymized chunk texts
            combined_text = "\n\n".join([c.text_anonymized for c in chunks])[:4000]

            from tacitgraph.prompt_loader import get_prompt

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt(
                            "silver",
                            "thread_summary",
                            "system_prompt",
                            "Summarize this email thread in 2-3 sentences. Focus on the main topic and outcome.",
                        ),
                    },
                    {"role": "user", "content": combined_text},
                ],
                temperature=get_prompt("silver", "thread_summary", "temperature", 0.3),
                max_tokens=get_prompt("silver", "thread_summary", "max_tokens", 150),
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            return extract_llm_content(response, context="thread summary")

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, network timeouts, etc. Falls back to extractive summary.
            logger.warning(
                f"LLM thread summary failed for thread_id={thread.conversation_id} subject='{thread.subject[:80]}': {e}"
            )
            return self._generate_simple_summary(thread, chunks)

    def _anonymize_participant(self, participant: str) -> str:
        """Anonymize participant name"""
        # Use the same anonymizer for consistency
        result = self.anonymizer.anonymize(participant, "en")
        return result.anonymized_text

    @staticmethod
    def _get_email_timestamps(email: dict[str, Any]) -> tuple[str, str]:
        """Extract sent and received timestamps from Bronze email.
        Timestamps are in document_metadata (from Bronze PST extraction).
        Returns (sent_timestamp, received_timestamp)."""
        meta = email.get("document_metadata", {})
        return meta.get("sent_time", ""), meta.get("received_time", "")

    @staticmethod
    def _safe_filename(raw_id: str) -> str:
        """Generate a safe filename from an ID: short hash + sanitized prefix for readability."""
        hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:10]
        safe_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_id)[:60]
        return f"{safe_prefix}_{hash_suffix}"

    def _save_personal(self, thread: EmailThread) -> None:
        """Save a personal thread/email to the personal/ folder with full Bronze data."""
        safe_id = self._safe_filename(thread.conversation_id)
        out_file = self.silver_path / "personal" / f"{safe_id}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "thread_id": thread.conversation_id,
            "subject": thread.subject,
            "participants": thread.participants,
            "email_count": thread.email_count,
            "classification": "personal",
            "reason": "Thread classified as personal — skipped processing",
            "skipped_at": datetime.now().isoformat(),
            "emails": thread.emails,  # Full Bronze email data for reference
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)

    def _save_individual_chunk(self, chunk: ThreadChunk) -> None:
        """Save chunk to Silver layer — routes to appropriate subdirectory by source_type."""
        subdir = "document_chunks" if chunk.source_type == "document" else "email_chunks"
        chunk_file = self.silver_path / "not_personal" / subdir / f"{chunk.chunk_id}.json"
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"    Saved email chunk: {chunk.chunk_id} (thread={chunk.thread_id})")

    def _save_thread_summary(self, summary: ThreadSummary) -> None:
        """Save thread summary to Silver layer"""
        # Sanitize filename
        safe_id = self._safe_filename(summary.thread_id)
        summary_file = self.silver_path / "not_personal" / "thread_summaries" / f"{safe_id}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _save_metadata(self) -> None:
        """Save processing metadata"""
        metadata_file = self.silver_path / "metadata" / "thread_processing_stats.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing
        existing = []
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.append(self.stats)

        with open(metadata_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        # Save LLM failures separately for easy auditing
        failures = self.stats.get("llm_failures", [])
        if failures:
            failures_file = self.silver_path / "metadata" / "llm_failures.json"
            with open(failures_file, "w") as f:
                json.dump(
                    {
                        "total_failures": len(failures),
                        "content_filter_blocks": self.stats.get("content_filter_blocks", 0),
                        "failures": failures,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"Saved {len(failures)} LLM failure records to {failures_file}")


# Convenience function
def process_threads_to_silver(
    bronze_path: str, silver_path: str, chunk_size: int = 1024, openai_api_key: str | None = None
) -> dict[str, int]:
    """
    Process Bronze layer emails into thread-aware Silver layer.

    Args:
        bronze_path: Path to Bronze layer
        silver_path: Path for Silver layer output
        chunk_size: Target chunk size in tokens
        openai_api_key: Optional OpenAI key for summaries

    Returns:
        Processing statistics
    """
    processor = ThreadAwareProcessor(
        bronze_path=bronze_path,
        silver_path=silver_path,
        chunk_size=chunk_size,
        openai_api_key=openai_api_key,
    )

    return processor.process()
