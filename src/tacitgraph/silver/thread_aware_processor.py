"""
Thread-Aware Silver Processor

Processes email threads with semantic context preservation:
1. Groups emails into conversation threads
2. Concatenates thread emails chronologically
3. Chunks with thread boundaries respected
4. Generates thread summaries for high-level retrieval
5. Maintains consistent anonymization across thread
"""

import json
import logging
import os
from collections.abc import Callable
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
from tacitgraph.silver.thread_attachments import ThreadAttachmentProcessing
from tacitgraph.silver.thread_models import (
    ThreadChunk,
)

logger = logging.getLogger(__name__)


class ThreadAwareProcessor(ThreadAttachmentProcessing):
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

            # PIIDetector and OpenAIPIIDetector share the detect() interface used by Anonymizer
            self.pii_detector: Any = OpenAIPIIDetector(
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
        self.sensitivity_classifier: EmailSensitivityClassifier | LLMSensitivityClassifier
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

    def _classify_email(self, email: dict[str, Any]) -> SensitivityResult:
        """Classify a single email as personal or not_personal."""
        return self.sensitivity_classifier.classify(email)

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
                        chunk_entities = kg_entity_dicts or []
                        chunk_rels = kg_relationships or []
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
                        chunk_entities: list[dict[str, Any]] = []
                        chunk_entities_raw: list[KGEntity] = []
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
                    chunk_entities = kg_entity_dicts or []
                    chunk_rels = kg_relationships or []
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
                    for seg_chunk in seg_chunks:
                        seg_chunk.summary = summary_text
                        self._save_individual_chunk(seg_chunk)

            seg_counter += 1

        return all_chunks

    def _process_thread(self, thread: EmailThread) -> list[ThreadChunk]:
        """Process a multi-email thread — each email chunked individually.

        Each email is classified individually. Personal emails are skipped.
        Each work email is chunked, summarized, and saved to email_chunks/.
        A thread summary is generated across all work emails for full-thread context.
        """
        chunks: list[ThreadChunk] = []

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
        attachment_ids: list[str] = []
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
        chunks: list[ThreadChunk] = []
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
