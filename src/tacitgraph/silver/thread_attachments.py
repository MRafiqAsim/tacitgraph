"""Attachment processing: text quality checks, chunking and summaries for email attachments."""

import logging
from pathlib import Path

from tacitgraph.silver.thread_models import ThreadChunk
from tacitgraph.silver.thread_summaries import ThreadSummarization

logger = logging.getLogger(__name__)


class ThreadAttachmentProcessing(ThreadSummarization):
    """Attachment processing: text quality checks, chunking and summaries for email attachments."""

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
