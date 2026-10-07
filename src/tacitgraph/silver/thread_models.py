"""Silver-layer records produced by thread-aware processing."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
