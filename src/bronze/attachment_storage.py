"""
Attachment Storage Module

Handles storage and indexing of email attachments with full metadata
and relationship tracking for the Bronze layer.
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class AttachmentAnalysis:
    """Analysis results for routing decision."""
    is_image_based: bool = False
    has_extractable_text: bool = False
    estimated_pages: int = 0
    chars_per_page: float = 0.0
    has_embedded_fonts: bool = True
    image_ratio: float = 0.0
    has_tables: bool = False
    has_forms: bool = False
    complexity_score: float = 0.0
    recommended_processor: str = "pending"
    routing_reason: str = ""


@dataclass
class AttachmentMetadata:
    """Complete metadata for an attachment."""
    attachment_id: str
    source_email_id: str
    source_email_subject: str
    source_email_date: str
    source_email_sender: str
    filename: str
    content_type: str
    size: int
    file_hash: str = ""
    extraction_time: str = ""
    analysis: Optional[AttachmentAnalysis] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "attachment_id": self.attachment_id,
            "source_email_id": self.source_email_id,
            "source_email_subject": self.source_email_subject,
            "source_email_date": self.source_email_date,
            "source_email_sender": self.source_email_sender,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "file_hash": self.file_hash,
            "extraction_time": self.extraction_time,
        }
        if self.analysis:
            result["analysis"] = asdict(self.analysis)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttachmentMetadata":
        """Create from dictionary."""
        analysis_data = data.pop("analysis", None)
        analysis = AttachmentAnalysis(**analysis_data) if analysis_data else None
        return cls(**data, analysis=analysis)


@dataclass
class AttachmentIndex:
    """Index for quick attachment lookups."""
    by_email: Dict[str, List[str]] = field(default_factory=dict)
    by_type: Dict[str, List[str]] = field(default_factory=dict)
    by_processor: Dict[str, List[str]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "by_email": self.by_email,
            "by_type": self.by_type,
            "by_processor": self.by_processor,
            "statistics": self.statistics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttachmentIndex":
        """Create from dictionary."""
        return cls(
            by_email=data.get("by_email", {}),
            by_type=data.get("by_type", {}),
            by_processor=data.get("by_processor", {}),
            statistics=data.get("statistics", {})
        )


class AttachmentStorage:
    """
    Manages attachment storage with full metadata and relationship tracking.

    Directory Structure:
    bronze_path/
    ├── attachments/
    │   └── {attachment_id}/
    │       ├── original/
    │       │   └── {filename}
    │       └── metadata.json
    └── attachment_index.json
    """

    def __init__(self, bronze_path: str):
        """
        Initialize attachment storage.

        Args:
            bronze_path: Path to bronze layer directory
        """
        self.bronze_path = Path(bronze_path)
        self.attachments_dir = self.bronze_path / "attachments"
        self.index_path = self.bronze_path / "attachment_index.json"

        # Ensure directories exist
        self.attachments_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self.index = self._load_index()

        logger.info(f"AttachmentStorage initialized at {self.bronze_path}")

    def _load_index(self) -> AttachmentIndex:
        """Load existing index or create new one."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return AttachmentIndex.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, creating new")
        return AttachmentIndex()

    def _save_index(self):
        """Save index to disk."""
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index.to_dict(), f, indent=2)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return f"sha256:{sha256_hash.hexdigest()}"

    def store_attachment(
        self,
        attachment_id: str,
        filename: str,
        content: bytes,
        content_type: str,
        email_id: str,
        email_subject: str,
        email_date: str,
        email_sender: str
    ) -> AttachmentMetadata:
        """
        Store an attachment with full metadata.

        Args:
            attachment_id: Unique identifier for attachment
            filename: Original filename
            content: Raw attachment bytes
            content_type: MIME type
            email_id: Source email message ID
            email_subject: Source email subject
            email_date: Source email date
            email_sender: Source email sender

        Returns:
            AttachmentMetadata with storage details
        """
        # Create attachment directory
        att_dir = self.attachments_dir / attachment_id
        original_dir = att_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)
        file_path = original_dir / safe_filename

        # Write file
        with open(file_path, 'wb') as f:
            f.write(content)

        # Create metadata
        metadata = AttachmentMetadata(
            attachment_id=attachment_id,
            source_email_id=email_id,
            source_email_subject=email_subject,
            source_email_date=email_date,
            source_email_sender=email_sender,
            filename=filename,
            content_type=content_type,
            size=len(content),
            file_hash=self._compute_file_hash(file_path),
            extraction_time=datetime.now().isoformat(),
            analysis=AttachmentAnalysis(recommended_processor="pending")
        )

        # Save metadata
        metadata_path = att_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update index
        self._update_index(metadata)

        logger.debug(f"Stored attachment: {attachment_id} ({filename})")
        return metadata

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Replace problematic characters
        unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\x00']
        safe_name = filename
        for char in unsafe_chars:
            safe_name = safe_name.replace(char, '_')

        # Limit length
        if len(safe_name) > 200:
            name, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
            safe_name = name[:190] + ('.' + ext if ext else '')

        return safe_name or "unnamed_attachment"

    def _update_index(self, metadata: AttachmentMetadata):
        """Update index with new attachment."""
        att_id = metadata.attachment_id

        # By email
        if metadata.source_email_id not in self.index.by_email:
            self.index.by_email[metadata.source_email_id] = []
        if att_id not in self.index.by_email[metadata.source_email_id]:
            self.index.by_email[metadata.source_email_id].append(att_id)

        # By content type
        if metadata.content_type not in self.index.by_type:
            self.index.by_type[metadata.content_type] = []
        if att_id not in self.index.by_type[metadata.content_type]:
            self.index.by_type[metadata.content_type].append(att_id)

        # By processor (if analysis available)
        if metadata.analysis:
            processor = metadata.analysis.recommended_processor
            if processor not in self.index.by_processor:
                self.index.by_processor[processor] = []
            if att_id not in self.index.by_processor[processor]:
                self.index.by_processor[processor].append(att_id)

        # Update statistics
        self._update_statistics()

        # Save index
        self._save_index()

    def _update_statistics(self):
        """Update index statistics."""
        total_attachments = sum(len(atts) for atts in self.index.by_email.values())

        type_counts = {}
        for content_type, atts in self.index.by_type.items():
            # Simplify type name
            simple_type = content_type.split('/')[-1].split('.')[-1][:10]
            type_counts[simple_type] = len(atts)

        processor_counts = {
            proc: len(atts)
            for proc, atts in self.index.by_processor.items()
        }

        self.index.statistics = {
            "total_attachments": total_attachments,
            "unique_emails": len(self.index.by_email),
            "by_type_count": type_counts,
            "by_processor_count": processor_counts,
            "last_updated": datetime.now().isoformat()
        }

    def get_attachment_metadata(self, attachment_id: str) -> Optional[AttachmentMetadata]:
        """Get metadata for an attachment."""
        metadata_path = self.attachments_dir / attachment_id / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return AttachmentMetadata.from_dict(data)
        return None

    def get_attachment_path(self, attachment_id: str) -> Optional[Path]:
        """Get path to the original attachment file."""
        original_dir = self.attachments_dir / attachment_id / "original"
        if original_dir.exists():
            files = list(original_dir.iterdir())
            if files:
                return files[0]
        return None

    def get_attachments_for_email(self, email_id: str) -> List[AttachmentMetadata]:
        """Get all attachments for an email."""
        attachment_ids = self.index.by_email.get(email_id, [])
        return [
            self.get_attachment_metadata(att_id)
            for att_id in attachment_ids
            if self.get_attachment_metadata(att_id) is not None
        ]

    def get_attachments_by_type(self, content_type: str) -> List[str]:
        """Get attachment IDs by content type."""
        return self.index.by_type.get(content_type, [])

    def get_attachments_by_processor(self, processor: str) -> List[str]:
        """Get attachment IDs by recommended processor."""
        return self.index.by_processor.get(processor, [])

    def update_analysis(self, attachment_id: str, analysis: AttachmentAnalysis):
        """Update attachment analysis results."""
        metadata = self.get_attachment_metadata(attachment_id)
        if metadata:
            # Remove from old processor bucket
            if metadata.analysis and metadata.analysis.recommended_processor:
                old_processor = metadata.analysis.recommended_processor
                if old_processor in self.index.by_processor:
                    if attachment_id in self.index.by_processor[old_processor]:
                        self.index.by_processor[old_processor].remove(attachment_id)

            # Update metadata
            metadata.analysis = analysis

            # Save updated metadata
            metadata_path = self.attachments_dir / attachment_id / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Add to new processor bucket
            if analysis.recommended_processor not in self.index.by_processor:
                self.index.by_processor[analysis.recommended_processor] = []
            if attachment_id not in self.index.by_processor[analysis.recommended_processor]:
                self.index.by_processor[analysis.recommended_processor].append(attachment_id)

            # Update statistics and save
            self._update_statistics()
            self._save_index()

            logger.debug(f"Updated analysis for {attachment_id}: {analysis.recommended_processor}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.index.statistics

    def list_all_attachments(self) -> List[str]:
        """List all attachment IDs."""
        all_ids = set()
        for att_list in self.index.by_email.values():
            all_ids.update(att_list)
        return list(all_ids)

    def attachment_exists(self, attachment_id: str) -> bool:
        """Check if attachment exists."""
        return (self.attachments_dir / attachment_id / "metadata.json").exists()
