"""
Attachment Content Processor — Bronze Layer (Pure Extraction)

Extracts text content from email attachments. No classification or filtering.
Files stay in attachments/{att_id}/ with co-located metadata JSON.

Classification happens in Silver layer (AttachmentClassifier).

Usage:
    processor = AttachmentProcessor(bronze_path="./data/bronze")
    content = processor.process_attachment(attachment_id, filename)

    # Or process all attachments for an email
    all_content = processor.get_email_attachment_content(email_id)
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .document_parser import DocumentParser

logger = logging.getLogger(__name__)


@dataclass
class AttachmentContent:
    """Extracted content from an attachment"""

    attachment_id: str
    filename: str
    email_id: str  # Parent email ID

    # Extracted content
    text: str
    doc_type: str
    page_count: int = 0

    # Metadata
    file_size: int = 0
    extraction_time: datetime = field(default_factory=datetime.now)
    extraction_success: bool = True
    error_message: str = ""

    # For KG processing
    has_tables: bool = False
    tables: list[dict[str, Any]] = field(default_factory=list)

    # Classification (populated by Silver layer, not Bronze)
    classification: str = ""
    classification_confidence: float = 0.0
    classification_signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attachment_id": self.attachment_id,
            "filename": self.filename,
            "email_id": self.email_id,
            "text": self.text,
            "doc_type": self.doc_type,
            "page_count": self.page_count,
            "file_size": self.file_size,
            "extraction_time": self.extraction_time.isoformat(),
            "extraction_success": self.extraction_success,
            "error_message": self.error_message,
            "has_tables": self.has_tables,
            "classification": self.classification,
            "classification_confidence": self.classification_confidence,
            "classification_signals": self.classification_signals,
            "char_count": len(self.text),
            "word_count": len(self.text.split()) if self.text else 0,
        }


class AttachmentProcessor:
    """
    Extracts text from email attachments (Bronze layer — pure extraction).

    Files stay in attachments/{att_id}/ with co-located {filename}.json metadata.
    No classification, no subdirectories, no file moving.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".txt",
        ".csv",
        ".rtf",
        ".html",
        ".htm",
    }

    def __init__(self, bronze_path: str, extract_tables: bool = True):
        """
        Initialize attachment processor.

        Args:
            bronze_path: Path to Bronze layer
            extract_tables: Extract tables from documents
        """
        self.bronze_path = Path(bronze_path)
        self.attachments_dir = self.bronze_path / "attachments"
        self.extract_tables = extract_tables

        # Initialize document parser
        self.parser = DocumentParser(extract_tables=extract_tables)

        # Statistics
        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "cached": 0,
            "unsupported": 0,
        }

        # Load email-attachment mapping
        self.email_attachments = self._load_attachment_mapping()

    def _load_attachment_mapping(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load mapping of emails to their attachments from email metadata.

        Returns:
            Dict mapping email_id to list of attachment info
        """
        mapping = {}
        emails_dir = self.bronze_path / "emails"

        if not emails_dir.exists():
            logger.warning(f"Emails directory not found: {emails_dir}")
            return mapping

        for email_file in emails_dir.rglob("*.json"):
            if "metadata" in email_file.parts:
                continue
            try:
                with open(email_file, encoding="utf-8") as f:
                    email_data = json.load(f)

                email_id = email_data.get("record_id") or email_file.stem

                meta = email_data.get("document_metadata", {})
                has_attachments = meta.get("has_attachments", False)
                attachment_count = meta.get("attachment_count", 0)

                if has_attachments and attachment_count > 0:
                    attachments = email_data.get("attachments", [])
                    if attachments:
                        mapping[email_id] = attachments
                    else:
                        mapping[email_id] = [
                            {"filename": f"attachment_{i}", "attachment_id": f"{email_id}_{i}"}
                            for i in range(attachment_count)
                        ]

            except Exception as e:
                logger.debug(f"Error reading email file {email_file}: {e}")

        logger.info(f"Found {len(mapping)} emails with attachments")
        return mapping

    # ------------------------------------------------------------------
    # Metadata (co-located alongside binary file)
    # ------------------------------------------------------------------

    def _get_metadata_path(self, attachment_id: str, filename: str = "") -> Path:
        """Metadata JSON: attachments/{att_id}/{att_id}.json (safe name, no special chars)."""
        return self.attachments_dir / attachment_id / f"{attachment_id}.json"

    def _load_metadata(self, attachment_id: str, filename: str = "") -> AttachmentContent | None:
        """Load extracted content from metadata JSON: attachments/{att_id}/{att_id}.json"""
        meta_path = self._get_metadata_path(attachment_id)
        if meta_path.exists():
            return self._parse_metadata_json(meta_path)
        return None

    def _parse_metadata_json(self, path: Path) -> AttachmentContent | None:
        """Parse a metadata JSON file into an AttachmentContent."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            return AttachmentContent(
                attachment_id=data["attachment_id"],
                filename=data["filename"],
                email_id=data["email_id"],
                text=data["text"],
                doc_type=data["doc_type"],
                page_count=data.get("page_count", 0),
                file_size=data.get("file_size", 0),
                extraction_success=data.get("extraction_success", True),
                error_message=data.get("error_message", ""),
                has_tables=data.get("has_tables", False),
                classification=data.get("classification", ""),
                classification_confidence=data.get("classification_confidence", 0.0),
                classification_signals=data.get("classification_signals", {}),
            )
        except Exception as e:
            logger.debug(f"Metadata load error for {path}: {e}")
            return None

    def _save_metadata(self, content: AttachmentContent) -> None:
        """Save extracted content as co-located JSON alongside the binary file."""
        meta_path = self._get_metadata_path(content.attachment_id)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(content.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.debug(f"Metadata save error for {content.attachment_id}: {e}")

    # ------------------------------------------------------------------
    # File lookup
    # ------------------------------------------------------------------

    def find_attachment_file(self, attachment_id: str, filename: str) -> Path | None:
        """
        Find the attachment binary file on disk (skips .json metadata files).

        Files are stored as: attachments/{attachment_id}/{attachment_id}.{ext}
        """
        att_dir = self.attachments_dir / attachment_id
        if not att_dir.exists():
            return None

        # Find by attachment_id name
        ext = Path(filename).suffix
        if ext:
            path = att_dir / f"{attachment_id}{ext}"
            if path.exists():
                return path

        # Fallback: any non-JSON file in the directory
        files = [f for f in att_dir.iterdir() if not f.name.endswith(".json")]
        if files:
            return files[0]

        return None

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_attachment(
        self, attachment_id: str, filename: str, email_id: str, file_path: Path | None = None
    ) -> AttachmentContent:
        """
        Process a single attachment: extract text and save metadata.

        No classification — that happens in Silver.

        Args:
            attachment_id: Unique attachment identifier
            filename: Original filename
            email_id: Parent email ID
            file_path: Optional direct path to file

        Returns:
            AttachmentContent with extracted text
        """
        self.stats["processed"] += 1

        # Check cached metadata first
        cached = self._load_metadata(attachment_id, filename)
        if cached:
            self.stats["cached"] += 1
            return cached

        # Check file extension
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            self.stats["unsupported"] += 1
            content = AttachmentContent(
                attachment_id=attachment_id,
                filename=filename,
                email_id=email_id,
                text="",
                doc_type=ext or "unknown",
                extraction_success=False,
                error_message=f"Unsupported file type: {ext or 'none'}",
            )
            self._save_metadata(content)
            return content

        # Find the file
        if file_path is None:
            file_path = self.find_attachment_file(attachment_id, filename)

        if file_path is None or not file_path.exists():
            self.stats["failed"] += 1
            return AttachmentContent(
                attachment_id=attachment_id,
                filename=filename,
                email_id=email_id,
                text="",
                doc_type=ext,
                extraction_success=False,
                error_message="Attachment file not found",
            )

        # Extract text using DocumentParser
        try:
            parsed = self.parser.parse(file_path)

            content = AttachmentContent(
                attachment_id=attachment_id,
                filename=filename,
                email_id=email_id,
                text=parsed.text,
                doc_type=parsed.doc_type.value,
                page_count=parsed.page_count,
                file_size=file_path.stat().st_size,
                extraction_success=True,
                has_tables=len(parsed.tables) > 0,
                tables=parsed.tables,
            )

            self.stats["success"] += 1
            self._save_metadata(content)
            return content

        except Exception as e:
            self.stats["failed"] += 1
            logger.warning(f"Failed to extract {filename}: {e}")

            return AttachmentContent(
                attachment_id=attachment_id,
                filename=filename,
                email_id=email_id,
                text="",
                doc_type=ext,
                extraction_success=False,
                error_message=str(e),
            )

    def get_email_attachment_content(self, email_id: str) -> list[AttachmentContent]:
        """
        Get all attachment content for an email.

        Args:
            email_id: Email message ID

        Returns:
            List of AttachmentContent for all attachments
        """
        attachments = self.email_attachments.get(email_id, [])
        contents = []

        for att_info in attachments:
            att_id = att_info.get("attachment_id", "")
            filename = att_info.get("filename", "unknown")

            content = self.process_attachment(
                attachment_id=att_id, filename=filename, email_id=email_id
            )
            contents.append(content)

        return contents

    def get_combined_attachment_text(
        self, email_id: str, separator: str = "\n\n---ATTACHMENT---\n\n"
    ) -> str:
        """
        Get combined text from all attachments for an email.

        Args:
            email_id: Email message ID
            separator: Text separator between attachments

        Returns:
            Combined text from all attachments
        """
        contents = self.get_email_attachment_content(email_id)

        texts = []
        for content in contents:
            if content.extraction_success and content.text.strip():
                header = f"[Attachment: {content.filename}]\n"
                texts.append(header + content.text)

        return separator.join(texts)

    def process_all_attachments(
        self, progress_callback: Callable[..., Any] | None = None
    ) -> dict[str, Any]:
        """
        Process all attachments in the Bronze layer.

        Args:
            progress_callback: Optional callback(count, message)

        Returns:
            Processing statistics
        """
        total_emails = len(self.email_attachments)

        for i, (email_id, attachments) in enumerate(self.email_attachments.items()):
            for att_info in attachments:
                att_id = att_info.get("attachment_id", "")
                filename = att_info.get("filename", "unknown")

                self.process_attachment(attachment_id=att_id, filename=filename, email_id=email_id)

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(
                    i + 1, f"Processed {i + 1}/{total_emails} emails with attachments"
                )

        return self.stats

    def scan_attachment_files(self) -> list[dict[str, Any]]:
        """
        Scan the attachments directory to find all attachment files.

        Returns:
            List of attachment file info
        """
        files = []

        if not self.attachments_dir.exists():
            logger.warning(f"Attachments directory not found: {self.attachments_dir}")
            return files

        for file_path in self.attachments_dir.rglob("*"):
            if file_path.name.endswith(".json"):
                continue
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                att_id = (
                    file_path.parent.name
                    if file_path.parent != self.attachments_dir
                    else file_path.stem
                )

                files.append(
                    {
                        "attachment_id": att_id,
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "extension": file_path.suffix.lower(),
                    }
                )

        logger.info(f"Found {len(files)} attachment files")
        return files


def extract_attachment_text(bronze_path: str, email_id: str) -> str:
    """
    Convenience function to extract text from email attachments.

    Args:
        bronze_path: Path to Bronze layer
        email_id: Email message ID

    Returns:
        Combined text from all attachments
    """
    processor = AttachmentProcessor(bronze_path=bronze_path)
    return processor.get_combined_attachment_text(email_id)
