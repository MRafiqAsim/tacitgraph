"""
Bronze Layer Loader

Loads raw extracted data into the Bronze layer of the medallion architecture.
Supports local filesystem and Azure Data Lake Storage.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Iterator

import ftfy

from .pst_extractor import EmailMessage
from .document_parser import ParsedDocument

logger = logging.getLogger(__name__)


def _sanitize_strings(obj: Any) -> Any:
    """Recursively apply ftfy.fix_text to every string in a nested structure."""
    if isinstance(obj, str):
        return ftfy.fix_text(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_strings(v) for v in obj]
    return obj


class BronzeLayerLoader:
    """
    Load raw data into the Bronze layer.

    Bronze layer structure:
    bronze/
    ├── emails/
    │   ├── {year}/{month}/
    │   │   └── {message_id}.json
    │   └── metadata/
    │       └── extraction_stats.json
    ├── documents/
    │   ├── pdf/
    │   ├── docx/
    │   └── xlsx/
    ├── attachments/
    │   └── {attachment_id}/
    │       └── {filename}
    └── metadata/
        └── ingestion_log.json
    """

    def __init__(
        self,
        bronze_path: str,
        storage_type: str = "local",
        azure_config: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Bronze layer loader.

        Bronze is a pure extraction layer — no classification or filtering.
        Classification happens in Silver layer.

        Args:
            bronze_path: Base path for Bronze layer
            storage_type: "local" or "azure"
            azure_config: Azure storage configuration (for ADLS)
        """
        self.bronze_path = Path(bronze_path)
        self.storage_type = storage_type
        self.azure_config = azure_config

        # Create directory structure for local storage
        if storage_type == "local":
            self._create_directories()

        # Statistics
        self.stats = {
            "emails_loaded": 0,
            "documents_loaded": 0,
            "errors": 0,
            "start_time": datetime.now().isoformat(),
        }

    def _create_directories(self) -> None:
        """Create Bronze layer directory structure"""
        directories = [
            self.bronze_path / "emails",
            self.bronze_path / "emails" / "metadata",
            self.bronze_path / "documents" / "pdf",
            self.bronze_path / "documents" / "docx",
            self.bronze_path / "documents" / "xlsx",
            self.bronze_path / "documents" / "pptx",
            self.bronze_path / "documents" / "other",
            self.bronze_path / "attachments",
            self.bronze_path / "metadata",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_email(self, email: EmailMessage) -> str:
        """
        Load a single email into the Bronze layer.

        Args:
            email: EmailMessage to load

        Returns:
            Path where email was saved
        """
        try:
            # Determine path based on date
            if email.sent_time:
                year = email.sent_time.year
                month = f"{email.sent_time.month:02d}"
            else:
                year = "unknown"
                month = "unknown"

            email_dir = self.bronze_path / "emails" / str(year) / month
            email_dir.mkdir(parents=True, exist_ok=True)

            # Save email as JSON
            email_path = email_dir / f"{email.message_id}.json"

            email_data = _sanitize_strings(email.to_dict())

            if self.storage_type == "local":
                json_str = json.dumps(email_data, indent=2, ensure_ascii=False, default=str)
                with open(email_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
            else:
                self._write_to_azure(str(email_path), json.dumps(email_data, default=str))

            self.stats["emails_loaded"] += 1

            return str(email_path)

        except Exception as e:
            logger.error(f"Error loading email {email.message_id}: {e}")
            self.stats["errors"] += 1
            raise

    def load_emails_batch(
        self,
        emails: Iterator[EmailMessage],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Load multiple emails in batches.

        Args:
            emails: Iterator of EmailMessage objects
            batch_size: Number of emails per batch

        Returns:
            Statistics dictionary
        """
        batch = []
        batch_num = 0

        for email in emails:
            try:
                self.load_email(email)
                batch.append(email.message_id)

                if len(batch) >= batch_size:
                    batch_num += 1
                    logger.info(f"Loaded batch {batch_num}: {len(batch)} emails")
                    batch = []

            except Exception as e:
                logger.warning(f"Failed to load email: {e}")

        # Final batch
        if batch:
            logger.info(f"Loaded final batch: {len(batch)} emails")

        return self.stats.copy()

    def load_document(self, document: ParsedDocument) -> str:
        """
        Load a parsed document into the Bronze layer.

        Args:
            document: ParsedDocument to load

        Returns:
            Path where document was saved
        """
        try:
            # Determine subdirectory based on type
            doc_type = document.doc_type.value
            doc_dir = self.bronze_path / "documents" / doc_type

            if not doc_dir.exists():
                doc_dir = self.bronze_path / "documents" / "other"

            doc_dir.mkdir(parents=True, exist_ok=True)

            # Save document metadata and content
            doc_path = doc_dir / f"{document.doc_id}.json"

            doc_data = document.to_dict()
            doc_data["text"] = document.text
            doc_data = _sanitize_strings(doc_data)

            if self.storage_type == "local":
                json_str = json.dumps(doc_data, indent=2, ensure_ascii=False, default=str)
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
            else:
                self._write_to_azure(str(doc_path), json.dumps(doc_data, default=str))

            self.stats["documents_loaded"] += 1

            return str(doc_path)

        except Exception as e:
            logger.error(f"Error loading document {document.doc_id}: {e}")
            self.stats["errors"] += 1
            raise

    def load_documents_batch(
        self,
        documents: Iterator[ParsedDocument],
        batch_size: int = 50
    ) -> Dict[str, int]:
        """
        Load multiple documents in batches.

        Args:
            documents: Iterator of ParsedDocument objects
            batch_size: Number of documents per batch

        Returns:
            Statistics dictionary
        """
        batch = []
        batch_num = 0

        for doc in documents:
            try:
                self.load_document(doc)
                batch.append(doc.doc_id)

                if len(batch) >= batch_size:
                    batch_num += 1
                    logger.info(f"Loaded batch {batch_num}: {len(batch)} documents")
                    batch = []

            except Exception as e:
                logger.warning(f"Failed to load document: {e}")

        if batch:
            logger.info(f"Loaded final batch: {len(batch)} documents")

        return self.stats.copy()

    def save_metadata(self) -> str:
        """Save ingestion metadata and statistics"""
        self.stats["end_time"] = datetime.now().isoformat()

        metadata_path = self.bronze_path / "metadata" / "ingestion_log.json"

        # Load existing logs if present
        existing_logs = []
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    existing_logs = json.load(f)
            except Exception:
                existing_logs = []

        existing_logs.append(self.stats)

        with open(metadata_path, "w") as f:
            json.dump(existing_logs, f, indent=2, default=str)

        return str(metadata_path)

    def _write_to_azure(self, path: str, content: str) -> None:
        """Write content to Azure Data Lake"""
        # This would use azure-storage-file-datalake
        raise NotImplementedError("Azure storage not yet implemented")

    def _write_bytes_to_azure(self, path: str, content: bytes) -> None:
        """Write bytes to Azure Data Lake"""
        raise NotImplementedError("Azure storage not yet implemented")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()

    # =========================================================================
    # Reading from Bronze Layer
    # =========================================================================

    def iter_emails(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over emails in Bronze layer.

        Args:
            year: Filter by year (optional)
            month: Filter by month (optional)

        Yields:
            Email data dictionaries
        """
        emails_dir = self.bronze_path / "emails"

        if year:
            if month:
                search_path = emails_dir / str(year) / f"{month:02d}"
            else:
                search_path = emails_dir / str(year)
        else:
            search_path = emails_dir

        for json_file in search_path.rglob("*.json"):
            if "metadata" in str(json_file):
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    yield json.load(f)
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")

    def iter_documents(
        self,
        doc_type: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over documents in Bronze layer.

        Args:
            doc_type: Filter by document type (pdf, docx, etc.)

        Yields:
            Document data dictionaries
        """
        docs_dir = self.bronze_path / "documents"

        if doc_type:
            search_path = docs_dir / doc_type
        else:
            search_path = docs_dir

        for json_file in search_path.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    yield json.load(f)
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")

    def get_email_count(self) -> int:
        """Get total number of emails in Bronze layer"""
        emails_dir = self.bronze_path / "emails"
        return sum(1 for _ in emails_dir.rglob("*.json")
                  if "metadata" not in str(_))

    def get_document_count(self) -> int:
        """Get total number of documents in Bronze layer"""
        docs_dir = self.bronze_path / "documents"
        return sum(1 for _ in docs_dir.rglob("*.json"))


# Convenience function
def create_bronze_loader(output_dir: str) -> BronzeLayerLoader:
    """
    Create a Bronze layer loader with default settings.

    Args:
        output_dir: Output directory path

    Returns:
        Configured BronzeLayerLoader
    """
    return BronzeLayerLoader(bronze_path=output_dir)
