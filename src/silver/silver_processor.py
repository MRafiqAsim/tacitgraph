"""
Silver Layer Processor

Processes Bronze layer data into Silver layer with:
- Chunking
- Language detection
- PII detection and anonymization
- NER extraction
- Summarization
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Callable

from .pii_detector import PIIDetector, PIIEntity, PIIType
from .anonymizer import Anonymizer, AnonymizationResult, AnonymizationStrategy

# Import from sibling modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from silver.chunker import SemanticChunker, Chunk
from silver.language_detector import LanguageDetector

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """Fully processed chunk for Silver layer"""

    # Identification
    chunk_id: str
    doc_id: str
    chunk_index: int

    # Content
    text_original: str
    text_anonymized: str
    token_count: int

    # NER results
    entities: List[Dict[str, Any]] = field(default_factory=list)
    pii_entities: List[Dict[str, Any]] = field(default_factory=list)
    pii_count: int = 0

    # Metadata
    language: str = "en"
    source_file: Optional[str] = None
    source_date: Optional[datetime] = None

    # Processing metadata
    processing_time: datetime = field(default_factory=datetime.now)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text_original": self.text_original,
            "text_anonymized": self.text_anonymized,
            "token_count": self.token_count,
            "entities": self.entities,
            "pii_entities": self.pii_entities,
            "pii_count": self.pii_count,
            "language": self.language,
            "source_file": self.source_file,
            "source_date": self.source_date.isoformat() if self.source_date else None,
            "processing_time": self.processing_time.isoformat(),
            "confidence_scores": self.confidence_scores,
        }


class SilverLayerProcessor:
    """
    Process Bronze layer data into Silver layer.

    Pipeline:
    1. Load from Bronze layer
    2. Detect language
    3. Chunk text
    4. Detect PII
    5. Anonymize
    6. Extract named entities
    7. Save to Silver layer
    """

    def __init__(
        self,
        bronze_path: str,
        silver_path: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        anonymization_strategy: AnonymizationStrategy = AnonymizationStrategy.REPLACE,
        confidence_threshold: float = 0.5,
        languages: List[str] = None
    ):
        """
        Initialize the Silver layer processor.

        Args:
            bronze_path: Path to Bronze layer
            silver_path: Path to Silver layer output
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            anonymization_strategy: Strategy for PII anonymization
            confidence_threshold: Minimum confidence for PII detection
            languages: Supported languages
        """
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.languages = languages or ["en", "nl"]

        # Initialize components
        self.chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.language_detector = LanguageDetector()

        self.pii_detector = PIIDetector(
            languages=self.languages,
            confidence_threshold=confidence_threshold
        )

        self.anonymizer = Anonymizer(
            detector=self.pii_detector,
            strategy=anonymization_strategy,
            consistent_replacement=True
        )

        # Create Silver layer directories
        self._create_directories()

        # Statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "pii_detected": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

    def _create_directories(self) -> None:
        """Create Silver layer directory structure"""
        directories = [
            self.silver_path / "chunks",
            self.silver_path / "chunks_anonymized",
            self.silver_path / "ner_results",
            self.silver_path / "pii_mappings",
            self.silver_path / "metadata",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_document(
        self,
        doc_data: Dict[str, Any],
        doc_type: str = "email"
    ) -> List[ProcessedChunk]:
        """
        Process a single document from Bronze layer.

        Args:
            doc_data: Document data dictionary
            doc_type: Type of document (email, document)

        Returns:
            List of ProcessedChunk objects
        """
        processed_chunks = []

        try:
            # Extract text based on document type
            if doc_type == "email":
                text = self._extract_email_text(doc_data)
                doc_id = doc_data.get("record_id", "unknown")
                meta = doc_data.get("document_metadata", {})
                source_date = self._parse_date(meta.get("sent_time"))
            else:
                text = doc_data.get("text", "")
                doc_id = doc_data.get("doc_id", "unknown")
                source_date = self._parse_date(doc_data.get("created_date"))

            if not text or not text.strip():
                return []

            # Detect language
            lang_result = self.language_detector.detect(text)
            language = lang_result.language

            # Chunk the text
            chunks = self.chunker.chunk(
                text=text,
                doc_id=doc_id,
                metadata={
                    "source_file": doc_data.get("source_file"),
                    "source_date": source_date,
                    "language": language,
                }
            )

            # Process each chunk
            for chunk in chunks:
                processed = self._process_chunk(chunk, language)
                processed_chunks.append(processed)
                self.stats["chunks_created"] += 1

            self.stats["documents_processed"] += 1

        except Exception as e:
            logger.error(f"Error processing document {doc_data.get('doc_id', 'unknown')}: {e}")
            self.stats["errors"] += 1

        return processed_chunks

    def _process_chunk(
        self,
        chunk: Chunk,
        language: str
    ) -> ProcessedChunk:
        """Process a single chunk"""
        # Detect and anonymize PII
        anon_result = self.anonymizer.anonymize(chunk.text, language)

        self.stats["pii_detected"] += anon_result.entity_count

        # Extract named entities (non-PII)
        entities = self._extract_entities(chunk.text, language)

        # Calculate confidence scores
        confidence_scores = {
            "language": self.language_detector.detect(chunk.text).confidence,
            "pii_avg": (
                sum(e.confidence for e in anon_result.entities) / len(anon_result.entities)
                if anon_result.entities else 1.0
            ),
        }

        return ProcessedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            text_original=chunk.text,
            text_anonymized=anon_result.anonymized_text,
            token_count=chunk.token_count,
            entities=entities,
            pii_entities=[e.to_dict() for e in anon_result.entities],
            pii_count=anon_result.entity_count,
            language=language,
            source_file=chunk.source_file,
            source_date=chunk.source_date,
            confidence_scores=confidence_scores,
        )

    def _extract_email_text(self, email_data: Dict) -> str:
        """Extract text from email data (grouped bronze format)"""
        headers = email_data.get("email_headers", {})
        meta = email_data.get("document_metadata", {})
        parts = []

        if headers.get("subject"):
            parts.append(f"Subject: {headers['subject']}")

        if headers.get("sender"):
            parts.append(f"From: {headers['sender']}")

        if headers.get("recipients_to"):
            recipients = headers["recipients_to"]
            if isinstance(recipients, list):
                to_strs = []
                for r in recipients:
                    if isinstance(r, dict):
                        name = r.get("name", "")
                        email = r.get("email", "")
                        to_strs.append(f"{name} <{email}>" if name else email)
                    else:
                        to_strs.append(str(r))
                recipients = ", ".join(to_strs)
            parts.append(f"To: {recipients}")

        if meta.get("sent_time"):
            parts.append(f"Date: {meta['sent_time']}")

        parts.append("")  # Blank line

        if email_data.get("email_body_text"):
            parts.append(email_data["email_body_text"])

        return "\n".join(parts)

    def _extract_entities(
        self,
        text: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy"""
        entities = []

        try:
            import spacy

            model_name = "en_core_web_trf" if language == "en" else "nl_core_news_lg"

            try:
                nlp = spacy.load(model_name)
            except OSError:
                # Fallback to smaller model
                model_name = "en_core_web_sm" if language == "en" else "nl_core_news_sm"
                nlp = spacy.load(model_name)

            doc = nlp(text)

            for ent in doc.ents:
                # Skip PII types (handled by anonymizer)
                if ent.label_ in ["PERSON", "PER"]:
                    continue

                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })

        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

        return entities

    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats"""
        if date_value is None:
            return None

        if isinstance(date_value, datetime):
            return date_value

        if isinstance(date_value, str):
            try:
                from dateutil import parser
                return parser.parse(date_value)
            except Exception:
                return None

        return None

    def process_bronze_layer(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, int]:
        """
        Process entire Bronze layer into Silver layer.

        Args:
            progress_callback: Optional callback(count, message)

        Returns:
            Statistics dictionary
        """
        self.stats["start_time"] = datetime.now().isoformat()

        # Process emails
        emails_dir = self.bronze_path / "emails"
        if emails_dir.exists():
            self._process_directory(
                emails_dir,
                doc_type="email",
                progress_callback=progress_callback
            )

        # Process documents
        docs_dir = self.bronze_path / "documents"
        if docs_dir.exists():
            self._process_directory(
                docs_dir,
                doc_type="document",
                progress_callback=progress_callback
            )

        self.stats["end_time"] = datetime.now().isoformat()

        # Save statistics
        self._save_metadata()

        return self.stats.copy()

    def _process_directory(
        self,
        directory: Path,
        doc_type: str,
        progress_callback: Optional[Callable] = None
    ) -> None:
        """Process all JSON files in a directory"""
        json_files = list(directory.rglob("*.json"))

        # Filter out metadata files
        json_files = [f for f in json_files if "metadata" not in str(f)]

        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)

                processed_chunks = self.process_document(doc_data, doc_type)

                # Save processed chunks
                for chunk in processed_chunks:
                    self._save_chunk(chunk)

                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback(i + 1, f"Processed {i + 1}/{len(json_files)} {doc_type}s")

            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                self.stats["errors"] += 1

    def _save_chunk(self, chunk: ProcessedChunk) -> None:
        """Save processed chunk to Silver layer"""
        # Save full chunk data
        chunk_file = self.silver_path / "chunks" / f"{chunk.chunk_id}.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        # Save anonymized version separately (for downstream processing)
        anon_file = self.silver_path / "chunks_anonymized" / f"{chunk.chunk_id}.json"
        anon_data = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text_anonymized,
            "language": chunk.language,
            "token_count": chunk.token_count,
        }
        with open(anon_file, "w", encoding="utf-8") as f:
            json.dump(anon_data, f, indent=2, ensure_ascii=False)

        # Save NER results
        if chunk.entities:
            ner_file = self.silver_path / "ner_results" / f"{chunk.chunk_id}.json"
            with open(ner_file, "w", encoding="utf-8") as f:
                json.dump(chunk.entities, f, indent=2)

    def _save_metadata(self) -> None:
        """Save processing metadata"""
        metadata_file = self.silver_path / "metadata" / "processing_stats.json"

        # Load existing if present
        existing = []
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.append(self.stats)

        with open(metadata_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        # Save PII mapping
        mapping_file = self.silver_path / "pii_mappings" / "mapping.json"
        mapping = self.anonymizer.get_mapping()
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=2)

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics"""
        return self.stats.copy()


# Convenience function
def process_bronze_to_silver(
    bronze_path: str,
    silver_path: str,
    chunk_size: int = 512
) -> Dict[str, int]:
    """
    Process Bronze layer data into Silver layer.

    Args:
        bronze_path: Path to Bronze layer
        silver_path: Path for Silver layer output
        chunk_size: Target chunk size in tokens

    Returns:
        Processing statistics
    """
    processor = SilverLayerProcessor(
        bronze_path=bronze_path,
        silver_path=silver_path,
        chunk_size=chunk_size
    )

    return processor.process_bronze_layer()
