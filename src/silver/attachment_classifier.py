"""
Multi-Signal Attachment Classifier

Classifies email attachments as 'knowledge' or 'transactional' using
a weighted scoring model across four signals:
  1. Content patterns (0.45) — regex on first ~2000 chars
  2. Email body context (0.20) — keywords near "attach" mentions
  3. Document structure (0.20) — tables, page count, doc_type
  4. Filename/extension (0.15) — weakest, fallback only

Produces ClassificationResult with classification, confidence, and audit trail.
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of attachment classification with audit trail."""

    classification: str  # "knowledge" | "transactional"
    confidence: float  # 0.0–1.0
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "confidence": self.confidence,
            "signals": self.signals,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassificationResult":
        return cls(
            classification=data.get("classification", "knowledge"),
            confidence=data.get("confidence", 0.5),
            signals=data.get("signals", {}),
        )


class AttachmentClassifier:
    """
    Multi-signal attachment classifier for Bronze layer.

    Scores each attachment across content, email context, structure,
    and filename signals. Positive scores → knowledge, negative → transactional.
    """

    # Signal weights (must sum to 1.0)
    WEIGHT_CONTENT = 0.45
    WEIGHT_EMAIL_CONTEXT = 0.20
    WEIGHT_STRUCTURE = 0.20
    WEIGHT_FILENAME = 0.15

    # Content sample size
    CONTENT_SAMPLE_CHARS = 2000

    def __init__(self, bronze_path: str):
        self.bronze_path = Path(bronze_path)
        self._email_body_cache: Dict[str, Optional[str]] = {}

    def classify(self, att_content) -> ClassificationResult:
        """
        Classify an attachment using all available signals.

        Args:
            att_content: AttachmentContent instance with text, filename,
                         doc_type, has_tables, page_count, email_id, tables.

        Returns:
            ClassificationResult with classification, confidence, signals.
        """
        signals = {}

        # Signal 1: Content patterns
        content_score, content_details = self._score_content(
            att_content.text, att_content.doc_type
        )
        signals["content"] = {"score": content_score, **content_details}

        # Signal 2: Email body context
        email_body = self._load_email_body(att_content.email_id)
        context_score, context_details = self._score_email_context(
            email_body, att_content.filename
        )
        signals["email_context"] = {"score": context_score, **context_details}

        # Signal 3: Document structure
        structure_score, structure_details = self._score_structure(att_content)
        signals["structure"] = {"score": structure_score, **structure_details}

        # Signal 4: Filename/extension
        filename_score, filename_details = self._score_filename(att_content.filename)
        signals["filename"] = {"score": filename_score, **filename_details}

        # Weighted sum
        weighted_sum = (
            self.WEIGHT_CONTENT * content_score
            + self.WEIGHT_EMAIL_CONTEXT * context_score
            + self.WEIGHT_STRUCTURE * structure_score
            + self.WEIGHT_FILENAME * filename_score
        )

        # Classification
        classification = "knowledge" if weighted_sum >= 0 else "transactional"

        # Confidence via sigmoid mapping
        raw = abs(weighted_sum)
        confidence = 0.5 + 0.5 * (1 - math.exp(-3 * raw))

        signals["weighted_sum"] = round(weighted_sum, 4)

        logger.debug(
            f"Classified '{att_content.filename}' as {classification} "
            f"(confidence={confidence:.2f}, weighted_sum={weighted_sum:.3f})"
        )

        return ClassificationResult(
            classification=classification,
            confidence=round(confidence, 4),
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Signal 1: Content patterns
    # ------------------------------------------------------------------

    def _score_content(self, text: str, doc_type: str) -> Tuple[float, Dict[str, Any]]:
        """Score based on content patterns in first ~2000 chars."""
        details: Dict[str, Any] = {"matched_patterns": []}

        if not text or not text.strip():
            return 0.0, details

        sample = text[: self.CONTENT_SAMPLE_CHARS]
        lines = sample.split("\n")
        score = 0.0

        # --- Transactional indicators ---

        # Sheet markers (from xlsx extraction)
        if re.search(r"=== Sheet:", sample):
            score -= 0.20
            details["matched_patterns"].append("sheet_markers")

        # Tab-separated columnar rows (3+ tabs per line, >=5 such lines)
        tab_lines = sum(1 for line in lines if line.count("\t") >= 3)
        if tab_lines >= 5:
            score -= 0.25
            details["matched_patterns"].append(f"tab_columns({tab_lines}_lines)")

        # Currency amounts
        currency_matches = re.findall(r"\$[\d,]+\.?\d{0,2}", sample)
        if len(currency_matches) >= 3:
            score -= 0.15
            details["matched_patterns"].append(f"currency_amounts({len(currency_matches)})")

        # Financial headers
        financial_headers = re.findall(
            r"(?i)Invoice\s*#|Invoice\s*Number|Check\s*Number|Loan\s*Number|"
            r"Total\s*Amount|Fee\s*Amount|Due\s*Date",
            sample,
        )
        if len(financial_headers) >= 2:
            score -= 0.20
            details["matched_patterns"].append(f"financial_headers({len(financial_headers)})")

        # SQL fragments
        if re.search(r"(?i)INSERT\s+INTO|SELECT\s+.*FROM|CREATE\s+TABLE", sample):
            score -= 0.20
            details["matched_patterns"].append("sql_fragments")

        # ID column headers
        id_headers = re.findall(
            r"(?i)Sr\.\s*#|Document\s*ID|File\s*Number|Pdf\s*Number", sample
        )
        if len(id_headers) >= 1:
            score -= 0.15
            details["matched_patterns"].append(f"id_column_headers({len(id_headers)})")

        # --- Knowledge indicators ---

        # Section numbering
        section_nums = re.findall(r"(?m)^\d+\.?\d*\s+[A-Z]", sample)
        section_words = re.findall(r"(?i)Chapter|Section|Part\s+\d", sample)
        total_sections = len(section_nums) + len(section_words)
        if total_sections >= 3:
            score += 0.25
            details["matched_patterns"].append(f"section_numbering({total_sections})")

        # Table of Contents
        if re.search(r"(?i)Table\s+of\s+Contents", sample):
            score += 0.20
            details["matched_patterns"].append("table_of_contents")

        # Prose density (avg words per non-empty line)
        non_empty = [line for line in lines if line.strip()]
        if non_empty:
            avg_words = sum(len(line.split()) for line in non_empty) / len(non_empty)
            if avg_words >= 8:
                score += 0.20
                details["matched_patterns"].append(f"prose_density(avg={avg_words:.1f})")

        # Policy/legal language
        legal_terms = re.findall(
            r"(?i)\bshall\b|\bhereby\b|\bpursuant\b|\bpolicy\b|\bprocedure\b|\bcompliance\b",
            sample,
        )
        if len(legal_terms) >= 3:
            score += 0.15
            details["matched_patterns"].append(f"legal_language({len(legal_terms)})")

        # Long paragraph lines (>100 chars)
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines >= 3:
            score += 0.15
            details["matched_patterns"].append(f"long_paragraphs({long_lines})")

        # Narrative connectors
        connectors = re.findall(
            r"(?i)\bhowever\b|\btherefore\b|\bfurthermore\b|\bin addition\b",
            sample,
        )
        if len(connectors) >= 2:
            score += 0.10
            details["matched_patterns"].append(f"narrative_connectors({len(connectors)})")

        # Clamp to [-1.0, +1.0]
        score = max(-1.0, min(1.0, score))
        return round(score, 4), details

    # ------------------------------------------------------------------
    # Signal 2: Email body context
    # ------------------------------------------------------------------

    def _score_email_context(
        self, email_body: Optional[str], attachment_filename: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Score based on keywords near 'attach' mentions in email body."""
        details: Dict[str, Any] = {"windows": []}

        if not email_body:
            return 0.0, details

        body_lower = email_body.lower()
        score = 0.0

        transactional_kw = {
            "invoice", "receipt", "billing", "payment", "check",
            "data", "spreadsheet", "numbers", "ocr",
        }
        knowledge_kw = {
            "report", "document", "manual", "guide", "rules",
            "handbook", "policy", "reference", "analysis",
        }

        # Find all "attach" occurrences and extract windows
        for m in re.finditer(r"attach", body_lower):
            start = max(0, m.start() - 30)
            end = min(len(body_lower), m.end() + 30)
            window = body_lower[start:end]
            details["windows"].append(window)

            for kw in transactional_kw:
                if kw in window:
                    score -= 0.30
            for kw in knowledge_kw:
                if kw in window:
                    score += 0.30

        # Also check filename mentions in body
        fname_lower = Path(attachment_filename).stem.lower()
        if fname_lower in body_lower:
            # Check context around filename mention
            for m in re.finditer(re.escape(fname_lower), body_lower):
                start = max(0, m.start() - 30)
                end = min(len(body_lower), m.end() + 30)
                window = body_lower[start:end]
                for kw in transactional_kw:
                    if kw in window:
                        score -= 0.20
                for kw in knowledge_kw:
                    if kw in window:
                        score += 0.20

        score = max(-1.0, min(1.0, score))
        return round(score, 4), details

    def _load_email_body(self, email_id: str) -> Optional[str]:
        """Load email body text from Bronze cache."""
        if email_id in self._email_body_cache:
            return self._email_body_cache[email_id]

        emails_dir = self.bronze_path / "emails"
        if not emails_dir.exists():
            self._email_body_cache[email_id] = None
            return None

        # Search for email JSON file
        for email_file in emails_dir.rglob("*.json"):
            if email_id in email_file.stem or email_file.stem == email_id:
                try:
                    with open(email_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    mid = data.get("record_id", "")
                    if mid == email_id or email_file.stem == email_id:
                        body = data.get("email_body_text", "")
                        self._email_body_cache[email_id] = body
                        return body
                except Exception as e:
                    logger.debug(f"Error loading email body {email_file}: {e}")

        self._email_body_cache[email_id] = None
        return None

    # ------------------------------------------------------------------
    # Signal 3: Document structure
    # ------------------------------------------------------------------

    def _score_structure(self, att_content) -> Tuple[float, Dict[str, Any]]:
        """Score based on document structure (tables, page count, doc_type)."""
        details: Dict[str, Any] = {}
        score = 0.0

        # Has tables
        if att_content.has_tables:
            score -= 0.20
            details["has_tables"] = True

        # Doc type
        doc_type = att_content.doc_type.lower().lstrip(".")
        transactional_types = {"xlsx", "xls", "csv"}
        knowledge_types = {"docx", "doc", "pptx", "txt", "html", "rtf"}

        if doc_type in transactional_types:
            score -= 0.20
            details["doc_type_signal"] = "transactional"
        elif doc_type in knowledge_types:
            score += 0.10
            details["doc_type_signal"] = "knowledge"
        else:
            details["doc_type_signal"] = "neutral"

        # Page count
        page_count = getattr(att_content, "page_count", 0) or 0
        if page_count >= 5:
            score += 0.15
            details["multi_page"] = True

        # Table density
        tables = getattr(att_content, "tables", []) or []
        if page_count > 0 and len(tables) / max(page_count, 1) > 0.5:
            score -= 0.15
            details["high_table_density"] = True

        score = max(-1.0, min(1.0, score))
        return round(score, 4), details

    # ------------------------------------------------------------------
    # Signal 4: Filename / extension
    # ------------------------------------------------------------------

    def _score_filename(self, filename: str) -> Tuple[float, Dict[str, Any]]:
        """Score based on filename and extension."""
        details: Dict[str, Any] = {}
        score = 0.0

        ext = Path(filename).suffix.lower()
        stem = Path(filename).stem.lower()

        # Extension scoring
        transactional_ext = {".xlsx", ".xls", ".csv"}
        knowledge_ext = {".docx", ".doc", ".pptx", ".txt", ".html", ".rtf"}

        if ext in transactional_ext:
            score -= 0.80
            details["extension_signal"] = "transactional"
        elif ext in knowledge_ext:
            score += 0.60
            details["extension_signal"] = "knowledge"
        else:
            details["extension_signal"] = "neutral"  # .pdf etc.

        # Filename keyword scan (first match wins)
        transactional_keywords = [
            "invoice", "receipt", "statement", "purchase", "order", "payment",
            "billing", "form", "template", "schedule", "timesheet",
        ]
        knowledge_keywords = [
            "report", "manual", "guide", "analysis", "notes", "memo",
            "summary", "spec", "design", "architecture", "policy",
            "procedure", "rules", "handbook", "reference",
        ]

        keyword_found = False
        for kw in transactional_keywords:
            if kw in stem:
                score -= 0.30
                details["keyword_match"] = kw
                details["keyword_signal"] = "transactional"
                keyword_found = True
                break

        if not keyword_found:
            for kw in knowledge_keywords:
                if kw in stem:
                    score += 0.30
                    details["keyword_match"] = kw
                    details["keyword_signal"] = "knowledge"
                    break

        score = max(-1.0, min(1.0, score))
        return round(score, 4), details
