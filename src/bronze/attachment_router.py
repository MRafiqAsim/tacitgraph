"""
Attachment Router Module

Smart routing system that analyzes attachments and determines
the appropriate extraction method (local parser vs OpenAI Vision).
"""

import logging
from pathlib import Path
from typing import Optional, Set
from dataclasses import dataclass

from .attachment_storage import AttachmentAnalysis

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Configuration for attachment routing."""
    # Thresholds for PDF analysis
    min_chars_per_page: int = 100
    complexity_threshold: float = 0.6
    image_ratio_threshold: float = 0.7

    # Cost optimization
    prefer_local: bool = True
    max_file_size_for_vision: int = 20 * 1024 * 1024  # 20MB


class AttachmentRouter:
    """
    Routes attachments to appropriate processor based on content analysis.

    Routing Logic:
    1. ALWAYS LOCAL: txt, csv, json, xml, html, xlsx, docx, pptx
    2. ALWAYS VISION: png, jpg, gif, bmp, tiff, webp, heic
    3. NEEDS ANALYSIS: pdf
       - Try text extraction
       - If chars_per_page > threshold → local
       - If no fonts / high image ratio → vision
       - If complex structure → vision
    """

    # File types that always use local parser
    ALWAYS_LOCAL: Set[str] = {
        '.txt', '.csv', '.json', '.xml', '.html', '.htm',
        '.xlsx', '.xls', '.xlsm',
        '.docx', '.doc',
        '.pptx', '.ppt',
        '.rtf', '.md', '.rst',
        '.log', '.ini', '.cfg', '.yaml', '.yml'
    }

    # File types that always need OpenAI Vision
    ALWAYS_VISION: Set[str] = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp',
        '.tiff', '.tif', '.webp', '.heic', '.heif',
        '.ico', '.svg'
    }

    # File types that need content analysis
    NEEDS_ANALYSIS: Set[str] = {'.pdf'}

    # File types we cannot process
    UNSUPPORTED: Set[str] = {
        '.exe', '.dll', '.so', '.dylib',
        '.zip', '.rar', '.7z', '.tar', '.gz',
        '.mp3', '.wav', '.mp4', '.avi', '.mov',
        '.pst', '.ost', '.mbox'
    }

    def __init__(self, config: Optional[RouterConfig] = None):
        """
        Initialize the router.

        Args:
            config: Router configuration
        """
        self.config = config or RouterConfig()
        logger.info("AttachmentRouter initialized")

    def analyze_and_route(
        self,
        file_path: Path,
        content_type: Optional[str] = None
    ) -> AttachmentAnalysis:
        """
        Analyze attachment and determine processing method.

        Args:
            file_path: Path to the attachment file
            content_type: MIME type (optional, used as hint)

        Returns:
            AttachmentAnalysis with routing decision
        """
        ext = file_path.suffix.lower()
        analysis = AttachmentAnalysis()

        # Check file size
        try:
            file_size = file_path.stat().st_size
        except Exception:
            file_size = 0

        # Route by extension
        if ext in self.ALWAYS_LOCAL:
            analysis.recommended_processor = "local_parser"
            analysis.has_extractable_text = True
            analysis.routing_reason = f"File type {ext} uses local parser"
            logger.debug(f"Routing {file_path.name} to local_parser (type: {ext})")

        elif ext in self.ALWAYS_VISION:
            analysis.recommended_processor = "openai_vision"
            analysis.is_image_based = True
            analysis.routing_reason = f"Image file {ext} requires Vision API"
            logger.debug(f"Routing {file_path.name} to openai_vision (image)")

        elif ext in self.NEEDS_ANALYSIS:
            analysis = self._analyze_pdf(file_path, analysis)
            logger.debug(f"Routing {file_path.name} to {analysis.recommended_processor} (PDF analysis)")

        elif ext in self.UNSUPPORTED:
            analysis.recommended_processor = "unsupported"
            analysis.routing_reason = f"File type {ext} is not supported"
            logger.debug(f"Skipping {file_path.name} (unsupported type)")

        else:
            # Unknown type - try to infer from content_type or default to local
            if content_type and 'image' in content_type.lower():
                analysis.recommended_processor = "openai_vision"
                analysis.is_image_based = True
                analysis.routing_reason = f"Image content type: {content_type}"
            else:
                analysis.recommended_processor = "local_parser"
                analysis.routing_reason = f"Unknown type {ext}, trying local parser"
            logger.debug(f"Routing {file_path.name} to {analysis.recommended_processor} (fallback)")

        # Check file size limits for Vision API
        if (analysis.recommended_processor == "openai_vision" and
                file_size > self.config.max_file_size_for_vision):
            analysis.routing_reason += f" (WARNING: file too large for Vision API: {file_size} bytes)"
            logger.warning(f"File {file_path.name} may be too large for Vision API")

        return analysis

    def _analyze_pdf(self, file_path: Path, analysis: AttachmentAnalysis) -> AttachmentAnalysis:
        """
        Deep analysis of PDF to determine extraction method.

        Analysis steps:
        1. Try text extraction with pypdf
        2. Count characters per page
        3. Check for embedded fonts
        4. Estimate image ratio
        5. Detect tables/forms

        Args:
            file_path: Path to PDF file
            analysis: Analysis object to populate

        Returns:
            Updated AttachmentAnalysis
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            analysis.estimated_pages = len(reader.pages)

            # Extract text and count characters
            total_chars = 0
            pages_with_text = 0

            for page in reader.pages:
                try:
                    text = page.extract_text() or ""
                    page_chars = len(text.strip())
                    total_chars += page_chars
                    if page_chars > 50:  # Meaningful text
                        pages_with_text += 1
                except Exception:
                    pass

            analysis.text_char_count = total_chars
            analysis.chars_per_page = total_chars / max(1, analysis.estimated_pages)

            # Check for embedded fonts (heuristic)
            analysis.has_embedded_fonts = self._check_fonts(reader)

            # Estimate image ratio (simplified)
            analysis.image_ratio = self._estimate_image_ratio(reader, pages_with_text)

            # Detect tables/forms (simplified heuristic)
            analysis.has_tables = self._detect_tables_heuristic(reader)
            analysis.has_forms = self._detect_forms(reader)

            # Calculate complexity score
            analysis.complexity_score = self._calculate_complexity(analysis)

            # Make routing decision based on analysis
            analysis = self._make_pdf_routing_decision(analysis)

        except ImportError:
            analysis.recommended_processor = "openai_vision"
            analysis.routing_reason = "pypdf not available, using Vision API"
            logger.warning("pypdf not installed, defaulting to Vision API for PDFs")

        except Exception as e:
            # If analysis fails, default to OpenAI Vision (safer choice)
            analysis.recommended_processor = "openai_vision"
            analysis.routing_reason = f"PDF analysis failed: {str(e)[:50]}, using Vision API"
            logger.warning(f"PDF analysis failed for {file_path.name}: {e}")

        return analysis

    def _check_fonts(self, reader) -> bool:
        """Check if PDF has embedded fonts (indicates text-based)."""
        try:
            for page in reader.pages[:3]:  # Check first 3 pages
                if hasattr(page, 'get') and '/Font' in str(page.get('/Resources', {})):
                    return True
                # Alternative: check if text extraction returns anything
                text = page.extract_text() or ""
                if len(text.strip()) > 100:
                    return True
        except Exception:
            pass
        return False

    def _estimate_image_ratio(self, reader, pages_with_text: int) -> float:
        """
        Estimate what portion of the PDF is images.

        Simple heuristic: if most pages have little text, assume image-heavy.
        """
        if not reader.pages:
            return 0.0

        total_pages = len(reader.pages)
        text_ratio = pages_with_text / total_pages

        # Invert to get image ratio estimate
        return 1.0 - text_ratio

    def _detect_tables_heuristic(self, reader) -> bool:
        """
        Simple heuristic to detect if PDF contains tables.

        Looks for patterns like tab characters, aligned columns.
        """
        try:
            for page in reader.pages[:3]:
                text = page.extract_text() or ""
                # Look for tab-separated or pipe-separated data
                if '\t' in text or '|' in text:
                    lines = text.split('\n')
                    # Check if multiple lines have similar structure
                    separator_counts = [line.count('\t') + line.count('|') for line in lines]
                    if len([c for c in separator_counts if c > 2]) > 3:
                        return True
        except Exception:
            pass
        return False

    def _detect_forms(self, reader) -> bool:
        """Check if PDF has form fields."""
        try:
            if hasattr(reader, 'get_fields'):
                fields = reader.get_fields()
                return fields is not None and len(fields) > 0
        except Exception:
            pass
        return False

    def _calculate_complexity(self, analysis: AttachmentAnalysis) -> float:
        """
        Calculate overall complexity score (0-1).

        Higher score = more likely to need Vision API.
        """
        score = 0.0

        # Low text = likely scanned
        if analysis.chars_per_page < self.config.min_chars_per_page:
            score += 0.4

        # No embedded fonts = likely scanned
        if not analysis.has_embedded_fonts:
            score += 0.3

        # High image ratio
        if analysis.image_ratio > 0.5:
            score += 0.2 * analysis.image_ratio

        # Has tables or forms
        if analysis.has_tables:
            score += 0.1
        if analysis.has_forms:
            score += 0.1

        return min(1.0, score)

    def _make_pdf_routing_decision(self, analysis: AttachmentAnalysis) -> AttachmentAnalysis:
        """Make final routing decision for PDF based on analysis."""

        # Clear scanned document
        if analysis.chars_per_page < self.config.min_chars_per_page:
            analysis.is_image_based = True
            analysis.recommended_processor = "openai_vision"
            analysis.routing_reason = f"Low text density ({analysis.chars_per_page:.0f} chars/page, threshold: {self.config.min_chars_per_page})"

        # No fonts = scanned
        elif not analysis.has_embedded_fonts:
            analysis.is_image_based = True
            analysis.recommended_processor = "openai_vision"
            analysis.routing_reason = "No embedded fonts detected (likely scanned)"

        # Image-heavy
        elif analysis.image_ratio > self.config.image_ratio_threshold:
            analysis.recommended_processor = "openai_vision"
            analysis.routing_reason = f"High image ratio ({analysis.image_ratio:.0%})"

        # Complex structure
        elif analysis.complexity_score > self.config.complexity_threshold:
            analysis.recommended_processor = "openai_vision"
            analysis.routing_reason = f"Complex structure (score: {analysis.complexity_score:.2f})"

        # Good text-based PDF
        else:
            analysis.has_extractable_text = True
            analysis.recommended_processor = "local_parser"
            analysis.routing_reason = f"Text-based PDF ({analysis.chars_per_page:.0f} chars/page)"

        return analysis

    def get_processor_for_type(self, extension: str) -> str:
        """
        Quick lookup for processor by file extension.

        Args:
            extension: File extension (with or without dot)

        Returns:
            Processor name: 'local_parser', 'openai_vision', 'needs_analysis', or 'unsupported'
        """
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = '.' + ext

        if ext in self.ALWAYS_LOCAL:
            return "local_parser"
        elif ext in self.ALWAYS_VISION:
            return "openai_vision"
        elif ext in self.NEEDS_ANALYSIS:
            return "needs_analysis"
        elif ext in self.UNSUPPORTED:
            return "unsupported"
        else:
            return "unknown"

    def batch_analyze(self, file_paths: list) -> dict:
        """
        Analyze multiple files and return routing summary.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            Dictionary with routing summary and details
        """
        results = {
            "local_parser": [],
            "openai_vision": [],
            "unsupported": [],
            "errors": []
        }

        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists():
                    analysis = self.analyze_and_route(path)
                    processor = analysis.recommended_processor
                    if processor in results:
                        results[processor].append({
                            "file": str(path),
                            "reason": analysis.routing_reason
                        })
                    else:
                        results["local_parser"].append({
                            "file": str(path),
                            "reason": analysis.routing_reason
                        })
                else:
                    results["errors"].append({
                        "file": str(file_path),
                        "error": "File not found"
                    })
            except Exception as e:
                results["errors"].append({
                    "file": str(file_path),
                    "error": str(e)
                })

        # Add summary
        results["summary"] = {
            "total": len(file_paths),
            "local_parser": len(results["local_parser"]),
            "openai_vision": len(results["openai_vision"]),
            "unsupported": len(results["unsupported"]),
            "errors": len(results["errors"])
        }

        return results
