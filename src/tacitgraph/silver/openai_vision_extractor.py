"""
OpenAI Vision Extractor Module

Extracts text content from images and scanned PDFs using OpenAI Vision API (GPT-4o).
Supports both Azure OpenAI and OpenAI direct API.
"""

import base64
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tacitgraph.prompt_loader import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of vision-based extraction."""

    attachment_id: str
    filename: str
    extracted_text: str
    pages_processed: int
    extraction_method: str = "openai_vision"
    extraction_model: str = ""
    token_count: int = 0
    extraction_cost: float = 0.0
    extraction_time: str = ""
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "attachment_id": self.attachment_id,
            "filename": self.filename,
            "extracted_text": self.extracted_text,
            "text_length": len(self.extracted_text),
            "pages_processed": self.pages_processed,
            "extraction_method": self.extraction_method,
            "extraction_model": self.extraction_model,
            "token_count": self.token_count,
            "extraction_cost": self.extraction_cost,
            "extraction_time": self.extraction_time,
            "extraction_success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class VisionConfig:
    """Configuration for Vision API."""

    # Azure OpenAI settings
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_api_version: str = "2024-02-15-preview"
    azure_deployment: str = "gpt-4o"

    # OpenAI direct settings
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"

    # Processing settings
    max_image_size: int = 20 * 1024 * 1024  # 20MB
    detail_level: str = "high"  # "low", "high", or "auto"
    max_pages_per_request: int = 5
    temperature: float = 0.0

    # Cost tracking (per 1M tokens, approximate)
    input_cost_per_1m: float = 2.50
    output_cost_per_1m: float = 10.00

    @classmethod
    def from_env(cls) -> "VisionConfig":
        """Create config from environment variables."""
        return cls(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        )


class OpenAIVisionExtractor:
    """
    Extracts text from images and scanned PDFs using OpenAI Vision API.

    Supports:
    - Direct image files (PNG, JPG, etc.)
    - Scanned PDFs (converted to images)
    - Multi-page documents

    Uses Azure OpenAI by default, falls back to OpenAI direct API.
    """

    SYSTEM_PROMPT = get_prompt("silver", "vision_ocr", "system_prompt")
    EXTRACTION_PROMPT = get_prompt("silver", "vision_ocr", "user_prompt")

    def __init__(self, config: VisionConfig | None = None):
        """
        Initialize the Vision extractor.

        Args:
            config: Vision API configuration
        """
        self.config = config or VisionConfig.from_env()
        self.client = None
        self.use_azure = False
        self._initialize_client()

        logger.info(f"OpenAIVisionExtractor initialized (Azure: {self.use_azure})")

    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            # Try Azure OpenAI first
            if self.config.azure_endpoint and self.config.azure_api_key:
                from openai import AzureOpenAI

                self.client = AzureOpenAI(
                    azure_endpoint=self.config.azure_endpoint,
                    api_key=self.config.azure_api_key,
                    api_version=self.config.azure_api_version,
                )
                self.use_azure = True
                logger.info("Using Azure OpenAI for Vision")
                return

            # Fall back to OpenAI direct
            if self.config.openai_api_key:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.config.openai_api_key)
                self.use_azure = False
                logger.info("Using OpenAI direct for Vision")
                return

            logger.warning("No OpenAI credentials found - Vision extraction will fail")

        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def extract_from_image(self, image_path: Path, attachment_id: str) -> ExtractionResult:
        """
        Extract text from a single image file.

        Args:
            image_path: Path to image file
            attachment_id: Unique identifier for the attachment

        Returns:
            ExtractionResult with extracted text
        """
        datetime.now()
        result = ExtractionResult(
            attachment_id=attachment_id,
            filename=image_path.name,
            extracted_text="",
            pages_processed=1,
            extraction_model=self._get_model_name(),
        )

        if not self.client:
            result.success = False
            result.error_message = "OpenAI client not initialized"
            return result

        try:
            # Read and encode image
            image_data = self._encode_image(image_path)
            if not image_data:
                result.success = False
                result.error_message = "Failed to encode image"
                return result

            # Call Vision API
            response = self._call_vision_api([image_data])

            # Extract text from response
            result.extracted_text = self._parse_response(response)
            result.token_count = self._get_token_count(response)
            result.extraction_cost = self._calculate_cost(response)
            result.extraction_time = datetime.now().isoformat()
            result.success = True

            logger.info(f"Extracted {len(result.extracted_text)} chars from {image_path.name}")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Vision extraction failed for {image_path.name}: {e}")

        return result

    def extract_from_pdf(
        self, pdf_path: Path, attachment_id: str, max_pages: int | None = None
    ) -> ExtractionResult:
        """
        Extract text from a scanned PDF using Vision API.

        Converts PDF pages to images and processes them.

        Args:
            pdf_path: Path to PDF file
            attachment_id: Unique identifier for the attachment
            max_pages: Maximum number of pages to process (None = all)

        Returns:
            ExtractionResult with extracted text
        """
        result = ExtractionResult(
            attachment_id=attachment_id,
            filename=pdf_path.name,
            extracted_text="",
            pages_processed=0,
            extraction_model=self._get_model_name(),
        )

        if not self.client:
            result.success = False
            result.error_message = "OpenAI client not initialized"
            return result

        try:
            # Convert PDF to images
            images = self._pdf_to_images(pdf_path, max_pages)
            if not images:
                result.success = False
                result.error_message = "Failed to convert PDF to images"
                return result

            result.pages_processed = len(images)
            all_text = []
            total_tokens = 0
            total_cost = 0.0

            # Process images in batches
            batch_size = self.config.max_pages_per_request
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]

                # Encode images
                encoded_images = []
                for img_data in batch:
                    encoded = base64.b64encode(img_data).decode("utf-8")
                    encoded_images.append(encoded)

                # Call Vision API with batch
                response = self._call_vision_api(encoded_images, is_base64=True)

                # Extract text
                page_text = self._parse_response(response)
                all_text.append(f"--- Page {i + 1} to {i + len(batch)} ---\n{page_text}")

                total_tokens += self._get_token_count(response)
                total_cost += self._calculate_cost(response)

            result.extracted_text = "\n\n".join(all_text)
            result.token_count = total_tokens
            result.extraction_cost = total_cost
            result.extraction_time = datetime.now().isoformat()
            result.success = True

            logger.info(
                f"Extracted {len(result.extracted_text)} chars from {result.pages_processed} pages of {pdf_path.name}"
            )

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"PDF Vision extraction failed for {pdf_path.name}: {e}")

        return result

    def _encode_image(self, image_path: Path) -> str | None:
        """Read and base64 encode an image file."""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Check size
            if len(image_data) > self.config.max_image_size:
                logger.warning(f"Image {image_path.name} exceeds size limit")
                return None

            return base64.b64encode(image_data).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to encode image {image_path.name}: {e}")
            return None

    def _pdf_to_images(self, pdf_path: Path, max_pages: int | None = None) -> list[bytes]:
        """
        Convert PDF pages to images.

        Uses pdf2image if available, falls back to PyMuPDF.
        """
        images = []

        try:
            # Try pdf2image first (requires poppler)
            from pdf2image import convert_from_path

            pages = convert_from_path(
                str(pdf_path),
                dpi=150,  # Balance quality vs size
                fmt="png",
                first_page=1,
                last_page=max_pages,
            )

            for page in pages:
                import io

                img_buffer = io.BytesIO()
                page.save(img_buffer, format="PNG")
                images.append(img_buffer.getvalue())

            logger.debug(f"Converted {len(images)} pages using pdf2image")
            return images

        except ImportError:
            logger.debug("pdf2image not available, trying PyMuPDF")

        try:
            # Fall back to PyMuPDF
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            page_count = min(len(doc), max_pages) if max_pages else len(doc)

            for page_num in range(page_count):
                page = doc[page_num]
                # Render at 150 DPI
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat)
                images.append(pix.tobytes("png"))

            doc.close()
            logger.debug(f"Converted {len(images)} pages using PyMuPDF")
            return images

        except ImportError:
            logger.error("Neither pdf2image nor PyMuPDF available for PDF conversion")
            return []

        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            return []

    def _call_vision_api(self, images: list[str], is_base64: bool = False) -> Any:
        """
        Call the Vision API with images.

        Args:
            images: List of base64-encoded images
            is_base64: Whether images are already base64 encoded

        Returns:
            API response
        """
        # Build content array with images
        content = [{"type": "text", "text": self.EXTRACTION_PROMPT}]

        for img in images:
            image_data = img
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}",
                        "detail": self.config.detail_level,
                    },
                }
            )

        # Make API call
        model = self.config.azure_deployment if self.use_azure else self.config.openai_model

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=4096,
            temperature=self.config.temperature,
        )

        return response

    def _parse_response(self, response: Any) -> str:
        """Extract text content from API response with finish_reason checks."""
        try:
            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            return extract_llm_content(response, context="vision extraction")
        except Exception:
            # Vision extractor is best-effort — return empty string on any failure
            return ""

    def _get_token_count(self, response: Any) -> int:
        """Get token count from response."""
        try:
            usage = response.usage
            return (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
        except AttributeError:
            return 0

    def _calculate_cost(self, response: Any) -> float:
        """Calculate approximate cost from response."""
        try:
            usage = response.usage
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0

            input_cost = (input_tokens / 1_000_000) * self.config.input_cost_per_1m
            output_cost = (output_tokens / 1_000_000) * self.config.output_cost_per_1m

            return input_cost + output_cost
        except AttributeError:
            return 0.0

    def _get_model_name(self) -> str:
        """Get the model name being used."""
        if self.use_azure:
            return f"azure/{self.config.azure_deployment}"
        return self.config.openai_model

    def extract(
        self, file_path: Path, attachment_id: str, max_pages: int | None = None
    ) -> ExtractionResult:
        """
        Extract text from any supported file type.

        Automatically detects file type and uses appropriate method.

        Args:
            file_path: Path to file
            attachment_id: Unique identifier
            max_pages: Maximum pages to process (for PDFs)

        Returns:
            ExtractionResult with extracted text
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self.extract_from_pdf(file_path, attachment_id, max_pages)
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}:
            return self.extract_from_image(file_path, attachment_id)
        else:
            return ExtractionResult(
                attachment_id=attachment_id,
                filename=file_path.name,
                extracted_text="",
                pages_processed=0,
                success=False,
                error_message=f"Unsupported file type for Vision API: {ext}",
            )

    def is_available(self) -> bool:
        """Check if Vision API is available and configured."""
        return self.client is not None
