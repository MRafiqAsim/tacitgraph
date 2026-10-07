"""
Local Text Summarizer using DistilBART

Uses sshleifer/distilbart-cnn-12-6 for abstractive summarization without any API calls.
Model is loaded lazily on first use and cached for subsequent calls.
Uses MPS (Metal) acceleration on Apple Silicon, falls back to CPU.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton model cache — loaded once, reused across all calls
_model = None
_tokenizer = None
_device = None
_load_attempted = False


def _get_device():
    """Detect best available device: MPS (Apple Silicon) > CPU."""
    import torch

    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Metal) acceleration")
        return torch.device("mps")
    logger.info("Using CPU (MPS not available)")
    return torch.device("cpu")


def _load_model():
    """Lazy-load the BART model and tokenizer (singleton)."""
    global _model, _tokenizer, _device, _load_attempted

    if _model is not None:
        return _model, _tokenizer, _device

    if _load_attempted:
        return None, None, None

    _load_attempted = True
    try:
        from transformers import BartForConditionalGeneration, BartTokenizer

        _device = _get_device()
        model_name = "sshleifer/distilbart-cnn-12-6"
        logger.info(f"Loading summarization model ({model_name}) on {_device}...")
        _tokenizer = BartTokenizer.from_pretrained(model_name)
        _model = BartForConditionalGeneration.from_pretrained(model_name)
        _model.eval()
        _model.to(_device)
        logger.info(f"Model loaded on {_device}")
        return _model, _tokenizer, _device
    except Exception as e:
        logger.warning(f"Failed to load BART model: {e}. Falling back to extractive.")
        return None, None, None


def summarize_text(
    text: str,
    max_length: int = 130,
    min_length: int = 30,
) -> Optional[str]:
    """
    Summarize text using DistilBART.

    Args:
        text: Input text to summarize
        max_length: Maximum summary length in tokens
        min_length: Minimum summary length in tokens

    Returns:
        Summary string, or None if model is unavailable
    """
    model, tokenizer, device = _load_model()
    if model is None:
        return None

    # BART has a 1024 token input limit — truncate to fit
    truncated = text[:4000]

    # Skip very short texts
    if len(truncated.split()) < 20:
        return None

    try:
        import torch

        inputs = tokenizer(
            truncated,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=1,
                length_penalty=1.0,
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logger.debug(f"BART summarization failed: {e}")
        return None


def summarize_thread(
    subject: str,
    participants: list,
    email_count: int,
    text: str,
) -> str:
    """
    Summarize an email thread. Falls back to extractive if model unavailable.
    """
    summary = summarize_text(text, max_length=150, min_length=40)

    if summary:
        meta = f"Thread with {email_count} emails about '{subject}'. "
        meta += f"Participants: {', '.join(participants[:5])}. "
        return meta + summary

    # Fallback: extractive
    preview = text[:200].strip()
    meta = f"Thread with {email_count} emails about '{subject}'. "
    meta += f"Participants: {', '.join(participants[:3])}. "
    meta += f"Preview: {preview}..."
    return meta


def summarize_attachment(
    filename: str,
    chunk_count: int,
    text: str,
) -> str:
    """
    Summarize an attachment. Falls back to extractive if model unavailable.
    """
    summary = summarize_text(text, max_length=130, min_length=30)

    if summary:
        return f"Attachment '{filename}' ({chunk_count} chunk(s)). {summary}"

    # Fallback: extractive
    preview = text[:300].strip()
    return f"Attachment '{filename}' with {chunk_count} chunk(s). Preview: {preview}..."
