"""
Language Detector

Detects the language of text content for proper NLP model selection.
Supports English and Dutch as primary languages.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetectionResult:
    """Result of language detection"""

    language: str                    # ISO 639-1 code (en, nl, etc.)
    confidence: float                # 0.0 to 1.0
    is_reliable: bool                # Confidence > threshold
    all_languages: List[Tuple[str, float]]  # All detected languages with scores


class LanguageDetector:
    """
    Detect language of text content.

    Uses multiple detection methods for reliability:
    1. langdetect (Google's language detection)
    2. langid (fallback)
    3. Character/word patterns (simple fallback)

    Primary focus on English (en) and Dutch (nl).
    """

    # Supported languages for NLP processing
    SUPPORTED_LANGUAGES = {"en", "nl"}

    # Default language when detection fails
    DEFAULT_LANGUAGE = "en"

    # Confidence threshold for reliable detection
    CONFIDENCE_THRESHOLD = 0.8

    # Common Dutch words for pattern matching
    DUTCH_INDICATORS = {
        "de", "het", "een", "van", "en", "in", "is", "dat", "op", "te",
        "voor", "met", "zijn", "aan", "niet", "ook", "maar", "bij", "naar",
        "dit", "dan", "nog", "wel", "als", "om", "tot", "geen", "kan",
        "hebben", "worden", "deze", "zou", "moet", "meer", "veel", "over",
        "alleen", "andere", "goed", "mensen", "jaar", "waar", "maken",
        "geweest", "gedaan", "graag", "beste", "groeten", "bedankt", "alstublieft"
    }

    # Common English words
    ENGLISH_INDICATORS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "best", "regards", "thanks", "please"
    }

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        default_language: str = "en"
    ):
        """
        Initialize the detector.

        Args:
            confidence_threshold: Minimum confidence for reliable detection
            default_language: Language to use when detection fails
        """
        self.confidence_threshold = confidence_threshold
        self.default_language = default_language

        # Check available detection libraries
        self._langdetect_available = self._check_langdetect()
        self._langid_available = self._check_langid()

    def _check_langdetect(self) -> bool:
        """Check if langdetect is available"""
        try:
            import langdetect
            return True
        except ImportError:
            logger.debug("langdetect not available")
            return False

    def _check_langid(self) -> bool:
        """Check if langid is available"""
        try:
            import langid
            return True
        except ImportError:
            logger.debug("langid not available")
            return False

    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of text.

        Args:
            text: Text to analyze

        Returns:
            LanguageDetectionResult with language code and confidence
        """
        if not text or len(text.strip()) < 10:
            return LanguageDetectionResult(
                language=self.default_language,
                confidence=0.0,
                is_reliable=False,
                all_languages=[]
            )

        # Try langdetect first (most accurate)
        if self._langdetect_available:
            result = self._detect_with_langdetect(text)
            if result.is_reliable:
                return result

        # Try langid as fallback
        if self._langid_available:
            result = self._detect_with_langid(text)
            if result.is_reliable:
                return result

        # Pattern-based fallback
        return self._detect_with_patterns(text)

    def detect_batch(self, texts: List[str]) -> List[LanguageDetectionResult]:
        """
        Detect language for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of detection results
        """
        return [self.detect(text) for text in texts]

    def get_spacy_model(self, language: str) -> str:
        """
        Get the appropriate spaCy model for a language.

        Args:
            language: ISO 639-1 language code

        Returns:
            spaCy model name
        """
        model_map = {
            "en": "en_core_web_trf",      # English transformer model
            "nl": "nl_core_news_lg",       # Dutch large model
        }

        return model_map.get(language, model_map["en"])

    def _detect_with_langdetect(self, text: str) -> LanguageDetectionResult:
        """Detect using langdetect library"""
        import langdetect
        from langdetect import detect_langs

        try:
            # Get all language probabilities
            detections = detect_langs(text)

            all_languages = [(d.lang, d.prob) for d in detections]

            if detections:
                top = detections[0]
                # Map to supported languages
                language = self._map_language(top.lang)

                return LanguageDetectionResult(
                    language=language,
                    confidence=top.prob,
                    is_reliable=top.prob >= self.confidence_threshold,
                    all_languages=all_languages
                )

        except langdetect.LangDetectException:
            pass

        return LanguageDetectionResult(
            language=self.default_language,
            confidence=0.0,
            is_reliable=False,
            all_languages=[]
        )

    def _detect_with_langid(self, text: str) -> LanguageDetectionResult:
        """Detect using langid library"""
        import langid

        try:
            lang, confidence = langid.classify(text)

            # langid returns negative log probability, convert to 0-1
            # Higher magnitude = more confident
            prob = min(1.0, abs(confidence) / 100)

            language = self._map_language(lang)

            return LanguageDetectionResult(
                language=language,
                confidence=prob,
                is_reliable=prob >= self.confidence_threshold,
                all_languages=[(lang, prob)]
            )

        except Exception:
            pass

        return LanguageDetectionResult(
            language=self.default_language,
            confidence=0.0,
            is_reliable=False,
            all_languages=[]
        )

    def _detect_with_patterns(self, text: str) -> LanguageDetectionResult:
        """Simple pattern-based detection as fallback"""
        # Tokenize
        words = set(text.lower().split())

        # Count indicator matches
        dutch_count = len(words & self.DUTCH_INDICATORS)
        english_count = len(words & self.ENGLISH_INDICATORS)

        total = dutch_count + english_count

        if total == 0:
            return LanguageDetectionResult(
                language=self.default_language,
                confidence=0.0,
                is_reliable=False,
                all_languages=[]
            )

        # Calculate ratios
        dutch_ratio = dutch_count / total
        english_ratio = english_count / total

        if dutch_ratio > english_ratio:
            language = "nl"
            confidence = dutch_ratio
        else:
            language = "en"
            confidence = english_ratio

        # Pattern matching is less reliable
        confidence *= 0.8

        return LanguageDetectionResult(
            language=language,
            confidence=confidence,
            is_reliable=confidence >= self.confidence_threshold,
            all_languages=[("nl", dutch_ratio * 0.8), ("en", english_ratio * 0.8)]
        )

    def _map_language(self, lang_code: str) -> str:
        """Map detected language to supported languages"""
        # Direct mapping
        if lang_code in self.SUPPORTED_LANGUAGES:
            return lang_code

        # Common variations
        language_mapping = {
            "nld": "nl",
            "dut": "nl",
            "dutch": "nl",
            "eng": "en",
            "english": "en",
        }

        if lang_code.lower() in language_mapping:
            return language_mapping[lang_code.lower()]

        # Default to English for unsupported languages
        return self.default_language

    def is_mixed_language(self, text: str, threshold: float = 0.3) -> bool:
        """
        Check if text contains significant content in multiple languages.

        Args:
            text: Text to analyze
            threshold: Minimum ratio for secondary language

        Returns:
            True if mixed language content detected
        """
        result = self.detect(text)

        if len(result.all_languages) < 2:
            return False

        # Check if second language is significant
        if result.all_languages[1][1] >= threshold:
            return True

        return False


# Convenience function
def detect_language(text: str) -> str:
    """
    Detect language of text.

    Args:
        text: Text to analyze

    Returns:
        ISO 639-1 language code (en, nl)
    """
    detector = LanguageDetector()
    result = detector.detect(text)
    return result.language


def get_nlp_model_for_text(text: str) -> str:
    """
    Get appropriate spaCy model name for text.

    Args:
        text: Text to analyze

    Returns:
        spaCy model name
    """
    detector = LanguageDetector()
    result = detector.detect(text)
    return detector.get_spacy_model(result.language)
