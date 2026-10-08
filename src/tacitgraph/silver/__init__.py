# Silver Layer Module
# Classification, PII detection, anonymization, KG extraction, chunking
#
# Supports three processing modes:
# - local: Use local models (Presidio/spaCy/regex) - no LLM cost
# - llm: Use Azure OpenAI GPT-4o for PII detection + summarization
# - hybrid: Local first, LLM verifies low-confidence cases

from .anonymizer import AnonymizationResult, Anonymizer
from .attachment_classifier import AttachmentClassifier, ClassificationResult
from .chunker import Chunk, SemanticChunker
from .email_sensitivity_classifier import (
    EmailSensitivityClassifier,
    LLMSensitivityClassifier,
    SensitivityResult,
)
from .identity_registry import Identity, IdentityRegistry
from .kg_entity_extractor import (
    KGEntity,
    KGEntityExtractor,
    SpaCyKGExtractor,
    create_kg_extractor,
)
from .language_detector import LanguageDetector
from .pii_detector import PIIDetector, PIIEntity, PIIType
from .privacy_metrics import (
    AnonymizedRecord,
    EquivalenceClass,
    PrivacyMetricsCalculator,
    PrivacyMetricsResult,
    QuasiIdentifier,
    SensitiveAttribute,
    TextPrivacyAnalyzer,
    analyze_text_privacy,
    calculate_privacy_metrics,
)
from .relationship_extractor import (
    RelationshipExtractor,
    create_relationship_extractor,
)
from .silver_processor import SilverLayerProcessor


# Lazy imports for OpenAI components (requires openai package)
def get_openai_detector():
    from .openai_pii_detector import OpenAIPIIDetector

    return OpenAIPIIDetector


def get_openai_anonymizer():
    from .openai_pii_detector import OpenAIAnonymizer

    return OpenAIAnonymizer


def get_openai_summarizer():
    from .openai_summarizer import OpenAISummarizer

    return OpenAISummarizer


def get_unified_processor():
    from .unified_processor import UnifiedProcessor

    return UnifiedProcessor


__all__ = [
    "AnonymizationResult",
    "AnonymizedRecord",
    "Anonymizer",
    "AttachmentClassifier",
    "Chunk",
    "ClassificationResult",
    "EmailSensitivityClassifier",
    "EquivalenceClass",
    "Identity",
    "IdentityRegistry",
    "KGEntity",
    "KGEntityExtractor",
    "LLMSensitivityClassifier",
    "LanguageDetector",
    "PIIDetector",
    "PIIEntity",
    "PIIType",
    "PrivacyMetricsCalculator",
    "PrivacyMetricsResult",
    "QuasiIdentifier",
    "RelationshipExtractor",
    "SemanticChunker",
    "SensitiveAttribute",
    "SensitivityResult",
    "SilverLayerProcessor",
    "SpaCyKGExtractor",
    "TextPrivacyAnalyzer",
    "analyze_text_privacy",
    "calculate_privacy_metrics",
    "create_kg_extractor",
    "create_relationship_extractor",
    "get_openai_anonymizer",
    "get_openai_detector",
    "get_openai_summarizer",
    "get_unified_processor",
]
