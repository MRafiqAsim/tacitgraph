# Silver Layer Module
# Classification, PII detection, anonymization, KG extraction, chunking
#
# Supports three processing modes:
# - local: Use local models (Presidio/spaCy/regex) - no LLM cost
# - llm: Use Azure OpenAI GPT-4o for PII detection + summarization
# - hybrid: Local first, LLM verifies low-confidence cases

from .pii_detector import PIIDetector, PIIEntity, PIIType
from .anonymizer import Anonymizer, AnonymizationResult
from .identity_registry import IdentityRegistry, Identity
from .silver_processor import SilverLayerProcessor
from .kg_entity_extractor import (
    KGEntityExtractor,
    KGEntity,
    SpaCyKGExtractor,
    create_kg_extractor,
)
from .relationship_extractor import (
    RelationshipExtractor,
    create_relationship_extractor,
)
from .attachment_classifier import AttachmentClassifier, ClassificationResult
from .email_sensitivity_classifier import EmailSensitivityClassifier, LLMSensitivityClassifier, SensitivityResult
from .language_detector import LanguageDetector
from .chunker import SemanticChunker, Chunk
from .privacy_metrics import (
    PrivacyMetricsCalculator,
    PrivacyMetricsResult,
    TextPrivacyAnalyzer,
    AnonymizedRecord,
    QuasiIdentifier,
    SensitiveAttribute,
    EquivalenceClass,
    calculate_privacy_metrics,
    analyze_text_privacy,
)

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
    # PII Detection
    "PIIDetector", "PIIEntity", "PIIType",
    # Anonymization
    "Anonymizer", "AnonymizationResult",
    # Identity Registry
    "IdentityRegistry", "Identity",
    # Silver Layer Processing
    "SilverLayerProcessor",
    # KG Extraction
    "KGEntityExtractor", "KGEntity", "SpaCyKGExtractor", "create_kg_extractor",
    "RelationshipExtractor", "create_relationship_extractor",
    # Language Detection & Chunking
    "LanguageDetector", "SemanticChunker", "Chunk",
    # Privacy Metrics
    "PrivacyMetricsCalculator", "PrivacyMetricsResult", "TextPrivacyAnalyzer",
    "AnonymizedRecord", "QuasiIdentifier", "SensitiveAttribute", "EquivalenceClass",
    "calculate_privacy_metrics", "analyze_text_privacy",
    # OpenAI Components (lazy loaded)
    "get_openai_detector", "get_openai_anonymizer", "get_openai_summarizer",
    "get_unified_processor",
]
