"""
PII Detector

Detects Personally Identifiable Information (PII) in text using
multiple detection methods: Presidio, spaCy NER, and regex patterns.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected"""

    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    IBAN = "IBAN"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"  # Social Security Number (US)
    BSN = "BSN"  # Burger Service Nummer (NL)
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    ID_NUMBER = "ID_NUMBER"
    LICENSE_PLATE = "LICENSE_PLATE"
    PASSPORT = "PASSPORT"


@dataclass
class PIIEntity:
    """Detected PII entity"""

    text: str  # The detected text
    pii_type: PIIType  # Type of PII
    start: int  # Start character position
    end: int  # End character position
    confidence: float  # Detection confidence (0-1)
    detection_method: str  # "presidio", "spacy", "regex"

    # For anonymization
    replacement: str | None = None  # What to replace with

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "type": self.pii_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "method": self.detection_method,
        }


class PIIDetector:
    """
    Detect PII in text using multiple methods.

    Detection methods (in order of application):
    1. Microsoft Presidio - Primary detection engine
    2. spaCy NER - Named entity recognition for PERSON, ORG, LOCATION
    3. Regex patterns - Fallback for structured PII (emails, phones, etc.)

    Supports English and Dutch text.
    """

    # Regex patterns for structured PII
    PATTERNS = {
        PIIType.EMAIL: re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        PIIType.PHONE: re.compile(
            # Phone numbers - captures common formats without trailing punctuation
            r"(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}(?:[-.\s]?\d{1,4})?"
        ),
        PIIType.IBAN: re.compile(
            # IBAN - supports with/without spaces (e.g., NL91 ABNA 0417 1643 00)
            r"\b[A-Z]{2}\s?\d{2}\s?[A-Z0-9]{4}\s?\d{4}\s?\d{4}\s?\d{2,4}(?:\s?\d{2})?\b"
        ),
        PIIType.CREDIT_CARD: re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        PIIType.IP_ADDRESS: re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        PIIType.URL: re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*"),
        PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        PIIType.BSN: re.compile(
            r"\b\d{9}\b"  # Dutch BSN is 9 digits
        ),
        PIIType.LICENSE_PLATE: re.compile(
            # Dutch and Belgian formats
            r"\b[A-Z]{2}[-\s]?\d{2,3}[-\s]?[A-Z]{2}\b|\b\d[-\s]?[A-Z]{3}[-\s]?\d{3}\b"
        ),
    }

    # Entity types to detect
    # Note: DATE_OF_BIRTH removed - regular dates are not PII
    # Only actual birth dates with context (e.g., "DOB: 01/15/1985") should be PII
    DEFAULT_ENTITIES = [
        PIIType.PERSON,
        PIIType.EMAIL,
        PIIType.PHONE,
        PIIType.IBAN,
        PIIType.ADDRESS,
        PIIType.CREDIT_CARD,
        PIIType.IP_ADDRESS,
    ]

    # PERSON blocklist — common false positives from spaCy/Presidio
    PERSON_BLOCKLIST = {
        "tel",
        "telephone",
        "cell",
        "fax",
        "mobile",
        "goodbye",
        "regards",
        "thanks",
        "cheers",
        "sincerely",
        "sent",
        "received",
        "from",
        "to",
        "cc",
        "bcc",
        "subject",
        "date",
        "time",
        "attachment",
        "forwarded",
        "se",
        "advisory se",
        "ocr status",
        "ocr",
        "dear",
        "hi",
        "hello",
        "good morning",
        "good afternoon",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "re",
        "fw",
        "fwd",
    }

    # Type priority for merge conflicts — higher number wins
    TYPE_PRIORITY = {
        PIIType.EMAIL: 10,
        PIIType.IP_ADDRESS: 9,
        PIIType.IBAN: 9,
        PIIType.CREDIT_CARD: 9,
        PIIType.SSN: 9,
        PIIType.BSN: 8,
        PIIType.URL: 7,
        PIIType.PHONE: 5,
        PIIType.PERSON: 6,
        PIIType.ORGANIZATION: 4,
        PIIType.LOCATION: 4,
        PIIType.ADDRESS: 4,
        PIIType.LICENSE_PLATE: 6,
        PIIType.PASSPORT: 8,
        PIIType.ID_NUMBER: 7,
        PIIType.DATE_OF_BIRTH: 3,
    }

    def __init__(
        self,
        entities: list[PIIType] | None = None,
        languages: list[str] | None = None,
        use_presidio: bool = True,
        use_spacy: bool = True,
        use_regex: bool = True,
        confidence_threshold: float = 0.5,
        identity_registry=None,
    ):
        """
        Initialize the PII detector.

        Args:
            entities: PII types to detect (default: common types)
            languages: Languages to support (default: ["en", "nl"])
            use_presidio: Use Microsoft Presidio
            use_spacy: Use spaCy NER
            use_regex: Use regex patterns
            confidence_threshold: Minimum confidence for detection
            identity_registry: Optional IdentityRegistry for PERSON validation
        """
        self.entities = entities or self.DEFAULT_ENTITIES
        self.languages = languages or ["en", "nl"]
        self.use_presidio = use_presidio
        self.use_spacy = use_spacy
        self.use_regex = use_regex
        self.confidence_threshold = confidence_threshold
        self.identity_registry = identity_registry

        # Initialize detection engines
        self._presidio_analyzer = None
        self._spacy_models = {}

        if use_presidio:
            self._init_presidio()

        if use_spacy:
            self._init_spacy()

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer"""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Entity labels to ignore (not PII)
            # These are spaCy NER labels that aren't personally identifiable
            labels_to_ignore = [
                "CARDINAL",  # Numbers (one, two, 100)
                "ORDINAL",  # Ordinal numbers (first, second)
                "QUANTITY",  # Quantities (3 gallons, 5 kg)
                "PERCENT",  # Percentages (50%)
                "MONEY",  # Money amounts ($100)
                "TIME",  # Times (4pm, noon)
                "DATE",  # Dates (May 2013) - not DOB
                "FAC",  # Facilities (buildings, airports)
                "WORK_OF_ART",  # Titles of works
                "LAW",  # Laws, regulations
                "LANGUAGE",  # Languages
                "EVENT",  # Named events
                "PRODUCT",  # Products
                "NORP",  # Nationalities, religions
            ]

            # Configure NLP engine with labels to ignore
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "en", "model_name": "en_core_web_lg"},
                    {"lang_code": "nl", "model_name": "nl_core_news_lg"},
                ],
                "ner_model_configuration": {
                    "labels_to_ignore": labels_to_ignore,
                },
            }

            try:
                provider = NlpEngineProvider(nlp_configuration=configuration)
                nlp_engine = provider.create_engine()
                self._presidio_analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            except Exception as e:
                logger.warning(f"Failed to load custom NLP config: {e}, using default")
                self._presidio_analyzer = AnalyzerEngine()

            logger.info("Presidio analyzer initialized")

        except ImportError:
            logger.warning(
                "Presidio not available. Install: pip install presidio-analyzer presidio-anonymizer"
            )
            self.use_presidio = False
        except Exception as e:
            # Catch other errors like Python version incompatibilities
            logger.warning(f"Presidio initialization failed: {e}")
            self.use_presidio = False

    def _init_spacy(self) -> None:
        """Initialize spaCy models"""
        try:
            import spacy

            model_map = {
                "en": "en_core_web_trf",  # or "en_core_web_lg" for faster processing
                "nl": "nl_core_news_lg",
            }

            for lang in self.languages:
                model_name = model_map.get(lang)
                if model_name:
                    try:
                        self._spacy_models[lang] = spacy.load(model_name)
                        logger.info(f"Loaded spaCy model: {model_name}")
                    except OSError:
                        # Try to download
                        logger.info(f"Downloading spaCy model: {model_name}")
                        try:
                            spacy.cli.download(model_name)
                            self._spacy_models[lang] = spacy.load(model_name)
                        except Exception as e:
                            logger.warning(f"Failed to load {model_name}: {e}")

        except ImportError:
            logger.warning("spaCy not available. Install: pip install spacy")
            self.use_spacy = False
        except Exception as e:
            # Catch other errors like Python version incompatibilities
            logger.warning(f"spaCy initialization failed: {e}")
            self.use_spacy = False

    def detect(self, text: str, language: str = "en") -> list[PIIEntity]:
        """
        Detect PII in text.

        Args:
            text: Text to analyze
            language: Text language (en, nl)

        Returns:
            List of detected PII entities
        """
        if not text or not text.strip():
            return []

        all_entities = []

        # 1. Presidio detection
        if self.use_presidio and self._presidio_analyzer:
            presidio_entities = self._detect_with_presidio(text, language)
            all_entities.extend(presidio_entities)

        # 2. spaCy NER
        if self.use_spacy and language in self._spacy_models:
            spacy_entities = self._detect_with_spacy(text, language)
            all_entities.extend(spacy_entities)

        # 3. Regex patterns
        if self.use_regex:
            regex_entities = self._detect_with_regex(text)
            all_entities.extend(regex_entities)

        # Suppress PERSON entities that are really signature labels (Tel:, Fax:)
        # in signature blocks
        cleaned = []
        for entity in all_entities:
            # In signature blocks, suppress PERSON if preceded by phone label
            if entity.pii_type == PIIType.PERSON:
                preceding = text[max(0, entity.start - 10) : entity.start].strip()
                if re.match(r"(?:Tel|Cell|Fax|Mobile|Ph)\s*:?\s*$", preceding, re.IGNORECASE):
                    continue
            cleaned.append(entity)

        # Deduplicate and merge overlapping entities
        merged_entities = self._merge_entities(cleaned)

        # Filter by confidence
        filtered = [e for e in merged_entities if e.confidence >= self.confidence_threshold]

        # Sort by position
        return sorted(filtered, key=lambda e: e.start)

    def _detect_with_presidio(self, text: str, language: str) -> list[PIIEntity]:
        """Detect PII using Presidio"""
        entities = []

        # Map our PIIType to Presidio entity types
        presidio_entities = []
        for pii_type in self.entities:
            presidio_name = self._map_to_presidio(pii_type)
            if presidio_name:
                presidio_entities.append(presidio_name)

        try:
            results = self._presidio_analyzer.analyze(
                text=text, entities=presidio_entities, language=language
            )

            for result in results:
                pii_type = self._map_from_presidio(result.entity_type)
                if pii_type:
                    entity = PIIEntity(
                        text=text[result.start : result.end],
                        pii_type=pii_type,
                        start=result.start,
                        end=result.end,
                        confidence=result.score,
                        detection_method="presidio",
                    )
                    entities.append(entity)

        except Exception as e:
            logger.warning(f"Presidio detection failed: {e}")

        return entities

    def _detect_with_spacy(self, text: str, language: str) -> list[PIIEntity]:
        """Detect named entities using spaCy"""
        entities = []

        nlp = self._spacy_models.get(language)
        if not nlp:
            return entities

        try:
            doc = nlp(text)

            # Map spaCy entity types to PII types
            spacy_mapping = {
                "PERSON": PIIType.PERSON,
                "PER": PIIType.PERSON,  # Dutch model
                "ORG": PIIType.ORGANIZATION,
                "GPE": PIIType.LOCATION,
                "LOC": PIIType.LOCATION,
            }

            for ent in doc.ents:
                pii_type = spacy_mapping.get(ent.label_)
                if pii_type and pii_type in self.entities:
                    entity = PIIEntity(
                        text=ent.text,
                        pii_type=pii_type,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.85,  # spaCy doesn't provide confidence
                        detection_method="spacy",
                    )
                    entities.append(entity)

        except Exception as e:
            logger.warning(f"spaCy detection failed: {e}")

        return entities

    def _detect_with_regex(self, text: str) -> list[PIIEntity]:
        """Detect PII using regex patterns"""
        entities = []

        for pii_type, pattern in self.PATTERNS.items():
            if pii_type not in self.entities:
                continue

            for match in pattern.finditer(text):
                matched_text = match.group()
                start = match.start()
                end = match.end()

                # Strip trailing punctuation for phone numbers
                if pii_type == PIIType.PHONE:
                    while matched_text and matched_text[-1] in ".,;:!?)":
                        matched_text = matched_text[:-1]
                        end -= 1

                # Validate the match (with full text context for phone validation)
                if self._validate_regex_match(matched_text, pii_type, text, start):
                    entity = PIIEntity(
                        text=matched_text,
                        pii_type=pii_type,
                        start=start,
                        end=end,
                        confidence=0.9,  # High confidence for regex matches
                        detection_method="regex",
                    )
                    entities.append(entity)

        return entities

    # Patterns that look like phone numbers but aren't
    _DATE_PATTERN = re.compile(
        r"^\d{4}[-/]\d{2}[-/]\d{2}$|"  # YYYY-MM-DD or YYYY/MM/DD
        r"^\d{2}[-/]\d{2}[-/]\d{4}$|"  # DD/MM/YYYY or MM/DD/YYYY
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$"  # D/M/YY variants
    )
    _IP_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    _NON_PHONE_CONTEXT = re.compile(
        r"(?:invoice|ref|ticket|order|lat|lon|longitude|latitude|"
        r"version|v\.|no\.|nr\.|#|id|case)\s*[:.]?\s*$",
        re.IGNORECASE,
    )

    def _validate_regex_match(
        self, text: str, pii_type: PIIType, full_text: str = "", start: int = 0
    ) -> bool:
        """Validate regex match to reduce false positives"""
        if pii_type == PIIType.PHONE:
            return self._validate_phone(text, full_text, start)

        if pii_type == PIIType.IP_ADDRESS:
            # Validate IP address format
            parts = text.split(".")
            try:
                return all(0 <= int(p) <= 255 for p in parts)
            except ValueError:
                return False

        if pii_type == PIIType.BSN:
            # Dutch BSN must pass 11-test
            return self._validate_bsn(text)

        return True

    def _validate_phone(self, text: str, full_text: str = "", start: int = 0) -> bool:
        """Validate phone number, rejecting dates, IPs, and non-phone number sequences."""
        # Reject dates (YYYY-MM-DD, DD/MM/YYYY, etc.)
        if self._DATE_PATTERN.match(text.strip()):
            return False

        # Reject IP addresses
        if self._IP_PATTERN.match(text.strip()):
            return False

        # Must have at least 10 digits (international) or phone-like formatting
        digits = re.sub(r"\D", "", text)

        # Short digit sequences without phone formatting are not phones
        has_phone_format = bool(re.search(r"[+\(\)]", text) or re.search(r"\d[-.\s]\d", text))

        if len(digits) < 7:
            return False

        if len(digits) < 10 and not has_phone_format:
            return False

        # Check context: preceding words like "invoice", "ref", "ticket"
        if full_text and start > 0:
            preceding = full_text[max(0, start - 30) : start]
            if self._NON_PHONE_CONTEXT.search(preceding):
                return False

        return True

    def _validate_bsn(self, bsn: str) -> bool:
        """Validate Dutch BSN using 11-test"""
        bsn = re.sub(r"\D", "", bsn)
        if len(bsn) != 9:
            return False

        try:
            total = sum(int(bsn[i]) * (9 - i) for i in range(8))
            total -= int(bsn[8])
            return total % 11 == 0
        except (ValueError, IndexError):
            return False

    def _merge_entities(self, entities: list[PIIEntity]) -> list[PIIEntity]:
        """Merge overlapping entities using type priority.

        When two entities overlap, the one with higher TYPE_PRIORITY wins.
        This prevents, e.g., an IP address from being swallowed by a PHONE match.
        """
        if not entities:
            return []

        # Clean spaCy boundary artifacts from PERSON entities
        entities = [self._clean_entity_boundaries(e) for e in entities]

        # Validate PERSON entities against blocklist and registry
        entities = [e for e in entities if self._validate_person_entity(e)]

        if not entities:
            return []

        # Sort by start position, then by type priority (desc), then confidence (desc)
        sorted_entities = sorted(
            entities, key=lambda e: (e.start, -self.TYPE_PRIORITY.get(e.pii_type, 0), -e.confidence)
        )

        merged = []
        current = sorted_entities[0]

        for next_entity in sorted_entities[1:]:
            # Check for overlap
            if next_entity.start < current.end:
                # Use type priority to resolve
                current_priority = self.TYPE_PRIORITY.get(current.pii_type, 0)
                next_priority = self.TYPE_PRIORITY.get(next_entity.pii_type, 0)

                if next_priority > current_priority or (
                    next_priority == current_priority
                    and next_entity.confidence > current.confidence
                ):
                    current = next_entity
                # If same type, extend the span
                elif next_entity.pii_type == current.pii_type:
                    current = PIIEntity(
                        text=current.text
                        if current.end >= next_entity.end
                        else current.text + next_entity.text[current.end - next_entity.start :],
                        pii_type=current.pii_type,
                        start=current.start,
                        end=max(current.end, next_entity.end),
                        confidence=max(current.confidence, next_entity.confidence),
                        detection_method=current.detection_method,
                    )
            else:
                merged.append(current)
                current = next_entity

        merged.append(current)
        return merged

    def _clean_entity_boundaries(self, entity: PIIEntity) -> PIIEntity:
        """Clean spaCy entity boundary artifacts.

        Strips trailing control characters, header labels, and
        newline-appended text from PERSON entities.
        """
        if entity.pii_type != PIIType.PERSON:
            return entity

        text = entity.text

        # Strip trailing \r\n and whitespace
        text = text.rstrip()

        # Strip trailing header fragments: \nDate, \nCc:, \r\nAcme Ltd
        text = re.sub(r"[\r\n]+.*$", "", text)

        # Strip trailing punctuation artifacts
        text = text.rstrip(".,;:!?")

        if text != entity.text:
            return PIIEntity(
                text=text,
                pii_type=entity.pii_type,
                start=entity.start,
                end=entity.start + len(text),
                confidence=entity.confidence,
                detection_method=entity.detection_method,
            )

        return entity

    def _validate_person_entity(self, entity: PIIEntity) -> bool:
        """Validate PERSON entity, rejecting blocklisted false positives."""
        if entity.pii_type != PIIType.PERSON:
            return True

        text = entity.text.strip()

        # Reject very short names (< 3 chars) that aren't in the registry
        if len(text) < 3:
            if self.identity_registry:
                return self.identity_registry.lookup_by_name(text) is not None
            return False

        # Reject blocklisted names
        if text.lower() in self.PERSON_BLOCKLIST:
            return False

        # Reject if text is purely digits
        if text.replace(" ", "").isdigit():
            return False

        # Boost confidence for names in registry
        if self.identity_registry:
            identity = self.identity_registry.lookup_by_name(text)
            if identity:
                # Known person — boost confidence
                entity_obj = PIIEntity(
                    text=entity.text,
                    pii_type=entity.pii_type,
                    start=entity.start,
                    end=entity.end,
                    confidence=max(entity.confidence, 0.95),
                    detection_method=entity.detection_method,
                )
                # Mutate in place since we can't easily return modified
                entity.confidence = entity_obj.confidence

        return True

    def _is_in_signature_block(self, text: str, start: int) -> bool:
        """Detect if position is within an email signature block."""
        # Look backwards for signature markers
        preceding = text[:start]
        signature_markers = [
            r"(?:Best |Kind |Warm )?[Rr]egards\s*[,.]?\s*$",
            r"[Tt]hanks?\s*[,.]?\s*$",
            r"[Cc]heers\s*[,.]?\s*$",
            r"[Ss]incerely\s*[,.]?\s*$",
            r"--\s*$",
            r"_{3,}\s*$",
        ]
        for marker in signature_markers:
            if re.search(marker, preceding[-200:] if len(preceding) > 200 else preceding):
                return True
        return False

    def _map_to_presidio(self, pii_type: PIIType) -> str | None:
        """Map PIIType to Presidio entity type"""
        mapping = {
            PIIType.PERSON: "PERSON",
            PIIType.EMAIL: "EMAIL_ADDRESS",
            PIIType.PHONE: "PHONE_NUMBER",
            PIIType.ADDRESS: "ADDRESS",
            PIIType.IBAN: "IBAN_CODE",
            PIIType.CREDIT_CARD: "CREDIT_CARD",
            PIIType.SSN: "US_SSN",
            PIIType.IP_ADDRESS: "IP_ADDRESS",
            PIIType.URL: "URL",
            PIIType.LOCATION: "LOCATION",
            # PIIType.DATE_OF_BIRTH: "DATE_TIME",  # Disabled: regular dates are not PII
        }
        return mapping.get(pii_type)

    def _map_from_presidio(self, presidio_type: str) -> PIIType | None:
        """Map Presidio entity type to PIIType"""
        mapping = {
            "PERSON": PIIType.PERSON,
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "PHONE_NUMBER": PIIType.PHONE,
            "ADDRESS": PIIType.ADDRESS,
            "IBAN_CODE": PIIType.IBAN,
            "CREDIT_CARD": PIIType.CREDIT_CARD,
            "US_SSN": PIIType.SSN,
            "IP_ADDRESS": PIIType.IP_ADDRESS,
            "URL": PIIType.URL,
            "LOCATION": PIIType.LOCATION,
            # "DATE_TIME": PIIType.DATE_OF_BIRTH,  # Disabled: regular dates are not PII
            "NRP": PIIType.ID_NUMBER,  # National Registration Number
        }
        return mapping.get(presidio_type)


# Convenience function
def detect_pii(text: str, language: str = "en") -> list[PIIEntity]:
    """
    Detect PII in text.

    Args:
        text: Text to analyze
        language: Text language

    Returns:
        List of detected PII entities
    """
    detector = PIIDetector()
    return detector.detect(text, language)
