"""
OpenAI-based PII Detector

Uses OpenAI API for PII detection with structured output.
"""

import json
import logging

import httpx

from tacitgraph.prompt_loader import format_prompt, get_prompt

from .pii_detector import PIIEntity, PIIType

logger = logging.getLogger(__name__)


class OpenAIPIIDetector:
    """
    PII Detector using OpenAI API.

    Uses GPT models to identify PII with high accuracy,
    especially for context-dependent and ambiguous cases.
    Supports both OpenAI and Azure OpenAI endpoints.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        confidence_threshold: float = 0.5,
        use_azure: bool = False,
        azure_endpoint: str | None = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment: str | None = None,
        identity_registry=None,
    ):
        """
        Initialize OpenAI PII Detector.

        Args:
            api_key: OpenAI/Azure API key
            model: Model to use (default: gpt-4o)
            temperature: Temperature for generation (default: 0.0 for deterministic)
            confidence_threshold: Minimum confidence to include entity
            use_azure: Whether to use Azure OpenAI instead of OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure API version
            azure_deployment: Azure deployment name (used as model for Azure)
            identity_registry: Optional IdentityRegistry for known-person context
        """
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.identity_registry = identity_registry
        self.use_azure = use_azure

        try:
            if use_azure:
                from openai import AzureOpenAI

                self.model = azure_deployment or model
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_api_version,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    max_retries=2,
                )
                logger.info(
                    f"Azure OpenAI PII Detector initialized: {azure_endpoint}, deployment={self.model}"
                )
            else:
                from openai import OpenAI

                self.model = model
                self.client = OpenAI(api_key=api_key)
                logger.info(f"OpenAI PII Detector initialized with model: {model}")
        except ImportError as exc:
            raise ImportError("OpenAI package not installed. Run: pip install openai") from exc

    def detect(self, text: str, language: str = "en") -> list[PIIEntity]:
        """
        Detect PII in text using OpenAI.

        Args:
            text: Text to analyze
            language: Text language (used for context)

        Returns:
            List of detected PII entities
        """
        if not text or not text.strip():
            return []

        try:
            # Build system prompt with optional identity context
            system_prompt = get_prompt("silver", "pii_detection", "system_prompt")
            if self.identity_registry and self.identity_registry.identity_count > 0:
                known_names = sorted(self.identity_registry.get_all_known_names())[:50]
                suffix = get_prompt("silver", "pii_detection", "known_people_suffix", "")
                system_prompt += suffix.format(known_names=", ".join(known_names))

            user_prompt = format_prompt(
                get_prompt("silver", "pii_detection", "user_prompt"), text=text
            )

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=get_prompt("silver", "pii_detection", "temperature", 0.0),
                response_format={"type": "json_object"},
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            content = extract_llm_content(response, context="PII detection")
            result = json.loads(content)

            # Extract entities array from response
            entities_data = result.get("entities", []) if isinstance(result, dict) else result

            # Convert to PIIEntity objects
            entities = []
            for item in entities_data:
                try:
                    pii_type = PIIType(item["type"])
                    confidence = float(item.get("confidence", 0.9))

                    if confidence >= self.confidence_threshold:
                        entity = PIIEntity(
                            text=item["text"],
                            pii_type=pii_type,
                            start=int(item["start"]),
                            end=int(item["end"]),
                            confidence=confidence,
                            detection_method="openai",
                        )
                        entities.append(entity)

                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse entity: {item}, error: {e}")
                    continue

            # Validate positions
            entities = self._validate_positions(entities, text)

            return sorted(entities, key=lambda e: e.start)

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, JSONDecodeError, network timeouts, etc.
            logger.error(f"OpenAI PII detection failed: {e}")
            return []

    def _find_all_occurrences(self, text: str, substring: str) -> list[int]:
        """Find all start positions of substring in text."""
        positions = []
        start = 0
        while True:
            idx = text.find(substring, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + 1
        return positions

    def _validate_positions(self, entities: list[PIIEntity], text: str) -> list[PIIEntity]:
        """
        Validate entity positions and deduplicate.

        GPT-4o may return slightly wrong positions or duplicate entries for
        repeated text (e.g., a name appearing 4 times in a thread).  We:
        1. Find ALL real occurrences of the entity text in the source.
        2. Match each GPT-4o entity to the nearest real occurrence.
        3. Deduplicate so each (start, end) span appears only once.
        """
        # Cache: entity_text -> list of real positions in source text
        occurrence_cache: dict[str, list[int]] = {}

        validated = []
        for entity in entities:
            # Build/reuse the list of real occurrences
            if entity.text not in occurrence_cache:
                occurrence_cache[entity.text] = self._find_all_occurrences(text, entity.text)

            occurrences = occurrence_cache[entity.text]
            if not occurrences:
                logger.warning(f"Could not find entity text in source: {entity.text}")
                continue

            # Check if the GPT-4o position is already correct
            actual_text = (
                text[entity.start : entity.end]
                if 0 <= entity.start < entity.end <= len(text)
                else ""
            )
            if actual_text == entity.text:
                validated.append(entity)
                continue

            # Find the nearest real occurrence to the GPT-4o position
            nearest = min(occurrences, key=lambda pos: abs(pos - entity.start))
            validated.append(
                PIIEntity(
                    text=entity.text,
                    pii_type=entity.pii_type,
                    start=nearest,
                    end=nearest + len(entity.text),
                    confidence=entity.confidence,
                    detection_method=entity.detection_method,
                )
            )

        # Deduplicate: keep one entity per unique (start, end, type) span
        seen: dict[tuple, PIIEntity] = {}
        for entity in validated:
            key = (entity.start, entity.end, entity.pii_type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return sorted(seen.values(), key=lambda e: e.start)

    def detect_batch(self, texts: list[str], language: str = "en") -> list[list[PIIEntity]]:
        """
        Detect PII in multiple texts.

        Args:
            texts: List of texts to analyze
            language: Text language

        Returns:
            List of entity lists, one per input text
        """
        results = []
        for text in texts:
            entities = self.detect(text, language)
            results.append(entities)
        return results


class OpenAIAnonymizer:
    """
    Anonymizer using OpenAI API.

    Can generate realistic synthetic replacements for PII.
    Supports both OpenAI and Azure OpenAI endpoints.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        generate_synthetic: bool = True,
        use_azure: bool = False,
        azure_endpoint: str | None = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment: str | None = None,
    ):
        """
        Initialize OpenAI Anonymizer.

        Args:
            api_key: OpenAI/Azure API key
            model: Model to use
            generate_synthetic: Generate realistic replacements
            use_azure: Whether to use Azure OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_version: Azure API version
            azure_deployment: Azure deployment name
        """
        self.generate_synthetic = generate_synthetic
        self._replacement_cache: dict[str, str] = {}

        try:
            if use_azure:
                from openai import AzureOpenAI

                self.model = azure_deployment or model
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_api_version,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    max_retries=2,
                )
            else:
                from openai import OpenAI

                self.model = model
                self.client = OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError("OpenAI package not installed. Run: pip install openai") from exc

    def anonymize(self, text: str, entities: list[PIIEntity], strategy: str = "replace") -> str:
        """
        Anonymize text by replacing PII entities.

        Args:
            text: Original text
            entities: Detected PII entities
            strategy: "replace" (placeholders), "synthetic" (realistic), "redact" (remove)

        Returns:
            Anonymized text
        """
        if not entities:
            return text

        # Sort entities by position (reverse order for replacement)
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        result = text

        for entity in sorted_entities:
            if strategy == "redact":
                replacement = "[REDACTED]"
            elif strategy == "synthetic" and self.generate_synthetic:
                replacement = self._get_synthetic_replacement(entity)
            else:
                # Default placeholder replacement
                replacement = self._get_placeholder(entity)

            result = result[: entity.start] + replacement + result[entity.end :]

        return result

    def _get_placeholder(self, entity: PIIEntity) -> str:
        """Generate placeholder replacement"""
        type_name = entity.pii_type.value
        # Use hash for consistency
        hash_id = hash(entity.text) % 1000
        return f"[{type_name}_{hash_id}]"

    def _get_synthetic_replacement(self, entity: PIIEntity) -> str:
        """Generate synthetic realistic replacement"""
        # Check cache for consistency
        cache_key = f"{entity.pii_type.value}:{entity.text}"
        if cache_key in self._replacement_cache:
            return self._replacement_cache[cache_key]

        try:
            prompt = format_prompt(
                get_prompt("silver", "synthetic_replacement", "user_prompt"),
                pii_type=entity.pii_type.value,
                original_text=entity.text,
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt("silver", "synthetic_replacement", "system_prompt"),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=get_prompt("silver", "synthetic_replacement", "temperature", 0.7),
                max_tokens=get_prompt("silver", "synthetic_replacement", "max_tokens", 50),
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            replacement = extract_llm_content(response, context="synthetic replacement")
            self._replacement_cache[cache_key] = replacement
            return replacement

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, network timeouts, etc.
            logger.warning(f"Failed to generate synthetic replacement: {e}")
            return self._get_placeholder(entity)

    def anonymize_with_detection(
        self, text: str, language: str = "en", strategy: str = "replace"
    ) -> tuple:
        """
        Detect PII and anonymize in one call.

        Returns:
            (anonymized_text, detected_entities)
        """
        detector = OpenAIPIIDetector(api_key=self.client.api_key, model=self.model)
        entities = detector.detect(text, language)
        anonymized = self.anonymize(text, entities, strategy)
        return anonymized, entities
