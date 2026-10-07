"""
Pipeline Configuration

Central configuration for the data processing pipeline.
Supports three processing modes:
- OPENAI: Use OpenAI API for all processing (default)
- LOCAL: Use local models (Presidio/spaCy/regex)
- HYBRID: Combine local for high-confidence, OpenAI for complex cases
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


class ProcessingMode(Enum):
    """Processing mode for the pipeline"""
    OPENAI = "openai"    # Use OpenAI/Azure API for PII detection + summarization (alias: LLM)
    LOCAL = "local"      # Use local models only (Presidio/spaCy/regex)
    HYBRID = "hybrid"    # Combine local + OpenAI based on confidence

    @classmethod
    def from_string(cls, value: str) -> "ProcessingMode":
        """Parse mode from string, supporting aliases."""
        value = value.lower().strip()
        aliases = {"llm": "openai", "openai": "openai", "local": "local", "hybrid": "hybrid"}
        return cls(aliases.get(value, value))


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: Optional[str] = None
    model: str = "gpt-4o"  # Default model for PII/summarization
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0  # Deterministic for PII detection
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3

    def __post_init__(self):
        # Try to get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI API configuration"""
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    deployment: str = "gpt-4o"
    embedding_deployment: str = "text-embedding-3-large"
    api_version: str = "2024-12-01-preview"

    def __post_init__(self):
        if self.endpoint is None:
            self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if self.api_key is None:
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.deployment or self.deployment == "gpt-4o":
            self.deployment = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT", "gpt-4o")


@dataclass
class IdentityRegistryConfig:
    """Identity Registry configuration"""
    enabled: bool = True
    registry_path: str = "./data/identity_registry.json"
    auto_build: bool = True  # Build from bronze if registry file doesn't exist


@dataclass
class PIIConfig:
    """PII detection configuration"""
    # Confidence thresholds
    confidence_threshold: float = 0.5
    hybrid_confidence_threshold: float = 0.7  # Below this, use OpenAI in hybrid mode

    # Entity types to detect
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "EMAIL", "PHONE", "ADDRESS", "IBAN",
        "CREDIT_CARD", "SSN", "BSN", "DATE_OF_BIRTH",
        "IP_ADDRESS", "LOCATION", "ORGANIZATION"
    ])

    # Languages to support
    languages: List[str] = field(default_factory=lambda: ["en", "nl"])

    # Local detection settings
    use_presidio: bool = True
    use_spacy: bool = True
    use_regex: bool = True


@dataclass
class AnonymizationConfig:
    """Anonymization configuration"""
    # Anonymization strategy: "replace", "mask", "hash", "redact", "synthetic"
    default_strategy: str = "replace"

    # Use OpenAI to generate realistic synthetic replacements
    generate_synthetic: bool = True

    # Maintain consistency (same name -> same replacement)
    consistent_replacements: bool = True


@dataclass
class SummarizationConfig:
    """Summarization configuration"""
    enabled: bool = True
    max_summary_length: int = 500  # Max characters for summary
    summary_style: str = "concise"  # "concise", "detailed", "bullet_points"
    preserve_key_entities: bool = True  # Keep important entities in summary


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    # Processing mode (OPENAI/LLM, LOCAL, or HYBRID)
    mode: ProcessingMode = ProcessingMode.OPENAI

    # Sub-configurations
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    anonymization: AnonymizationConfig = field(default_factory=AnonymizationConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    identity_registry: IdentityRegistryConfig = field(default_factory=IdentityRegistryConfig)

    # Data paths
    bronze_path: str = "./data/bronze"
    silver_path: str = "./data/silver"
    gold_path: str = "./data/gold"

    # Logging
    log_level: str = "INFO"
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables"""
        mode_str = os.getenv("PIPELINE_MODE", "openai").lower()
        mode = ProcessingMode.from_string(mode_str)

        return cls(
            mode=mode,
            openai=OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            ),
            azure_openai=AzureOpenAIConfig(
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                deployment=os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT", "gpt-4o"),
                embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
            ),
            identity_registry=IdentityRegistryConfig(
                enabled=os.getenv("IDENTITY_REGISTRY_ENABLED", "true").lower() == "true",
                registry_path=os.getenv("IDENTITY_REGISTRY_PATH", "./data/identity_registry.json"),
            ),
            bronze_path=os.getenv("BRONZE_PATH", "./data/bronze"),
            silver_path=os.getenv("SILVER_PATH", "./data/silver"),
            gold_path=os.getenv("GOLD_PATH", "./data/gold"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    @classmethod
    def load_from_file(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML or JSON file"""
        import json
        import yaml

        path = Path(path)

        if path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        # Convert mode string to enum
        if "mode" in data:
            data["mode"] = ProcessingMode(data["mode"])

        # Build nested configs
        if "openai" in data:
            data["openai"] = OpenAIConfig(**data["openai"])
        if "pii" in data:
            data["pii"] = PIIConfig(**data["pii"])
        if "anonymization" in data:
            data["anonymization"] = AnonymizationConfig(**data["anonymization"])
        if "summarization" in data:
            data["summarization"] = SummarizationConfig(**data["summarization"])

        return cls(**data)

    def save_to_file(self, path: str) -> None:
        """Save configuration to file"""
        import json
        import yaml

        path = Path(path)

        data = {
            "mode": self.mode.value,
            "openai": {
                "model": self.openai.model,
                "embedding_model": self.openai.embedding_model,
                "temperature": self.openai.temperature,
                "max_tokens": self.openai.max_tokens,
            },
            "pii": {
                "confidence_threshold": self.pii.confidence_threshold,
                "hybrid_confidence_threshold": self.pii.hybrid_confidence_threshold,
                "entity_types": self.pii.entity_types,
                "languages": self.pii.languages,
            },
            "anonymization": {
                "default_strategy": self.anonymization.default_strategy,
                "generate_synthetic": self.anonymization.generate_synthetic,
                "consistent_replacements": self.anonymization.consistent_replacements,
            },
            "summarization": {
                "enabled": self.summarization.enabled,
                "max_summary_length": self.summarization.max_summary_length,
                "summary_style": self.summarization.summary_style,
            },
            "bronze_path": self.bronze_path,
            "silver_path": self.silver_path,
            "gold_path": self.gold_path,
        }

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)


# Global configuration instance
_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = PipelineConfig.from_env()
    return _config


def set_config(config: PipelineConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config


def init_config(
    mode: str = "openai",
    openai_api_key: Optional[str] = None,
    **kwargs
) -> PipelineConfig:
    """Initialize configuration with specified settings"""
    global _config

    _config = PipelineConfig(
        mode=ProcessingMode(mode),
        openai=OpenAIConfig(api_key=openai_api_key),
        **kwargs
    )
    return _config
