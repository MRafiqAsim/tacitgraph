"""Pipeline configuration defaults, environment loading and file round-trips."""

import pytest

from tacitgraph import settings
from tacitgraph.settings import (
    AzureOpenAIConfig,
    OpenAIConfig,
    PipelineConfig,
    ProcessingMode,
)

ENV_VARS = [
    "PIPELINE_MODE",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_GPT4O_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "IDENTITY_REGISTRY_ENABLED",
    "IDENTITY_REGISTRY_PATH",
    "BRONZE_PATH",
    "SILVER_PATH",
    "GOLD_PATH",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(settings, "_config", None)


@pytest.mark.parametrize(
    ("value", "mode"),
    [
        ("llm", ProcessingMode.OPENAI),
        ("OpenAI", ProcessingMode.OPENAI),
        (" local ", ProcessingMode.LOCAL),
        ("hybrid", ProcessingMode.HYBRID),
    ],
)
def test_processing_mode_from_string(value, mode):
    assert ProcessingMode.from_string(value) is mode


def test_processing_mode_rejects_unknown_values():
    with pytest.raises(ValueError):
        ProcessingMode.from_string("cloud")


def test_defaults():
    config = PipelineConfig()
    assert config.mode is ProcessingMode.OPENAI
    assert config.openai.temperature == 0.0
    assert config.pii.languages == ["en", "nl"]
    assert "PERSON" in config.pii.entity_types
    assert config.identity_registry.enabled


def test_api_keys_fall_back_to_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    assert OpenAIConfig().api_key == "sk-test"
    assert AzureOpenAIConfig().endpoint == "https://example.openai.azure.com/"


def test_from_env(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "llm")
    monkeypatch.setenv("SILVER_PATH", "/tmp/silver")
    monkeypatch.setenv("IDENTITY_REGISTRY_ENABLED", "false")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embeddings")
    config = PipelineConfig.from_env()
    assert config.mode is ProcessingMode.OPENAI
    assert config.silver_path == "/tmp/silver"
    assert not config.identity_registry.enabled
    assert config.azure_openai.embedding_deployment == "embeddings"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_save_and_load_round_trip(tmp_path, suffix):
    original = PipelineConfig(mode=ProcessingMode.HYBRID, gold_path="/data/gold")
    original.pii.confidence_threshold = 0.65
    path = tmp_path / "nested" / f"config{suffix}"

    original.save_to_file(str(path))
    loaded = PipelineConfig.load_from_file(str(path))

    assert loaded.mode is ProcessingMode.HYBRID
    assert loaded.gold_path == "/data/gold"
    assert loaded.pii.confidence_threshold == 0.65
    assert loaded.openai.model == original.openai.model


def test_load_rejects_unknown_format(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("")
    with pytest.raises(ValueError, match="Unsupported"):
        PipelineConfig.load_from_file(str(path))


def test_get_config_is_cached_and_replaceable():
    first = settings.get_config()
    assert settings.get_config() is first

    replacement = PipelineConfig(mode=ProcessingMode.LOCAL)
    settings.set_config(replacement)
    assert settings.get_config() is replacement


def test_init_config_sets_global():
    config = settings.init_config(mode="local", openai_api_key="sk-test", gold_path="/g")
    assert settings.get_config() is config
    assert config.mode is ProcessingMode.LOCAL
    assert config.openai.api_key == "sk-test"
    assert config.gold_path == "/g"


@pytest.mark.parametrize(
    ("alias", "expected"), [("llm", "openai"), ("LLM", "openai"), ("local", "local")]
)
def test_init_config_accepts_mode_aliases(alias, expected):
    config = settings.init_config(mode=alias, openai_api_key="sk-test")
    assert config.mode.value == expected


def test_load_from_file_accepts_mode_alias(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"mode": "llm"}', encoding="utf-8")
    assert PipelineConfig.load_from_file(str(path)).mode.value == "openai"
