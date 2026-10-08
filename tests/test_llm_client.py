"""Provider selection for the query-time chat client."""

import json

import httpx
import pytest

from tacitgraph import llm_client, model_config

PROVIDER_ENV = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "LOCAL_LLM_BASE_URL",
    "LOCAL_LLM_MODEL",
)


@pytest.fixture
def models_config(tmp_path, monkeypatch):
    """Point config/models.json at a temporary file and clear provider env vars."""
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(model_config, "CONFIG_DIR", tmp_path)

    def write(local: dict) -> None:
        (tmp_path / "models.json").write_text(json.dumps({"llm": {"local": local}}))

    return write


@pytest.fixture
def server_up(monkeypatch):
    monkeypatch.setattr(llm_client, "_local_server_reachable", lambda url: True)


@pytest.fixture
def server_down(monkeypatch):
    monkeypatch.setattr(llm_client, "_local_server_reachable", lambda url: False)


def test_no_provider_means_no_client(models_config, server_down):
    models_config({"base_url": "http://localhost:11434/v1"})
    assert llm_client.llm_provider() is None
    assert llm_client.create_chat_client() is None


def test_local_model_comes_from_config(models_config, server_up):
    models_config({"base_url": "http://ollama:11434/v1", "model": "qwen2.5:7b"})
    assert llm_client.llm_provider() == "local"
    assert llm_client.chat_model() == "qwen2.5:7b"
    client = llm_client.create_chat_client()
    assert str(client.base_url).startswith("http://ollama:11434/v1")


def test_without_base_url_there_is_no_local_provider(models_config, server_up):
    models_config({"model": "llama3.1:8b"})
    assert llm_client.llm_provider() is None


def test_env_overrides_config(models_config, server_up, monkeypatch):
    models_config({"base_url": "http://localhost:11434/v1", "model": "a"})
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://host.docker.internal:11434/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "mistral:7b")
    assert llm_client.llm_provider() == "local"
    assert llm_client.chat_model() == "mistral:7b"


def test_cloud_providers_take_precedence(models_config, server_up, monkeypatch):
    models_config({"base_url": "http://localhost:11434/v1"})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm_client.llm_provider() == "openai"
    assert llm_client.chat_model() == llm_client.DEFAULT_CLOUD_MODEL
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-gpt")
    assert llm_client.llm_provider() == "azure"
    assert llm_client.chat_model() == "my-gpt"


def test_unreachable_server_is_detected():
    assert llm_client._local_server_reachable("http://127.0.0.1:9/v1") is False


def test_unreachable_server_message_names_model_and_fix(models_config, server_down):
    models_config({"base_url": "http://localhost:11434/v1", "model": "llama3.1:8b"})
    message = llm_client.chat_model_unavailable_message()
    assert "llama3.1:8b" in message and "http://localhost:11434/v1" in message
    assert "ollama serve" in message


def test_missing_configuration_message(models_config, server_down):
    models_config({})
    assert "No chat model is configured" in llm_client.chat_model_unavailable_message()


def test_no_message_when_a_provider_is_available(models_config, server_up):
    models_config({"base_url": "http://localhost:11434/v1"})
    assert llm_client.chat_model_unavailable_message() is None


def test_only_successful_probes_are_remembered(monkeypatch):
    calls = []

    class Response:
        status_code = 200

    def fake_get(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise httpx.ConnectError("down")
        return Response()

    monkeypatch.setattr(httpx, "get", fake_get)
    url = "http://localhost:11434/v1"
    assert llm_client._local_server_reachable(url) is False
    assert llm_client._local_server_reachable(url) is True  # server came up
    assert llm_client._local_server_reachable(url) is True
    assert len(calls) == 2
