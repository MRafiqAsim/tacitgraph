"""Chat-completion client for query time: Azure OpenAI, OpenAI or a local server.

The first configured provider wins:

1. Azure OpenAI  — ``AZURE_OPENAI_ENDPOINT`` + ``AZURE_OPENAI_API_KEY``, model ``AZURE_OPENAI_DEPLOYMENT``
2. OpenAI        — ``OPENAI_API_KEY``, model ``OPENAI_MODEL``
3. Local server  — ``llm.local`` in ``config/models.json``: any OpenAI-compatible endpoint such as
   Ollama, llama.cpp or vLLM. ``LOCAL_LLM_BASE_URL`` / ``LOCAL_LLM_MODEL`` override it.

Chat needs one of them: when none is reachable, callers get None and show
:func:`chat_model_unavailable_message` to the user.
"""

import logging
import os
from functools import cache
from typing import Any

from tacitgraph.model_config import load_models_config

logger = logging.getLogger(__name__)

DEFAULT_CLOUD_MODEL = "gpt-4o"
DEFAULT_LOCAL_MODEL = "llama3.1:8b"
TIMEOUT_SECONDS = 120.0
LOCAL_TIMEOUT_SECONDS = 300.0  # CPU inference on long contexts is slow


@cache
def _local_config() -> dict[str, Any]:
    """The ``llm.local`` section of config/models.json, with environment overrides applied."""
    config = dict(load_models_config().get("llm", {}).get("local", {}))
    if os.getenv("LOCAL_LLM_BASE_URL"):
        config["base_url"] = os.environ["LOCAL_LLM_BASE_URL"]
    if os.getenv("LOCAL_LLM_MODEL"):
        config["model"] = os.environ["LOCAL_LLM_MODEL"]
    return config


_REACHABLE_SERVERS: set[str] = set()


def _local_server_reachable(base_url: str) -> bool:
    """True when the OpenAI-compatible server answers ``GET /models``.

    Only successes are remembered, so a server started later is picked up.
    """
    if base_url in _REACHABLE_SERVERS:
        return True
    import httpx

    try:
        ok = httpx.get(f"{base_url.rstrip('/')}/models", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        ok = False
    if ok:
        _REACHABLE_SERVERS.add(base_url)
    return ok


def forget_local_server() -> None:
    """Re-probe the local server on next use, e.g. after a request to it failed."""
    _REACHABLE_SERVERS.clear()


def chat_model_unavailable_message() -> str | None:
    """User-facing explanation when no chat model can be reached, else None."""
    if llm_provider() is not None:
        return None
    local = _local_config()
    base_url = local.get("base_url")
    if not base_url:
        return (
            "⚠️ **No chat model is configured.**\n\n"
            "Set `llm.local.base_url` in `config/models.json` to a local server such as "
            "Ollama, or add Azure OpenAI / OpenAI credentials to `.env`."
        )
    model = local.get("model") or DEFAULT_LOCAL_MODEL
    return (
        "⚠️ **The local language model is not reachable.**\n\n"
        f"TacitGraph is configured to answer with `{model}` at `{base_url}`, "
        "but the server did not respond. Start it with `ollama serve` "
        f"(the first time also run `ollama pull {model}`), then ask again — "
        "no restart needed."
    )


def llm_provider() -> str | None:
    """Name of the configured provider: ``azure``, ``openai``, ``local`` or None."""
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        return "azure"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    base_url = _local_config().get("base_url")
    if base_url and _local_server_reachable(base_url):
        return "local"
    return None


def chat_model() -> str:
    """Model or deployment name for the configured provider."""
    provider = llm_provider()
    if provider == "local":
        return _local_config().get("model") or DEFAULT_LOCAL_MODEL
    if provider == "openai":
        return os.getenv("OPENAI_MODEL") or DEFAULT_CLOUD_MODEL
    return os.getenv("AZURE_OPENAI_DEPLOYMENT") or DEFAULT_CLOUD_MODEL


def create_chat_client() -> Any | None:
    """Return an OpenAI-SDK client for the configured provider, or None if none is available."""
    provider = llm_provider()
    if provider is None:
        return None
    try:
        from openai import AzureOpenAI, OpenAI

        if provider == "azure":
            return AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                timeout=TIMEOUT_SECONDS,
                max_retries=2,
            )
        if provider == "openai":
            return OpenAI(
                api_key=os.environ["OPENAI_API_KEY"], timeout=TIMEOUT_SECONDS, max_retries=2
            )
        local = _local_config()
        logger.info(f"Using local LLM {chat_model()} at {local['base_url']}")
        return OpenAI(
            base_url=local["base_url"],
            api_key=os.getenv("LOCAL_LLM_API_KEY") or "not-needed",
            timeout=LOCAL_TIMEOUT_SECONDS,
            max_retries=1,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize {provider} LLM client: {e}")
        return None
