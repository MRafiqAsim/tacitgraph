"""
Central prompt loader — reads all LLM prompts from config/prompts.json.

The JSON is organized by pipeline layer → prompt section → key:

    {
      "domain_context": "You are analysing ...",
      "silver": {
        "pii_detection": {
          "system_prompt": "{domain_context}\\n\\nYou are a PII detection expert ...",
          "user_prompt": "...",
          "temperature": 0.0
        }
      }
    }

``{domain_context}`` in any prompt string is replaced with the top-level
``domain_context`` value, or with the ``PROMPT_DOMAIN_CONTEXT`` environment
variable when set. This lets each deployment describe its own corpus once
instead of editing every prompt.

Usage:
    from prompt_loader import get_prompt

    system = get_prompt("silver", "pii_detection", "system_prompt")
    temp   = get_prompt("silver", "pii_detection", "temperature")
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DOMAIN_CONTEXT_PLACEHOLDER = "{domain_context}"
DOMAIN_CONTEXT_ENV = "PROMPT_DOMAIN_CONTEXT"

_CACHE: Optional[dict] = None


def _load() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    prompts_path = Path(__file__).resolve().parent.parent / "config" / "prompts.json"
    if prompts_path.exists():
        with open(prompts_path, "r", encoding="utf-8") as f:
            _CACHE = json.load(f)
            logger.info(f"Loaded prompts from {prompts_path}")
    else:
        logger.warning(f"Prompts config not found at {prompts_path}, using empty dict")
        _CACHE = {}

    return _CACHE


def get_domain_context() -> str:
    """Return the corpus description injected into system prompts."""
    return os.getenv(DOMAIN_CONTEXT_ENV) or _load().get("domain_context", "")


def _resolve(value: Any) -> Any:
    if isinstance(value, str) and DOMAIN_CONTEXT_PLACEHOLDER in value:
        return value.replace(DOMAIN_CONTEXT_PLACEHOLDER, get_domain_context()).strip()
    return value


def get_prompt(layer: str, section: str, key: str, default: Any = None) -> Any:
    """
    Get a prompt value from config/prompts.json.

    Args:
        layer: Pipeline layer ("bronze", "silver", "gold", "retrieval")
        section: Prompt group (e.g. "pii_detection", "generation")
        key: Key within the group (e.g. "system_prompt", "temperature")
        default: Fallback if not found

    Returns:
        The prompt string (with ``{domain_context}`` resolved), number, or list
    """
    data = _load()
    return _resolve(data.get(layer, {}).get(section, {}).get(key, default))


def format_prompt(template: str, **kwargs) -> str:
    """
    Safely format a prompt template, preserving literal braces.

    Replaces only {key} placeholders that match kwargs keys.
    All other braces (including JSON examples) are left untouched.
    """
    # First, escape ALL braces
    safe = template.replace("{", "{{").replace("}", "}}")

    # Then un-escape only the known kwargs placeholders
    for key in kwargs:
        safe = safe.replace("{{" + key + "}}", "{" + key + "}")

    return safe.format(**kwargs)


def get_section(layer: str, section: str) -> dict:
    """Get an entire prompt section."""
    data = _load()
    return {k: _resolve(v) for k, v in data.get(layer, {}).get(section, {}).items()}
