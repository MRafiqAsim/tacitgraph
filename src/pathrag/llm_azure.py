"""
Lightweight LLM module for PathRAG with Azure OpenAI support.

This replaces the heavy llm.py that requires modelscope, vllm, etc.
Matches the calling convention used by PathRAG.__post_init__ which wraps
the LLM function with ``partial(func, hashing_kv=..., **llm_model_kwargs)``.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional

import numpy as np
from openai import AsyncAzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .utils import EmbeddingFunc

import logging
logger = logging.getLogger(__name__)


def get_azure_client():
    """Get Azure OpenAI client."""
    return AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
)
async def azure_openai_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = None,
    keyword_extraction: bool = False,
    max_tokens: int = 2000,
    temperature: float = 0.3,
    stream: bool = False,
    **kwargs,
) -> str:
    """
    Complete using Azure OpenAI.

    Compatible with PathRAG's expected LLM function signature.
    PathRAG.__post_init__ wraps this with partial(func, hashing_kv=..., **kwargs),
    so we pop PathRAG-internal keys that the Azure API doesn't understand.
    """
    # --- Pop PathRAG-internal kwargs that must not reach the Azure API ---
    kwargs.pop("hashing_kv", None)

    kw_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    response_format = kwargs.pop("response_format", None)

    if kw_extraction:
        response_format = {"type": "json_object"}

    client = get_azure_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    api_kwargs: Dict[str, Any] = {
        "model": deployment,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if response_format:
        api_kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**api_kwargs)

    if stream:
        async def stream_generator():
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return stream_generator()

    # extract_llm_content raises LLMContentError for content_filter / length.
    # PathRAG callers catch exceptions at the operate.py level.
    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from llm_response import extract_llm_content, LLMContentError
    return extract_llm_content(response, context="PathRAG LLM")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
)
async def azure_openai_embedding(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings using Azure OpenAI.

    Returns np.ndarray of shape (len(texts), embedding_dim) — this is the
    format expected by NanoVectorDBStorage and other PathRAG storage backends.
    """
    client = get_azure_client()
    deployment = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
    )

    batch_size = 100
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(model=deployment, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


def create_azure_embedding_func(
    embedding_dim: int = 1536,
    max_token_size: int = 8191,
    concurrent_limit: int = 16,
) -> EmbeddingFunc:
    """
    Create an EmbeddingFunc wrapper around azure_openai_embedding.

    NanoVectorDBStorage accesses ``embedding_func.embedding_dim`` (storage.py:76),
    so a plain async function won't work — it must be wrapped in EmbeddingFunc.
    """
    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=azure_openai_embedding,
        concurrent_limit=concurrent_limit,
    )


# Compatibility aliases
openai_complete = azure_openai_complete
openai_embedding = azure_openai_embedding
