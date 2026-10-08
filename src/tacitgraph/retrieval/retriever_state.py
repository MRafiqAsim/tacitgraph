"""Instance attributes shared by the HybridRetriever layers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tacitgraph.retrieval.hybrid_retriever import HybridConfig
    from tacitgraph.retrieval.react_retriever import ReActRetriever
    from tacitgraph.retrieval.retrieval_tools import RetrievalToolkit


class RetrieverState:
    """Attributes assigned by ``HybridRetriever.__init__`` and used by its layers."""

    gold_path: Path
    silver_path: Path | None
    config: HybridConfig
    mode: str
    toolkit: RetrievalToolkit
    react_retriever: ReActRetriever
    llm_client: Any  # Azure OpenAI client when an LLM is configured, else None
