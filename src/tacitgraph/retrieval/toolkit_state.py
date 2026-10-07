"""Instance attributes shared by the RetrievalToolkit layers."""

from pathlib import Path

from tacitgraph.retrieval.lexical_index import BM25Index


class ToolkitState:
    """Attributes assigned by ``RetrievalToolkit.__init__`` and used by its layers."""

    gold_path: Path
    silver_path: Path | None
    mode: str
    _lexical_index: BM25Index | None
