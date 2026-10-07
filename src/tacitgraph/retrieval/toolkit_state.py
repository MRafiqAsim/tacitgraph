"""Instance attributes shared by the RetrievalToolkit layers."""

from pathlib import Path


class ToolkitState:
    """Attributes assigned by ``RetrievalToolkit.__init__`` and used by its layers."""

    gold_path: Path
    silver_path: Path | None
    mode: str
