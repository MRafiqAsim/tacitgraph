"""Model choices from config/models.json (LLM server and embedding model).

Changing a model is a one-line edit of that file; environment variables still
take precedence so containers can override it without rebuilding.
"""

import json
from functools import cache
from typing import Any

from tacitgraph.paths import CONFIG_DIR

MODELS_CONFIG_FILE = "models.json"


@cache
def load_models_config() -> dict[str, Any]:
    """Parsed config/models.json, or an empty dict when the file is absent."""
    path = CONFIG_DIR / MODELS_CONFIG_FILE
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
