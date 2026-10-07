"""Filesystem locations shared across the package.

Both locations can be overridden with environment variables, which is how the
Docker image and the Synapse notebooks point the package at their own layout.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("TACITGRAPH_HOME", Path(__file__).resolve().parents[2]))
CONFIG_DIR = Path(os.getenv("TACITGRAPH_CONFIG_DIR", PROJECT_ROOT / "config"))
LOG_DIR = PROJECT_ROOT / "logs"
