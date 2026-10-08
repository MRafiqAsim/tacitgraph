"""Every module must import with only the core dependencies installed.

Heavy optional dependencies (spaCy, torch, Azure SDKs, RAGAS) are imported lazily,
so a plain `uv sync` is enough to load the whole package.
"""

import importlib
import pkgutil

import pytest

import tacitgraph

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(tacitgraph.__path__, f"{tacitgraph.__name__}.")
)


def test_package_exposes_version():
    assert tacitgraph.__version__


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
