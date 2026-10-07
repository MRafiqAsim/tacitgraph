"""Shared fixtures: isolate module-level caches between tests."""

import pytest

from tacitgraph import entity_registry, prompt_loader


@pytest.fixture(autouse=True)
def _reset_caches():
    entity_registry._catalog = None
    prompt_loader._CACHE = None
    yield
    entity_registry._catalog = None
    prompt_loader._CACHE = None


@pytest.fixture
def catalog_file(tmp_path):
    """A minimal entity catalog as written by the Gold layer."""
    path = tmp_path / "entity_catalog.json"
    path.write_text(
        """{
          "entities": {
            "Berlin Office": {"standard_name": "Berlin Office", "type": "FACILITY",
                              "aliases": ["BER", "Ber"]},
            "Jane Doe": {"standard_name": "Jane Doe", "type": "PERSON", "aliases": []}
          },
          "metadata": {}
        }""",
        encoding="utf-8",
    )
    return path
