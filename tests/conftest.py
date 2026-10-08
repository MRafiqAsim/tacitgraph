"""Shared fixtures: isolate module-level caches between tests."""

import pytest

from tacitgraph import entity_registry, llm_client, model_config, prompt_loader


@pytest.fixture(autouse=True)
def _reset_caches():
    def clear():
        entity_registry._catalog = None
        prompt_loader._CACHE = None
        model_config.load_models_config.cache_clear()
        llm_client._local_config.cache_clear()
        llm_client.forget_local_server()

    clear()
    yield
    clear()


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
