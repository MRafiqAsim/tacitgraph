"""The local embedding model is recorded with the index and reused at query time."""

import json

import numpy as np

from tacitgraph.gold.embedding_generator import EmbeddingConfig, EmbeddingGenerator


def generator(tmp_path, local_model="BAAI/bge-m3"):
    gen = object.__new__(EmbeddingGenerator)
    gen.config = EmbeddingConfig(local_model=local_model)
    gen.mode = "local"
    gen.embeddings_path = tmp_path
    return gen


def test_save_records_the_local_model(tmp_path):
    generator(tmp_path).save_embeddings(["a", "b"], np.zeros((2, 4)), "chunks")
    saved = json.loads((tmp_path / "chunks_config.json").read_text())
    assert saved["local_model"] == "BAAI/bge-m3"
    assert saved["model"] == "BAAI/bge-m3"
    assert saved["dimensions"] == 4


def test_indexed_model_is_read_back(tmp_path):
    generator(tmp_path).save_embeddings(["a"], np.zeros((1, 4)), "chunks")
    assert generator(tmp_path, "all-MiniLM-L6-v2")._indexed_local_model() == "BAAI/bge-m3"


def test_index_without_model_record_returns_none(tmp_path):
    (tmp_path / "chunks_config.json").write_text('{"model": "text-embedding-3-small"}')
    assert generator(tmp_path)._indexed_local_model() is None


def test_configured_model_comes_from_env(monkeypatch):
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    assert EmbeddingConfig().local_model == "intfloat/multilingual-e5-small"
