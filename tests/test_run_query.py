"""The query CLI must honour --mode (regression: it used to always run in llm mode)."""

import sys

import pytest

from tacitgraph.pipeline import run_query


class _FakeRetriever:
    instances: list["_FakeRetriever"] = []

    def __init__(self, gold_path, silver_path=None, mode="llm"):
        self.mode = mode
        _FakeRetriever.instances.append(self)


@pytest.mark.parametrize("mode", ["local", "llm", "hybrid"])
def test_cli_passes_mode_to_retriever(monkeypatch, tmp_path, mode):
    gold = tmp_path / f"gold_{mode}"
    gold.mkdir()
    _FakeRetriever.instances.clear()
    monkeypatch.setattr(run_query, "HybridRetriever", _FakeRetriever)
    monkeypatch.setattr(run_query, "run_interactive", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["tacitgraph-query", "--mode", mode, "--gold", str(gold)])

    run_query.main()

    assert [r.mode for r in _FakeRetriever.instances] == [mode]
