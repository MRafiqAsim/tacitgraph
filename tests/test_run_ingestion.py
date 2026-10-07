"""Argument validation of the ingestion CLI."""

import sys

import pytest

from tacitgraph.pipeline import run_ingestion


def test_cli_without_inputs_reports_usage_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tacitgraph-ingest"])
    with pytest.raises(SystemExit) as exc:
        run_ingestion.main()
    assert exc.value.code == 2  # argparse usage error, not a crash
