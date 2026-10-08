"""Embedded Outlook .msg attachments (forwarded emails) must keep their text."""

import sys
import types

import pytest

from tacitgraph.bronze.document_parser import DocumentParser


class _FakeMessage:
    closed = False

    def __init__(self, path):
        self.subject = "Project Atlas go-live"
        self.sender = "Jane Doe <jane.doe@example.com>"
        self.to = "John Smith <john.smith@example.com>"
        self.date = "Mon, 3 Mar 2025 09:00:00 +0100"
        self.body = "The go-live moved to Friday after the canary checks passed."
        self.htmlBody = None

    def close(self):
        _FakeMessage.closed = True


@pytest.fixture
def fake_extract_msg(monkeypatch):
    monkeypatch.setitem(sys.modules, "extract_msg", types.SimpleNamespace(Message=_FakeMessage))


def test_embedded_msg_text_is_extracted(fake_extract_msg, tmp_path):
    path = tmp_path / "forwarded.msg"
    path.write_bytes(b"not a real msg; the parser is faked")

    doc = DocumentParser()._parse_msg(path)

    assert not doc.parse_errors
    assert doc.title == "Project Atlas go-live"
    assert "Subject: Project Atlas go-live" in doc.text
    assert "From: Jane Doe" in doc.text
    assert "go-live moved to Friday" in doc.text
    assert _FakeMessage.closed
