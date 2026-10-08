"""Answer generation when the chat model is missing, unreachable or failing."""

from tacitgraph.retrieval import answer_generation as ag
from tacitgraph.retrieval.answer_generation import AnswerGeneration, _drop_repeated_paragraphs

CHUNKS = [
    {
        "text_english": "Jane Doe will lead the ERP migration from April.",
        "thread_subject": "ERP migration kickoff",
        "email_sender": "John Smith",
        "sent_timestamp": "2024-03-01T09:00:00",
    }
]


class _FailingCompletions:
    def create(self, **kwargs):
        raise ConnectionError("server went away")


class _FailingClient:
    class chat:
        completions = _FailingCompletions()


def _generator(client):
    generator = object.__new__(AnswerGeneration)
    generator.llm_client = client
    generator.config = type("Config", (), {"answer_model": "llama3.1:8b"})()
    return generator


def test_missing_chat_model_alerts_the_user(monkeypatch):
    notice = "⚠️ **The local language model is not reachable.**"
    monkeypatch.setattr(ag, "create_chat_client", lambda: None)
    monkeypatch.setattr(ag, "chat_model_unavailable_message", lambda: notice)
    text, grounded, missing, tokens = _generator(None)._generate_answer("Who leads it?", CHUNKS)
    assert text == notice
    assert not grounded and missing and tokens == 0


def test_reconnects_when_the_server_comes_back(monkeypatch):
    monkeypatch.setattr(ag, "create_chat_client", lambda: "client")
    monkeypatch.setattr(ag, "chat_model", lambda: "qwen2.5:7b")
    generator = _generator(None)
    assert generator._llm_unavailable_notice() is None
    assert generator.llm_client == "client"
    assert generator.config.answer_model == "qwen2.5:7b"


def test_failed_request_to_vanished_local_server_alerts_the_user(monkeypatch):
    monkeypatch.setattr(ag, "llm_provider", lambda: "local")
    monkeypatch.setattr(ag, "chat_model_unavailable_message", lambda: "server down")
    generator = _generator(_FailingClient())
    text, grounded, _, tokens = generator._generate_answer("Who leads it?", CHUNKS)
    assert text == "server down" and not grounded and tokens == 0
    assert generator.llm_client is None  # reconnects on the next question


def test_failed_cloud_request_reports_the_error(monkeypatch):
    monkeypatch.setattr(ag, "llm_provider", lambda: "azure")
    text, grounded, _, _ = _generator(_FailingClient())._generate_answer("Who?", CHUNKS)
    assert "request failed" in text and "ConnectionError" in text
    assert not grounded


def test_repeated_paragraphs_are_removed():
    assert _drop_repeated_paragraphs("A.\n\nB.\n\nA.") == "A.\n\nB."
