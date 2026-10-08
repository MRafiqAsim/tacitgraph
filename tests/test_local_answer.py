"""Extractive answers used when no LLM is available."""

from tacitgraph.retrieval import answer_generation as ag
from tacitgraph.retrieval.answer_generation import AnswerGeneration, _drop_repeated_paragraphs

CHUNKS = [
    {
        "text_english": (
            "Thanks for the update. Jane Doe will lead the ERP migration from April. "
            "Lunch is at noon in the Berlin office."
        ),
        "thread_subject": "ERP migration kickoff",
        "email_sender": "John Smith",
        "sent_timestamp": "2024-03-01T09:00:00",
    },
    {
        "text_english": "The cutover weekend for the ERP migration is fixed for 14 June.",
        "thread_subject": "Cutover plan",
        "email_sender": "Jane Doe",
        "sent_timestamp": "2024-05-20T16:30:00",
    },
]


def answer(query, chunks=CHUNKS):
    return AnswerGeneration._generate_local_answer(object.__new__(AnswerGeneration), query, chunks)


def test_quotes_only_sentences_that_match_the_query():
    text, grounded, missing = answer("Who leads the ERP migration?")
    assert grounded and missing is None
    assert "> Jane Doe will lead the ERP migration from April." in text
    assert "Lunch is at noon" not in text


def test_each_quote_cites_its_email():
    text, _, _ = answer("When is the ERP cutover?")
    assert "— [2] Cutover plan · Jane Doe · 2024-05-20" in text


def test_no_matching_sentence_is_reported_as_not_grounded():
    text, grounded, missing = answer("What is the budget for marketing?")
    assert not grounded
    assert missing
    assert text.startswith("No retrieved passage mentions")


def test_repeated_paragraphs_are_removed():
    assert _drop_repeated_paragraphs("A.\n\nB.\n\nA.") == "A.\n\nB."


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


def test_unreachable_local_server_alerts_the_user(monkeypatch):
    notice = "⚠️ **The local language model is not reachable.**"
    monkeypatch.setattr(ag, "create_chat_client", lambda: None)
    monkeypatch.setattr(ag, "local_llm_unavailable_message", lambda: notice)
    text, grounded, missing, _ = _generator(None)._generate_answer("Who leads it?", CHUNKS)
    assert text == notice
    assert not grounded and missing


def test_no_configured_llm_quotes_sources(monkeypatch):
    monkeypatch.setattr(ag, "create_chat_client", lambda: None)
    monkeypatch.setattr(ag, "local_llm_unavailable_message", lambda: None)
    text, grounded, _, _ = _generator(None)._generate_answer("Who leads the ERP migration?", CHUNKS)
    assert grounded
    assert "> Jane Doe will lead the ERP migration from April." in text


def test_failed_request_to_vanished_local_server_alerts_the_user(monkeypatch):
    monkeypatch.setattr(ag, "llm_provider", lambda: None)
    monkeypatch.setattr(ag, "local_llm_unavailable_message", lambda: "server down")
    generator = _generator(_FailingClient())
    text, grounded, _, tokens = generator._generate_answer("Who leads it?", CHUNKS)
    assert text == "server down" and not grounded and tokens == 0
    assert generator.llm_client is None  # reconnects on the next question


def test_failed_cloud_request_reports_the_error(monkeypatch):
    monkeypatch.setattr(ag, "llm_provider", lambda: "azure")
    monkeypatch.setattr(ag, "local_llm_unavailable_message", lambda: None)
    text, grounded, _, _ = _generator(_FailingClient())._generate_answer("Who?", CHUNKS)
    assert "request failed" in text and "ConnectionError" in text
    assert not grounded
