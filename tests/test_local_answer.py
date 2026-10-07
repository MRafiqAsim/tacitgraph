"""Extractive answers used when no LLM is available."""

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
