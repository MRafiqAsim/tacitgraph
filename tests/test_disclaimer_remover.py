"""Legal boilerplate removal from email bodies."""

import pytest

from tacitgraph.silver.disclaimer_remover import DisclaimerRemover, remove_disclaimers


@pytest.mark.parametrize(
    "disclaimer",
    [
        "CONFIDENTIALITY NOTICE: This message may contain privileged information.",
        "DISCLAIMER: The views expressed are those of the sender.",
        "If you are not the intended recipient, please notify the sender and delete it.",
        "Please consider the environment before printing this email.",
        "This message has been scanned for viruses and checked for malware.",
    ],
)
def test_known_disclaimers_are_removed(disclaimer):
    text = f"Meeting moved to 3pm.\n\n{disclaimer}\n\nRegards, Jane"
    cleaned, count = DisclaimerRemover().remove(text)
    assert count >= 1
    assert "Meeting moved to 3pm." in cleaned
    assert disclaimer not in cleaned


def test_text_without_disclaimer_is_untouched():
    text = "Project Atlas ships on Friday.\n\nRegards, Jane"
    assert DisclaimerRemover().remove(text) == (text, 0)


def test_empty_text():
    assert DisclaimerRemover().remove("") == ("", 0)


def test_custom_patterns_are_applied():
    remover = DisclaimerRemover(custom_patterns=[r"Sent from my phone\.?"])
    cleaned, count = remover.remove("On my way.\n\nSent from my phone.")
    assert cleaned == "On my way."
    assert count == 1


def test_remove_from_email_updates_body_and_count():
    email = {"body_text": "See notes.\n\nLEGAL NOTICE: internal use only.\n\nJane"}
    result = DisclaimerRemover().remove_from_email(email)
    assert result["body_text"] == "See notes.\n\nJane"
    assert result["disclaimers_removed"] == 1


def test_remove_from_email_without_body_is_noop():
    assert DisclaimerRemover().remove_from_email({"subject": "x"}) == {"subject": "x"}


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: the broad 'This email ... confidential ... delete' "
    "pattern also matches ordinary sentences, removing real content.",
)
def test_ordinary_sentence_mentioning_confidential_is_kept():
    paragraph = (
        "This email is about the confidential Project Atlas budget. Please delete the old draft."
    )
    assert paragraph in remove_disclaimers(f"Hi team,\n\n{paragraph}\n\nThanks, Jane")
