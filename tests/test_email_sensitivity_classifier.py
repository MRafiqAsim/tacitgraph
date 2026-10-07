"""Rule-based personal / not_personal email classification."""

import math

import pytest

from tacitgraph.silver.email_sensitivity_classifier import (
    EmailSensitivityClassifier,
    SensitivityResult,
)


def make_email(subject="", body="", folder="", to=None, attachments=None, sender_email=""):
    return {
        "email_headers": {
            "subject": subject,
            "folder_path": folder,
            "recipients_to": to or [],
            "sender_email": sender_email,
        },
        "email_body_text": body,
        "attachments": attachments or [],
    }


@pytest.fixture
def classifier(tmp_path):
    # An empty rules file means built-in defaults, independent of config/ contents.
    rules = tmp_path / "rules.yaml"
    rules.write_text("overrides: []\n")
    return EmailSensitivityClassifier(rules_path=str(rules))


def test_work_email_is_not_personal(classifier):
    result = classifier.classify(
        make_email(
            subject="Deployment of Project Atlas release 2.1",
            body="The pipeline build failed with a stack trace; check the server logs.",
            folder="Inbox/Engineering",
            to=[{"email": "devops-team@example.com", "name": "DevOps Team"}],
            attachments=[{"filename": "deploy_config.yaml"}],
        )
    )
    assert result.classification == "not_personal"
    assert result.signals["weighted_sum"] > 0
    for signal in ("subject", "content", "folder", "recipients", "attachments"):
        assert result.signals[signal]["score"] > 0


def test_personal_email_is_personal(classifier):
    result = classifier.classify(
        make_email(
            subject="Happy birthday Jane!",
            body="Congratulations and best wishes! Let's have a party.",
            to=[{"email": "hr@example.com", "name": "HR"}],
            attachments=[{"filename": "payslip_march.pdf"}],
        )
    )
    assert result.classification == "personal"
    assert result.signals["weighted_sum"] < 0


def test_empty_email_defaults_to_not_personal(classifier):
    result = classifier.classify(make_email())
    assert result.classification == "not_personal"
    assert result.confidence == 0.5


def test_confidence_matches_formula(classifier):
    result = classifier.classify(make_email(subject="Happy birthday Jane!"))
    weighted = result.signals["weighted_sum"]
    assert result.confidence == pytest.approx(
        round(0.5 + 0.5 * (1 - math.exp(-3 * abs(weighted))), 4)
    )


def test_sender_override_wins(tmp_path):
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        "overrides:\n  - sender_email: Bot@Example.com\n    classification: personal\n"
    )
    classifier = EmailSensitivityClassifier(rules_path=str(rules))

    result = classifier.classify(
        make_email(subject="Deployment pipeline", sender_email="bot@example.com")
    )

    assert result.classification == "personal"
    assert result.confidence == 1.0
    assert result.signals["forced"] == "personal"


def test_custom_patterns_from_rules_file(tmp_path):
    rules = tmp_path / "rules.yaml"
    rules.write_text('sensitive_subject_patterns:\n  - "(?i)\\\\bproject atlas\\\\b"\n')
    classifier = EmailSensitivityClassifier(rules_path=str(rules))

    score, details = classifier._score_subject("Project Atlas update")

    assert score == pytest.approx(-0.4)
    assert details["matched"]


def test_invalid_patterns_are_skipped():
    compiled = EmailSensitivityClassifier._compile_patterns(["(unclosed", r"\bok\b"])
    assert [p.pattern for p in compiled] == [r"\bok\b"]


def test_collect_recipients_normalizes_strings_and_dicts():
    headers = {
        "recipients_to": ["a@example.com"],
        "recipients_cc": [{"email": "b@example.com", "name": "B"}],
    }
    assert EmailSensitivityClassifier._collect_recipients(headers) == [
        {"email": "a@example.com", "name": ""},
        {"email": "b@example.com", "name": "B"},
    ]


@pytest.mark.parametrize(
    ("results", "skip"),
    [
        ([None, SensitivityResult("not_personal", 0.6), SensitivityResult("personal", 0.7)], True),
        ([SensitivityResult("not_personal", 0.9)], False),
        ([None], False),
        ([], False),
    ],
)
def test_should_skip_thread(results, skip):
    assert EmailSensitivityClassifier.should_skip_thread(results) is skip


def test_result_to_dict_rounds_confidence():
    assert SensitivityResult("personal", 0.123456).to_dict() == {
        "classification": "personal",
        "confidence": 0.1235,
        "signals": {},
    }
