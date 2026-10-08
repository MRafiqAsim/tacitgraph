"""Multi-signal knowledge vs. transactional attachment classification."""

import json
import math
from types import SimpleNamespace

import pytest

from tacitgraph.silver.attachment_classifier import AttachmentClassifier, ClassificationResult

INVOICE_TEXT = "\n".join(
    [
        "Invoice # 1001",
        "Due Date: 2024-04-01",
        "Total Amount: $1,200.00",
        "Item\tQty\tPrice\tTotal",
        *["Widget\t1\t$10.00\t$10.00"] * 5,
    ]
)

DESIGN_DOC_TEXT = (
    "Table of Contents\n1 Introduction\n2 Architecture\n3 Deployment\n"
    + (
        "This design document describes how the platform shall be deployed. However, the "
        "procedure must follow the security policy and compliance rules. Therefore, every "
        "change goes through review. "
    )
    * 3
)


def attachment(**overrides):
    fields = {
        "text": "",
        "filename": "file.pdf",
        "doc_type": "pdf",
        "has_tables": False,
        "page_count": 0,
        "email_id": "e1",
        "tables": [],
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def classifier(tmp_path):
    return AttachmentClassifier(str(tmp_path))


def test_invoice_spreadsheet_is_transactional(classifier):
    result = classifier.classify(
        attachment(
            text=INVOICE_TEXT, filename="invoice_march.xlsx", doc_type="xlsx", has_tables=True
        )
    )
    assert result.classification == "transactional"
    assert result.signals["weighted_sum"] < 0
    assert set(result.signals["content"]["matched_patterns"]) >= {
        "tab_columns(6_lines)",
        "financial_headers(3)",
    }


def test_design_document_is_knowledge(classifier):
    result = classifier.classify(
        attachment(
            text=DESIGN_DOC_TEXT, filename="architecture_design.docx", doc_type="docx", page_count=6
        )
    )
    assert result.classification == "knowledge"
    assert result.signals["structure"]["multi_page"] is True
    assert result.signals["filename"]["keyword_signal"] == "knowledge"


def test_no_signals_defaults_to_knowledge_with_minimum_confidence(classifier):
    result = classifier.classify(attachment())
    assert result.classification == "knowledge"
    assert result.confidence == 0.5
    assert result.signals["weighted_sum"] == 0.0


@pytest.mark.parametrize("weighted_sum", [-0.5, 0.0, 0.25, 0.59])
def test_confidence_follows_documented_formula(classifier, weighted_sum, monkeypatch):
    # Force every signal to return the same score so the weighted sum is known.
    for name in ("_score_content", "_score_structure"):
        monkeypatch.setattr(classifier, name, lambda *a, s=weighted_sum: (s, {}))
    monkeypatch.setattr(classifier, "_score_email_context", lambda *a: (weighted_sum, {}))
    monkeypatch.setattr(classifier, "_score_filename", lambda *a: (weighted_sum, {}))

    result = classifier.classify(attachment())

    expected = 0.5 + 0.5 * (1 - math.exp(-3 * abs(weighted_sum)))
    assert result.confidence == pytest.approx(round(expected, 4))
    assert result.classification == ("knowledge" if weighted_sum >= 0 else "transactional")


@pytest.mark.parametrize(
    ("filename", "score", "signal"),
    [
        ("data.csv", -0.8, "transactional"),
        ("notes.txt", 0.9, "knowledge"),  # extension + keyword
        ("invoice_report.pdf", -0.3, None),  # transactional keyword wins over knowledge
        ("scan.pdf", 0.0, None),
    ],
)
def test_score_filename(classifier, filename, score, signal):
    value, details = classifier._score_filename(filename)
    assert value == pytest.approx(score)
    if signal:
        assert details["extension_signal"] == signal


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Please find the attached report on Project Atlas.", 0.3),
        ("Attached is the invoice for March.", -0.3),
        ("No mention of files here.", 0.0),
        (None, 0.0),
    ],
)
def test_score_email_context(classifier, body, expected):
    score, _ = classifier._score_email_context(body, "file.pdf")
    assert score == pytest.approx(expected)


def test_scores_are_clamped(classifier):
    many_markers = "=== Sheet: A\n" + "INSERT INTO t VALUES (1);\n" + INVOICE_TEXT + "\nSr. #"
    score, _ = classifier._score_content(many_markers, "xlsx")
    assert -1.0 <= score < 0


def test_email_body_is_loaded_from_bronze(tmp_path):
    emails_dir = tmp_path / "emails"
    emails_dir.mkdir()
    (emails_dir / "e1.json").write_text(
        json.dumps({"record_id": "e1", "email_body_text": "Attached is the user guide."})
    )
    classifier = AttachmentClassifier(str(tmp_path))

    result = classifier.classify(attachment(email_id="e1"))

    assert result.signals["email_context"]["score"] == pytest.approx(0.3)
    assert classifier._load_email_body("missing") is None


def test_classification_result_round_trip():
    result = ClassificationResult("transactional", 0.8, {"weighted_sum": -0.4})
    assert ClassificationResult.from_dict(result.to_dict()) == result
    assert ClassificationResult.from_dict({}) == ClassificationResult("knowledge", 0.5, {})
