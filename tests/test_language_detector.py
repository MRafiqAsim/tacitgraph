"""English/Dutch language detection."""

import pytest
from langdetect import DetectorFactory

from tacitgraph.silver.language_detector import (
    LanguageDetector,
    detect_language,
    get_nlp_model_for_text,
)

ENGLISH = (
    "Thank you for sending the project plan. We will review it with the team next week "
    "and share our feedback."
)
DUTCH = (
    "Bedankt voor het sturen van het projectplan. We zullen het volgende week met het team "
    "bekijken en onze feedback delen."
)


@pytest.fixture(autouse=True)
def deterministic_langdetect():
    DetectorFactory.seed = 0


@pytest.fixture
def detector():
    return LanguageDetector()


@pytest.mark.parametrize(("text", "language"), [(ENGLISH, "en"), (DUTCH, "nl")])
def test_detects_english_and_dutch(detector, text, language):
    result = detector.detect(text)
    assert result.language == language
    assert result.is_reliable
    assert result.confidence >= detector.confidence_threshold


@pytest.mark.parametrize("text", ["", "   ", "hi there"])
def test_short_text_falls_back_to_default(detector, text):
    result = detector.detect(text)
    assert result.language == "en"
    assert result.confidence == 0.0
    assert not result.is_reliable


def test_custom_default_language():
    assert LanguageDetector(default_language="nl").detect("ok").language == "nl"


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Beste Jan, bedankt voor de informatie. Met vriendelijke groeten", "nl"),
        ("Thanks for the update on the release, best regards", "en"),
    ],
)
def test_pattern_fallback(detector, text, language):
    result = detector._detect_with_patterns(text)
    assert result.language == language
    assert result.confidence <= 0.8  # pattern matches are discounted


def test_pattern_fallback_without_indicators(detector):
    result = detector._detect_with_patterns("xyzzy plugh qwerty")
    assert result.language == "en"
    assert result.confidence == 0.0


@pytest.mark.parametrize(
    ("code", "expected"),
    [("nl", "nl"), ("en", "en"), ("nld", "nl"), ("Dutch", "nl"), ("eng", "en"), ("de", "en")],
)
def test_map_language(detector, code, expected):
    assert detector._map_language(code) == expected


def test_spacy_model_selection(detector):
    assert detector.get_spacy_model("nl") == "nl_core_news_lg"
    assert detector.get_spacy_model("en") == "en_core_web_trf"
    assert detector.get_spacy_model("fr") == "en_core_web_trf"


def test_batch_detection(detector):
    assert [r.language for r in detector.detect_batch([ENGLISH, DUTCH])] == ["en", "nl"]


def test_convenience_functions():
    assert detect_language(DUTCH) == "nl"
    assert get_nlp_model_for_text(ENGLISH) == "en_core_web_trf"
