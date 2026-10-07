"""Safe content extraction from (Azure) OpenAI chat completion responses."""

from types import SimpleNamespace

import pytest

from tacitgraph.llm_response import (
    LLMContentError,
    extract_llm_content,
    extract_llm_content_or_none,
)


def make_response(content="  hello  ", finish_reason="stop", choices=True):
    """Build an object shaped like openai.types.chat.ChatCompletion."""
    if not choices:
        return SimpleNamespace(choices=[])
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


def test_returns_stripped_content():
    assert extract_llm_content(make_response()) == "hello"


def test_missing_finish_reason_is_treated_as_normal():
    assert extract_llm_content(make_response(content="ok", finish_reason=None)) == "ok"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (make_response(content=None, finish_reason="content_filter"), "content_filter"),
        (make_response(content='{"partial": ', finish_reason="length"), "length"),
        (make_response(choices=False), "no_choices"),
        (make_response(content=None, finish_reason="stop"), "stop"),
        (make_response(content=None, finish_reason=None), "none_content"),
    ],
)
def test_unusable_responses_raise(response, reason):
    with pytest.raises(LLMContentError) as exc_info:
        extract_llm_content(response, context="summary")
    assert exc_info.value.reason == reason
    assert exc_info.value.context == "summary"


def test_error_message_includes_context_and_reason():
    error = LLMContentError("content_filter", "PII detection")
    assert str(error) == "LLM response blocked [PII detection]: finish_reason='content_filter'"
    assert str(LLMContentError("length")) == "LLM response blocked: finish_reason='length'"


def test_or_none_variant_swallows_content_errors():
    assert extract_llm_content_or_none(make_response(finish_reason="content_filter")) is None
    assert extract_llm_content_or_none(make_response(content=" ok ")) == "ok"
