"""
Shared utility for safely extracting content from Azure OpenAI / OpenAI API responses.

Azure OpenAI can return responses in two failure modes:

1. finish_reason = "content_filter"  (SafetyGateError / content policy block)
   - The API still returns a choices[0] object but message.content is None.
   - We must check finish_reason BEFORE accessing .content, otherwise we get
     AttributeError: 'NoneType' object has no attribute 'strip'.

2. finish_reason = "length"  (OutputSchemaError / token limit hit)
   - The API returns a truncated response — content exists but may be incomplete JSON.
   - Callers that parse JSON should treat this as a soft error and fall back.

3. No choices at all  (server errors, rate limit, network timeout)
   - Caught by the surrounding try/except in each caller.
   - The SDK raises openai.RateLimitError, openai.APIStatusError, etc.
   - These bubble up and are handled by the broad except block.

Usage:
    from tacitgraph.llm_response import extract_llm_content, LLMContentError

    try:
        response = client.chat.completions.create(...)
        content = extract_llm_content(response, context="PII detection")
    except LLMContentError as e:
        logger.warning(str(e))
        # fall back to default
"""

import logging

logger = logging.getLogger(__name__)


class LLMContentError(Exception):
    """
    Raised when the LLM response cannot produce usable content.

    Attributes:
        reason: The finish_reason returned by the API (e.g. "content_filter", "length").
        context: Caller-supplied label for logging (e.g. "PII detection", "email summary").
    """

    def __init__(self, reason: str, context: str = ""):
        self.reason = reason
        self.context = context
        label = f" [{context}]" if context else ""
        super().__init__(f"LLM response blocked{label}: finish_reason='{reason}'")


def extract_llm_content(response, context: str = "") -> str:
    """
    Safely extract text content from an OpenAI ChatCompletion response.

    Checks finish_reason before accessing message.content so that content_filter
    and length truncations are surfaced as LLMContentError instead of crashing
    with AttributeError or silently returning None / malformed JSON.

    Args:
        response: ChatCompletion object returned by client.chat.completions.create().
        context:  Short label used in log messages (e.g. "PII detection").

    Returns:
        The message content string (stripped of leading/trailing whitespace).

    Raises:
        LLMContentError: If finish_reason is "content_filter" or "length",
                         or if choices is empty / content is None.
    """
    label = f"[{context}] " if context else ""

    # ── Guard: no choices at all (should not happen if SDK didn't raise, but be safe)
    if not response.choices:
        logger.error(f"{label}LLM returned empty choices list — no content available.")
        raise LLMContentError("no_choices", context)

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)

    # ── Case 1: Azure content safety filter blocked the output
    # finish_reason="content_filter" means the model's output was flagged by the
    # content policy. message.content will be None in this case.
    if finish_reason == "content_filter":
        logger.warning(
            f"{label}LLM output was blocked by the content safety filter "
            f"(finish_reason='content_filter'). Input may contain sensitive content."
        )
        raise LLMContentError("content_filter", context)

    # ── Case 2: Response was cut off because max_tokens was reached
    # Content exists but may be incomplete — callers that parse JSON should treat
    # this as unreliable and fall back rather than attempting to parse a partial response.
    if finish_reason == "length":
        logger.warning(
            f"{label}LLM response was truncated (finish_reason='length'). "
            f"The output may be incomplete. Consider increasing max_tokens."
        )
        raise LLMContentError("length", context)

    # ── Normal case: finish_reason="stop" (or None for some Azure versions)
    content = getattr(choice.message, "content", None)

    if content is None:
        # Defensive: finish_reason looked fine but content is still None
        logger.error(
            f"{label}LLM returned None content despite finish_reason='{finish_reason}'. "
            f"This may indicate an API schema change or unexpected error."
        )
        raise LLMContentError(finish_reason or "none_content", context)

    return content.strip()


def extract_llm_content_or_none(response, context: str = "") -> str | None:
    """
    Like extract_llm_content() but returns None instead of raising on error.

    Useful in callers that already have a broad except block and just want
    a safe None-check instead of catching LLMContentError separately.

    Args:
        response: ChatCompletion object.
        context:  Short label for log messages.

    Returns:
        Content string, or None if any error occurred.
    """
    try:
        return extract_llm_content(response, context)
    except LLMContentError:
        return None
