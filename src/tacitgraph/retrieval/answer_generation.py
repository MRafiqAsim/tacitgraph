"""Answer generation with the configured chat model, grounding checks and confidence scores."""

import logging
from typing import Any

from tacitgraph.llm_client import (
    chat_model,
    chat_model_unavailable_message,
    create_chat_client,
    forget_local_server,
    llm_provider,
)
from tacitgraph.prompt_loader import format_prompt, get_prompt
from tacitgraph.retrieval.thread_expansion import ThreadExpansion

logger = logging.getLogger(__name__)


def _drop_repeated_paragraphs(text: str) -> str:
    """Remove verbatim repeats; small local models sometimes emit the answer twice."""
    seen: set[str] = set()
    kept = []
    for paragraph in text.split("\n\n"):
        key = paragraph.strip()
        if key and key in seen:
            continue
        seen.add(key)
        kept.append(paragraph)
    return "\n\n".join(kept)


class AnswerGeneration(ThreadExpansion):
    """Answer generation with the configured chat model, grounding checks and confidence scores."""

    def _initialize_llm(self):
        """Initialize the LLM client for answer generation (None when no provider is set)."""
        self.llm_client = create_chat_client()

    def _llm_unavailable_notice(self) -> str | None:
        """Connect if a chat server has become available; otherwise explain why there is none.

        Returns None when an LLM client is ready.
        """
        if self.llm_client is None:
            self.llm_client = create_chat_client()
            if self.llm_client is not None:
                self.config.answer_model = chat_model()
                logger.info(f"Connected to chat model {self.config.answer_model}")
        if self.llm_client is not None:
            return None
        return chat_model_unavailable_message()

    def _generate_answer(
        self, query: str, chunks: list[dict[str, Any]], extra_context: str = ""
    ) -> tuple[str, bool, str | None, int]:
        """
        Generate answer using LLM with grounding check.

        Returns:
            (answer, is_grounded, missing_info, total_tokens)
        """
        if not chunks and not extra_context:
            return "", True, None, 0

        notice = self._llm_unavailable_notice()
        if notice:
            return notice, False, "Chat model not available", 0

        # Build context from chunks — label email vs attachment for LLM clarity
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get(
                "text",
                chunk.get("summary")
                or chunk.get("text_english")
                or chunk.get("text_anonymized", ""),
            )
            chunk_id = chunk.get("chunk_id", "unknown")
            thread = chunk.get("thread_subject", "")
            source_type = chunk.get("source_type", "email")
            # Include email date so LLM can reference it in answers
            recv = chunk.get("received_timestamp", "")
            sent = chunk.get("sent_timestamp", "")
            if recv:
                date_str = f" | Received: {recv[:10]}"
            elif sent:
                date_str = f" | Sent: {sent[:10]}"
            else:
                date_str = ""

            sender = chunk.get("email_sender", "")
            sender_str = f" | From: {sender}" if sender else ""

            if source_type == "attachment":
                filename = chunk.get("source_attachment_filename", "unknown")
                label = f"[Source {i}] [{chunk_id}] (Thread: {thread}{date_str}{sender_str} | Attachment: {filename})"
            else:
                label = f"[Source {i}] [{chunk_id}] (Thread: {thread}{date_str}{sender_str} | Email body)"

            context_parts.append(f"{label}\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        if extra_context:
            context = f"{extra_context}\n\n---\n\nEvidence:\n{context}"

        # Inject entity alias hints so LLM connects query terms with chunk content
        from tacitgraph.entity_registry import expand_entity_aliases

        alias_hints = []
        for word in query.split():
            clean = word.strip("?,.:;!()\"'")
            if len(clean) >= 2:
                aliases = expand_entity_aliases(clean)
                if len(aliases) > 1:
                    alias_hints.append(
                        f'"{clean}" is also known as: {", ".join(a for a in aliases if a != clean)}'
                    )
        if alias_hints:
            context = (
                "Entity aliases (treat these as the same entity):\n"
                + "\n".join(alias_hints)
                + "\n\n---\n\n"
                + context
            )

        # Format prompts from config/prompts.json
        system_prompt = get_prompt("retrieval", "generation", "system_prompt")
        user_prompt = format_prompt(
            get_prompt("retrieval", "generation", "user_prompt"),
            context=context,
            question=query,
        )

        # Build messages — include conversation history if available
        messages = [{"role": "system", "content": system_prompt}]
        conv_history = getattr(self, "_conversation_history", "")
        if conv_history:
            messages.append({"role": "user", "content": f"Conversation so far:\n{conv_history}"})
            messages.append(
                {
                    "role": "assistant",
                    "content": "I have the conversation context. Please provide the new question and evidence.",
                }
            )
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.answer_model,
                messages=messages,
                temperature=get_prompt("retrieval", "generation", "temperature"),
                max_tokens=get_prompt("retrieval", "generation", "max_tokens"),
            )
            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            answer = _drop_repeated_paragraphs(
                extract_llm_content(response, context="answer generation")
            )
            total_tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0
        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, network timeouts and a local server that went away.
            logger.error(f"Answer generation failed: {e}")
            if llm_provider() == "local":
                forget_local_server()
                notice = chat_model_unavailable_message()
                if notice:
                    self.llm_client = None
                    return notice, False, "Chat model not available", 0
            return (
                f"⚠️ **The language model request failed** ({type(e).__name__}). "
                "The retrieved sources are listed; please try again.",
                False,
                "Language model request failed",
                0,
            )

        # Grounding check
        is_grounded = True
        missing_info = None

        missing_indicators = get_prompt(
            "retrieval",
            "generation",
            "missing_indicators",
            [
                "don't have enough information",
                "not in the context",
                "cannot find",
                "no information",
                "not mentioned",
            ],
        )
        for indicator in missing_indicators:
            if indicator.lower() in answer.lower():
                is_grounded = False
                missing_info = "Required information not found in knowledge base"
                break

        return answer, is_grounded, missing_info, total_tokens

    def _check_grounding(self, answer: str) -> tuple[bool, str | None]:
        """Check if an answer indicates missing information."""
        from tacitgraph.prompt_loader import get_prompt

        missing_indicators = get_prompt(
            "retrieval",
            "generation",
            "missing_indicators",
            [
                "don't have enough information",
                "not in the context",
                "cannot find",
                "no information",
                "not mentioned",
            ],
        )
        for indicator in missing_indicators:
            if indicator.lower() in answer.lower():
                return False, "Required information not found in knowledge base"
        return True, None

    def _calculate_confidence(self, chunks: list[dict[str, Any]]) -> float:
        """Calculate confidence as average cosine similarity of retrieved chunks.
        Excludes expanded/sibling chunks (score=0) that weren't directly matched."""
        if not chunks:
            return 0.0

        scores = [c.get("similarity_score", 0) for c in chunks if c.get("similarity_score", 0) > 0]
        if not scores:
            return 0.0

        return sum(scores) / len(scores)
