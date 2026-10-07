"""Answer generation: LLM or local extractive quoting, grounding checks and confidence scores."""

import logging
import re
from typing import Any

from tacitgraph.llm_client import create_chat_client
from tacitgraph.prompt_loader import format_prompt, get_prompt
from tacitgraph.retrieval.lexical_index import tokenize
from tacitgraph.retrieval.thread_expansion import ThreadExpansion

logger = logging.getLogger(__name__)

LOCAL_ANSWER_CHUNKS = 5
LOCAL_ANSWER_SENTENCES = 4
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """Split text into trimmed sentences of a readable length."""
    sentences = (s.strip(" -*>\t") for s in _SENTENCE_RE.split(text))
    return [s for s in sentences if 20 <= len(s) <= 400]


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


def _cite(chunk: dict[str, Any], n: int) -> str:
    """One-line source label: subject, sender and date of the chunk's email."""
    parts = [chunk.get("thread_subject") or chunk.get("source_attachment_filename") or ""]
    parts.append(chunk.get("email_sender") or "")
    parts.append((chunk.get("sent_timestamp") or chunk.get("received_timestamp") or "")[:10])
    return f"— [{n}] " + " · ".join(p for p in parts if p)


class AnswerGeneration(ThreadExpansion):
    """Answer generation: LLM or local extractive quoting, grounding checks and confidence scores."""

    def _initialize_llm(self):
        """Initialize the LLM client for answer generation (None when no provider is set)."""
        self.llm_client = create_chat_client()

    def _generate_answer(
        self, query: str, chunks: list[dict[str, Any]], extra_context: str = ""
    ) -> tuple[str, bool, str | None, int]:
        """
        Generate answer using LLM with grounding check.
        Falls back to extractive summary in local mode.

        Returns:
            (answer, is_grounded, missing_info, total_tokens)
        """
        if not chunks and not extra_context:
            return "", True, None, 0

        # Local mode: quote matching sentences from the retrieved chunks
        if not self.llm_client:
            answer, is_grounded, missing_info = self._generate_local_answer(
                query, chunks, extra_context
            )
            return answer, is_grounded, missing_info, 0

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
            # APIStatusError, network timeouts, etc.
            logger.error(f"Answer generation failed: {e}")
            return "", True, None, 0

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

    def _generate_local_answer(
        self, query: str, chunks: list[dict[str, Any]], extra_context: str = ""
    ) -> tuple[str, bool, str | None]:
        """Answer without an LLM by quoting the retrieved sentences that match the query.

        Abstractive summarisers rewrite names and invent facts when fed several
        unrelated emails, so local mode only quotes source text and cites it.
        """
        terms = set(tokenize(query))
        candidates: list[tuple[int, int, int, str, str]] = []
        for rank, chunk in enumerate(chunks[:LOCAL_ANSWER_CHUNKS]):
            text = chunk.get("text_english") or chunk.get("text") or chunk.get("summary") or ""
            for pos, sentence in enumerate(_split_sentences(text)):
                hits = len(terms & set(tokenize(sentence)))
                if hits:
                    candidates.append((hits, -rank, -pos, sentence, _cite(chunk, rank + 1)))

        if not candidates:
            top = chunks[0]
            text = top.get("summary") or top.get("text_english") or top.get("text") or ""
            snippet = " ".join(_split_sentences(text)[:2])
            if not snippet:
                return "", False, "No matching passages found"
            return (
                "No retrieved passage mentions the terms in your question. "
                f"Closest match:\n\n> {snippet}\n\n{_cite(top, 1)}",
                False,
                "No passage mentions the query terms",
            )

        candidates.sort(reverse=True)
        quoted: list[str] = []
        seen: set[str] = set()
        for _, _, _, sentence, cite in candidates:
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            quoted.append(f"> {sentence}\n\n{cite}")
            if len(quoted) >= LOCAL_ANSWER_SENTENCES:
                break
        return "From the retrieved emails:\n\n" + "\n\n".join(quoted), True, None

    def _calculate_confidence(self, chunks: list[dict[str, Any]]) -> float:
        """Calculate confidence as average cosine similarity of retrieved chunks.
        Excludes expanded/sibling chunks (score=0) that weren't directly matched."""
        if not chunks:
            return 0.0

        scores = [c.get("similarity_score", 0) for c in chunks if c.get("similarity_score", 0) > 0]
        if not scores:
            return 0.0

        return sum(scores) / len(scores)
