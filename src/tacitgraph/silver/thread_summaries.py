"""Email, attachment and thread summaries (local summariser or LLM)."""

import json
import logging
import os
from typing import Any

from tacitgraph.bronze.thread_grouper import EmailThread
from tacitgraph.silver.email_text_cleaner import clean_email_text
from tacitgraph.silver.thread_knowledge import ThreadKnowledgeExtraction
from tacitgraph.silver.thread_models import (
    AttachmentSummary,
    EmailSummary,
    ThreadChunk,
    ThreadSummary,
)

logger = logging.getLogger(__name__)


class ThreadSummarization(ThreadKnowledgeExtraction):
    """Email, attachment and thread summaries (local summariser or LLM)."""

    def _generate_thread_summary(
        self,
        thread: EmailThread,
        chunks: list[ThreadChunk],
        language: str,
        attachment_ids: list[str] | None = None,
    ) -> ThreadSummary | None:
        """Generate a summary for the thread with attachment cross-references."""
        # Build date range string
        date_range = ""
        if thread.start_date and thread.end_date:
            if thread.start_date.date() == thread.end_date.date():
                date_range = thread.start_date.strftime("%Y-%m-%d")
            else:
                date_range = f"{thread.start_date.strftime('%Y-%m-%d')} to {thread.end_date.strftime('%Y-%m-%d')}"

        # Extract key topics (simple: from subject + common words)
        key_topics = self._extract_key_topics(thread, chunks)

        # Generate summary — LLM only in llm/hybrid mode
        if self.processing_mode in ("llm", "hybrid") and self.openai_api_key:
            summary_text = self._generate_llm_summary(thread, chunks)
        else:
            summary_text = self._generate_simple_summary(thread, chunks)

        summary = ThreadSummary(
            thread_id=thread.conversation_id,
            subject=thread.subject,
            participants=thread.participants,
            email_count=thread.email_count,
            date_range=date_range,
            summary=summary_text,
            key_topics=key_topics,
            chunk_ids=[c.chunk_id for c in chunks],
            attachment_ids=attachment_ids or [],
            source_email_ids=[e.get("record_id", "") for e in thread.emails if e.get("record_id")],
        )

        self._save_thread_summary(summary)
        self.stats["summaries_generated"] += 1

        return summary

    def _summarize_email(self, email: dict[str, Any], language: str) -> str | None:
        """Generate a work-focused summary for a single email.

        Strips personal/social chatter (congratulations, personal news, etc.)
        but keeps all professional content: names, roles, decisions, actions.
        Returns summary text only — no file is written.
        """
        body_text = email.get("email_body_text", "")
        if not body_text or not body_text.strip():
            return None

        body_text = clean_email_text(body_text)
        if not body_text.strip():
            return None

        if self.processing_mode in ("llm", "hybrid") and self.openai_api_key:
            return self._generate_llm_email_summary(email, body_text)
        else:
            return self._generate_simple_email_summary(email, body_text)

    def _generate_email_summary(
        self,
        email: dict[str, Any],
        thread: EmailThread,
        language: str,
        attachment_ids: list[str] | None = None,
    ) -> EmailSummary | None:
        """Generate a per-email summary and save to email_summaries/."""
        email_id = email.get("record_id", "")
        if not email_id:
            return None

        # Get email text
        body_text = email.get("email_body_text", "")
        if not body_text or not body_text.strip():
            return None

        # Clean the text
        body_text = clean_email_text(body_text)
        body_text = self._clean_text(body_text)

        if not body_text.strip():
            return None

        # Get sender and date
        headers = email.get("email_headers", {})
        sender = headers.get("sender", "") or headers.get("sender_email", "")
        doc_meta = email.get("document_metadata", {})
        date = doc_meta.get("delivery_time", "") or doc_meta.get("creation_time", "")

        # Generate summary text
        if self.processing_mode in ("llm", "hybrid") and self.openai_api_key:
            summary_text = self._generate_llm_email_summary(email, body_text)
        else:
            summary_text = self._generate_simple_email_summary(email, body_text)

        # Find chunk_ids that belong to this email
        chunk_ids = []
        # For single emails, chunk_id starts with record_id
        # For thread emails, we can't map precisely — leave empty
        if thread.email_count == 1:
            chunk_dir = self.silver_path / "not_personal" / "email_chunks"
            if chunk_dir.exists():
                for f in chunk_dir.glob(f"{email_id}_*.json"):
                    chunk_ids.append(f.stem)

        summary = EmailSummary(
            email_id=email_id,
            thread_id=thread.conversation_id,
            subject=headers.get("subject", thread.subject),
            sender=sender,
            date=str(date),
            summary=summary_text,
            chunk_ids=chunk_ids,
            attachment_ids=attachment_ids or [],
        )

        self._save_email_summary(summary)
        self.stats.setdefault("email_summaries_generated", 0)
        self.stats["email_summaries_generated"] += 1

        return summary

    def _generate_simple_email_summary(self, email: dict[str, Any], text: str) -> str:
        """Generate email summary using local BART model."""
        from tacitgraph.silver.local_summarizer import summarize_thread

        headers = email.get("email_headers", {})
        subject = headers.get("subject", "No Subject")
        sender = headers.get("sender", "")

        return summarize_thread(
            subject=subject,
            participants=[sender] if sender else [],
            email_count=1,
            text=text,
        )

    def _generate_llm_email_summary(self, email: dict[str, Any], text: str) -> str:
        """Generate intent-only email summary using LLM.

        Phase 2 anonymization: summaries strip irrelevant details and sensitive
        content, capturing only the intent — not specifics.
        """
        try:
            import httpx

            if self.use_azure:
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=self.openai_api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_api_version,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    max_retries=2,
                )
                model = self.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            else:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_api_key)
                model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

            headers = email.get("email_headers", {})
            subject = headers.get("subject", "")

            from tacitgraph.prompt_loader import get_prompt

            system_prompt = get_prompt(
                "silver",
                "email_summary",
                "system_prompt",
                "Summarize the intent of this email. Remove all sensitive content and specific details. Capture only what was discussed, decided, and needed.",
            )
            user_prompt_template = get_prompt(
                "silver",
                "email_summary",
                "user_prompt",
                "Summarize the intent of this email. Remove all sensitive content, names, and specific details.\n\nSubject: {subject}\n\n{text}\n\nIntent summary:",
            )
            user_prompt = user_prompt_template.format(subject=subject, text=text[:4000])

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=get_prompt("silver", "email_summary", "temperature", 0.3),
                max_tokens=get_prompt("silver", "email_summary", "max_tokens", 300),
            )

            from tacitgraph.llm_response import LLMContentError, extract_llm_content

            try:
                return extract_llm_content(response, context="email summary")
            except LLMContentError as lce:
                if "length" in str(lce):
                    # Truncated summary — use partial content instead of falling back
                    choice = response.choices[0] if response.choices else None
                    content = getattr(getattr(choice, "message", None), "content", None)
                    if content and len(content) > 20:
                        logger.warning(
                            f"Email summary truncated, using partial ({len(content)} chars)"
                        )
                        return content
                raise

        except Exception as e:
            email_id = email.get("record_id", "unknown")
            subject = email.get("email_headers", {}).get("subject", "")
            error_str = str(e)
            is_content_filter = "content_filter" in error_str
            if is_content_filter:
                self.stats["content_filter_blocks"] += 1
            self.stats["llm_failures"].append(
                {
                    "chunk_id": email_id,
                    "stage": "email_summary",
                    "reason": "content_filter" if is_content_filter else "error",
                    "message": error_str[:200],
                }
            )
            logger.warning(
                f"LLM email summary failed for email_id={email_id} subject='{subject[:80]}': {e}"
            )
            return self._generate_simple_email_summary(email, text)

    def _save_email_summary(self, summary: EmailSummary) -> None:
        """Save email summary to Silver layer."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in summary.email_id)[:100]
        summary_file = self.silver_path / "not_personal" / "email_summaries" / f"{safe_id}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _generate_attachment_summary(
        self,
        attachment_id: str,
        filename: str,
        thread_id: str,
        chunks: list[ThreadChunk],
        classification: str,
        language: str,
    ) -> AttachmentSummary | None:
        """Generate and save a per-attachment summary."""
        if not chunks:
            return None

        total_tokens = sum(c.token_count for c in chunks)
        chunk_ids = [c.chunk_id for c in chunks]

        # Generate summary text — LLM only in llm/hybrid mode
        if self.processing_mode in ("llm", "hybrid") and self.openai_api_key:
            summary_text = self._generate_llm_attachment_summary(filename, chunks)
        else:
            summary_text = self._generate_simple_attachment_summary(filename, chunks)

        att_summary = AttachmentSummary(
            attachment_id=attachment_id,
            thread_id=thread_id,
            filename=filename,
            summary=summary_text,
            chunk_ids=chunk_ids,
            classification=classification,
            token_count=total_tokens,
        )

        self._save_attachment_summary(att_summary)
        self.stats["attachment_summaries_generated"] += 1

        return att_summary

    def _generate_simple_attachment_summary(
        self,
        filename: str,
        chunks: list[ThreadChunk],
    ) -> str:
        """Generate an attachment summary using local BART model."""
        from tacitgraph.silver.local_summarizer import summarize_attachment

        combined_text = "\n\n".join(c.text_anonymized for c in chunks)
        return summarize_attachment(filename, len(chunks), combined_text)

    def _generate_llm_attachment_summary(
        self,
        filename: str,
        chunks: list[ThreadChunk],
    ) -> str:
        """Generate an LLM-based attachment summary scaled to document size."""
        try:
            if self.use_azure:
                import httpx
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=self.openai_api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_api_version,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    max_retries=2,
                )
                model = self.azure_deployment or "gpt-4o"
            else:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_api_key)
                model = "gpt-4o"

            # Scale context window with document size
            # Small docs (1-2 chunks): 4000 chars, large docs (10+): up to 12000
            max_chars = min(12000, 4000 + len(chunks) * 1000)
            combined_text = "\n\n".join([c.text_anonymized for c in chunks])[:max_chars]

            from tacitgraph.prompt_loader import format_prompt, get_prompt

            user_template = get_prompt(
                "silver",
                "attachment_summary",
                "user_prompt",
                "Summarize this attachment:\nFilename: {filename}\nChunks: {chunk_count}\n\nContent:\n{content}\n\nSummary:",
            )
            user_prompt = format_prompt(
                user_template,
                filename=filename,
                chunk_count=str(len(chunks)),
                content=combined_text,
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt(
                            "silver",
                            "attachment_summary",
                            "system_prompt",
                            "Summarize this document attachment proportional to its length.",
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=get_prompt("silver", "attachment_summary", "temperature", 0.3),
                max_tokens=get_prompt("silver", "attachment_summary", "max_tokens", 1000),
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            return extract_llm_content(response, context="attachment summary")

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, network timeouts, etc. Falls back to extractive summary.
            logger.warning(f"LLM attachment summary failed for '{filename}': {e}")
            return self._generate_simple_attachment_summary(filename, chunks)

    def _save_attachment_summary(self, summary: AttachmentSummary) -> None:
        """Save attachment summary to Silver layer."""
        summary_dir = self.silver_path / "not_personal" / "attachment_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in summary.attachment_id)[
            :100
        ]
        summary_file = summary_dir / f"{safe_id}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _extract_key_topics(self, thread: EmailThread, chunks: list[ThreadChunk]) -> list[str]:
        """Extract key topics from thread"""
        # Simple extraction: words from subject
        topics = []
        subject_words = thread.subject.lower().split()
        stopwords = {"re", "fw", "fwd", "the", "a", "an", "and", "or", "is", "are", "was", "were"}

        for word in subject_words:
            word = word.strip(":-.,!?()[]")
            if len(word) > 2 and word not in stopwords:
                topics.append(word)

        return topics[:5]

    def _generate_simple_summary(self, thread: EmailThread, chunks: list[ThreadChunk]) -> str:
        """Generate a thread summary using local BART model."""
        if not chunks:
            return "No content available"

        from tacitgraph.silver.local_summarizer import summarize_thread

        combined_text = "\n\n".join(c.text_anonymized for c in chunks)
        return summarize_thread(
            subject=thread.subject,
            participants=thread.participants,
            email_count=thread.email_count,
            text=combined_text,
        )

    def _generate_llm_summary(self, thread: EmailThread, chunks: list[ThreadChunk]) -> str:
        """Generate summary using LLM (supports both OpenAI and Azure OpenAI)"""
        try:
            if self.use_azure:
                import httpx
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=self.openai_api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_api_version,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                    max_retries=2,
                )
                model = self.azure_deployment or "gpt-4o"
            else:
                from openai import OpenAI

                client = OpenAI(api_key=self.openai_api_key)
                model = "gpt-4o"

            # Combine anonymized chunk texts
            combined_text = "\n\n".join([c.text_anonymized for c in chunks])[:4000]

            from tacitgraph.prompt_loader import get_prompt

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt(
                            "silver",
                            "thread_summary",
                            "system_prompt",
                            "Summarize this email thread in 2-3 sentences. Focus on the main topic and outcome.",
                        ),
                    },
                    {"role": "user", "content": combined_text},
                ],
                temperature=get_prompt("silver", "thread_summary", "temperature", 0.3),
                max_tokens=get_prompt("silver", "thread_summary", "max_tokens", 150),
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            return extract_llm_content(response, context="thread summary")

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, network timeouts, etc. Falls back to extractive summary.
            logger.warning(
                f"LLM thread summary failed for thread_id={thread.conversation_id} subject='{thread.subject[:80]}': {e}"
            )
            return self._generate_simple_summary(thread, chunks)
