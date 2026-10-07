"""Context expansion: adds sibling emails and summaries from the threads of retrieved chunks."""

import json
import logging
from typing import Any

from tacitgraph.retrieval.query_analysis import QueryAnalysis

from .date_filter import filter_chunks_by_date

logger = logging.getLogger(__name__)


class ThreadExpansion(QueryAnalysis):
    """Context expansion: adds sibling emails and summaries from the threads of retrieved chunks."""

    def _expand_by_thread(
        self, chunks: list[dict[str, Any]], max_siblings: int = 50
    ) -> list[dict[str, Any]]:
        """
        Expand retrieval results by fetching sibling chunks and summaries
        from the same thread.

        When a chunk is retrieved, also pull in:
        1. Sibling email/attachment chunks from the same thread
        2. Thread summary (if available)
        3. Attachment summaries (if available)

        This gives the LLM full context regardless of which entry point was hit.
        """
        if not self.silver_path:
            return chunks

        existing_ids = {c.get("chunk_id") for c in chunks}

        # Rank threads by their best chunk score, expand only top 3
        thread_best_score = {}
        for chunk in chunks:
            tid = chunk.get("thread_id")
            score = chunk.get("similarity_score", 0)
            if tid and score > thread_best_score.get(tid, 0):
                thread_best_score[tid] = score

        if not thread_best_score:
            return chunks

        # Only expand the top 3 most relevant threads
        sorted_threads = sorted(thread_best_score.items(), key=lambda x: x[1], reverse=True)
        thread_ids = {tid for tid, _ in sorted_threads[:3]}

        # 1. Search Silver layer for sibling chunks
        search_dirs = [
            self.silver_path / "not_personal" / "email_chunks",
            self.silver_path / "not_personal" / "attachment_chunks",
            self.silver_path / "not_personal" / "document_chunks",
        ]

        sibling_chunks = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for chunk_file in search_dir.glob("*.json"):
                if len(sibling_chunks) >= max_siblings:
                    break
                try:
                    with open(chunk_file, encoding="utf-8") as f:
                        chunk_data = json.load(f)
                    cid = chunk_data.get("chunk_id")
                    tid = chunk_data.get("thread_id")
                    if tid in thread_ids and cid not in existing_ids:
                        sibling_chunks.append(
                            {
                                "chunk_id": cid,
                                "text": chunk_data.get("summary")
                                or chunk_data.get("text_english")
                                or chunk_data.get("text_anonymized", ""),
                                "text_english": chunk_data.get("text_english", ""),
                                "summary": chunk_data.get("summary", ""),
                                "thread_id": tid,
                                "thread_subject": chunk_data.get("thread_subject"),
                                "source_type": chunk_data.get("source_type", "email"),
                                "source_attachment_filename": chunk_data.get(
                                    "source_attachment_filename", ""
                                ),
                                "has_attachments": chunk_data.get("has_attachments", False),
                                "email_sender": chunk_data.get("email_sender", ""),
                                "sent_timestamp": chunk_data.get("sent_timestamp", ""),
                                "similarity_score": 0.0,
                                "_expanded": True,
                            }
                        )
                        existing_ids.add(cid)
                except Exception:
                    continue

        # 2. Load thread summaries for matched threads
        summary_context = []
        thread_summaries_dir = self.silver_path / "not_personal" / "thread_summaries"
        if thread_summaries_dir.exists():
            for summary_file in thread_summaries_dir.glob("*.json"):
                try:
                    with open(summary_file, encoding="utf-8") as f:
                        summary_data = json.load(f)
                    if summary_data.get("thread_id") in thread_ids:
                        summary_text = summary_data.get("summary", "")
                        if summary_text:
                            summary_context.append(
                                {
                                    "chunk_id": f"summary_{summary_data['thread_id']}",
                                    "text": f"[Thread Summary] {summary_data.get('subject', '')}: {summary_text}",
                                    "thread_id": summary_data["thread_id"],
                                    "source_type": "thread_summary",
                                    "similarity_score": 0.0,
                                    "_expanded": True,
                                }
                            )
                            # 3. Follow cross-references to attachment summaries
                            for att_id in summary_data.get("attachment_ids", []):
                                att_summary = self._load_attachment_summary(att_id)
                                if att_summary:
                                    summary_context.append(
                                        {
                                            "chunk_id": f"att_summary_{att_id}",
                                            "text": f"[Attachment: {att_summary.get('filename', '')}] {att_summary.get('summary', '')}",
                                            "thread_id": summary_data["thread_id"],
                                            "source_type": "attachment_summary",
                                            "similarity_score": 0.0,
                                            "_expanded": True,
                                        }
                                    )
                except Exception:
                    continue

        # Per-email summaries are not added: sibling chunks + thread summary suffice.

        expanded_count = len(sibling_chunks) + len(summary_context)
        if expanded_count:
            logger.info(
                f"Thread expansion: added {len(sibling_chunks)} sibling chunks, "
                f"{len(summary_context)} summaries from {len(thread_ids)} threads"
            )

        return list(chunks) + sibling_chunks + summary_context

    def _load_attachment_summary(self, attachment_id: str) -> dict[str, Any] | None:
        """Load an attachment summary by ID from Silver layer."""
        if not self.silver_path:
            return None
        summary_dir = self.silver_path / "not_personal" / "attachment_summaries"
        if not summary_dir.exists():
            return None
        # Try exact match and sanitized match
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in attachment_id)[:100]
        for candidate in (f"{attachment_id}.json", f"{safe_id}.json"):
            path = summary_dir / candidate
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return None

    def _apply_temporal_filter(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter chunks by date range if a temporal query was detected."""
        date_range = getattr(self, "_date_range", None)
        if not date_range or not date_range.is_set:
            return chunks
        before = len(chunks)
        filtered = filter_chunks_by_date(chunks, date_range)
        if len(filtered) < before:
            logger.info(f"Temporal filter: {before} → {len(filtered)} chunks ({date_range})")
        return filtered
