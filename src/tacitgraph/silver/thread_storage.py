"""Silver-layer persistence: directories, checkpoints and JSON output files."""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from tacitgraph.bronze.thread_grouper import EmailThread
from tacitgraph.silver.thread_models import (
    ThreadChunk,
    ThreadSummary,
)

logger = logging.getLogger(__name__)


class ThreadStorage:
    """Silver-layer persistence: directories, checkpoints and JSON output files."""

    def _create_directories(self) -> None:
        """Create Silver layer directory structure.

        Layout:
            silver/
            ├── not_personal/         ← processed (chunked, KG extracted, summarized)
            │   ├── email_chunks/     ← per-email chunks (both single and thread emails)
            │   ├── attachment_chunks/← per-attachment chunks (knowledge/transactional)
            │   ├── thread_summaries/ ← full-thread outcome summaries
            │   └── attachment_summaries/
            ├── personal/             ← raw Bronze data, skipped
            └── metadata/
        """
        tech = self.silver_path / "not_personal"
        directories = [
            tech / "email_chunks",
            tech / "thread_summaries",
            tech / "attachment_chunks",
            tech / "attachment_summaries",
            self.silver_path / "personal",
            self.silver_path / "metadata",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self) -> set:
        """Load set of already-processed thread IDs from checkpoint file."""
        checkpoint_file = self.silver_path / "checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return set()

    def _save_checkpoint(self, processed_ids: set) -> None:
        """Save processed thread IDs to checkpoint file."""
        checkpoint_file = self.silver_path / "checkpoint.json"
        self.silver_path.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(list(processed_ids), f)

    @staticmethod
    def _safe_filename(raw_id: str) -> str:
        """Generate a safe filename from an ID: short hash + sanitized prefix for readability."""
        hash_suffix = hashlib.md5(raw_id.encode()).hexdigest()[:10]
        safe_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw_id)[:60]
        return f"{safe_prefix}_{hash_suffix}"

    def _save_personal(self, thread: EmailThread) -> None:
        """Save a personal thread/email to the personal/ folder with full Bronze data."""
        safe_id = self._safe_filename(thread.conversation_id)
        out_file = self.silver_path / "personal" / f"{safe_id}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "thread_id": thread.conversation_id,
            "subject": thread.subject,
            "participants": thread.participants,
            "email_count": thread.email_count,
            "classification": "personal",
            "reason": "Thread classified as personal — skipped processing",
            "skipped_at": datetime.now().isoformat(),
            "emails": thread.emails,  # Full Bronze email data for reference
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)

    def _save_individual_chunk(self, chunk: ThreadChunk) -> None:
        """Save chunk to Silver layer — routes to appropriate subdirectory by source_type."""
        subdir = "document_chunks" if chunk.source_type == "document" else "email_chunks"
        chunk_file = self.silver_path / "not_personal" / subdir / f"{chunk.chunk_id}.json"
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        logger.debug(f"    Saved email chunk: {chunk.chunk_id} (thread={chunk.thread_id})")

    def _save_thread_summary(self, summary: ThreadSummary) -> None:
        """Save thread summary to Silver layer"""
        # Sanitize filename
        safe_id = self._safe_filename(summary.thread_id)
        summary_file = self.silver_path / "not_personal" / "thread_summaries" / f"{safe_id}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    def _save_metadata(self) -> None:
        """Save processing metadata"""
        metadata_file = self.silver_path / "metadata" / "thread_processing_stats.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing
        existing = []
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.append(self.stats)

        with open(metadata_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        # Save LLM failures separately for easy auditing
        failures = self.stats.get("llm_failures", [])
        if failures:
            failures_file = self.silver_path / "metadata" / "llm_failures.json"
            with open(failures_file, "w") as f:
                json.dump(
                    {
                        "total_failures": len(failures),
                        "content_filter_blocks": self.stats.get("content_filter_blocks", 0),
                        "failures": failures,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"Saved {len(failures)} LLM failure records to {failures_file}")

    def _save_attachment_chunk(self, chunk: ThreadChunk) -> None:
        """Save attachment chunk to Silver layer not_personal/attachment_chunks/"""
        chunk_dir = self.silver_path / "not_personal" / "attachment_chunks"
        chunk_file = chunk_dir / f"{chunk.chunk_id}.json"
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        logger.debug(
            f"    Saved attachment chunk: {chunk.chunk_id} (file={chunk.source_attachment_filename}, thread={chunk.thread_id})"
        )

    @staticmethod
    def _get_email_timestamps(email: dict[str, Any]) -> tuple[str, str]:
        """Extract sent and received timestamps from Bronze email.
        Timestamps are in document_metadata (from Bronze PST extraction).
        Returns (sent_timestamp, received_timestamp)."""
        meta = email.get("document_metadata", {})
        return meta.get("sent_time", ""), meta.get("received_time", "")
