"""Shared state and text cleaning for the ThreadAwareProcessor layers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tacitgraph.bronze.attachment_processor import AttachmentProcessor
    from tacitgraph.silver.anonymizer import Anonymizer
    from tacitgraph.silver.chunker import SemanticChunker
    from tacitgraph.silver.identity_registry import IdentityRegistry
    from tacitgraph.silver.kg_entity_extractor import KGEntityExtractor
    from tacitgraph.silver.openai_vision_extractor import OpenAIVisionExtractor
    from tacitgraph.silver.relationship_extractor import RelationshipExtractor


class ThreadProcessorBase:
    """Attributes assigned by ``ThreadAwareProcessor.__init__`` and used by its layers."""

    silver_path: Path
    processing_mode: str
    stats: dict[str, Any]
    chunker: SemanticChunker
    anonymizer: Anonymizer
    identity_registry: IdentityRegistry | None
    kg_extractor: KGEntityExtractor
    relationship_extractor: RelationshipExtractor | None
    extract_relationships: bool
    attachment_processor: AttachmentProcessor | None
    vision_extractor: OpenAIVisionExtractor | None
    openai_api_key: str | None
    use_azure: bool
    azure_endpoint: str | None
    azure_deployment: str | None
    azure_api_version: str

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean text for embedding-optimized downstream use.

        Preserves all semantic content and context while removing noise
        that degrades embedding quality:
        - Email metadata headers (From:, Date:, Subject: lines already in chunk metadata)
        - Thread/email separator lines (--- Email 1/2 ---)
        - Quoted-reply markers (>)
        - Redundant whitespace and control characters
        - Email signatures and disclaimers
        - Forwarded message boilerplate

        Keeps: all substantive content, paragraph structure (as single newlines),
        entity names, technical terms, decisions, facts.
        """
        import re

        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines (will handle spacing later)
            if not stripped:
                continue

            # Skip thread/email metadata headers (already in chunk metadata fields)
            if re.match(r"^\[THREAD:", stripped, re.IGNORECASE):
                continue
            if re.match(r"^\[Participants:", stripped, re.IGNORECASE):
                continue
            if re.match(r"^\[Emails:\s*\d+\]", stripped, re.IGNORECASE):
                continue
            if re.match(r"^---\s*Email\s+\d+/\d+\s*---", stripped):
                continue
            if re.match(r"^---\s*Forwarded\s*---", stripped, re.IGNORECASE):
                continue

            # Skip email header lines (From:, Date:, Subject:, To:, Cc:, Sent:)
            # but NOT lines where these words appear mid-sentence
            if re.match(r"^(From|Date|Sent|To|Cc|Bcc|Subject):\s", stripped):
                continue

            # Remove quoted-reply markers but keep the content
            stripped = re.sub(r"^>+\s*", "", stripped)

            # Skip device/app boilerplate and disclaimers (keep regards/thanks for context)
            if re.match(
                r"^(sent from my|get outlook|disclaimer|confidential|this email)",
                stripped,
                re.IGNORECASE,
            ):
                continue
            if re.match(r"^[-_=]{5,}$", stripped):
                continue

            # Skip lines that are just a person's name (likely signature, 1-3 words, all title case)
            if (
                re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s*$", stripped)
                and len(stripped) < 40
            ):
                # Only skip if it looks like a standalone name (not part of content)
                if len(stripped.split()) <= 3:
                    continue

            # Normalize tabs to spaces
            stripped = stripped.replace("\t", " ")

            # Collapse multiple spaces
            stripped = re.sub(r" {2,}", " ", stripped)

            if stripped:
                cleaned_lines.append(stripped)

        # Join with single space — flat text is best for embedding models
        # Embedding models don't benefit from newlines; dense text = better vectors
        result = " ".join(cleaned_lines)

        # Final cleanup: collapse any remaining multiple spaces
        result = re.sub(r" {2,}", " ", result)

        return result.strip()
