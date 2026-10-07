"""
Thread Grouper - Groups emails by conversation/thread

Groups related emails together to preserve semantic context
before chunking and anonymization.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator, Set
from pathlib import Path

from silver.email_text_cleaner import clean_email_text

logger = logging.getLogger(__name__)


@dataclass
class EmailThread:
    """Represents a grouped email thread"""

    conversation_id: str
    subject: str
    emails: List[Dict[str, Any]] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    @property
    def email_count(self) -> int:
        return len(self.emails)

    @property
    def is_thread(self) -> bool:
        """True if this contains multiple emails (actual thread)"""
        return len(self.emails) > 1

    def add_email(self, email: Dict[str, Any]) -> None:
        """Add an email to the thread"""
        self.emails.append(email)

        headers = email.get('email_headers', {})
        meta = email.get('document_metadata', {})

        # Track participants — handle both structured and plain formats
        sender = headers.get('sender') or headers.get('sender_email', '')
        if sender and sender not in self.participants:
            self.participants.append(sender)

        # Also add recipient emails/names as participants
        for recipient_list in [headers.get('recipients_to', []), headers.get('recipients_cc', [])]:
            for r in recipient_list:
                if isinstance(r, dict):
                    name = r.get('name', '') or r.get('email', '')
                else:
                    name = str(r)
                if name and name not in self.participants:
                    self.participants.append(name)

        # Track date range
        sent_time = self._parse_date(meta.get('sent_time'))
        if sent_time:
            if self.start_date is None or sent_time < self.start_date:
                self.start_date = sent_time
            if self.end_date is None or sent_time > self.end_date:
                self.end_date = sent_time

    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats"""
        if date_value is None:
            return None
        if isinstance(date_value, datetime):
            return date_value
        if isinstance(date_value, str):
            try:
                from dateutil import parser
                return parser.parse(date_value)
            except Exception:
                return None
        return None

    def get_sorted_emails(self) -> List[Dict[str, Any]]:
        """Get emails sorted by date (oldest first)"""
        def get_date(email):
            meta = email.get('document_metadata', {})
            dt = self._parse_date(meta.get('sent_time'))
            return dt or datetime.min

        return sorted(self.emails, key=get_date)

    def to_concatenated_text(self, include_metadata: bool = True) -> str:
        """
        Concatenate all emails in thread into single text.

        Args:
            include_metadata: Include From/Date/Subject headers

        Returns:
            Concatenated thread text
        """
        parts = []

        # Thread header
        parts.append(f"[THREAD: {self.subject}]")
        parts.append(f"[Participants: {', '.join(self.participants[:5])}]")
        parts.append(f"[Emails: {self.email_count}]")
        parts.append("")

        # Add each email in chronological order
        for i, email in enumerate(self.get_sorted_emails(), 1):
            headers = email.get('email_headers', {})
            meta = email.get('document_metadata', {})

            if include_metadata:
                parts.append(f"--- Email {i}/{self.email_count} ---")
                if headers.get('sender'):
                    parts.append(f"From: {headers['sender']}")
                if meta.get('sent_time'):
                    parts.append(f"Date: {meta['sent_time']}")
                if i == 1 and headers.get('subject'):
                    parts.append(f"Subject: {headers['subject']}")
                parts.append("")

            # Email body
            body = email.get('email_body_text', '')
            if body:
                parts.append(clean_email_text(body))

            parts.append("")

        parts.append("[END THREAD]")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "conversation_id": self.conversation_id,
            "subject": self.subject,
            "email_count": self.email_count,
            "participants": self.participants,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "email_ids": [e.get('record_id') for e in self.emails],
            "is_thread": self.is_thread,
        }


class ThreadGrouper:
    """
    Groups emails into conversation threads.

    Uses multiple strategies to identify related emails:
    1. RFC 2822 threading (Message-ID, In-Reply-To, References)
    2. conversation_id (Outlook thread ID)
    3. Subject matching (RE:, FW: normalization)
    4. Fallback (message_id — no grouping)
    """

    def __init__(self,
                 use_rfc2822_threading: bool = True,
                 use_conversation_id: bool = True,
                 use_subject_matching: bool = True,
                 normalize_subject: bool = True):
        """
        Initialize thread grouper.

        Args:
            use_rfc2822_threading: Group by RFC 2822 Message-ID/In-Reply-To/References
            use_conversation_id: Group by Outlook conversation_id
            use_subject_matching: Group by normalized subject
            normalize_subject: Remove RE:/FW: prefixes
        """
        self.use_rfc2822_threading = use_rfc2822_threading
        self.use_conversation_id = use_conversation_id
        self.use_subject_matching = use_subject_matching
        self.normalize_subject = normalize_subject

        # RFC 2822 threading data
        self._message_id_index: Dict[str, str] = {}  # message_id -> thread_key
        self._rfc_threads: Dict[str, List[str]] = {}  # thread_key -> [message_ids]

    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject by removing RE:/FW: prefixes"""
        if not subject:
            return ""

        import re
        # Remove common reply/forward prefixes
        normalized = re.sub(
            r'^(RE:|FW:|FWD:|AW:|WG:|SV:|VS:|Antw:|TR:|R:|Re:|Fw:|Fwd:)\s*',
            '',
            subject.strip(),
            flags=re.IGNORECASE
        )
        # Recursively remove nested prefixes
        while normalized != subject:
            subject = normalized
            normalized = re.sub(
                r'^(RE:|FW:|FWD:|AW:|WG:|SV:|VS:|Antw:|TR:|R:|Re:|Fw:|Fwd:)\s*',
                '',
                subject.strip(),
                flags=re.IGNORECASE
            )

        return normalized.strip().lower()

    def _build_rfc2822_threads(self, emails: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Build thread mapping using RFC 2822 Message-ID/In-Reply-To/References.

        Creates parent-child links between emails using standard email headers,
        then groups connected components into thread keys.

        Args:
            emails: List of email dictionaries (with header_message_id, in_reply_to, references)

        Returns:
            Dict mapping email's message_id → thread_key (for emails that can be threaded)
        """
        # Union-Find for grouping connected Message-IDs
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Index: header_message_id → internal message_id
        header_to_internal: Dict[str, str] = {}

        # Pass 1: Register all Message-IDs
        for email in emails:
            headers = email.get('email_headers', {})
            header_mid = headers.get('message_id')
            internal_mid = email.get('record_id', '')
            if header_mid:
                header_to_internal[header_mid] = internal_mid
                parent.setdefault(header_mid, header_mid)

        # Pass 2: Link In-Reply-To and References
        for email in emails:
            headers = email.get('email_headers', {})
            header_mid = headers.get('message_id')
            if not header_mid:
                continue

            in_reply_to = headers.get('in_reply_to')
            if in_reply_to:
                parent.setdefault(in_reply_to, in_reply_to)
                union(header_mid, in_reply_to)

            references = headers.get('references', [])
            for ref in references:
                parent.setdefault(ref, ref)
                union(header_mid, ref)

        # Pass 3: Build thread_key mapping (record_id → thread_key)
        email_thread_map: Dict[str, str] = {}
        for email in emails:
            headers = email.get('email_headers', {})
            header_mid = headers.get('message_id')
            if header_mid:
                root = find(header_mid)
                email_thread_map[email.get('record_id', '')] = f"rfc:{root}"

        return email_thread_map

    def _get_thread_key(self, email: Dict[str, Any], rfc_map: Optional[Dict[str, str]] = None) -> str:
        """Generate a key for thread grouping"""
        headers = email.get('email_headers', {})

        # Priority 1: RFC 2822 threading (Message-ID/In-Reply-To/References)
        if self.use_rfc2822_threading and rfc_map:
            rfc_key = rfc_map.get(email.get('record_id', ''))
            if rfc_key:
                return rfc_key

        # Priority 2: conversation_id
        if self.use_conversation_id:
            conv_id = headers.get('conversation_id')
            if conv_id:
                return f"conv:{conv_id}"

        # Priority 3: Normalized subject
        if self.use_subject_matching:
            subject = headers.get('subject', '')
            if self.normalize_subject:
                subject = self._normalize_subject(subject)
            if subject:
                return f"subj:{subject}"

        # Fallback: Use record_id (no grouping)
        return f"msg:{email.get('record_id', 'unknown')}"

    def group_emails(self, emails: Iterator[Dict[str, Any]]) -> List[EmailThread]:
        """
        Group emails into threads.

        Args:
            emails: Iterator of email dictionaries

        Returns:
            List of EmailThread objects
        """
        # Materialize to list for RFC 2822 threading (needs two passes)
        email_list = list(emails)

        # Build RFC 2822 thread map if enabled
        rfc_map = None
        if self.use_rfc2822_threading:
            rfc_map = self._build_rfc2822_threads(email_list)
            rfc_threaded = sum(1 for v in rfc_map.values() if v) if rfc_map else 0
            logger.info(f"RFC 2822 threading: {rfc_threaded} emails matched to threads")

        threads: Dict[str, EmailThread] = {}

        for email in email_list:
            thread_key = self._get_thread_key(email, rfc_map)

            if thread_key not in threads:
                # Get original subject for display
                headers = email.get('email_headers', {})
                subject = headers.get('subject', 'No Subject')
                if self.normalize_subject:
                    display_subject = self._normalize_subject(subject) or subject
                else:
                    display_subject = subject

                threads[thread_key] = EmailThread(
                    conversation_id=thread_key,
                    subject=display_subject
                )

            threads[thread_key].add_email(email)

        # Convert to list and sort by date
        thread_list = list(threads.values())
        thread_list.sort(key=lambda t: t.start_date or datetime.min)

        logger.info(f"Grouped {sum(t.email_count for t in thread_list)} emails into {len(thread_list)} threads")
        logger.info(f"  - Multi-email threads: {sum(1 for t in thread_list if t.is_thread)}")
        logger.info(f"  - Single emails: {sum(1 for t in thread_list if not t.is_thread)}")

        return thread_list

    def group_from_bronze(self, bronze_path: str) -> List[EmailThread]:
        """
        Group emails from Bronze layer.

        Args:
            bronze_path: Path to Bronze layer

        Returns:
            List of EmailThread objects
        """
        import json

        bronze = Path(bronze_path)
        emails_dir = bronze / "emails"

        if not emails_dir.exists():
            logger.warning(f"Emails directory not found: {emails_dir}")
            return []

        def email_iterator():
            for json_file in emails_dir.rglob("*.json"):
                if "metadata" in str(json_file):
                    continue
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        yield json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading {json_file}: {e}")

        return self.group_emails(email_iterator())


# Convenience function
def group_emails_into_threads(emails: List[Dict[str, Any]]) -> List[EmailThread]:
    """
    Group a list of emails into threads.

    Args:
        emails: List of email dictionaries

    Returns:
        List of EmailThread objects
    """
    grouper = ThreadGrouper()
    return grouper.group_emails(iter(emails))
