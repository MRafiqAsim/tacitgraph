"""
Reply Splitter Module

Detects and splits quoted reply chains in email bodies.
Separates the primary message from inline quoted replies,
preserving metadata (sender, date, subject) from reply headers.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmailSegment:
    """A segment of an email — either primary content or a quoted reply."""

    text: str
    is_primary: bool = True
    parsed_sender: str = ""
    parsed_date: str = ""  # YYYY-MM-DD format
    parsed_subject: str = ""
    bronze_match_id: str = ""  # record_id if matched in Bronze
    status: str = "primary"  # primary | matched | orphan


# Reply header patterns (multilingual)
REPLY_PATTERNS = [
    # English Outlook: -----Original Message-----
    re.compile(
        r"^[-_]{3,}\s*(?:Original Message|Forwarded message)\s*[-_]{3,}\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Dutch Outlook: -----Oorspronkelijk bericht-----
    re.compile(
        r"^[-_]{3,}\s*(?:Oorspronkelijk bericht|Doorgestuurd bericht)\s*[-_]{3,}\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # "On {date}, {name} wrote:" / "Op {date} schreef {name}:"
    re.compile(r"^On\s+.+wrote:\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Op\s+.+schreef\s+.+:\s*$", re.MULTILINE | re.IGNORECASE),
]

# Header field patterns to extract sender/date/subject from quoted reply headers
HEADER_FIELD_PATTERNS = {
    "sender": re.compile(r"(?:^From|^Van|^De|^Von):\s*(.+?)$", re.MULTILINE | re.IGNORECASE),
    "date": re.compile(
        r"(?:^Sent|^Verzonden|^Date|^Datum|^Envoy[eé]):\s*(.+?)$", re.MULTILINE | re.IGNORECASE
    ),
    "subject": re.compile(
        r"(?:^Subject|^Onderwerp|^Objet|^Betreff):\s*(.+?)$", re.MULTILINE | re.IGNORECASE
    ),
}


def parse_date_flexible(date_str: str) -> str:
    """Parse various date formats to YYYY-MM-DD."""
    if not date_str:
        return ""
    try:
        from dateutil import parser as dateutil_parser

        dt = dateutil_parser.parse(date_str, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # Try basic regex extraction
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if match:
            return match.group(0)
        match = re.search(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", date_str)
        if match:
            d, m, y = match.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"
        return ""


def normalize_subject(subject: str) -> str:
    """Normalize subject for matching — strip RE:/FW:/AW: prefixes."""
    if not subject:
        return ""
    cleaned = re.sub(r"^(?:RE|FW|AW|Antw|Doorst):\s*", "", subject.strip(), flags=re.IGNORECASE)
    return cleaned.lower().strip()


def split_replies(email_body: str) -> list[EmailSegment]:
    """
    Split an email body into primary content and quoted replies.

    Returns a list of EmailSegment objects:
    - First element is always the primary message (is_primary=True)
    - Subsequent elements are quoted replies with parsed sender/date/subject
    """
    if not email_body or not email_body.strip():
        return [EmailSegment(text=email_body or "", is_primary=True, status="primary")]

    # Find all reply boundary positions
    boundaries = []
    for pattern in REPLY_PATTERNS:
        for match in pattern.finditer(email_body):
            boundaries.append(match.start())

    # Also detect From:/Van: header blocks that indicate a quoted reply
    # Pattern: blank line + From: line + Sent: line (within 200 chars)
    header_block_pattern = re.compile(
        r"\n\s*\n\s*(?:From|Van|De|Von):\s*.+\n\s*(?:Sent|Verzonden|Date|Datum):\s*.+\n",
        re.IGNORECASE,
    )
    for match in header_block_pattern.finditer(email_body):
        boundaries.append(match.start())

    if not boundaries:
        return [EmailSegment(text=email_body.strip(), is_primary=True, status="primary")]

    # Sort and deduplicate boundaries
    boundaries = sorted(set(boundaries))

    # Split text at boundaries
    segments = []

    # Primary content: everything before first boundary
    primary_text = email_body[: boundaries[0]].strip()
    if primary_text:
        segments.append(
            EmailSegment(
                text=primary_text,
                is_primary=True,
                status="primary",
            )
        )

    # Quoted replies: between boundaries
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(email_body)
        reply_text = email_body[start:end].strip()

        if not reply_text or len(reply_text) < 20:
            continue

        # Parse reply headers
        sender, date, subject = _parse_reply_headers(reply_text)

        segments.append(
            EmailSegment(
                text=reply_text,
                is_primary=False,
                parsed_sender=sender,
                parsed_date=parse_date_flexible(date),
                parsed_subject=subject,
                status="orphan",  # Default to orphan until matched
            )
        )

    # If no primary text found (edge case: entire email is quoted)
    if not segments:
        segments.append(EmailSegment(text=email_body.strip(), is_primary=True, status="primary"))

    return segments


def _parse_reply_headers(reply_text: str) -> tuple:
    """Extract sender, date, subject from the reply header block."""
    # Only scan the first 500 chars for headers
    header_area = reply_text[:500]

    sender = ""
    date = ""
    subject = ""

    sender_match = HEADER_FIELD_PATTERNS["sender"].search(header_area)
    if sender_match:
        sender = sender_match.group(1).strip()
        # Clean email address from sender (keep name only)
        sender = re.sub(r"<[^>]+>", "", sender).strip()
        sender = re.sub(r"\[mailto:[^\]]+\]", "", sender).strip()

    date_match = HEADER_FIELD_PATTERNS["date"].search(header_area)
    if date_match:
        date = date_match.group(1).strip()

    subject_match = HEADER_FIELD_PATTERNS["subject"].search(header_area)
    if subject_match:
        subject = subject_match.group(1).strip()

    return sender, date, subject


def build_bronze_index(bronze_emails: list) -> dict:
    """
    Build a lookup index from Bronze emails for matching quoted replies.

    Key: normalized_subject | sender_name_lower | date (YYYY-MM-DD)
    Value: record_id
    """
    index = {}
    for email in bronze_emails:
        headers = email.get("email_headers", {})
        subject = normalize_subject(headers.get("subject", ""))
        sender = (headers.get("sender") or "").lower().strip()

        # Get date from sent_timestamp
        sent = email.get("document_metadata", {}).get("sent_timestamp", "")
        if not sent:
            sent = headers.get("date", "")
        date = parse_date_flexible(str(sent)) if sent else ""

        record_id = email.get("record_id", "")
        if subject and sender and date and record_id:
            key = f"{subject}|{sender}|{date}"
            index[key] = record_id

    logger.info(f"Built Bronze index: {len(index)} entries")
    return index


def match_quoted_replies(
    segments: list[EmailSegment],
    bronze_index: dict,
    parent_subject: str = "",
) -> list[EmailSegment]:
    """
    Match quoted reply segments against Bronze index.

    Sets status to "matched" if found in Bronze, "orphan" if not.
    Uses parent_subject as fallback when quoted reply has no Subject: header.
    """
    fallback_subject = normalize_subject(parent_subject)

    for seg in segments:
        if seg.is_primary:
            continue

        subject = normalize_subject(seg.parsed_subject) or fallback_subject
        sender = seg.parsed_sender.lower().strip()
        date = seg.parsed_date

        if subject and sender and date:
            key = f"{subject}|{sender}|{date}"
            match_id = bronze_index.get(key)
            if match_id:
                seg.bronze_match_id = match_id
                seg.status = "matched"
                logger.debug(f"Quoted reply matched: {sender} ({date}) → {match_id}")
            else:
                # Try fuzzy sender match (first name only)
                first_name = sender.split()[0] if sender else ""
                if first_name:
                    for idx_key, rid in bronze_index.items():
                        if first_name in idx_key and subject in idx_key and date in idx_key:
                            seg.bronze_match_id = rid
                            seg.status = "matched"
                            logger.debug(f"Quoted reply fuzzy matched: {sender} ({date}) → {rid}")
                            break

        if seg.status == "orphan":
            logger.debug(f"Quoted reply orphan: {sender} ({date}) | {subject[:40]}")

    return segments
