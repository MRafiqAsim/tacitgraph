"""
PST Email Extractor

Extracts emails and attachments from Microsoft Outlook PST files.
Handles 35 years of email archives with proper encoding and error handling.
"""

import hashlib
import logging
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from email.parser import HeaderParser
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Any

import ftfy

logger = logging.getLogger(__name__)


@dataclass
class Attachment:
    """Email attachment"""

    filename: str
    content_type: str
    size: int
    content: bytes
    attachment_id: str = ""

    def __post_init__(self):
        if not self.attachment_id:
            self.attachment_id = uuid.uuid4().hex

    def save(self, output_dir: str) -> str:
        """Save attachment to disk using attachment_id as filename (preserves extension)."""
        ext = Path(self.filename).suffix
        safe_name = f"{self.attachment_id}{ext}"
        output_path = Path(output_dir) / self.attachment_id / safe_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(self.content)

        return str(output_path)


@dataclass
class EmailMessage:
    """Extracted email message"""

    # Identification
    message_id: str
    source_pst: str
    folder_path: str

    # Headers
    subject: str
    sender: str
    sender_email: str
    recipients_to: list[str] = field(default_factory=list)
    recipients_cc: list[str] = field(default_factory=list)
    recipients_bcc: list[str] = field(default_factory=list)

    # Timestamps
    sent_time: datetime | None = None
    received_time: datetime | None = None

    # Content
    body_text: str = ""
    body_html: str = ""

    # Threading
    conversation_id: str | None = None
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)

    # Transport headers (RFC 2822)
    header_message_id: str | None = None
    return_path: str | None = None
    x_originating_ip: str | None = None
    transport_headers_raw: str | None = None

    # Attachments
    attachments: list[Attachment] = field(default_factory=list)
    has_attachments: bool = False

    # Metadata
    importance: str = "normal"
    is_read: bool = True
    categories: list[str] = field(default_factory=list)

    # Processing metadata
    extraction_time: datetime = field(default_factory=datetime.now)
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to grouped dictionary for serialization.

        Structure:
        - record_id, source_file, file_type, ingestion_time (top-level)
        - email_headers: addressing, threading, transport fields
        - email_body_text, email_body_html: content
        - document_metadata: timestamps, flags, language
        - attachments: list of attachment metadata
        """
        result = {
            "record_id": self.message_id,
            "source_file": self.source_pst,
            "file_type": ".pst",
            "ingestion_time": self.extraction_time.isoformat(),
            "email_headers": {
                "subject": self.subject,
                "sender": self.sender,
                "sender_email": self.sender_email,
                "recipients_to": self.recipients_to,
                "recipients_cc": self.recipients_cc,
                "recipients_bcc": self.recipients_bcc,
                "folder_path": self.folder_path,
                "message_id": self.header_message_id,
                "in_reply_to": self.in_reply_to,
                "references": self.references,
                "conversation_id": self.conversation_id,
                "return_path": self.return_path,
                "x_originating_ip": self.x_originating_ip,
            },
            "email_body_text": self.body_text,
            "email_body_html": self.body_html,
            "document_metadata": {
                "sent_time": self.sent_time.isoformat() if self.sent_time else None,
                "received_time": self.received_time.isoformat() if self.received_time else None,
                "importance": self.importance,
                "language": self.language,
                "has_attachments": self.has_attachments,
                "attachment_count": len(self.attachments),
            },
            "attachments": [
                {
                    "attachment_id": att.attachment_id,
                    "filename": att.filename,
                    "content_type": att.content_type,
                    "size": att.size,
                }
                for att in self.attachments
            ],
        }

        return result

    def get_full_text(self) -> str:
        """Get full text content for processing"""
        parts = []

        if self.subject:
            parts.append(f"Subject: {self.subject}")
        if self.sender:
            parts.append(f"From: {self.sender}")
        if self.recipients_to:
            # Handle both structured dicts and plain strings
            to_strs = []
            for r in self.recipients_to:
                if isinstance(r, dict):
                    name = r.get("name", "")
                    email = r.get("email", "")
                    to_strs.append(f"{name} <{email}>" if name else email)
                else:
                    to_strs.append(str(r))
            parts.append(f"To: {', '.join(to_strs)}")
        if self.sent_time:
            parts.append(f"Date: {self.sent_time.strftime('%Y-%m-%d %H:%M')}")

        parts.append("")  # Blank line
        parts.append(self.body_text)

        return "\n".join(parts)


class PSTExtractor:
    """
    Extract emails from PST files.

    Supports:
    - Microsoft Outlook PST files (97-2003, 2007+)
    - MBOX files (as fallback)
    - Recursive folder traversal
    - Attachment extraction
    - Encoding handling for international content
    """

    def __init__(
        self,
        extract_attachments: bool = True,
        attachment_output_dir: str | None = None,
        max_attachment_size_mb: int = 50,
        supported_attachment_types: list[str] | None = None,
    ):
        """
        Initialize the PST extractor.

        Args:
            extract_attachments: Whether to extract attachments
            attachment_output_dir: Where to save attachments
            max_attachment_size_mb: Maximum attachment size to extract
            supported_attachment_types: List of extensions to extract (None = all)
        """
        self.extract_attachments = extract_attachments
        self.attachment_output_dir = attachment_output_dir
        self.max_attachment_size = max_attachment_size_mb * 1024 * 1024
        self.supported_attachment_types = supported_attachment_types or [
            ".pdf",
            ".docx",
            ".doc",
            ".xlsx",
            ".xls",
            ".pptx",
            ".ppt",
            ".txt",
            ".csv",
            ".rtf",
        ]

        # Extraction limits
        self.max_emails = None  # Set during extract()

        # Statistics
        self.stats = {
            "total_emails": 0,
            "total_attachments": 0,
            "attachment_read_failures": 0,
            "errors": 0,
            "folders_processed": 0,
        }

    def extract(
        self,
        pst_path: str,
        progress_callback: Callable[[int, str], None] | None = None,
        max_emails: int | None = None,
    ) -> Iterator[EmailMessage]:
        """
        Extract all emails from a PST file.

        Args:
            pst_path: Path to PST file
            progress_callback: Optional callback(count, status_message)
            max_emails: Maximum number of emails to extract (None = all)

        Yields:
            EmailMessage objects
        """
        self.max_emails = max_emails
        pst_path = Path(pst_path)

        if not pst_path.exists():
            raise FileNotFoundError(f"PST file not found: {pst_path}")

        logger.info(f"Starting extraction from: {pst_path}")

        # Try different extraction methods
        try:
            # Method 1: pypff (recommended for PST)
            yield from self._extract_with_pypff(pst_path, progress_callback)
        except ImportError:
            logger.warning("pypff not available, trying libpff-python")
            try:
                # Method 2: libpff
                yield from self._extract_with_libpff(pst_path, progress_callback)
            except ImportError:
                logger.warning("libpff not available, trying extract_msg")
                try:
                    # Method 3: extract_msg (for MSG files, convert PST first)
                    yield from self._extract_with_readpst(pst_path, progress_callback)
                except Exception as e:
                    raise RuntimeError(
                        f"No PST extraction library available. "
                        f"Install one of: pypff, libpff-python, or readpst. "
                        f"Error: {e}"
                    ) from e

        if self.stats["attachment_read_failures"] > 0:
            logger.warning(
                f"PST attachment index corrupted for {self.stats['attachment_read_failures']} messages "
                f"(emails saved without attachments — PST file damage, not a code issue)"
            )
        logger.info(f"Extraction complete. Stats: {self.stats}")

    def _extract_with_pypff(
        self, pst_path: Path, progress_callback: Callable | None
    ) -> Iterator[EmailMessage]:
        """Extract using pypff library"""
        import pypff

        pst = pypff.file()
        pst.open(str(pst_path))

        try:
            root = pst.get_root_folder()
            yield from self._process_folder_pypff(root, str(pst_path), "", progress_callback)
        finally:
            pst.close()

    def _process_folder_pypff(
        self, folder, pst_path: str, folder_path: str, progress_callback: Callable | None
    ) -> Iterator[EmailMessage]:
        """Recursively process folders using pypff"""

        current_path = f"{folder_path}/{folder.name}" if folder_path else folder.name
        self.stats["folders_processed"] += 1

        # Process messages in this folder
        for i in range(folder.number_of_sub_messages):
            # Check if we've reached the limit
            if self.max_emails and self.stats["total_emails"] >= self.max_emails:
                logger.info(f"Reached max_emails limit: {self.max_emails}")
                return

            try:
                message = folder.get_sub_message(i)
                email = self._convert_pypff_message(message, pst_path, current_path)

                if email:
                    self.stats["total_emails"] += 1

                    if progress_callback and self.stats["total_emails"] % 100 == 0:
                        progress_callback(self.stats["total_emails"], f"Processing: {current_path}")

                    yield email

            except Exception as e:
                logger.warning(f"Error processing message {i} in {current_path}: {e}")
                self.stats["errors"] += 1

        # Process subfolders
        for i in range(folder.number_of_sub_folders):
            # Check limit before processing subfolder
            if self.max_emails and self.stats["total_emails"] >= self.max_emails:
                return

            try:
                subfolder = folder.get_sub_folder(i)
                yield from self._process_folder_pypff(
                    subfolder, pst_path, current_path, progress_callback
                )
            except Exception as e:
                logger.warning(f"Error processing subfolder {i}: {e}")
                self.stats["errors"] += 1

    def _parse_transport_headers(self, message) -> dict[str, Any]:
        """
        Parse RFC 2822 transport headers from a pypff message.

        Extracts structured sender/recipient info, Message-ID,
        In-Reply-To, References, Return-Path, and X-Originating-IP
        from the raw transport message headers.

        Returns:
            Dict with parsed header fields
        """
        result = {
            "sender_name": "",
            "sender_email": "",
            "recipients_to": [],
            "recipients_cc": [],
            "header_message_id": None,
            "in_reply_to": None,
            "references": [],
            "return_path": None,
            "x_originating_ip": None,
            "transport_headers_raw": None,
        }

        # Get raw transport headers from pypff message
        raw_headers = None
        if hasattr(message, "transport_headers") and message.transport_headers:
            raw_headers = self._decode_body(message.transport_headers)

        if not raw_headers:
            return result

        result["transport_headers_raw"] = raw_headers

        try:
            parser = HeaderParser()
            headers = parser.parsestr(raw_headers)

            # Parse From header
            from_header = headers.get("From", "")
            if from_header:
                name, email_addr = parseaddr(from_header)
                if email_addr:
                    result["sender_email"] = email_addr.lower()
                if name:
                    result["sender_name"] = name

            # Parse To header → structured list
            to_header = headers.get("To", "")
            if to_header:
                result["recipients_to"] = [
                    {"name": name.strip(), "email": addr.lower()}
                    for name, addr in getaddresses([to_header])
                    if addr
                ]

            # Parse Cc header → structured list
            cc_header = headers.get("Cc", "")
            if cc_header:
                result["recipients_cc"] = [
                    {"name": name.strip(), "email": addr.lower()}
                    for name, addr in getaddresses([cc_header])
                    if addr
                ]

            # Message-ID
            msg_id = headers.get("Message-ID", "") or headers.get("Message-Id", "")
            if msg_id:
                result["header_message_id"] = msg_id.strip().strip("<>")

            # In-Reply-To
            in_reply = headers.get("In-Reply-To", "")
            if in_reply:
                result["in_reply_to"] = in_reply.strip().strip("<>")

            # References (space-separated Message-IDs)
            refs = headers.get("References", "")
            if refs:
                result["references"] = [r.strip().strip("<>") for r in refs.split() if r.strip()]

            # Return-Path
            return_path = headers.get("Return-Path", "")
            if return_path:
                _, addr = parseaddr(return_path)
                result["return_path"] = addr.lower() if addr else None

            # X-Originating-IP
            x_ip = headers.get("X-Originating-IP", "") or headers.get("x-originating-ip", "")
            if x_ip:
                result["x_originating_ip"] = x_ip.strip().strip("[]")

        except Exception as e:
            logger.debug(f"Error parsing transport headers: {e}")

        return result

    def _convert_pypff_message(
        self, message, pst_path: str, folder_path: str
    ) -> EmailMessage | None:
        """Convert pypff message to EmailMessage"""
        try:
            # Generate message ID
            message_id = hashlib.md5(
                f"{pst_path}:{folder_path}:{message.subject or ''}:{message.delivery_time}".encode()
            ).hexdigest()

            # Parse transport headers first (RFC 2822)
            th = self._parse_transport_headers(message)

            # Parse sender — prefer transport header over MAPI properties
            sender = message.sender_name or th["sender_name"] or ""
            sender_email = th["sender_email"] or ""
            if not sender_email and hasattr(message, "sender_email_address"):
                sender_email = (message.sender_email_address or "").lower()

            # Parse recipients — prefer structured transport headers
            if th["recipients_to"]:
                recipients_to = th["recipients_to"]
            else:
                # Fallback to MAPI display names
                raw_to = self._parse_recipients(message, "to")
                recipients_to = [{"name": r, "email": ""} for r in raw_to]

            if th["recipients_cc"]:
                recipients_cc = th["recipients_cc"]
            else:
                raw_cc = self._parse_recipients(message, "cc")
                recipients_cc = [{"name": r, "email": ""} for r in raw_cc]

            # Parse body
            body_text = ""
            body_html = ""

            if hasattr(message, "plain_text_body") and message.plain_text_body:
                body_text = self._decode_body(message.plain_text_body)
            if hasattr(message, "html_body") and message.html_body:
                body_html = self._decode_body(message.html_body)

            # If no plain text, extract from HTML
            if not body_text and body_html:
                body_text = self._html_to_text(body_html)

            # Fallback: try RTF body (common in pre-2003 Outlook)
            if not body_text and hasattr(message, "rtf_body") and message.rtf_body:
                try:
                    from striprtf.striprtf import rtf_to_text

                    rtf_raw = self._decode_body(message.rtf_body)
                    body_text = rtf_to_text(rtf_raw)
                except Exception as e:
                    logger.debug(f"RTF body extraction failed: {e}")

            # Parse timestamps
            sent_time = None
            received_time = None

            if hasattr(message, "client_submit_time") and message.client_submit_time:
                sent_time = message.client_submit_time
            if hasattr(message, "delivery_time") and message.delivery_time:
                received_time = message.delivery_time

            # Extract attachments — isolated so corrupted PST entries don't lose the email
            attachments = []
            if self.extract_attachments:
                try:
                    num_att = message.number_of_attachments
                    if num_att > 0:
                        attachments = self._extract_attachments_pypff(message)
                except Exception:
                    # PST local descriptor table is corrupted for this message.
                    # The email body/headers are still saved; only attachments are lost.
                    self.stats["attachment_read_failures"] += 1

            return EmailMessage(
                message_id=message_id,
                source_pst=pst_path,
                folder_path=folder_path,
                subject=message.subject or "",
                sender=sender,
                sender_email=sender_email,
                recipients_to=recipients_to,
                recipients_cc=recipients_cc,
                sent_time=sent_time,
                received_time=received_time,
                body_text=body_text,
                body_html=body_html,
                conversation_id=getattr(message, "conversation_topic", None),
                in_reply_to=th["in_reply_to"],
                references=th["references"],
                header_message_id=th["header_message_id"],
                return_path=th["return_path"],
                x_originating_ip=th["x_originating_ip"],
                transport_headers_raw=th["transport_headers_raw"],
                attachments=attachments,
                has_attachments=len(attachments) > 0,
                importance=self._parse_importance(message),
            )

        except Exception as e:
            logger.warning(f"Error converting message: {e}")
            return None

    def _parse_recipients(self, message, recipient_type: str) -> list[str]:
        """Parse recipients from message"""
        recipients = []

        try:
            if recipient_type == "to":
                # Try different attribute names
                for attr in ["display_to", "to"]:
                    if hasattr(message, attr):
                        value = getattr(message, attr)
                        if value:
                            recipients = [r.strip() for r in value.split(";") if r.strip()]
                            break
            elif recipient_type == "cc":
                for attr in ["display_cc", "cc"]:
                    if hasattr(message, attr):
                        value = getattr(message, attr)
                        if value:
                            recipients = [r.strip() for r in value.split(";") if r.strip()]
                            break
        except Exception:
            pass

        return recipients

    def _decode_body(self, body: Any) -> str:
        """Decode email body handling various encodings.

        Uses ftfy to fix mojibake, lone surrogates, null bytes, and
        control characters that would break JSON serialization.
        """
        if body is None:
            return ""

        if isinstance(body, str):
            text = body
        elif isinstance(body, bytes):
            text = None
            # Try common encodings
            for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
                try:
                    text = body.decode(encoding)
                    break
                except (UnicodeDecodeError, AttributeError):
                    continue

            if text is None:
                text = body.decode("utf-8", errors="replace")
        else:
            text = str(body)

        return ftfy.fix_text(text)

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text"""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "head"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n")

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            return "\n".join(line for line in lines if line)

        except ImportError:
            # Fallback: basic regex cleaning
            import re

            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

    def _extract_attachments_pypff(self, message) -> list[Attachment]:
        """Extract attachments from pypff message"""
        attachments = []

        try:
            num_attachments = message.number_of_attachments
            if num_attachments == 0:
                return attachments

            logger.info(f"Processing {num_attachments} attachments")

            for i in range(num_attachments):
                try:
                    att = message.get_attachment(i)

                    # Get filename from record sets (MAPI property 12289 = PR_ATTACH_FILENAME)
                    filename = self._get_attachment_filename(att, i)

                    # Check file type
                    ext = Path(filename).suffix.lower()
                    if (
                        self.supported_attachment_types
                        and ext not in self.supported_attachment_types
                    ):
                        # If no extension, try to extract anyway
                        if ext:
                            logger.debug(f"Skipping unsupported attachment type: {filename}")
                            continue

                    # Get size — try multiple pypff attributes
                    size = 0
                    for size_attr in ("size", "get_size"):
                        if hasattr(att, size_attr):
                            val = getattr(att, size_attr)
                            size = val() if callable(val) else val
                            if size and size > 0:
                                break

                    # Check size limit
                    if size > self.max_attachment_size:
                        logger.debug(f"Skipping large attachment: {filename} ({size} bytes)")
                        continue

                    # Read content — try read_buffer with size, then without
                    content = b""
                    if hasattr(att, "read_buffer"):
                        try:
                            content = att.read_buffer(size) if size > 0 else att.read_buffer()
                        except TypeError:
                            # read_buffer() might not accept args — try alternate
                            try:
                                content = att.read_buffer()
                            except Exception as e2:
                                logger.debug(f"read_buffer() fallback failed: {e2}")
                        except Exception as e:
                            logger.debug(f"read_buffer failed: {e}")

                    if not content:
                        logger.debug(f"Could not read attachment content: {filename}")
                        continue

                    # Update size from actual content
                    size = len(content)

                    attachment = Attachment(
                        filename=filename,
                        content_type=self._guess_content_type(filename),
                        size=size,
                        content=content,
                    )

                    # Save if output directory specified
                    if self.attachment_output_dir:
                        try:
                            attachment.save(self.attachment_output_dir)
                            logger.info(f"Saved attachment: {filename} ({size} bytes)")
                        except Exception as e:
                            logger.warning(f"Failed to save attachment {filename}: {e}")

                    attachments.append(attachment)
                    self.stats["total_attachments"] += 1

                except Exception as e:
                    logger.debug(f"Error extracting attachment {i}: {e}")

        except Exception as e:
            logger.debug(f"Error accessing attachments: {e}")

        return attachments

    def _get_attachment_filename(self, attachment, index: int) -> str:
        """Extract filename from attachment record sets"""
        # MAPI property IDs for attachment filename
        # 12289 = PR_ATTACH_FILENAME (short filename)
        # 14084 = PR_ATTACH_LONG_FILENAME
        # 14085 = PR_DISPLAY_NAME
        filename_props = [14084, 12289, 14085]

        try:
            for rs_idx in range(attachment.number_of_record_sets):
                record_set = attachment.get_record_set(rs_idx)

                for entry_idx in range(record_set.number_of_entries):
                    try:
                        entry = record_set.get_entry(entry_idx)
                        entry_type = entry.entry_type

                        if entry_type in filename_props:
                            if hasattr(entry, "get_data_as_string"):
                                filename = entry.get_data_as_string()
                                if filename:
                                    return filename
                    except Exception:
                        continue

        except Exception as e:
            logger.debug(f"Error getting attachment filename: {e}")

        return f"attachment_{index}"

    def _guess_content_type(self, filename: str) -> str:
        """Guess content type from filename extension"""
        ext = Path(filename).suffix.lower()
        content_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".html": "text/html",
            ".htm": "text/html",
            ".rtf": "application/rtf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }
        return content_types.get(ext, "application/octet-stream")

    def _parse_importance(self, message) -> str:
        """Parse message importance/priority"""
        try:
            if hasattr(message, "importance"):
                importance = message.importance
                if importance == 0:
                    return "low"
                elif importance == 2:
                    return "high"
        except Exception:
            pass
        return "normal"

    def _extract_with_libpff(
        self, pst_path: Path, progress_callback: Callable | None
    ) -> Iterator[EmailMessage]:
        """Fallback extraction using libpff"""
        # Similar to pypff but using libpff bindings
        raise ImportError("libpff extraction not implemented")

    def _extract_with_readpst(
        self, pst_path: Path, progress_callback: Callable | None
    ) -> Iterator[EmailMessage]:
        """Fallback extraction using readpst command-line tool"""
        import email
        import subprocess
        import tempfile
        from email import policy

        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run readpst
            result = subprocess.run(
                ["readpst", "-e", "-o", temp_dir, str(pst_path)], capture_output=True, text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"readpst failed: {result.stderr}")

            # Process extracted files
            for eml_path in Path(temp_dir).rglob("*.eml"):
                try:
                    with open(eml_path, "rb") as f:
                        msg = email.message_from_binary_file(f, policy=policy.default)

                    yield self._convert_email_message(msg, str(pst_path), str(eml_path.parent))
                    self.stats["total_emails"] += 1

                except Exception as e:
                    logger.warning(f"Error processing {eml_path}: {e}")
                    self.stats["errors"] += 1

    def _convert_email_message(self, msg, pst_path: str, folder_path: str) -> EmailMessage:
        """Convert standard email.message to EmailMessage"""
        # Parse Message-ID
        header_message_id = msg.get("Message-ID", "")
        if header_message_id:
            header_message_id = header_message_id.strip().strip("<>")

        # Generate internal ID
        message_id = hashlib.md5(
            f"{pst_path}:{msg.get('Subject', '')}:{msg.get('Date', '')}".encode()
        ).hexdigest()

        # Parse sender using email.utils
        from_header = msg.get("From", "")
        sender_name, sender_email = parseaddr(from_header)
        sender_email = sender_email.lower() if sender_email else ""

        # Parse structured recipients
        to_header = msg.get("To", "")
        recipients_to = (
            [
                {"name": name.strip(), "email": addr.lower()}
                for name, addr in getaddresses([to_header])
                if addr
            ]
            if to_header
            else []
        )

        cc_header = msg.get("Cc", "")
        recipients_cc = (
            [
                {"name": name.strip(), "email": addr.lower()}
                for name, addr in getaddresses([cc_header])
                if addr
            ]
            if cc_header
            else []
        )

        # Parse In-Reply-To
        in_reply_to = msg.get("In-Reply-To", "")
        if in_reply_to:
            in_reply_to = in_reply_to.strip().strip("<>")

        # Parse References
        refs_header = msg.get("References", "")
        references = (
            [r.strip().strip("<>") for r in refs_header.split() if r.strip()] if refs_header else []
        )

        # Return-Path
        return_path_header = msg.get("Return-Path", "")
        return_path = None
        if return_path_header:
            _, rp_addr = parseaddr(return_path_header)
            return_path = rp_addr.lower() if rp_addr else None

        # X-Originating-IP
        x_ip = msg.get("X-Originating-IP", "")
        x_originating_ip = x_ip.strip().strip("[]") if x_ip else None

        # Parse date
        sent_time = None
        if msg.get("Date"):
            try:
                sent_time = parsedate_to_datetime(msg.get("Date"))
            except Exception:
                pass

        # Get body
        body_text = ""
        body_html = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    body_text = part.get_content()
                elif content_type == "text/html":
                    body_html = part.get_content()
        else:
            body_text = msg.get_content()

        if not body_text and body_html:
            body_text = self._html_to_text(body_html)

        return EmailMessage(
            message_id=message_id,
            source_pst=pst_path,
            folder_path=folder_path,
            subject=msg.get("Subject", ""),
            sender=sender_name or sender_email,
            sender_email=sender_email,
            recipients_to=recipients_to,
            recipients_cc=recipients_cc,
            sent_time=sent_time,
            body_text=body_text,
            body_html=body_html,
            conversation_id=msg.get("Thread-Index"),
            in_reply_to=in_reply_to or None,
            references=references,
            header_message_id=header_message_id or None,
            return_path=return_path,
            x_originating_ip=x_originating_ip,
        )

    def get_stats(self) -> dict[str, int]:
        """Get extraction statistics"""
        return self.stats.copy()


# Convenience function
def extract_pst(pst_path: str, output_dir: str | None = None) -> list[EmailMessage]:
    """
    Extract all emails from a PST file.

    Args:
        pst_path: Path to PST file
        output_dir: Optional directory for attachments

    Returns:
        List of EmailMessage objects
    """
    extractor = PSTExtractor(extract_attachments=True, attachment_output_dir=output_dir)

    return list(extractor.extract(pst_path))
