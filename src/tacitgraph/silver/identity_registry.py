"""
Identity Registry

Maintains a mapping of email addresses to canonical identities,
enabling consistent pseudonymization across the entire corpus.

Name variations (e.g., "Jane", "Jane Doe", "Jane A. Doe") are
linked to a single identity via their shared email address.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Identity:
    """A single person identity with all known variations"""

    email_address: str  # Primary key (lowercase)
    canonical_name: str  # Longest observed name
    aliases: set[str] = field(default_factory=set)  # All name variations
    pseudonym_id: str = ""  # e.g., "PERSON_001"
    email_count: int = 0  # Emails sent by this person

    def to_dict(self) -> dict[str, Any]:
        return {
            "email_address": self.email_address,
            "canonical_name": self.canonical_name,
            "aliases": sorted(self.aliases),
            "pseudonym_id": self.pseudonym_id,
            "email_count": self.email_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Identity":
        return cls(
            email_address=data["email_address"],
            canonical_name=data["canonical_name"],
            aliases=set(data.get("aliases", [])),
            pseudonym_id=data.get("pseudonym_id", ""),
            email_count=data.get("email_count", 0),
        )


class IdentityRegistry:
    """
    Registry of known identities built from Bronze layer emails.

    Enables:
    - Consistent pseudonymization: same person → same PERSON_ID everywhere
    - Name resolution: fuzzy lookup of name variants
    - PII validation: boost confidence for known names, suppress false positives
    """

    def __init__(self):
        self._by_email: dict[str, Identity] = {}  # email → Identity
        self._by_name: dict[str, Identity] = {}  # normalized_name → Identity
        self._pseudonym_counter: int = 0
        self._build_stats: dict[str, Any] = {}

    @property
    def identity_count(self) -> int:
        return len(self._by_email)

    def register_identity(self, email: str, name: str) -> Identity:
        """
        Register or update an identity.

        If the email already exists, merges the name as an alias and
        updates canonical_name if the new name is longer.

        Args:
            email: Email address (will be lowercased)
            name: Display name

        Returns:
            The Identity object
        """
        email = email.strip().lower()
        name = name.strip().strip("'\"")

        if not email:
            return None

        if email in self._by_email:
            identity = self._by_email[email]
            identity.email_count += 1
            if name:
                identity.aliases.add(name)
                # Update canonical_name if new name is longer
                if len(name) > len(identity.canonical_name):
                    identity.canonical_name = name
                # Index the new name variant
                norm = self._normalize_name(name)
                if norm:
                    self._by_name[norm] = identity
        else:
            self._pseudonym_counter += 1
            identity = Identity(
                email_address=email,
                canonical_name=name or email.split("@")[0],
                aliases={name} if name else set(),
                pseudonym_id=f"PERSON_{self._pseudonym_counter:03d}",
                email_count=1,
            )
            self._by_email[email] = identity
            if name:
                norm = self._normalize_name(name)
                if norm:
                    self._by_name[norm] = identity

        return identity

    def build_from_bronze(self, bronze_path: str) -> dict[str, Any]:
        """
        Scan all Bronze layer emails and build the identity registry.

        Extracts sender and recipient email+name pairs from the
        enhanced Bronze JSON files.

        Args:
            bronze_path: Path to the Bronze layer root

        Returns:
            Build statistics
        """
        start = time.time()
        bronze = Path(bronze_path)
        emails_dir = bronze / "emails"

        if not emails_dir.exists():
            logger.warning(f"Emails directory not found: {emails_dir}")
            return {"error": "emails directory not found"}

        emails_scanned = 0

        for json_file in emails_dir.rglob("*.json"):
            if "metadata" in str(json_file):
                continue

            try:
                with open(json_file, encoding="utf-8") as f:
                    email = json.load(f)

                emails_scanned += 1

                headers = email.get("email_headers", {})

                # Register sender
                sender_email = headers.get("sender_email", "")
                sender_name = headers.get("sender", "")
                if sender_email:
                    self.register_identity(sender_email, sender_name)

                # Register To recipients
                for r in headers.get("recipients_to", []):
                    if isinstance(r, dict):
                        r_email = r.get("email", "")
                        r_name = r.get("name", "")
                        if r_email:
                            self.register_identity(r_email, r_name)

                # Register Cc recipients
                for r in headers.get("recipients_cc", []):
                    if isinstance(r, dict):
                        r_email = r.get("email", "")
                        r_name = r.get("name", "")
                        if r_email:
                            self.register_identity(r_email, r_name)

            except Exception as e:
                logger.debug(f"Error reading {json_file}: {e}")

        elapsed = time.time() - start
        total_aliases = sum(len(i.aliases) for i in self._by_email.values())

        self._build_stats = {
            "total_identities": self.identity_count,
            "total_aliases": total_aliases,
            "total_emails_scanned": emails_scanned,
            "build_time_seconds": round(elapsed, 2),
        }

        logger.info(
            f"Identity registry built: {self.identity_count} identities, "
            f"{total_aliases} aliases from {emails_scanned} emails in {elapsed:.1f}s"
        )

        return self._build_stats

    def lookup_by_email(self, email: str) -> Identity | None:
        """Look up identity by email address."""
        return self._by_email.get(email.strip().lower())

    def lookup_by_name(self, name: str) -> Identity | None:
        """
        Look up identity by name with fuzzy matching.

        Strategy:
        1. Exact normalized match
        2. Check if name is a substring of any canonical name or alias
        3. Levenshtein distance ≤ 2 for names with 4+ characters

        Args:
            name: Name to look up

        Returns:
            Identity if found, None otherwise
        """
        name = name.strip()
        if not name:
            return None

        norm = self._normalize_name(name)
        if not norm:
            return None

        # 1. Exact normalized match
        if norm in self._by_name:
            return self._by_name[norm]

        # 2. Substring match (for partial names like "Jane" matching "Jane Doe")
        for identity in self._by_email.values():
            norm_canonical = self._normalize_name(identity.canonical_name)
            if norm_canonical and (norm in norm_canonical or norm_canonical in norm):
                return identity
            for alias in identity.aliases:
                norm_alias = self._normalize_name(alias)
                if norm_alias and (norm in norm_alias or norm_alias in norm):
                    return identity

        # 3. Levenshtein distance ≤ 2 (only for names with 4+ chars)
        if len(norm) >= 4:
            for key, identity in self._by_name.items():
                if abs(len(key) - len(norm)) <= 2 and self._levenshtein(norm, key) <= 2:
                    return identity

        return None

    def get_pseudonym(self, email_or_name: str) -> str | None:
        """
        Get the stable pseudonym for an email or name.

        Args:
            email_or_name: Email address or display name

        Returns:
            Pseudonym string (e.g., "PERSON_001") or None
        """
        # Try email lookup first
        if "@" in email_or_name:
            identity = self.lookup_by_email(email_or_name)
        else:
            identity = self.lookup_by_name(email_or_name)

        return identity.pseudonym_id if identity else None

    def get_all_known_names(self) -> set[str]:
        """Get all known names (canonical + aliases) for validation."""
        names = set()
        for identity in self._by_email.values():
            names.add(identity.canonical_name)
            names.update(identity.aliases)
        return names

    def save(self, path: str) -> None:
        """Save registry to JSON file."""
        data = {
            "version": "1.0",
            "build_timestamp": datetime.now().isoformat(),
            "identities": [i.to_dict() for i in self._by_email.values()],
            "build_stats": self._build_stats,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Registry saved to {path} ({self.identity_count} identities)")

    def load(self, path: str) -> None:
        """Load registry from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._by_email.clear()
        self._by_name.clear()
        self._pseudonym_counter = 0

        for item in data.get("identities", []):
            identity = Identity.from_dict(item)
            self._by_email[identity.email_address] = identity

            # Rebuild name index
            for alias in identity.aliases:
                norm = self._normalize_name(alias)
                if norm:
                    self._by_name[norm] = identity

            norm_canonical = self._normalize_name(identity.canonical_name)
            if norm_canonical:
                self._by_name[norm_canonical] = identity

            # Track max pseudonym counter
            match = re.match(r"PERSON_(\d+)", identity.pseudonym_id)
            if match:
                num = int(match.group(1))
                if num > self._pseudonym_counter:
                    self._pseudonym_counter = num

        self._build_stats = data.get("build_stats", {})
        logger.info(f"Registry loaded from {path} ({self.identity_count} identities)")

    def report(self) -> str:
        """Generate a human-readable report of the registry."""
        lines = [
            "Identity Registry Report",
            "========================",
            f"Total identities: {self.identity_count}",
            f"Total aliases: {sum(len(i.aliases) for i in self._by_email.values())}",
            "",
            "Top contacts by email count:",
        ]

        sorted_identities = sorted(
            self._by_email.values(), key=lambda i: i.email_count, reverse=True
        )

        for identity in sorted_identities[:20]:
            aliases_str = ", ".join(sorted(identity.aliases)[:5])
            lines.append(
                f"  {identity.pseudonym_id}: {identity.canonical_name} "
                f"<{identity.email_address}> ({identity.email_count} emails) "
                f"aliases=[{aliases_str}]"
            )

        return "\n".join(lines)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a name for matching: lowercase, strip non-alpha, collapse spaces."""
        if not name:
            return ""
        # Remove non-alphanumeric except spaces
        normalized = re.sub(r"[^a-zA-Z\s]", "", name)
        # Collapse whitespace and lowercase
        return " ".join(normalized.lower().split())

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return IdentityRegistry._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]
