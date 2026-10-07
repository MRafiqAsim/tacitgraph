"""
Disclaimer Remover - Removes legal boilerplate from emails

Removes:
- Confidentiality notices
- Legal disclaimers
- Privacy policy statements

Keeps:
- Email thread info (From, Sent, To, Subject)
- Actual email content
- Signatures with names (useful for context)
"""

import re
from typing import List, Tuple


# Common disclaimer patterns (case-insensitive)
DISCLAIMER_PATTERNS = [
    # Confidentiality notices - match entire block
    r"Important\s+Notice:.*?(?:strictly\s+prohibited|unauthorized).*?(?:\n\n|\n*$)",
    r"(?:This|The)\s+(?:e-?mail|message|communication).*?(?:confidential|privileged).*?(?:prohibited|delete|destroy|notify).*?(?:\n\n|\n*$)",
    r"If\s+you\s+(?:are\s+not|have\s+received).*?(?:intended\s+recipient|in\s+error).*?(?:delete|notify|destroy).*?(?:\n\n|\n*$)",

    # Legal disclaimers
    r"(?:DISCLAIMER|LEGAL\s+NOTICE|CONFIDENTIALITY\s+NOTICE):?.*?(?:\n\n|\Z)",
    r"This\s+(?:electronic\s+)?(?:mail|message)\s+is\s+intended\s+only\s+for.*?(?:strictly\s+prohibited|unauthorized).*?(?:\n\n|\n*$)",

    # Privacy statements
    r"(?:Any|The)\s+(?:views|opinions).*?(?:author|sender).*?(?:do\s+not|does\s+not).*?(?:represent|reflect).*?(?:\n\n|\n*$)",
    r"(?:Please\s+)?(?:consider|think).*?(?:environment|planet).*?(?:before\s+printing).*?(?:\n\n|\n*$)",

    # Virus disclaimers
    r"(?:This\s+)?(?:e-?mail|message).*?(?:virus|malware).*?(?:scanned|checked).*?(?:\n\n|\n*$)",
    r"(?:We|The\s+company).*?(?:no\s+(?:liability|responsibility)).*?(?:virus|damage).*?(?:\n\n|\n*$)",
]


class DisclaimerRemover:
    """Remove legal disclaimers from email text"""

    def __init__(self, custom_patterns: List[str] = None):
        """
        Initialize the disclaimer remover.

        Args:
            custom_patterns: Additional regex patterns to remove
        """
        self.patterns = [re.compile(p, re.IGNORECASE | re.DOTALL)
                        for p in DISCLAIMER_PATTERNS]

        if custom_patterns:
            self.patterns.extend([
                re.compile(p, re.IGNORECASE | re.DOTALL)
                for p in custom_patterns
            ])

    def remove(self, text: str) -> Tuple[str, int]:
        """
        Remove disclaimers from text.

        Args:
            text: Email text

        Returns:
            (cleaned_text, number_of_removals)
        """
        if not text:
            return text, 0

        cleaned = text
        removal_count = 0

        for pattern in self.patterns:
            matches = pattern.findall(cleaned)
            if matches:
                removal_count += len(matches)
                cleaned = pattern.sub('', cleaned)

        # Clean up extra whitespace left behind
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()

        return cleaned, removal_count

    def remove_from_email(self, email_data: dict) -> dict:
        """
        Remove disclaimers from email body.

        Args:
            email_data: Email dictionary with body_text field

        Returns:
            Email dictionary with cleaned body_text
        """
        if 'body_text' in email_data and email_data['body_text']:
            cleaned, count = self.remove(email_data['body_text'])
            email_data['body_text'] = cleaned
            email_data['disclaimers_removed'] = count

        return email_data


# Convenience function
def remove_disclaimers(text: str) -> str:
    """
    Remove legal disclaimers from text.

    Args:
        text: Text containing potential disclaimers

    Returns:
        Cleaned text
    """
    remover = DisclaimerRemover()
    cleaned, _ = remover.remove(text)
    return cleaned
