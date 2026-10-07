"""
Temporal date filtering for retrieval.

Extracts date ranges from natural language queries and filters
chunks by their email timestamps (sent_timestamp / received_timestamp).

Usage:
    date_range = extract_date_range("What happened in Dec 2015 to Jan 2016?")
    # DateRange(start="2015-12-01", end="2016-01-31")

    filtered = filter_chunks_by_date(chunks, date_range)
"""

import calendar
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# Month name → number lookup
MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

MONTH_PATTERN = "|".join(MONTH_MAP.keys())


@dataclass
class DateRange:
    """A date range for temporal filtering. Dates are ISO format strings (YYYY-MM-DD)."""
    start: str = ""  # e.g. "2015-12-01"
    end: str = ""    # e.g. "2016-01-31"

    @property
    def is_set(self) -> bool:
        return bool(self.start or self.end)

    def __str__(self) -> str:
        if self.start and self.end:
            return f"{self.start} to {self.end}"
        if self.start:
            return f"from {self.start}"
        if self.end:
            return f"until {self.end}"
        return "none"


def _month_start(year: int, month: int) -> str:
    """First day of a month as ISO string."""
    return f"{year:04d}-{month:02d}-01"


def _month_end(year: int, month: int) -> str:
    """Last day of a month as ISO string."""
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{last_day:02d}"


def _year_start(year: int) -> str:
    return f"{year:04d}-01-01"


def _year_end(year: int) -> str:
    return f"{year:04d}-12-31"


def extract_date_range(query: str) -> Optional[DateRange]:
    """
    Extract a date range from a natural language query.

    Supported patterns:
        "Dec 2015 to Jan 2016"          → 2015-12-01 to 2016-01-31
        "from December 2015 to January 2016"
        "between 2015 and 2016"          → 2015-01-01 to 2016-12-31
        "in 2015"                         → 2015-01-01 to 2015-12-31
        "in March 2016"                   → 2016-03-01 to 2016-03-31
        "before March 2016"              → start=None, end=2016-03-31
        "after January 2015"             → start=2015-01-01, end=None
        "since 2015"                     → start=2015-01-01, end=None

    Returns None if no date pattern found.
    """
    q = query.lower()

    # Pattern 1: "Month Year to/- Month Year"
    range_pat = rf"({MONTH_PATTERN})\s+(\d{{4}})\s+(?:to|through|till|until|-|–)\s+({MONTH_PATTERN})\s+(\d{{4}})"
    m = re.search(range_pat, q)
    if m:
        m1, y1, m2, y2 = MONTH_MAP[m.group(1)], int(m.group(2)), MONTH_MAP[m.group(3)], int(m.group(4))
        return DateRange(start=_month_start(y1, m1), end=_month_end(y2, m2))

    # Pattern 2: "from/between Month Year to/and Month Year"
    range_pat2 = rf"(?:from|between)\s+({MONTH_PATTERN})\s+(\d{{4}})\s+(?:to|and|through|-|–)\s+({MONTH_PATTERN})\s+(\d{{4}})"
    m = re.search(range_pat2, q)
    if m:
        m1, y1, m2, y2 = MONTH_MAP[m.group(1)], int(m.group(2)), MONTH_MAP[m.group(3)], int(m.group(4))
        return DateRange(start=_month_start(y1, m1), end=_month_end(y2, m2))

    # Pattern 3: "between Year and Year"
    m = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", q)
    if m:
        return DateRange(start=_year_start(int(m.group(1))), end=_year_end(int(m.group(2))))

    # Pattern 4: "Year to/- Year"
    m = re.search(r"(\d{4})\s+(?:to|through|till|-|–)\s+(\d{4})", q)
    if m:
        return DateRange(start=_year_start(int(m.group(1))), end=_year_end(int(m.group(2))))

    # Pattern 5: "before/until Month Year"
    m = re.search(rf"(?:before|until|till)\s+({MONTH_PATTERN})\s+(\d{{4}})", q)
    if m:
        return DateRange(end=_month_end(int(m.group(2)), MONTH_MAP[m.group(1)]))

    # Pattern 6: "after/since Month Year"
    m = re.search(rf"(?:after|since|from)\s+({MONTH_PATTERN})\s+(\d{{4}})", q)
    if m:
        return DateRange(start=_month_start(int(m.group(2)), MONTH_MAP[m.group(1)]))

    # Pattern 7: "before/until Year"
    m = re.search(r"(?:before|until|till)\s+(\d{4})", q)
    if m:
        return DateRange(end=_year_end(int(m.group(1))))

    # Pattern 8: "after/since Year"
    m = re.search(r"(?:after|since|from)\s+(\d{4})", q)
    if m:
        return DateRange(start=_year_start(int(m.group(1))))

    # Pattern 9: "in Month Year"
    m = re.search(rf"in\s+({MONTH_PATTERN})\s+(\d{{4}})", q)
    if m:
        month, year = MONTH_MAP[m.group(1)], int(m.group(2))
        return DateRange(start=_month_start(year, month), end=_month_end(year, month))

    # Pattern 10: "in Year"
    m = re.search(r"in\s+(\d{4})", q)
    if m:
        year = int(m.group(1))
        return DateRange(start=_year_start(year), end=_year_end(year))

    return None


def _get_chunk_date(chunk: Dict) -> str:
    """Extract the best date string (YYYY-MM-DD) from a chunk."""
    ts = chunk.get("received_timestamp", "") or chunk.get("sent_timestamp", "")
    return ts[:10] if ts else ""


def filter_chunks_by_date(chunks: List[Dict], date_range: Optional[DateRange]) -> List[Dict]:
    """
    Filter chunks to only those within the date range.

    Chunks without timestamps are kept (don't penalize missing data).
    Returns original list if date_range is None or empty.
    """
    if not date_range or not date_range.is_set:
        return chunks

    filtered = []
    for chunk in chunks:
        chunk_date = _get_chunk_date(chunk)

        # Keep chunks without timestamps (backward compatibility)
        if not chunk_date:
            filtered.append(chunk)
            continue

        if date_range.start and chunk_date < date_range.start:
            continue
        if date_range.end and chunk_date > date_range.end:
            continue

        filtered.append(chunk)

    return filtered
