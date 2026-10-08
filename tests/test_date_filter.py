"""Natural-language date range extraction and chunk filtering."""

import pytest

from tacitgraph.retrieval.date_filter import DateRange, extract_date_range, filter_chunks_by_date


@pytest.mark.parametrize(
    ("query", "start", "end"),
    [
        ("What happened in Dec 2015 to Jan 2016?", "2015-12-01", "2016-01-31"),
        ("from December 2015 to January 2016", "2015-12-01", "2016-01-31"),
        ("Updates on Project Atlas between March 2020 and May 2020", "2020-03-01", "2020-05-31"),
        ("Who joined the Lisbon office between 2015 and 2016?", "2015-01-01", "2016-12-31"),
        ("Releases 2018 to 2019", "2018-01-01", "2019-12-31"),
        ("Budget emails in February 2024", "2024-02-01", "2024-02-29"),  # leap year
        ("Budget emails in February 2023", "2023-02-01", "2023-02-28"),
        ("What did Jane Doe send in 2017?", "2017-01-01", "2017-12-31"),
        ("Anything before March 2016", "", "2016-03-31"),
        ("Anything after January 2015", "2015-01-01", ""),
        ("Changes since 2019", "2019-01-01", ""),
        ("Contracts until 2012", "", "2012-12-31"),
    ],
)
def test_extract_date_range(query, start, end):
    assert extract_date_range(query) == DateRange(start=start, end=end)


def test_extract_date_range_returns_none_without_dates():
    assert extract_date_range("Who leads Project Atlas?") is None


def test_month_abbreviations_are_case_insensitive():
    assert extract_date_range("SEPT 2021 - OCT 2021") == DateRange("2021-09-01", "2021-10-31")


@pytest.mark.parametrize(
    ("date_range", "expected"),
    [
        (DateRange("2015-12-01", "2016-01-31"), "2015-12-01 to 2016-01-31"),
        (DateRange(start="2015-01-01"), "from 2015-01-01"),
        (DateRange(end="2016-03-31"), "until 2016-03-31"),
        (DateRange(), "none"),
    ],
)
def test_date_range_str(date_range, expected):
    assert str(date_range) == expected


def test_date_range_is_set():
    assert DateRange(start="2015-01-01").is_set
    assert not DateRange().is_set


CHUNKS = [
    {"chunk_id": "a", "sent_timestamp": "2015-11-30T09:00:00"},
    {"chunk_id": "b", "sent_timestamp": "2015-12-01T09:00:00"},
    {"chunk_id": "c", "received_timestamp": "2016-01-31T23:59:00"},
    {"chunk_id": "d", "sent_timestamp": "2016-02-01T00:00:00"},
    {"chunk_id": "e"},  # no timestamp: always kept
]


def _ids(chunks):
    return [c["chunk_id"] for c in chunks]


def test_filter_chunks_by_closed_range_is_inclusive():
    result = filter_chunks_by_date(CHUNKS, DateRange("2015-12-01", "2016-01-31"))
    assert _ids(result) == ["b", "c", "e"]


def test_filter_chunks_by_open_ranges():
    assert _ids(filter_chunks_by_date(CHUNKS, DateRange(start="2016-01-01"))) == ["c", "d", "e"]
    assert _ids(filter_chunks_by_date(CHUNKS, DateRange(end="2015-12-31"))) == ["a", "b", "e"]


def test_received_timestamp_takes_precedence_over_sent():
    chunk = {"sent_timestamp": "2014-01-01T00:00:00", "received_timestamp": "2016-06-01T00:00:00"}
    assert filter_chunks_by_date([chunk], DateRange(start="2016-01-01")) == [chunk]


@pytest.mark.parametrize("date_range", [None, DateRange()])
def test_filter_without_range_returns_input(date_range):
    assert filter_chunks_by_date(CHUNKS, date_range) is CHUNKS
