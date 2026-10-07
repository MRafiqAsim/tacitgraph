"""Quoted-reply detection, header parsing and Bronze matching."""

import pytest

from tacitgraph.bronze.reply_splitter import (
    EmailSegment,
    build_bronze_index,
    match_quoted_replies,
    normalize_subject,
    parse_date_flexible,
    split_replies,
)

OUTLOOK_REPLY = """Hi Jane,

Looks good, approved.

-----Original Message-----
From: John Smith <john.smith@example.com>
Sent: Monday, March 4, 2024 10:15 AM
To: Jane Doe
Subject: RE: Project Atlas budget

Can you approve the budget for Project Atlas?"""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Monday, March 4, 2024 10:15 AM", "2024-03-04"),
        ("2024-03-04T10:15:00", "2024-03-04"),
        ("4 March 2024", "2024-03-04"),
        ("", ""),
        ("no date here", ""),
    ],
)
def test_parse_date_flexible(raw, expected):
    assert parse_date_flexible(raw) == expected


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("RE: Project Atlas", "project atlas"),
        ("fw: Project Atlas", "project atlas"),
        ("Antw: Project Atlas", "project atlas"),
        ("  Project Atlas  ", "project atlas"),
        ("", ""),
        # Only a single leading prefix is stripped.
        ("RE: FW: Project Atlas", "fw: project atlas"),
    ],
)
def test_normalize_subject(subject, expected):
    assert normalize_subject(subject) == expected


def test_empty_body_returns_single_primary_segment():
    segments = split_replies("   ")
    assert len(segments) == 1
    assert segments[0].is_primary
    assert segments[0].status == "primary"


def test_body_without_replies_is_one_primary_segment():
    segments = split_replies("  Just a short update on Project Atlas.  ")
    assert segments == [EmailSegment(text="Just a short update on Project Atlas.", is_primary=True)]


def test_outlook_original_message_is_split_and_headers_parsed():
    primary, reply = split_replies(OUTLOOK_REPLY)

    assert primary.is_primary
    assert primary.text == "Hi Jane,\n\nLooks good, approved."

    assert not reply.is_primary
    assert reply.status == "orphan"
    assert reply.parsed_sender == "John Smith"  # email address stripped
    assert reply.parsed_date == "2024-03-04"
    assert reply.parsed_subject == "RE: Project Atlas budget"
    assert reply.text.startswith("-----Original Message-----")


def test_on_date_wrote_marker_is_detected():
    body = (
        "Thanks!\n\n"
        "On Mon, Mar 4, 2024 at 10:15 AM John Smith wrote:\n"
        "> Can you approve the Project Atlas budget please?"
    )
    primary, reply = split_replies(body)
    assert primary.text == "Thanks!"
    assert reply.text.startswith("On Mon, Mar 4, 2024")
    assert reply.parsed_sender == ""  # no From: header in this style


def test_bare_from_sent_header_block_is_detected():
    body = (
        "Short note\n\nFrom: John Smith\nSent: 2024-03-04\nSubject: Atlas\n\n"
        "Older content that is long enough."
    )
    primary, reply = split_replies(body)
    assert primary.text == "Short note"
    assert (reply.parsed_sender, reply.parsed_date, reply.parsed_subject) == (
        "John Smith",
        "2024-03-04",
        "Atlas",
    )


def test_dutch_reply_marker_and_headers():
    body = (
        "Akkoord.\n\n"
        "-----Oorspronkelijk bericht-----\n"
        "Van: Jane Doe\n"
        "Verzonden: 2024-03-04\n"
        "Onderwerp: Project Atlas planning\n\n"
        "Kun je de planning bekijken?"
    )
    _, reply = split_replies(body)
    assert reply.parsed_sender == "Jane Doe"
    assert reply.parsed_date == "2024-03-04"
    assert reply.parsed_subject == "Project Atlas planning"


def _bronze_email(record_id, subject, sender, sent):
    return {
        "record_id": record_id,
        "email_headers": {"subject": subject, "sender": sender},
        "document_metadata": {"sent_timestamp": sent},
    }


def test_build_bronze_index_keys_and_skips_incomplete_records():
    index = build_bronze_index(
        [
            _bronze_email("r1", "RE: Project Atlas budget", "John Smith", "2024-03-04T10:15:00"),
            _bronze_email("r2", "", "John Smith", "2024-03-04"),  # no subject
            _bronze_email("", "Atlas", "John Smith", "2024-03-04"),  # no record id
        ]
    )
    assert index == {"project atlas budget|john smith|2024-03-04": "r1"}


def test_match_quoted_replies_exact_and_orphan():
    index = build_bronze_index(
        [_bronze_email("r1", "Project Atlas budget", "John Smith", "2024-03-04")]
    )
    segments = split_replies(OUTLOOK_REPLY)
    orphan = EmailSegment(
        text="x" * 30,
        is_primary=False,
        parsed_sender="Somebody Else",
        parsed_date="2020-01-01",
        parsed_subject="Unrelated",
        status="orphan",
    )

    match_quoted_replies([*segments, orphan], index)

    assert segments[0].status == "primary"
    assert segments[1].status == "matched"
    assert segments[1].bronze_match_id == "r1"
    assert orphan.status == "orphan"
    assert orphan.bronze_match_id == ""


def test_match_uses_parent_subject_and_first_name_fallback():
    index = build_bronze_index(
        [_bronze_email("r9", "Project Atlas budget", "John Smith", "2024-03-04")]
    )
    segment = EmailSegment(
        text="x" * 30,
        is_primary=False,
        parsed_sender="John",  # first name only
        parsed_date="2024-03-04",
        parsed_subject="",  # falls back to the parent subject
        status="orphan",
    )

    match_quoted_replies([segment], index, parent_subject="RE: Project Atlas budget")

    assert segment.status == "matched"
    assert segment.bronze_match_id == "r9"
