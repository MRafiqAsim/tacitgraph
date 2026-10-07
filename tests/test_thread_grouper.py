"""Email thread grouping: RFC 2822 chains, conversation IDs and subjects."""

import json
from datetime import datetime

import pytest

from tacitgraph.bronze.thread_grouper import EmailThread, ThreadGrouper, group_emails_into_threads


def make_email(record_id, subject, *, sender="Jane Doe", sent=None, body="", **headers):
    return {
        "record_id": record_id,
        "email_headers": {"subject": subject, "sender": sender, **headers},
        "document_metadata": {"sent_time": sent},
        "email_body_text": body,
    }


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("RE: FW: Re: Project Atlas", "project atlas"),  # nested prefixes
        ("Fwd: AW: Budget", "budget"),
        ("WG: SV: Antw: Planning", "planning"),
        ("Report: Q3", "report: q3"),  # words that merely start with a prefix letter
        ("", ""),
    ],
)
def test_normalize_subject(subject, expected):
    assert ThreadGrouper()._normalize_subject(subject) == expected


def test_rfc2822_references_win_over_subject():
    emails = [
        make_email("a", "Project Atlas kickoff", message_id="<1@example.com>"),
        make_email(
            "b",
            "RE: Project Atlas kickoff",
            message_id="<2@example.com>",
            in_reply_to="<1@example.com>",
        ),
        make_email(
            "c",
            "Totally different subject",
            message_id="<3@example.com>",
            references=["<1@example.com>", "<2@example.com>"],
        ),
    ]
    threads = ThreadGrouper().group_emails(iter(emails))

    assert len(threads) == 1
    thread = threads[0]
    assert thread.conversation_id.startswith("rfc:")
    assert [e["record_id"] for e in thread.emails] == ["a", "b", "c"]
    assert thread.subject == "project atlas kickoff"


def test_conversation_id_groups_across_subjects():
    emails = [
        make_email("f", "Status", conversation_id="CONV1"),
        make_email("g", "Other subject", conversation_id="CONV1"),
    ]
    (thread,) = ThreadGrouper().group_emails(iter(emails))
    assert thread.conversation_id == "conv:CONV1"
    assert thread.email_count == 2
    assert thread.is_thread


def test_subject_matching_and_fallback_to_record_id():
    emails = [
        make_email("d", "Lunch"),
        make_email("e", "RE: Lunch"),
        make_email("h", ""),
    ]
    threads = {t.conversation_id: t for t in ThreadGrouper().group_emails(iter(emails))}

    assert set(threads) == {"subj:lunch", "msg:h"}
    assert threads["subj:lunch"].email_count == 2
    assert not threads["msg:h"].is_thread


def test_strategies_can_be_disabled():
    email = make_email("x", "RE: Hi", conversation_id="C9")
    grouper = ThreadGrouper(use_conversation_id=False, use_subject_matching=False)
    assert grouper._get_thread_key(email) == "msg:x"
    assert ThreadGrouper(use_conversation_id=False)._get_thread_key(email) == "subj:hi"


def test_participants_and_date_range_are_tracked():
    thread = EmailThread(conversation_id="t1", subject="Lunch")
    thread.add_email(make_email("d", "Lunch", sent="2024-01-02T09:00:00"))
    thread.add_email(
        make_email(
            "e",
            "RE: Lunch",
            sender="John Smith",
            sent="2024-01-01T09:00:00",
            recipients_to=[
                {"name": "Jane Doe", "email": "jane.doe@example.com"},
                {"email": "ops@example.com"},
            ],
            recipients_cc=["plain@example.com"],
        )
    )

    assert thread.participants == ["Jane Doe", "John Smith", "ops@example.com", "plain@example.com"]
    assert thread.start_date == datetime(2024, 1, 1, 9)
    assert thread.end_date == datetime(2024, 1, 2, 9)
    assert [e["record_id"] for e in thread.get_sorted_emails()] == ["e", "d"]


def test_threads_are_sorted_by_start_date():
    emails = [
        make_email("late", "Late topic", sent="2024-05-01"),
        make_email("early", "Early topic", sent="2023-01-01"),
    ]
    threads = group_emails_into_threads(emails)
    assert [t.emails[0]["record_id"] for t in threads] == ["early", "late"]


def test_to_concatenated_text_orders_emails_and_cleans_bodies():
    thread = EmailThread(conversation_id="t1", subject="project atlas")
    thread.add_email(
        make_email("b", "RE: Project Atlas", sender="John Smith", sent="2024-03-02", body="Second")
    )
    thread.add_email(
        make_email("a", "Project Atlas", sent="2024-03-01", body="First &amp; foremost  ")
    )

    text = thread.to_concatenated_text()

    assert text.startswith("[THREAD: project atlas]\n[Participants: John Smith, Jane Doe]")
    assert text.index("First & foremost") < text.index("Second")
    assert "Subject: Project Atlas" in text  # subject only on the first email
    assert "Subject: RE: Project Atlas" not in text
    assert text.endswith("[END THREAD]")


def test_to_dict():
    thread = EmailThread(conversation_id="t1", subject="atlas")
    thread.add_email(make_email("a", "Atlas", sent="2024-03-01T00:00:00"))
    assert thread.to_dict() == {
        "conversation_id": "t1",
        "subject": "atlas",
        "email_count": 1,
        "participants": ["Jane Doe"],
        "start_date": "2024-03-01T00:00:00",
        "end_date": "2024-03-01T00:00:00",
        "email_ids": ["a"],
        "is_thread": False,
    }


def test_group_from_bronze_reads_json_files(tmp_path):
    emails_dir = tmp_path / "emails" / "2024"
    emails_dir.mkdir(parents=True)
    for email in (make_email("a", "Atlas"), make_email("b", "RE: Atlas")):
        (emails_dir / f"{email['record_id']}.json").write_text(json.dumps(email))
    (emails_dir / "broken.json").write_text("{not json")

    threads = ThreadGrouper().group_from_bronze(str(tmp_path))

    assert len(threads) == 1
    assert threads[0].email_count == 2


def test_group_from_bronze_missing_directory(tmp_path):
    assert ThreadGrouper().group_from_bronze(str(tmp_path)) == []
