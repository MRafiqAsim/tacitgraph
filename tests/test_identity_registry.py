"""Email-address based identity resolution and stable pseudonyms."""

import json

import pytest

from tacitgraph.silver.identity_registry import Identity, IdentityRegistry


@pytest.fixture
def registry():
    reg = IdentityRegistry()
    reg.register_identity("  Jane.Doe@Example.com ", "Jane")
    reg.register_identity("jane.doe@example.com", "'Jane A. Doe'")
    reg.register_identity("john.smith@example.com", "")
    return reg


def test_same_email_merges_into_one_identity(registry):
    jane = registry.lookup_by_email("JANE.DOE@example.com")
    assert registry.identity_count == 2
    assert jane.canonical_name == "Jane A. Doe"  # longest observed name, quotes stripped
    assert jane.aliases == {"Jane", "Jane A. Doe"}
    assert jane.email_count == 2


def test_pseudonyms_are_sequential_and_stable(registry):
    assert registry.get_pseudonym("jane.doe@example.com") == "PERSON_001"
    assert registry.get_pseudonym("john.smith@example.com") == "PERSON_002"
    registry.register_identity("jane.doe@example.com", "J. Doe")
    assert registry.get_pseudonym("jane.doe@example.com") == "PERSON_001"


def test_identity_without_name_uses_email_local_part(registry):
    assert registry.lookup_by_email("john.smith@example.com").canonical_name == "john.smith"


def test_blank_email_is_ignored(registry):
    assert registry.register_identity("   ", "Nobody") is None
    assert registry.identity_count == 2


@pytest.mark.parametrize(
    "name",
    [
        "jane a doe",  # exact after normalization
        "Doe",  # substring of the canonical name
        "Jane A Dou",  # Levenshtein distance 1
    ],
)
def test_lookup_by_name_variants(registry, name):
    assert registry.lookup_by_name(name).pseudonym_id == "PERSON_001"


@pytest.mark.parametrize("name", ["", "   ", "Nobody Known", "123"])
def test_lookup_by_name_misses(registry, name):
    assert registry.lookup_by_name(name) is None


def test_very_short_names_match_by_substring(registry):
    # Documents current behavior: two-letter inputs match any identity containing them.
    assert registry.get_pseudonym("Jo") == "PERSON_002"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Dr. Jane-Ann  O'Doe ", "dr janeann odoe"),
        ("JOHN   SMITH", "john smith"),
        ("", ""),
    ],
)
def test_normalize_name(raw, expected):
    assert IdentityRegistry._normalize_name(raw) == expected


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("kitten", "sitting", 3), ("", "abc", 3), ("same", "same", 0), ("ab", "ba", 2)],
)
def test_levenshtein(a, b, distance):
    assert IdentityRegistry._levenshtein(a, b) == distance


def test_save_and_load_round_trip_keeps_counter(registry, tmp_path):
    path = tmp_path / "nested" / "registry.json"
    registry.save(str(path))

    loaded = IdentityRegistry()
    loaded.load(str(path))

    assert loaded.identity_count == 2
    assert loaded.get_pseudonym("Jane A. Doe") == "PERSON_001"
    new = loaded.register_identity("new.person@example.com", "New Person")
    assert new.pseudonym_id == "PERSON_003"


def test_build_from_bronze(tmp_path):
    emails_dir = tmp_path / "emails"
    emails_dir.mkdir()
    email = {
        "email_headers": {
            "sender_email": "jane.doe@example.com",
            "sender": "Jane Doe",
            "recipients_to": [{"email": "john.smith@example.com", "name": "John Smith"}],
            "recipients_cc": [{"email": "ops@example.com", "name": ""}, "ignored-string"],
        }
    }
    (emails_dir / "a.json").write_text(json.dumps(email))
    (emails_dir / "broken.json").write_text("{not json")

    registry = IdentityRegistry()
    stats = registry.build_from_bronze(str(tmp_path))

    assert stats["total_identities"] == 3
    assert stats["total_emails_scanned"] == 1
    assert registry.lookup_by_name("John Smith").email_address == "john.smith@example.com"


def test_build_from_missing_bronze(tmp_path):
    assert IdentityRegistry().build_from_bronze(str(tmp_path / "nope")) == {
        "error": "emails directory not found"
    }


def test_known_names_and_report(registry):
    assert registry.get_all_known_names() == {"Jane", "Jane A. Doe", "john.smith"}
    report = registry.report()
    assert "Total identities: 2" in report
    assert "PERSON_001: Jane A. Doe <jane.doe@example.com> (2 emails)" in report


def test_identity_dict_round_trip():
    identity = Identity("a@example.com", "A", {"A", "Ay"}, "PERSON_007", 3)
    assert Identity.from_dict(identity.to_dict()) == identity
