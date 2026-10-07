"""Semantic-safe email body cleaning."""

import pytest

from tacitgraph.silver.email_text_cleaner import clean_email_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\r\nb\rc", "a\nb\nc"),  # line endings
        ("Tom &amp; Jerry &lt;x&gt;", "Tom & Jerry <x>"),  # HTML entities
        ("a\x00b\x07c\td", "abc\td"),  # control chars, tab kept
        ("zero​width﻿", "zerowidth"),  # invisible characters
        ("non break", "non break"),  # unicode whitespace
        ("see [cid:image001.png@01D] and cid:image002.png here", "see and here"),
        ("------=_Part_123\nbody", "body"),  # MIME boundary
        ("-----Original Message-----\nhi", "--- Forwarded ---\nhi"),
        (">>> deep\n> > > deeper", "> deep\n> deeper"),  # quote nesting collapsed
        ("a     b", "a b"),
        ("x  \ny", "x\ny"),  # trailing spaces
        ("a\n\n\n\n\nb", "a\n\nb"),  # paragraph breaks kept, extra blank lines removed
        ("  hi  ", "hi"),
    ],
)
def test_clean_email_text(raw, expected):
    assert clean_email_text(raw) == expected


@pytest.mark.parametrize("empty", ["", None])
def test_empty_input_is_returned_unchanged(empty):
    assert clean_email_text(empty) == empty


def test_removes_encoded_blobs_but_keeps_text():
    jwt = "eyJ" + "a" * 120
    base64_blob = "A" * 600
    hex_blob = "0f" * 120
    text = f"Token {jwt} then {base64_blob} and {hex_blob} done"
    assert clean_email_text(text) == "Token then and done"


def test_removes_pgp_block():
    text = (
        "Signed update on Project Atlas.\n"
        "-----BEGIN PGP SIGNATURE-----\nabc123\n-----END PGP SIGNATURE-----\n"
        "Thanks"
    )
    assert clean_email_text(text) == "Signed update on Project Atlas.\n\nThanks"


def test_preserves_leading_indentation():
    assert clean_email_text("Steps:\n    1. build\n    2. deploy") == (
        "Steps:\n    1. build\n    2. deploy"
    )
