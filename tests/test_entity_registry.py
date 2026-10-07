"""Config-driven entity and relationship normalization."""

import pytest

from tacitgraph import entity_registry as er


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ORGANIZATION", "ORG"),
        ("organization", "ORG"),
        ("GEO", "GPE"),
        ("FAC", "LOC"),  # spaCy label
        ("PERSON", "PERSON"),
        ("VESSEL", "VESSEL"),  # unknown types pass through unchanged
    ],
)
def test_get_standard_type(raw, expected):
    assert er.get_standard_type(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("involved in", "WORKS_ON"),
        ("EMPLOYED_BY", "WORKS_AT"),
        ("employs", "WORKS_AT"),  # direction flip maps to its standard type
        ("SOMETHING_NEW", "SOMETHING_NEW"),
    ],
)
def test_normalize_relationship_type(raw, expected):
    assert er.normalize_relationship_type(raw) == expected


def test_should_flip_relationship():
    assert er.should_flip_relationship("EMPLOYS")
    assert er.should_flip_relationship("hires")
    assert not er.should_flip_relationship("WORKS_AT")


def test_structural_and_semantic_edge_types_are_disjoint():
    structural = er.get_structural_edge_types()
    semantic = er.get_semantic_edge_types()
    assert "MENTIONED_IN" in structural
    assert "WORKS_AT" in semantic
    assert not structural & semantic


def test_resolve_entity_name_uses_static_aliases():
    assert er.resolve_entity_name("ACME Corp.") == "Acme"
    assert er.resolve_entity_name("acme corporation") == "Acme"  # case-insensitive


def test_resolve_entity_name_uses_catalog(catalog_file):
    er.load_catalog(str(catalog_file))
    assert er.resolve_entity_name("BER") == "Berlin Office"
    assert er.resolve_entity_name("Unknown Thing") == "Unknown Thing"


def test_expand_entity_aliases_from_catalog(catalog_file):
    er.load_catalog(str(catalog_file))
    assert er.expand_entity_aliases("Berlin Office") == ["Berlin Office", "BER", "Ber"]
    assert er.expand_entity_aliases("ber") == ["ber", "Berlin Office", "BER", "Ber"]


def test_expand_entity_aliases_uses_catalog_only():
    # Static config aliases are applied by resolve_entity_name during graph
    # building; query-time expansion relies on the runtime catalog.
    assert er.expand_entity_aliases("Acme Research Centre") == ["Acme Research Centre"]


def test_expand_unknown_entity_returns_itself():
    assert er.expand_entity_aliases("Nothing Known") == ["Nothing Known"]


def test_missing_catalog_file_yields_empty_catalog(tmp_path):
    catalog = er.load_catalog(str(tmp_path / "missing.json"))
    assert catalog == {"entities": {}, "metadata": {}}
