"""BM25 keyword index and rank fusion used alongside vector search."""

from tacitgraph.retrieval.lexical_index import BM25Index, reciprocal_rank_fusion, tokenize

DOCS = [
    ("intro", "Welcome our new colleague who joins the team in March."),
    ("ticket", "Jane Doe created ticket OPS-386 for the ERP migration."),
    ("status", "The ERP migration status meeting moved to Thursday."),
]


def test_tokenize_drops_stopwords_and_single_characters():
    assert tokenize("Who is Jane Doe, and what is OPS-386?") == ["jane", "doe", "ops", "386"]


def test_rare_name_ranks_the_document_that_contains_it():
    index = BM25Index(DOCS)
    assert index.search("who is jane doe?")[0][0] == "ticket"


def test_rarer_terms_outweigh_common_ones():
    index = BM25Index(DOCS)
    ranked = [doc_id for doc_id, _ in index.search("ERP migration ticket")]
    assert ranked[0] == "ticket"
    assert set(ranked) == {"ticket", "status"}


def test_no_matching_terms_returns_nothing():
    assert BM25Index(DOCS).search("who is it?") == []
    assert BM25Index([]).search("jane") == []


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = [doc_id for doc_id, _ in reciprocal_rank_fusion(["a", "b", "c"], ["b", "d"])]
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}
