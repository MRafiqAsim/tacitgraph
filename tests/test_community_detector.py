"""Leiden community detection on a small synthetic graph (no LLM summaries)."""

import itertools

import pytest

from tacitgraph.gold.community_detector import Community, CommunityConfig, CommunityDetector
from tacitgraph.gold.graph_builder import GraphEdge, GraphNode, KnowledgeGraph

TEAM_BERLIN = ["Jane Doe", "John Smith", "Ana Lima", "Project Atlas"]
TEAM_LISBON = ["Maria Silva", "Tom Berg", "Lea Novak", "ERP migration"]


def build_two_team_graph() -> KnowledgeGraph:
    """Two fully connected teams with no edges between them, plus one chunk node."""
    kg = KnowledgeGraph()
    kg.add_node(GraphNode("chunk-1", "Chunk: chunk-1", "CHUNK", source_chunks=["chunk-1"]))
    for team in (TEAM_BERLIN, TEAM_LISBON):
        for i, name in enumerate(team):
            node_type = "PERSON" if i < 3 else "PROJECT"
            kg.add_node(
                GraphNode(name, name, node_type, mention_count=10 - i, source_chunks=[f"c-{name}"])
            )
            kg.add_edge(GraphEdge(f"m-{name}", name, "chunk-1", "MENTIONED_IN"))
        for a, b in itertools.combinations(team, 2):
            kg.add_edge(GraphEdge(f"{a}->{b}", a, b, "WORKS_ON"))
    return kg


@pytest.fixture
def detector(tmp_path, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = CommunityConfig(num_levels=2, min_community_size=3, use_llm_summarization=False)
    return CommunityDetector(build_two_team_graph(), str(tmp_path / "gold"), config)


def test_detects_one_community_per_team(detector):
    communities = detector.detect_communities()
    level0 = communities[0].values()
    assert sorted(sorted(c.node_ids) for c in level0) == sorted(
        [sorted(TEAM_BERLIN), sorted(TEAM_LISBON)]
    )
    for community in level0:
        assert community.density == pytest.approx(1.0)  # each team is a clique
        assert "chunk-1" not in community.node_ids  # CHUNK nodes are excluded


def test_hierarchy_links_levels(detector):
    communities = detector.detect_communities()
    parents = communities[1]
    for child in communities[0].values():
        assert child.parent_id in parents
        assert child.community_id in parents[child.parent_id].child_ids


def test_extractive_summaries_and_topics(detector):
    detector.detect_communities()
    berlin = detector.find_community_for_entity("Jane Doe")
    assert berlin.summary.startswith("Community with 4 entities")
    assert "People:" in berlin.summary
    assert berlin.key_topics == ["Project Atlas"]
    assert berlin.key_entities[0]["name"] == "Jane Doe"  # highest mention count first
    assert set(berlin.source_chunk_ids) == {f"c-{n}" for n in TEAM_BERLIN}


def test_save_and_load_round_trip(detector, tmp_path):
    detector.detect_communities()
    detector.save()

    reloaded = CommunityDetector(
        detector.graph, str(tmp_path / "gold"), CommunityConfig(use_llm_summarization=False)
    )
    communities = reloaded.load()
    assert {cid for level in communities.values() for cid in level} == {
        cid for level in detector.communities.values() for cid in level
    }
    some_id = next(iter(communities[0]))
    assert reloaded.get_community(some_id).summary == detector.get_community(some_id).summary
    assert len(reloaded.get_communities_at_level(0)) == 2


def test_load_without_index_raises(detector):
    with pytest.raises(FileNotFoundError):
        detector.load()


def test_min_community_size_filters_small_groups(tmp_path):
    kg = KnowledgeGraph()
    for name in ("Jane Doe", "John Smith"):
        kg.add_node(GraphNode(name, name, "PERSON"))
    kg.add_edge(GraphEdge("e", "Jane Doe", "John Smith", "SENT_TO"))
    config = CommunityConfig(num_levels=1, min_community_size=3, use_llm_summarization=False)
    assert CommunityDetector(kg, str(tmp_path), config).detect_communities()[0] == {}


def test_community_dict_round_trip():
    community = Community(community_id="comm_l0_0000", level=0, node_ids=["a", "b"], summary="s")
    assert Community.from_dict(community.to_dict()) == community
