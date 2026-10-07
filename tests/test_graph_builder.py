"""Knowledge graph construction from Silver chunk files (offline)."""

import json

import pytest

from tacitgraph.gold.graph_builder import GraphBuilder, GraphEdge, GraphNode, KnowledgeGraph

CHUNKS = {
    "c1": {
        "chunk_id": "c1",
        "thread_id": "t1",
        "thread_subject": "Project Atlas kickoff",
        "kg_entities": [
            {"entity": "Jane Doe", "type": "PERSON", "confidence": 0.9},
            {"entity": "Acme Corporation", "type": "ORGANIZATION", "aliases": ["ACME"]},
        ],
        "kg_relationships": [
            {
                "source": "Acme",
                "target": "Jane Doe",
                "relationship": "EMPLOYS",
                "source_type": "ORG",
                "target_type": "PERSON",
                "confidence": 0.8,
            }
        ],
        "email_sender": "Jane Doe",
        "email_sender_address": "jane.doe@example.com",
        "email_recipients_to": [{"name": "John Smith", "email": "john.smith@example.com"}],
        "sent_timestamp": "2024-03-01T10:00:00",
        "attachment_filenames": ["atlas_plan.pdf"],
    },
    "c2": {
        "chunk_id": "c2",
        "thread_id": "t1",
        "kg_entities": [{"entity": "Project Atlas", "type": "PROJECT"}],
        "thread_participants": ["Jane Doe", "John Smith"],
    },
}


@pytest.fixture
def layers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate the ./data identity-registry fallback
    chunk_dir = tmp_path / "silver" / "not_personal" / "email_chunks"
    chunk_dir.mkdir(parents=True)
    for chunk_id, data in CHUNKS.items():
        (chunk_dir / f"{chunk_id}.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path / "silver", tmp_path / "gold"


@pytest.fixture
def built(layers):
    builder = GraphBuilder(*map(str, layers))
    builder.build_from_chunks()
    return builder


def node_by_name(graph, name, node_type):
    matches = [n for n in graph.get_nodes_by_type(node_type) if n.name == name]
    assert len(matches) == 1, f"{node_type} {name!r}: {matches}"
    return matches[0]


def edge_types_between(graph, source_id, target_id):
    return {e.edge_type for e in graph.get_edges_from(source_id) if e.target_id == target_id}


def test_builds_chunk_entity_and_thread_nodes(built):
    graph = built.graph
    assert {n.node_id for n in graph.get_nodes_by_type("CHUNK")} == {"c1", "c2"}
    assert node_by_name(graph, "Project Atlas", "PROJECT")
    assert len(graph.get_nodes_by_type("THREAD")) == 1
    assert graph.metadata["total_chunks_processed"] == 2


def test_static_aliases_resolve_entity_names(built):
    acme = node_by_name(built.graph, "Acme", "ORG")
    assert "c1" in acme.source_chunks


def test_reversed_relationships_are_flipped(built):
    graph = built.graph
    jane = node_by_name(graph, "Jane Doe", "PERSON")
    acme = node_by_name(graph, "Acme", "ORG")
    assert "WORKS_AT" in edge_types_between(graph, jane.node_id, acme.node_id)
    assert not edge_types_between(graph, acme.node_id, jane.node_id)


def test_email_structure_edges(built):
    graph = built.graph
    jane = node_by_name(graph, "Jane Doe", "PERSON")
    john = node_by_name(graph, "John Smith", "PERSON")
    assert jane.properties["email"] == "jane.doe@example.com"
    assert "SENT" in edge_types_between(graph, jane.node_id, "c1")
    sent_to = [e for e in graph.get_edges_from(jane.node_id) if e.edge_type == "SENT_TO"]
    assert [(e.target_id, e.properties["date"]) for e in sent_to] == [(john.node_id, "2024-03-01")]
    assert "MENTIONED_IN" in edge_types_between(graph, jane.node_id, "c1")


def test_participants_are_linked_when_sender_is_missing(built):
    graph = built.graph
    john = node_by_name(graph, "John Smith", "PERSON")
    assert "PARTICIPATED_IN" in edge_types_between(graph, john.node_id, "c2")


def test_attachments_and_thread_membership(built):
    graph = built.graph
    doc = node_by_name(graph, "atlas_plan.pdf", "DOCUMENT")
    thread = graph.get_nodes_by_type("THREAD")[0]
    assert "HAS_ATTACHMENT" in edge_types_between(graph, "c1", doc.node_id)
    for chunk_id in ("c1", "c2"):
        assert "PART_OF_THREAD" in edge_types_between(graph, chunk_id, thread.node_id)


def test_edge_weights_are_normalized(built):
    weights = [e.weight for e in built.graph.edges.values()]
    assert max(weights) == pytest.approx(1.0)
    assert all(0 < w <= 1 for w in weights)


def test_catalog_collects_entities_and_aliases(built):
    entities = built._catalog["entities"]
    assert entities["Acme"]["type"] == "ORG"
    assert entities["Acme"]["aliases"] == ["ACME"]
    assert entities["Jane Doe"]["mention_count"] == 1


def test_save_and_load_round_trip(built, layers):
    built.save()
    kg_dir = layers[1] / "knowledge_graph"
    for name in ("nodes.json", "edges.json", "graph_stats.json", "graph.graphml"):
        assert (kg_dir / name).exists()
    assert (layers[1] / "entity_catalog.json").exists()

    reloaded = GraphBuilder(*map(str, layers)).load()
    assert reloaded.stats()["total_nodes"] == built.graph.stats()["total_nodes"]
    assert reloaded.stats()["total_edges"] == built.graph.stats()["total_edges"]


def test_load_without_saved_graph_raises(layers):
    with pytest.raises(FileNotFoundError):
        GraphBuilder(*map(str, layers)).load()


def test_identity_registry_merges_person_name_variants(layers):
    registry = {
        "identities": [
            {
                "canonical_name": "Jane Doe (jane.doe@example.com)",
                "aliases": ["Doe Jane", "J. Doe", "jane.doe@example.com"],
                "email_count": 12,
            }
        ]
    }
    (layers[0].parent / "identity_registry.json").write_text(json.dumps(registry))
    builder = GraphBuilder(*map(str, layers))
    assert builder._resolve_person_name("Doe Jane") == "Jane Doe"
    assert builder._resolve_person_name("J. Doe") == "Jane Doe"
    assert builder._resolve_person_name("Jane") == "Jane"  # bare first names are ambiguous


@pytest.mark.parametrize(
    ("a", "b"),
    [("The Acme", "acme"), ("  Project Atlas ", "project atlas"), ("An Office", "office")],
)
def test_node_ids_ignore_case_whitespace_and_articles(layers, a, b):
    builder = GraphBuilder(*map(str, layers))
    assert builder._generate_node_id(a, "ORG") == builder._generate_node_id(b, "ORG")
    assert builder._generate_node_id(a, "ORG") != builder._generate_node_id(a, "PERSON")


def make_node(node_id, node_type="PERSON", mentions=1):
    return GraphNode(
        node_id, f"name-{node_id}", node_type, mention_count=mentions, source_chunks=["c"]
    )


def test_knowledge_graph_merges_duplicates():
    kg = KnowledgeGraph()
    kg.add_node(make_node("n1", mentions=2))
    kg.add_node(make_node("n1", mentions=3))
    kg.add_edge(GraphEdge("e1", "n1", "n2", "RELATED_TO", weight=0.5))
    kg.add_edge(GraphEdge("e1", "n1", "n2", "RELATED_TO", weight=0.25))
    assert kg.nodes["n1"].mention_count == 5
    assert kg.nodes["n1"].source_chunks == ["c", "c"]
    assert kg.edges["e1"].weight == pytest.approx(0.75)
    assert kg.stats()["edges_by_type"] == {"RELATED_TO": 1}


def test_knowledge_graph_neighbors_and_dict_round_trip():
    kg = KnowledgeGraph()
    for node_id in ("a", "b", "c"):
        kg.add_node(make_node(node_id))
    kg.add_edge(GraphEdge("ab", "a", "b", "WORKS_ON"))
    kg.add_edge(GraphEdge("ca", "c", "a", "REPORTS_TO"))
    assert sorted(kg.get_neighbors("a")) == ["b", "c"]

    restored = KnowledgeGraph.from_dict(kg.to_dict())
    assert restored.stats() == kg.stats()
    assert restored.to_networkx().number_of_edges() == 2
