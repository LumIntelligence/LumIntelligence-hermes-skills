"""SqliteGraphStore's query-side methods and the GraphStore protocol."""

from __future__ import annotations

from lumintelligence.graph import GraphStore, SqliteGraphStore


def _graph():
    kg = SqliteGraphStore()
    s = kg.add_source("Acme Corp signed a deal with Globex.")
    e = kg.add_entity("Acme Corp", context={"source_id": s, "text": "Acme Corp"})
    kg.link_mentions(s, e)
    kg.set_entity_tags(e, ["Corporate Partnership"])
    c = kg.add_concept("Corporate Partnership")
    kg.link_is_concept(e, c)
    cen = kg.add_centroid("Business")
    kg.assign_concept_to_centroid(c, cen)
    kg.mark_source_processed(s)
    return kg, s, e, c, cen


def test_knowledge_graph_satisfies_protocol():
    assert isinstance(SqliteGraphStore(), GraphStore)


def test_traversal_methods():
    kg, s, e, c, cen = _graph()
    assert set(kg.ancestors(cen)) == {s, e, c}
    assert kg.shortest_path(s, cen) == [s, e, c, cen]
    assert kg.shortest_path(cen, s) is None
    assert kg.shortest_path("missing", s) is None
    edges = {(x["source"], x["target"]) for x in kg.subgraph_edges([s, e, c])}
    assert edges == {(s, e), (e, c)}


def test_match_entities_by_keyword_on_name_and_tags():
    kg, _, e, _, _ = _graph()
    assert [h["id"] for h in kg.match_entities_by_keyword("acme")] == [e]
    assert [h["id"] for h in kg.match_entities_by_keyword("partnership")] == [e]
    assert kg.match_entities_by_keyword("nothing") == []


def test_counts_and_reset_for_reprocessing():
    kg, s, *_ = _graph()
    assert (kg.node_count(), kg.edge_count(), kg.source_node_count()) == (4, 3, 1)
    assert kg.sources_to_process() == []
    kg.reset_sources_for_reprocessing([s, "missing"])
    assert kg.sources_to_process() == [s]
