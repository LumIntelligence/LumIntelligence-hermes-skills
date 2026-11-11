"""SQLite-specific behaviour: one kb.db with graph + vectors, build transactions seen
from a second connection, provenance origins, traversal edge cases, and graph.json import."""

from __future__ import annotations

import json

import pytest

from lumintelligence.graph import GraphStore, SqliteGraphStore, load_graph_json
from lumintelligence.models.records import VectorRecord
from lumintelligence.storage import SqliteVecStore
from lumintelligence.storage.sqlite_conn import connect


def _mini(kg: SqliteGraphStore) -> dict:
    src = kg.add_source("Acme funded the west wing.")
    ent = kg.add_entity("Acme", context={"source_id": src, "text": "Acme funded the west wing."})
    con = kg.add_concept("Funding Source", embedding=[0.1, 0.2, 0.3])
    cen = kg.add_centroid("Funding")
    kg.link_mentions(src, ent)
    kg.link_is_concept(ent, con)
    kg.assign_concept_to_centroid(con, cen)
    return {"src": src, "ent": ent, "con": con, "cen": cen}


def _record(node_id: str, dim: int = 3) -> VectorRecord:
    return VectorRecord(id=node_id, content="text", embedding=[1.0] + [0.0] * (dim - 1),
                        metadata={"node_id": node_id, "node_type": "Concept", "name": node_id})


def test_store_satisfies_the_protocol():
    assert isinstance(SqliteGraphStore(), GraphStore)


# ------------------------------------------------------------------ transactions
def test_reader_sees_last_committed_state_during_a_build(tmp_path):
    db = str(tmp_path / "kb.db")
    writer = SqliteGraphStore(path=db)
    writer.add_source("committed before the build")
    reader = SqliteGraphStore(path=db)

    writer.begin()
    writer.add_source("written inside the build")
    assert writer.source_node_count() == 2
    assert reader.source_node_count() == 1        # WAL: the reader sees the last commit only
    writer.commit()
    assert reader.source_node_count() == 2


def test_rollback_discards_graph_and_vector_writes_together(tmp_path):
    conn = connect(str(tmp_path / "kb.db"))
    kg = SqliteGraphStore(conn=conn)
    vec = SqliteVecStore(3, conn=conn)
    ids = _mini(kg)
    vec.replace_all([_record(ids["con"])])

    kg.begin()
    kg.add_source("half-built")
    vec.replace_all([_record(ids["con"]), _record("CONCEPT_other")])
    kg.rollback()

    assert kg.source_node_count() == 1
    assert vec.count() == 1


def test_replace_all_keeps_concept_embeddings():
    kg = SqliteGraphStore()
    vec = SqliteVecStore(3, conn=kg.conn)
    ids = _mini(kg)
    vec.replace_all([_record(ids["con"])])
    vec.replace_all([])
    assert kg.concept_embedding(ids["con"]) == [0.1, 0.2, 0.3]


def test_vector_dimension_is_fixed_per_kb(tmp_path):
    db = str(tmp_path / "kb.db")
    SqliteVecStore(3, path=db).close()
    with pytest.raises(ValueError, match="3-dimension"):
        SqliteVecStore(8, path=db)


# ----------------------------------------------------------------------- origins
def test_origin_round_trip_and_sha_lookup():
    kg = SqliteGraphStore()
    oid = kg.add_origin("file", "abc123", label="report.pdf", extra={"path": "/docs/report.pdf"})

    origin = kg.get_origin(oid)
    assert origin["kind"] == "file" and origin["label"] == "report.pdf"
    assert origin["extra"] == {"path": "/docs/report.pdf"}
    assert origin["created_at"]
    assert kg.find_origin_by_sha("abc123") == oid
    assert kg.find_origin_by_sha("nope") is None
    assert [o["id"] for o in kg.list_origins()] == [oid]


def test_sources_carry_origin_and_position_and_filter_by_origin():
    kg = SqliteGraphStore()
    pdf = kg.add_origin("file", "sha-pdf", label="report.pdf")
    chat = kg.add_origin("chat", "sha-chat", label="Board prep")
    s1 = kg.add_source("page one text", origin_id=pdf, position={"page": 1, "lines": [1, 15]})
    s2 = kg.add_source("a chat message", origin_id=chat, position={"messages": [3, 5]})
    kg.add_source("no origin yet")

    node = kg.get_node(s1)
    assert node["origin_id"] == pdf and node["position"] == {"page": 1, "lines": [1, 15]}
    assert kg.sources_by_origin([pdf]) == [s1]
    assert kg.sources_by_origin([pdf, chat]) == [s1, s2]


def test_add_source_rejects_unknown_origin():
    with pytest.raises(KeyError):
        SqliteGraphStore().add_source("text", origin_id="ORIGIN_missing")


# --------------------------------------------------------------------- traversal
def test_traversal_from_every_tier():
    kg = SqliteGraphStore()
    ids = _mini(kg)
    for start in ("ent", "con", "cen"):
        assert kg.trace_sources(ids[start]) == [ids["src"]]
    assert set(kg.ancestors(ids["cen"])) == {ids["src"], ids["ent"], ids["con"]}
    assert kg.shortest_path(ids["src"], ids["cen"]) == [ids["src"], ids["ent"], ids["con"], ids["cen"]]
    assert kg.shortest_path(ids["cen"], ids["src"]) is None


def test_ancestors_terminates_on_a_cycle():
    kg = SqliteGraphStore()
    a, b = kg.add_concept("A"), kg.add_concept("B")
    kg.conn.execute("insert into edges(id, src, dst, type) values ('e1', ?, ?, 'X'), ('e2', ?, ?, 'X')",
                    (a, b, b, a))
    assert kg.ancestors(a) == [b]


def test_keyword_match_on_name_and_tags():
    kg = SqliteGraphStore()
    e = kg.add_entity("Harbor Fund")
    kg.set_entity_tags(e, ["Grant Maker"])
    kg.add_entity("Lisbon")
    assert [h["id"] for h in kg.match_entities_by_keyword("harbor")] == [e]
    assert [h["id"] for h in kg.match_entities_by_keyword("grant")] == [e]


# ------------------------------------------------------------------------ import
def test_graph_json_import_keeps_ids_edges_and_moves_embeddings(tmp_path):
    old = SqliteGraphStore()
    ids = _mini(old)
    data = old.export_json()
    for n in data["nodes"]:          # old graph.json files kept the embedding on the concept node
        if n["id"] == ids["con"]:
            n["attributes"]["embedding"] = [0.5, 0.5]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data))

    new = load_graph_json(str(path))
    assert new.node_count() == old.node_count() and new.edge_count() == old.edge_count()
    assert new.trace_sources(ids["cen"]) == [ids["src"]]
    assert new.concept_embedding(ids["con"]) == [0.5, 0.5]
    assert "embedding" not in new.get_node(ids["con"])["attributes"]
