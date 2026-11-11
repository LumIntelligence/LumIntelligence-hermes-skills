"""Incremental vectorization: ``SqliteVecStore.sync`` embeds only new or changed record
texts, deletes vectors of removed nodes, and re-embeds everything when the embedding model changes."""

from __future__ import annotations

import pytest

from lumintelligence.graph import SqliteGraphStore
from lumintelligence.llm.base import EmbedPurpose
from lumintelligence.llm.mock import MockProvider
from lumintelligence.models.records import VectorRecord
from lumintelligence.storage import SqliteVecStore
from lumintelligence.storage.records import build_records
from lumintelligence.storage.sqlite_conn import connect


class CountingEmbedder(MockProvider):
    """Mock embedder that logs every text it embeds; ``model`` sets its model id."""

    def __init__(self, embed_dim: int = 8, model: str | None = "m1"):
        super().__init__(embed_dim=embed_dim)
        self.model = model
        self.texts: list[str] = []

    @property
    def model_id(self) -> str | None:
        return self.model

    def embed(self, texts: list[str], purpose: EmbedPurpose = EmbedPurpose.DOCUMENT) -> list[list[float]]:
        self.texts.extend(texts)
        return super().embed(texts, purpose)


def _rec(nid: str, text: str) -> VectorRecord:
    return VectorRecord(id=nid, content=text, metadata={"node_id": nid, "node_type": "Concept", "name": nid})


def _contents(store: SqliteVecStore) -> dict[str, str]:
    return dict(store._conn.execute("select node_id, content from vec_items").fetchall())


def test_first_sync_embeds_everything():
    emb = CountingEmbedder()
    store = SqliteVecStore(emb.dimension)
    counts = store.sync([_rec("a", "A"), _rec("b", "B")], emb)
    assert counts == {"added": 2, "updated": 0, "deleted": 0, "unchanged": 0, "embedded": 2, "full": False}
    assert store.count() == 2


def test_unchanged_records_are_not_embedded_again():
    emb = CountingEmbedder()
    store = SqliteVecStore(emb.dimension)
    store.sync([_rec("a", "A"), _rec("b", "B")], emb)
    emb.texts.clear()
    counts = store.sync([_rec("a", "A"), _rec("b", "B")], emb)
    assert emb.texts == []
    assert counts["unchanged"] == 2 and counts["embedded"] == 0


def test_only_new_and_changed_texts_are_embedded():
    emb = CountingEmbedder()
    store = SqliteVecStore(emb.dimension)
    store.sync([_rec("a", "A"), _rec("b", "B")], emb)
    emb.texts.clear()
    counts = store.sync([_rec("a", "A"), _rec("b", "B2"), _rec("c", "C")], emb)
    assert sorted(emb.texts) == ["B2", "C"]
    assert counts["added"] == 1 and counts["updated"] == 1 and counts["unchanged"] == 1
    assert _contents(store) == {"a": "A", "b": "B2", "c": "C"}


def test_removed_nodes_are_deleted():
    emb = CountingEmbedder()
    store = SqliteVecStore(emb.dimension)
    store.sync([_rec("a", "A"), _rec("b", "B")], emb)
    counts = store.sync([_rec("a", "A")], emb)
    assert counts["deleted"] == 1
    assert _contents(store) == {"a": "A"}


def test_full_flag_re_embeds_everything():
    emb = CountingEmbedder()
    store = SqliteVecStore(emb.dimension)
    store.sync([_rec("a", "A"), _rec("b", "B")], emb)
    emb.texts.clear()
    counts = store.sync([_rec("a", "A"), _rec("b", "B")], emb, full=True)
    assert sorted(emb.texts) == ["A", "B"]
    assert counts["full"] is True and counts["updated"] == 2


def test_changed_embedding_model_forces_a_full_re_embed():
    store = SqliteVecStore(8)
    store.sync([_rec("a", "A")], CountingEmbedder(model="m1"))
    other = CountingEmbedder(model="m2")
    counts = store.sync([_rec("a", "A")], other)
    assert counts["full"] is True and other.texts == ["A"]
    again = store.sync([_rec("a", "A")], other)           # model now recorded: back to incremental
    assert again["full"] is False and again["embedded"] == 0


def test_kb_without_a_recorded_model_is_re_embedded_once():
    store = SqliteVecStore(8)
    old = CountingEmbedder(model=None)                      # older KBs: vectors, no model recorded
    store.sync([_rec("a", "A")], old)
    emb = CountingEmbedder(model="m1")
    assert store.sync([_rec("a", "A")], emb)["full"] is True
    assert store.sync([_rec("a", "A")], emb)["full"] is False


def test_sync_matches_a_full_rebuild_on_a_growing_graph():
    emb = CountingEmbedder()
    kg = SqliteGraphStore()
    inc, full = SqliteVecStore(emb.dimension), SqliteVecStore(emb.dimension)
    src = kg.add_source("Acme funded the west wing.")
    e1 = kg.add_entity("Acme", context={"source_id": src, "text": "Acme funded"})
    c1 = kg.add_concept("Funder")
    kg.link_is_concept(e1, c1)
    cen = kg.add_centroid("Funding")
    kg.assign_concept_to_centroid(c1, cen)
    inc.sync(build_records(kg), emb)

    src2 = kg.add_source("Globex also funded it.")
    e2 = kg.add_entity("Globex", context={"source_id": src2, "text": "Globex also funded"})
    c2 = kg.add_concept("Donor")
    kg.link_is_concept(e2, c2)
    kg.assign_concept_to_centroid(c2, cen)                  # centroid text changes (new member)
    emb.texts.clear()
    inc.sync(build_records(kg), emb)
    assert len(emb.texts) == 3                              # Globex, Donor, the grown centroid
    full.sync(build_records(kg), emb, full=True)

    def rows(s: SqliteVecStore) -> dict:
        return {nid: (content, bytes(vec)) for nid, content, vec in
                s._conn.execute("select node_id, content, embedding from vec_items").fetchall()}
    assert rows(inc) == rows(full)


def test_failed_build_rolls_back_a_sync(tmp_path):
    conn = connect(str(tmp_path / "kb.db"))
    kg = SqliteGraphStore(conn=conn)
    vec = SqliteVecStore(8, conn=conn)
    emb = CountingEmbedder()
    vec.sync([_rec("a", "A")], emb)

    kg.begin()
    vec.sync([_rec("b", "B")], emb)
    kg.rollback()
    assert _contents(vec) == {"a": "A"}


def test_embedder_failure_leaves_the_index_unchanged():
    class Broken(CountingEmbedder):
        def embed(self, texts, purpose=EmbedPurpose.DOCUMENT):
            raise RuntimeError("embedder down")

    store = SqliteVecStore(8)
    store.sync([_rec("a", "A")], CountingEmbedder())
    with pytest.raises(RuntimeError):
        store.sync([_rec("b", "B")], Broken())
    assert _contents(store) == {"a": "A"}
