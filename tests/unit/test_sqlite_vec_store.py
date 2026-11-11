"""SqliteVecStore: the VectorStore contract (cosine scores, top_k, node_type metadata, file-backed)."""

from __future__ import annotations

from lumintelligence.graph import SqliteGraphStore
from lumintelligence.llm.mock import MockProvider
from lumintelligence.storage import SqliteVecStore


def _sample_graph() -> SqliteGraphStore:
    kg = SqliteGraphStore()
    src = kg.add_source("Acme Corp signed a deal with Globex.")
    e1 = kg.add_entity("Acme Corp", context={"source_id": src, "text": "Acme Corp signed a deal"})
    e2 = kg.add_entity("Globex", context={"source_id": src, "text": "deal with Globex"})
    concept = kg.add_concept("Corporate Partnership")
    kg.link_is_concept(e1, concept)
    kg.link_is_concept(e2, concept)
    return kg


def test_build_and_write_then_query():
    embedder = MockProvider(embed_dim=8)
    store = SqliteVecStore(dimension=embedder.dimension)
    kg = _sample_graph()

    store.build_and_write(kg, embedder)
    assert store.count() == 3  # 2 entities + 1 concept, no centroids yet

    results = store.query("Acme Corp", embedder, top_k=3)
    assert results
    assert any(r["metadata"]["name"] == "Acme Corp" for r in results)


def test_reset_clears_index():
    embedder = MockProvider(embed_dim=8)
    store = SqliteVecStore(dimension=embedder.dimension)
    store.build_and_write(_sample_graph(), embedder)
    assert store.count() > 0

    store.reset()
    assert store.count() == 0


def test_file_backed_store_persists_and_rewrites_by_node_id(tmp_path):
    embedder = MockProvider(embed_dim=8)
    path = str(tmp_path / "vectors.db")
    store = SqliteVecStore(dimension=embedder.dimension, path=path)
    store.build_and_write(_sample_graph(), embedder)
    store.build_and_write(_sample_graph(), embedder)   # same node ids: replaced, not duplicated
    n = store.count()
    store.close()

    reopened = SqliteVecStore(dimension=embedder.dimension, path=path)
    assert reopened.count() == n
    assert reopened.query("Globex", embedder, top_k=3)


def test_scores_are_cosine_distance():
    embedder = MockProvider(embed_dim=8)
    store = SqliteVecStore(dimension=embedder.dimension)
    store.build_and_write(_sample_graph(), embedder)
    hits = store.query("concept: Corporate Partnership", embedder, top_k=3)
    assert hits[0]["metadata"]["name"] == "Corporate Partnership"
    assert abs(hits[0]["score"]) < 1e-5          # identical text -> distance 0
    assert all(0.0 <= h["score"] <= 2.0 + 1e-6 for h in hits)
    assert hits[0]["metadata"]["node_type"] == "Concept"


def test_replace_all_swaps_the_file_in_one_step(tmp_path):
    import os

    embedder = MockProvider(embed_dim=8)
    path = str(tmp_path / "vectors.db")
    store = SqliteVecStore(dimension=embedder.dimension, path=path)
    store.build_and_write(_sample_graph(), embedder)
    assert store.count() == 3

    kg = SqliteGraphStore()
    src = kg.add_source("Only Initech here.")
    kg.add_entity("Initech", context={"source_id": src, "text": "Initech"})
    store.build_and_write(kg, embedder)                     # full rebuild via a temp file + rename
    assert store.count() == 1
    assert [r["metadata"]["name"] for r in store.query("Initech", embedder, top_k=5)] == ["Initech"]
    assert not os.path.exists(f"{path}.tmp")
    assert SqliteVecStore(dimension=embedder.dimension, path=path).count() == 1
