"""Content-hash idempotency at the SqliteGraphStore/Ingestor layer: dedup within one ingest call,
and dedup across two separate calls on the same graph."""

from __future__ import annotations

from lumintelligence.graph import SqliteGraphStore
from lumintelligence.ingestion.ingest import Ingestor


def test_same_unit_twice_in_one_call_yields_one_source():
    kg = SqliteGraphStore()
    ids = Ingestor().run(kg, [{"content": "hello world"}, {"content": "hello world"}])

    assert ids[0] == ids[1]
    assert kg.node_count() == 1


def test_same_unit_across_two_calls_with_graph_fed_back_yields_one_source():
    kg = SqliteGraphStore()
    first_ids = Ingestor().run(kg, [{"content": "hello world"}])

    # A retried or repeated ingest of the same unit on the same graph reuses the source.
    second_ids = Ingestor().run(kg, [{"content": "hello world"}])

    assert first_ids == second_ids
    assert kg.node_count() == 1


def test_different_metadata_is_not_deduped():
    kg = SqliteGraphStore()
    ids = Ingestor().run(kg, [
        {"content": "hello world", "metadata": {"source": "a"}},
        {"content": "hello world", "metadata": {"source": "b"}},
    ])

    assert ids[0] != ids[1]
    assert kg.node_count() == 2


def test_always_new_opts_out_of_dedup():
    kg = SqliteGraphStore()
    ids = Ingestor().run(kg, [{"content": "hello world"}, {"content": "hello world"}],
                         idempotency="always_new")

    assert ids[0] != ids[1]
    assert kg.node_count() == 2


def test_import_backfills_content_hash_for_legacy_sources_missing_it():
    """A Source saved to graph.json before content_hash existed has no hash; importing it must
    compute one so later ingests still dedup against it."""
    kg = SqliteGraphStore()
    sid = kg.add_source("hello world")
    data = kg.export_json()
    del data["nodes"][0]["attributes"]["content_hash"]   # simulate a legacy node

    reloaded = SqliteGraphStore()
    reloaded.import_node_link(data)
    ids = Ingestor().run(reloaded, [{"content": "hello world"}])
    assert ids[0] == sid   # still deduped correctly despite the missing hash on load
