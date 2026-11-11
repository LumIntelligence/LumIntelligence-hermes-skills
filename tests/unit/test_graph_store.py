"""Direct SqliteGraphStore coverage for methods only exercised transitively
elsewhere: concept/centroid creation, lookups, linking, incremental queues, embedding
accessors, centroid helpers, persistence, and error paths."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lumintelligence.graph import SqliteGraphStore


def test_add_concept_dedups_by_name_and_ignores_new_embedding_on_reuse():
    kg = SqliteGraphStore()
    first = kg.add_concept("Partnership", embedding=[0.1, 0.2])
    second = kg.add_concept("partnership ", embedding=[0.9, 0.9])  # different case/whitespace

    assert first == second
    # Re-adding an existing concept does not overwrite its embedding with the new call's value.
    assert kg.concept_embedding(first) == [0.1, 0.2]


def test_add_centroid_never_dedups_even_with_same_name():
    kg = SqliteGraphStore()
    first = kg.add_centroid("Cluster A")
    second = kg.add_centroid("Cluster A")

    assert first != second


def test_find_by_name_returns_none_when_absent():
    kg = SqliteGraphStore()
    assert kg.find_source_by_hash("nope") is None
    assert kg.find_entity_by_name("nobody") is None
    assert kg.find_concept_by_name("nothing") is None


def test_link_is_concept_records_original_trigger():
    kg = SqliteGraphStore()
    src = kg.add_source("content")
    entity = kg.add_entity("Acme")
    concept = kg.add_concept("Company")

    kg.link_is_concept(entity, concept, original_trigger="Company")

    edge_data = kg.edge_data(entity, concept)
    assert edge_data["attributes"]["original_trigger"] == "Company"
    assert src  # keep source referenced so the fixture reads as a full mini-graph


def test_link_belongs_to_is_idempotent_and_returns_same_edge_id():
    kg = SqliteGraphStore()
    concept = kg.add_concept("Company")
    centroid = kg.add_centroid("Org cluster")

    first = kg.link_belongs_to(concept, centroid)
    second = kg.link_belongs_to(concept, centroid)

    assert first == second
    assert kg.edge_count() == 1  # DiGraph allows only one edge per ordered pair


def test_append_entity_context_adds_to_existing_context_list():
    kg = SqliteGraphStore()
    src = kg.add_source("content")
    entity = kg.add_entity("Acme", context={"source_id": src, "text": "first mention"})

    kg.append_entity_context(entity, {"source_id": src, "text": "second mention"})

    node = kg.get_node(entity)
    assert [c["text"] for c in node["attributes"]["context"]] == ["first mention", "second mention"]
    assert node["updated_at"] is not None


def test_set_entity_tags_overwrites_and_removes_entity_from_tag_queue():
    kg = SqliteGraphStore()
    entity = kg.add_entity("Acme")
    assert entity in kg.entities_to_tag()

    kg.set_entity_tags(entity, ["company", "vendor"])

    assert kg.get_node(entity)["attributes"]["tags"] == ["company", "vendor"]
    assert entity not in kg.entities_to_tag()


def test_set_entity_tags_with_empty_list_still_removes_from_queue():
    """entities_to_tag() checks `tags is not None`, not non-empty — an entity tagged with
    zero tags (e.g. all LLM retry attempts exhausted) must still leave the queue, or NoD
    would retry it forever."""
    kg = SqliteGraphStore()
    entity = kg.add_entity("Acme")

    kg.set_entity_tags(entity, [])

    assert entity not in kg.entities_to_tag()


def test_mark_source_processed_removes_source_from_queue():
    kg = SqliteGraphStore()
    src = kg.add_source("content")
    assert src in kg.sources_to_process()

    kg.mark_source_processed(src)

    assert src not in kg.sources_to_process()


def test_neighbors_returns_empty_list_for_unknown_node():
    kg = SqliteGraphStore()
    assert kg.neighbors("nope") == []


def test_neighbors_out_in_and_both_directions():
    kg = SqliteGraphStore()
    src = kg.add_source("content")
    entity = kg.add_entity("Acme")
    kg.link_mentions(src, entity)

    assert kg.neighbors(src, "out") == [entity]
    assert kg.neighbors(entity, "in") == [src]
    assert set(kg.neighbors(src, "both")) == {entity}


def test_neighbors_raises_on_invalid_direction_for_known_node():
    kg = SqliteGraphStore()
    src = kg.add_source("content")

    with pytest.raises(ValueError, match="Invalid direction"):
        kg.neighbors(src, "sideways")


def test_neighbors_returns_empty_before_validating_direction_for_unknown_node():
    """Edge case: an unknown node_id short-circuits to [] before the direction argument
    is even checked, so an invalid direction with an unknown node silently returns []
    instead of raising — documenting this so a future refactor doesn't accidentally
    change the observable order of these two checks without noticing."""
    kg = SqliteGraphStore()
    assert kg.neighbors("nope", "sideways") == []


def test_trace_sources_follows_full_lineage_from_centroid_to_source():
    kg = SqliteGraphStore()
    src = kg.add_source("content")
    entity = kg.add_entity("Acme")
    concept = kg.add_concept("Company")
    centroid = kg.add_centroid("Org cluster")
    kg.link_mentions(src, entity)
    kg.link_is_concept(entity, concept)
    kg.link_belongs_to(concept, centroid)

    assert kg.trace_sources(centroid) == [src]


def test_trace_sources_raises_for_unknown_start_id():
    kg = SqliteGraphStore()
    with pytest.raises(KeyError):
        kg.trace_sources("nope")


def test_concepts_unconsolidated_excludes_concepts_with_belongs_to_edge():
    kg = SqliteGraphStore()
    concept = kg.add_concept("Company")
    assert concept in kg.concepts_unconsolidated()

    centroid = kg.add_centroid("Org cluster")
    kg.link_belongs_to(concept, centroid)

    assert concept not in kg.concepts_unconsolidated()


def test_concept_embedding_accessors():
    kg = SqliteGraphStore()
    concept = kg.add_concept("Company")
    assert kg.concept_embedding(concept) is None
    assert concept in kg.concepts_missing_embedding()

    kg.set_concept_embedding(concept, [0.1, 0.2, 0.3])

    assert kg.concept_embedding(concept) == [0.1, 0.2, 0.3]
    assert concept not in kg.concepts_missing_embedding()


def test_assign_concept_to_centroid_is_idempotent_in_members_list():
    kg = SqliteGraphStore()
    concept = kg.add_concept("Company")
    centroid = kg.add_centroid("Org cluster")

    kg.assign_concept_to_centroid(concept, centroid)
    kg.assign_concept_to_centroid(concept, centroid)  # double-call must not duplicate

    assert kg.centroid_members(centroid) == [concept]
    assert kg.get_node(centroid)["attributes"]["members"] == [concept]


def test_clear_centroids_removes_centroids_but_preserves_concepts_and_their_edges():
    kg = SqliteGraphStore()
    entity = kg.add_entity("Acme")
    concept = kg.add_concept("Company")
    centroid = kg.add_centroid("Org cluster")
    kg.link_is_concept(entity, concept)
    kg.assign_concept_to_centroid(concept, centroid)

    kg.clear_centroids()

    assert kg.nodes_by_type("Centroid") == []
    assert concept in kg.nodes_by_type("Concept")
    assert kg.neighbors(entity, "out") == [concept]  # IS_CONCEPT edge survives
    assert concept in kg.concepts_unconsolidated()  # its BELONGS_TO edge is gone


def test_kb_file_reopens_with_the_same_contents(tmp_path):
    path = tmp_path / "kb.db"
    kg = SqliteGraphStore(path=str(path))
    src = kg.add_source("content")
    entity = kg.add_entity("Acme", context={"source_id": src, "text": "mention"})
    kg.link_mentions(src, entity)

    kg.close()
    kg = SqliteGraphStore(path=str(path))
    reloaded = SqliteGraphStore(path=str(path))

    assert reloaded.node_count() == kg.node_count()
    assert reloaded.edge_count() == kg.edge_count()
    assert reloaded.find_entity_by_name("Acme") == entity
    assert reloaded.find_source_by_hash(kg.get_node(src)["attributes"]["content_hash"]) == src


def test_add_edge_raises_key_error_for_missing_endpoint():
    kg = SqliteGraphStore()
    entity = kg.add_entity("Acme")

    with pytest.raises(KeyError):
        kg.link_mentions("nonexistent_source", entity)


def test_add_source_raises_on_blank_content():
    kg = SqliteGraphStore()
    with pytest.raises(ValidationError):
        kg.add_source("   ")


def test_add_entity_raises_on_blank_name():
    kg = SqliteGraphStore()
    with pytest.raises(ValidationError):
        kg.add_entity("   ")


def test_add_entity_stores_inferred_type_on_creation():
    kg = SqliteGraphStore()
    entity = kg.add_entity("Acme Corp", inferred_type="Organization")
    assert kg.get_node(entity)["attributes"]["inferred_type"] == "Organization"


def test_add_entity_inferred_type_is_first_write_wins_on_reuse():
    kg = SqliteGraphStore()
    entity = kg.add_entity("Acme Corp", inferred_type="Organization")
    same_entity = kg.add_entity("Acme Corp", inferred_type="Nonprofit")

    assert same_entity == entity
    assert kg.get_node(entity)["attributes"]["inferred_type"] == "Organization"


def test_add_entity_inferred_type_can_be_set_later_if_initially_absent():
    kg = SqliteGraphStore()
    entity = kg.add_entity("Acme Corp")
    kg.add_entity("Acme Corp", inferred_type="Organization")

    assert kg.get_node(entity)["attributes"]["inferred_type"] == "Organization"
