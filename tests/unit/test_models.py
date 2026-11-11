"""Direct coverage for lumintelligence/models/{nodes,edges,records}.py — no
test anywhere previously constructed these Pydantic models directly to assert validation
behavior (existing test_stage4_schemas.py covers the unrelated api/schemas.py module)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lumintelligence.constants import EdgeType, NodeType
from lumintelligence.models.edges import Edge
from lumintelligence.models.nodes import (
    CentroidNode,
    ConceptNode,
    ContextEntry,
    EntityNode,
    SourceAttributes,
    SourceNode,
    content_hash,
)
from lumintelligence.models.records import VectorRecord


def test_source_attributes_rejects_blank_content():
    with pytest.raises(ValidationError):
        SourceAttributes(content="   ")


def test_source_attributes_backfills_content_hash_when_omitted():
    attrs = SourceAttributes(content="hello world", metadata={"a": 1})
    assert attrs.content_hash == content_hash("hello world", {"a": 1})


def test_source_attributes_preserves_explicit_content_hash():
    attrs = SourceAttributes(content="hello world", content_hash="explicit-hash")
    assert attrs.content_hash == "explicit-hash"


def test_named_node_rejects_blank_name():
    with pytest.raises(ValidationError):
        EntityNode(name="   ")


def test_context_entry_rejects_unexpected_keys():
    with pytest.raises(ValidationError):
        ContextEntry(source_id="s1", text="hi", extra_field="not allowed")


def test_source_attributes_rejects_unexpected_keys():
    with pytest.raises(ValidationError):
        SourceAttributes(content="hello", not_a_real_field=True)


def test_source_node_defaults_and_type():
    node = SourceNode(attributes=SourceAttributes(content="hello"))
    assert node.id.startswith("SRC_")
    assert node.type == NodeType.SOURCE
    assert node.is_new is True


def test_entity_node_defaults_empty_attributes_when_omitted():
    node = EntityNode(name="Acme Corp")
    assert node.id.startswith("DISC_")
    assert node.attributes.context == []
    assert node.attributes.tags is None  # sentinel: not yet tagged
    assert node.attributes.inferred_type is None  # free-form NoD classification


def test_concept_node_and_centroid_node_id_prefixes_and_defaults():
    concept = ConceptNode(name="Company")
    centroid = CentroidNode(name="Org cluster")

    assert concept.id.startswith("CONCEPT_")
    assert concept.attributes.embedding is None
    assert centroid.id.startswith("CENTROID_")
    assert centroid.attributes.members == []


def test_use_enum_values_makes_model_dump_emit_plain_strings():
    node = EntityNode(name="Acme Corp")
    assert node.model_dump()["type"] == "Discovered_entity"


def test_edge_defaults_and_enum_dump():
    edge = Edge(type=EdgeType.MENTIONS)
    assert edge.id.startswith("EDGE_")
    assert edge.attributes == {}
    assert edge.model_dump()["type"] == "MENTIONS"


def test_edge_rejects_unexpected_keys():
    with pytest.raises(ValidationError):
        Edge(type=EdgeType.MENTIONS, not_a_real_field=True)


def test_vector_record_basic_construction_and_defaults():
    record = VectorRecord(id="SRC_1", content="hello world")
    assert record.metadata == {}
    assert record.embedding is None


def test_vector_record_rejects_unexpected_keys():
    with pytest.raises(ValidationError):
        VectorRecord(id="SRC_1", content="hello", not_a_real_field=True)


def test_content_hash_is_stable_regardless_of_metadata_key_order():
    a = content_hash("hello", {"x": 1, "y": 2})
    b = content_hash("hello", {"y": 2, "x": 1})
    assert a == b


def test_content_hash_differs_for_different_content():
    assert content_hash("hello", {}) != content_hash("goodbye", {})
