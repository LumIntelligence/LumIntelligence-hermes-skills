"""Vector-record building — one record per Entity/Concept/Centroid node, from any GraphStore.

Backend-agnostic: works with any GraphStore and VectorStore.
"""

from __future__ import annotations

from lumintelligence.constants import Attr, NodeType
from lumintelligence.graph.protocol import GraphStore
from lumintelligence.models.records import VectorRecord


def entity_text(node: dict) -> str:
    tags = node["attributes"].get(Attr.TAGS) or []
    inferred_type = node["attributes"].get(Attr.INFERRED_TYPE)
    type_part = f" | inferred_type: {inferred_type}" if inferred_type else ""
    return f"entity: {node['name']} | tags: {', '.join(tags)}{type_part}"


def concept_text(node: dict) -> str:
    return f"concept: {node['name']}"


def centroid_text(kg: GraphStore, centroid_id: str, node: dict) -> str:
    members = [n["name"] for m in kg.centroid_members(centroid_id) if (n := kg.get_node(m)) is not None]
    return f"centroid: {node['name']} | members: {', '.join(members)}"


def build_records(kg: GraphStore) -> list[VectorRecord]:
    records: list[VectorRecord] = []
    for node_type in (NodeType.ENTITY, NodeType.CONCEPT, NodeType.CENTROID):
        for nid in kg.nodes_by_type(node_type):
            node = kg.get_node(nid)
            assert node is not None
            if node_type is NodeType.ENTITY:
                text = entity_text(node)
            elif node_type is NodeType.CONCEPT:
                text = concept_text(node)
            else:
                text = centroid_text(kg, nid, node)
            meta = {"node_id": nid, "node_type": node["type"], "name": node["name"]}
            records.append(VectorRecord(id=nid, content=text, metadata=meta))
    return records
