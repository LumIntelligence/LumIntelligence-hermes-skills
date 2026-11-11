"""Pydantic node schemas for the four tiers.

These enforce the data model at creation time. ``use_enum_values=True`` makes
``model_dump()`` emit plain strings (``type: "Source"``), so stored node data stays
JSON-serializable and backward-compatible with existing graph files.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lumintelligence.constants import ID_PREFIX, NodeType, new_id, now_iso


def content_hash(content: str, metadata: dict) -> str:
    """sha256 of content + sorted metadata items — stable across dict key order.
    Used for ingest dedup; not yet wired to any
    lookup here, just computed and stored so later stages can compare cheaply."""
    payload = content + str(sorted(metadata.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Attribute sub-models (the typed `attributes` container per tier)
# --------------------------------------------------------------------------- #
class ContextEntry(BaseModel):
    """One mention of an entity by a source."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    text: str


class SourceAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str                                    # the substance — NoD/embeddings read this
    metadata: dict = Field(default_factory=dict)    # logical search only; never parsed/embedded
    content_hash: str | None = None              # sha256(content + sorted(metadata)); dedup key

    @field_validator("content")
    @classmethod
    def _content_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Source content must be non-empty")
        return v

    def model_post_init(self, context, /) -> None:
        if self.content_hash is None:
            self.content_hash = content_hash(self.content, self.metadata)


class EntityAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: list[ContextEntry] = Field(default_factory=list)
    tags: list[str] | None = None                # None = not yet tagged (incremental signal)
    inferred_type: str | None = None             # free-form NoD classification; canonicalized
                                                  # via embedding similarity, not a fixed taxonomy


class ConceptAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    embedding: list[float] | None = None         # cached clustering-space vector


class CentroidAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    members: list[str] = Field(default_factory=list)  # convenience; edges are authoritative


# --------------------------------------------------------------------------- #
# Node models
# --------------------------------------------------------------------------- #
class _BaseNode(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    created_at: str = Field(default_factory=now_iso)   # Source: ingest fallback if metadata lacks one;
                                                        # Entity/Concept/Centroid: extraction time
    updated_at: str | None = None                   # touched by mutation (context append, tagging, etc.)


class _NamedNode(_BaseNode):
    name: str

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class SourceNode(_BaseNode):
    id: str = Field(default_factory=lambda: new_id(ID_PREFIX[NodeType.SOURCE]))
    type: NodeType = NodeType.SOURCE
    is_new: bool = True
    attributes: SourceAttributes


class EntityNode(_NamedNode):
    id: str = Field(default_factory=lambda: new_id(ID_PREFIX[NodeType.ENTITY]))
    type: NodeType = NodeType.ENTITY
    attributes: EntityAttributes = Field(default_factory=EntityAttributes)


class ConceptNode(_NamedNode):
    id: str = Field(default_factory=lambda: new_id(ID_PREFIX[NodeType.CONCEPT]))
    type: NodeType = NodeType.CONCEPT
    attributes: ConceptAttributes = Field(default_factory=ConceptAttributes)


class CentroidNode(_NamedNode):
    id: str = Field(default_factory=lambda: new_id(ID_PREFIX[NodeType.CENTROID]))
    type: NodeType = NodeType.CENTROID
    attributes: CentroidAttributes = Field(default_factory=CentroidAttributes)
