"""Canonical vocabulary for the knowledge graph — the single source of truth.

These are the *fixed* names that define the data model and must be identical
everywhere. Referencing ``NodeType.SOURCE`` instead of the bare string ``"Source"``
turns a typo into an import/attribute error at definition time, rather than a silent
KeyError deep in the pipeline at runtime (the class of bug behind the old
``attributes['Content']`` mismatch).

Distinct from :mod:`lumintelligence.config`, which holds *tunable* values
(models, thresholds, provider choice) that change per run or environment.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum


class NodeType(str, Enum):
    """The four network tiers. Values match the strings stored
    in existing graph JSON, so str-Enum keeps backward compatibility."""

    SOURCE = "Source"
    ENTITY = "Discovered_entity"
    CONCEPT = "Concept"
    CENTROID = "Centroid"


class EdgeType(str, Enum):
    """Edge relationships. All point *upward* toward abstraction, so lineage tracing
    is a single uniform predecessor-BFS to Source."""

    MENTIONS = "MENTIONS"        # Source  -> Entity
    IS_CONCEPT = "IS_CONCEPT"    # Entity  -> Concept
    BELONGS_TO = "BELONGS_TO"    # Concept -> Centroid


# Type-prefixed UUID4 IDs keep node ids human-scannable.
ID_PREFIX: dict[NodeType, str] = {
    NodeType.SOURCE: "SRC_",
    NodeType.ENTITY: "DISC_",
    NodeType.CONCEPT: "CONCEPT_",
    NodeType.CENTROID: "CENTROID_",
}

EDGE_ID_PREFIX = "EDGE_"


def new_id(prefix: str) -> str:
    """Generate a type-prefixed UUID4 id, e.g. ``SRC_3f9a...`` — unique, scannable.
    UUID gives uniqueness, not dedup; dedup is handled by name index, not by id."""
    return f"{prefix}{uuid.uuid4().hex}"


def now_iso() -> str:
    """UTC timestamp, canonical format for every timestamp field in the API/library:
    ``YYYY-MM-DDTHH:MM:SS`` — ISO-8601, second precision, no fractional seconds, no UTC
    offset suffix. Always UTC; callers must not assume local time. Matches the CRM's own
    ``initial_data_modeling.md`` convention so timestamps round-trip byte-for-byte."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")


class Attr:
    """Canonical attribute-dict keys used across tiers. Replaces scattered magic
    strings (notably the capital-C ``Content`` that silently broke on data lacking
    that exact column)."""

    CONTENT = "content"        # Source: the substance; NoD/embeddings read this
    METADATA = "metadata"      # Source: ancillary; logical search only, never parsed/embedded
    CONTEXT = "context"        # Entity: [{source_id, text}, ...]
    TAGS = "tags"              # Entity: [str] added by ADHD tagging
    INFERRED_TYPE = "inferred_type"  # Entity: free-form type classification from NoD
    EMBEDDING = "embedding"    # Concept: cached clustering-space vector
    MEMBERS = "members"        # Centroid: [concept_id] (authoritative membership is edges)
    IS_NEW = "is_new"          # Source: incremental marker (ingest sets True, NoD clears)
