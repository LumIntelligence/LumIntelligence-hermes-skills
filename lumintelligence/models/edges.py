"""Pydantic edge schema.

Edge *data* (id, type, attributes); the source/target node pair lives on the graph
endpoints, not in the model. All edge directions point upward:
MENTIONS (Source→Entity), IS_CONCEPT (Entity→Concept), BELONGS_TO (Concept→Centroid).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from lumintelligence.constants import EDGE_ID_PREFIX, EdgeType, new_id


class Edge(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    id: str = Field(default_factory=lambda: new_id(EDGE_ID_PREFIX))
    type: EdgeType
    attributes: dict = Field(default_factory=dict)
