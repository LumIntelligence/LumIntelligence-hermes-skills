"""VectorRecord — a vector-store entry built from a graph node (Step 4)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VectorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str                                         # = the source node's id (1:1 with the graph)
    content: str                                    # text to embed
    metadata: dict = Field(default_factory=dict)    # {node_id, type, name}
    embedding: list[float] | None = None
