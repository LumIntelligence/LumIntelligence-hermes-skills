"""VectorStore — the contract every vector backend implements.

Query's entry-point search and the pipeline's vectorize step call only these methods, so the
backend (sqlite-vec today) can be swapped without touching the engine.

Score semantics are fixed across backends: ``score`` is **cosine distance** (``1 - cosine
similarity``, lower is closer), so ``Config.entry_cutoff`` means the same thing everywhere.
Hits are ``{"node_id", "score", "metadata": {"node_id", "node_type", "name"}, "content"}``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lumintelligence.llm.base import EmbeddingProvider
from lumintelligence.models.records import VectorRecord


@runtime_checkable
class VectorStore(Protocol):
    dimension: int

    def write(self, records: list[VectorRecord]) -> None: ...
    def replace_all(self, records: list[VectorRecord]) -> None:
        """Replace the whole index; readers see the old or the new index, never an empty one."""
        ...
    def sync(self, records: list[VectorRecord], embedder: EmbeddingProvider, full: bool = False) -> dict:
        """Make the index match ``records`` (un-embedded): embed only new or changed texts, delete
        ids not in ``records``. ``full`` (or a changed embedding model) re-embeds everything.
        Returns counts: added, updated, deleted, unchanged, embedded, full."""
        ...
    def reset(self) -> None: ...
    def count(self) -> int: ...
    def query(self, query_text: str, embedder: EmbeddingProvider, top_k: int = 5,
              max_distance: float | None = None) -> list[dict]: ...
