"""Shared, backend-agnostic record-embedding helper.

Batches ``embed()`` calls for any VectorStore backend; nothing here is backend-specific.
"""

from __future__ import annotations

from lumintelligence.llm.base import EmbeddingProvider, EmbedPurpose
from lumintelligence.models.records import VectorRecord

_EMBED_BATCH = 128  # cap per embed() call: avoids provider timeouts / request-size limits


def embed_records(records: list[VectorRecord], embedder: EmbeddingProvider) -> list[VectorRecord]:
    if not records:
        return records
    for i in range(0, len(records), _EMBED_BATCH):
        chunk = records[i:i + _EMBED_BATCH]
        vectors = embedder.embed([r.content for r in chunk], EmbedPurpose.DOCUMENT)
        for record, vector in zip(chunk, vectors):
            record.embedding = vector
    return records
