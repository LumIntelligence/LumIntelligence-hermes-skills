"""Storage layer: the VectorStore contract, backend-agnostic record building and embedding,
and the default sqlite-vec backend."""

from lumintelligence.storage.embedding import embed_records
from lumintelligence.storage.protocol import VectorStore
from lumintelligence.storage.records import build_records
from lumintelligence.storage.sqlite_vec_store import SqliteVecStore

__all__ = ["SqliteVecStore", "VectorStore", "build_records", "embed_records"]
