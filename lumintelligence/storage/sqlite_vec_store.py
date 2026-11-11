"""SqliteVecStore — the default VectorStore: sqlite-vec, in memory, in a file, or on a shared
connection (one ``kb.db`` per knowledge base with the graph).

One row per Entity/Concept/Centroid node, keyed by node id (writing a node again replaces
it). Distances are cosine (``distance_metric=cosine``), so scores follow the VectorStore
contract (``1 - cosine``).

``SqliteVecStore(dim)`` is in memory; ``path=`` opens a file; ``conn=`` shares a connection.
The dimension is recorded in the ``meta`` table, and a mismatch is refused (a KB is tied to one
embedding model). ``replace_all`` is one transaction, so readers on other connections see the
old index or the new one, never an empty one.
"""

from __future__ import annotations

import sqlite_vec

from lumintelligence.graph.protocol import GraphStore
from lumintelligence.llm.base import EmbeddingProvider, EmbedPurpose
from lumintelligence.models.records import VectorRecord
from lumintelligence.storage.embedding import embed_records
from lumintelligence.storage.records import build_records
from lumintelligence.storage.sqlite_conn import Connection, atomic, connect

_TABLE = "vec_items"


class SqliteVecStore:
    def __init__(self, dimension: int, path: str | None = None, conn: Connection | None = None):
        self.dimension = dimension
        self.path = path
        self._owns_conn = conn is None
        self._conn = conn if conn is not None else connect(path)
        self._check_dimension()
        self._ensure_table()

    def _check_dimension(self) -> None:
        if not self._table_exists("meta"):
            self._conn.execute("create table if not exists meta(key text primary key, value text)")
        row = self._conn.execute("select value from meta where key = 'embedding_dim'").fetchone()
        if row is None:
            self._conn.execute("insert into meta(key, value) values ('embedding_dim', ?)", (str(self.dimension),))
        elif int(row[0]) != self.dimension:
            raise ValueError(f"This knowledge base uses {row[0]}-dimension embeddings; the configured "
                             f"embedding model has {self.dimension}. Use the model the KB was built with.")

    def _table_exists(self, name: str) -> bool:
        return self._conn.execute("select 1 from sqlite_master where name = ?", (name,)).fetchone() is not None

    def _ensure_table(self) -> None:
        if self._table_exists(_TABLE):   # existing KB: no write, so opening never waits on a build
            return
        self._conn.execute(
            f"create virtual table if not exists {_TABLE} using vec0("
            f"node_id text primary key, embedding float[{self.dimension}] distance_metric=cosine, "
            f"+node_type text, +name text, +content text)"
        )

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    def reset(self) -> None:
        self._conn.execute(f"delete from {_TABLE}")

    def write(self, records: list[VectorRecord]) -> None:
        records = [r for r in records if r.embedding is not None]
        if not records:
            return
        with atomic(self._conn):
            self._insert(records)

    def _insert(self, records: list[VectorRecord]) -> None:
        self._conn.executemany(f"delete from {_TABLE} where node_id = ?", [(r.id,) for r in records])
        self._conn.executemany(
            f"insert into {_TABLE}(node_id, embedding, node_type, name, content) values (?, ?, ?, ?, ?)",
            [
                (r.id, sqlite_vec.serialize_float32(r.embedding),  # type: ignore[arg-type]
                 r.metadata.get("node_type"), r.metadata.get("name"), r.content)
                for r in records
            ],
        )

    def replace_all(self, records: list[VectorRecord]) -> None:
        """Full rebuild in one transaction (or inside the build's transaction)."""
        records = [r for r in records if r.embedding is not None]
        with atomic(self._conn):
            self._conn.execute(f"delete from {_TABLE}")
            if records:
                self._insert(records)

    def sync(self, records: list[VectorRecord], embedder: EmbeddingProvider, full: bool = False) -> dict:
        """Incremental update: compare each record's text with the stored ``content``; embed
        and write only new or changed records; delete ids that are no longer in ``records``. A
        recorded embedding model that differs from ``embedder.model_id`` (or none recorded while
        vectors exist) forces a full re-embed. One transaction, or a savepoint in the build's."""
        model = embedder.model_id
        stored = self._conn.execute("select value from meta where key = 'embedding_model'").fetchone()
        existing = dict(self._conn.execute(f"select node_id, content from {_TABLE}").fetchall())
        if model is not None and (stored[0] if stored else None) != model and existing:
            full = True
        wanted = {r.id: r for r in records}
        deleted = [nid for nid in existing if nid not in wanted]
        if full:
            todo = list(wanted.values())
            added = sum(1 for r in todo if r.id not in existing)
            updated = len(todo) - added
        else:
            todo = [r for r in wanted.values() if existing.get(r.id) != r.content]
            added = sum(1 for r in todo if r.id not in existing)
            updated = len(todo) - added
        embed_records(todo, embedder)
        with atomic(self._conn):
            if deleted:
                self._conn.executemany(f"delete from {_TABLE} where node_id = ?", [(n,) for n in deleted])
            todo = [r for r in todo if r.embedding is not None]
            if todo:
                self._insert(todo)
            if model is not None:
                self._conn.execute("insert or replace into meta(key, value) values ('embedding_model', ?)",
                                   (model,))
        return {"added": added, "updated": updated, "deleted": len(deleted),
                "unchanged": len(wanted) - added - updated, "embedded": len(todo), "full": full}

    def build_and_write(self, kg: GraphStore, embedder: EmbeddingProvider) -> int:
        """Full rebuild from the graph: build records, embed, replace the index. Returns records written."""
        records = embed_records(build_records(kg), embedder)
        self.replace_all(records)
        return len(records)

    def count(self) -> int:
        row = self._conn.execute(f"select count(*) from {_TABLE}").fetchone()
        return row[0] if row else 0

    def query(self, query_text: str, embedder: EmbeddingProvider, top_k: int = 5,
              max_distance: float | None = None) -> list[dict]:
        """Nearest neighbours by cosine distance (lower is closer), at most ``top_k``."""
        if self.count() == 0:
            return []
        qv = embedder.embed_one(query_text, EmbedPurpose.QUERY)
        rows = self._conn.execute(
            f"select node_id, distance, node_type, name, content from {_TABLE} "
            f"where embedding match ? and k = ? order by distance",
            (sqlite_vec.serialize_float32(qv), top_k),
        ).fetchall()
        return [
            {"node_id": node_id, "score": distance,
             "metadata": {"node_id": node_id, "node_type": node_type, "name": name}, "content": content}
            for node_id, distance, node_type, name, content in rows
            if max_distance is None or distance <= max_distance
        ]
