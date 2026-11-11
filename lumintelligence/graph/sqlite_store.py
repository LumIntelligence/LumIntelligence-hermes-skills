"""SqliteGraphStore — the graph in SQLite, with provenance origins.

Every node/edge write is validated by the same Pydantic models as before, so the data model is
enforced in one place. ``get_node`` returns the same dict shape the stages have always read
(``id``, ``type``, ``name``, ``attributes``, ...); Sources also carry ``origin_id``/``position``.

Layout (schema version 1):
- ``nodes``: fixed, indexed columns for lookups and work queues; everything else in ``data`` (JSON
  of the node's ``attributes``). Dedup of entities/concepts is by normalized name (unique index).
- ``edges``: one typed edge per (src, dst) pair. All edges point upward
  (Source → Entity → Concept → Centroid), so lineage is a recursive query over incoming edges.
- ``concept_embeddings``: clustering vectors, kept out of node rows and out of the query index.
- ``origins``: one row per ingested original (file, chat, note, records).

``SqliteGraphStore()`` is in memory; pass ``path`` for a file, or ``conn`` to share a connection
with ``SqliteVecStore`` (one ``kb.db``). ``begin/commit/rollback`` wrap a whole build.
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections import deque

from lumintelligence.constants import Attr, EdgeType, NodeType, new_id, now_iso
from lumintelligence.models.edges import Edge
from lumintelligence.models.nodes import (
    CentroidNode,
    ConceptNode,
    ContextEntry,
    EntityNode,
    SourceNode,
    content_hash,
)
from lumintelligence.storage.sqlite_conn import Connection, atomic, connect

ContextLike = ContextEntry | dict

SCHEMA_VERSION = "1"
ORIGIN_PREFIX = "ORIGIN_"

_SCHEMA = """
create table if not exists meta(key text primary key, value text);
create table if not exists origins(
  id text primary key, kind text not null, created_at text not null, sha256 text not null,
  label text, extra text);
create index if not exists origins_sha on origins(sha256);
create table if not exists nodes(
  id text primary key, type text not null, name text, name_key text,
  content_hash text, is_new integer, has_tags integer not null default 0,
  origin_id text, position text, created_at text, updated_at text, data text not null);
create unique index if not exists nodes_entity_name on nodes(name_key) where type = 'Discovered_entity';
create unique index if not exists nodes_concept_name on nodes(name_key) where type = 'Concept';
create index if not exists nodes_source_hash on nodes(content_hash) where type = 'Source';
create index if not exists nodes_type on nodes(type, is_new);
create index if not exists nodes_origin on nodes(origin_id) where origin_id is not null;
create table if not exists edges(
  id text primary key, src text not null, dst text not null, type text not null, data text,
  unique(src, dst));
create index if not exists edges_dst on edges(dst, type);
create table if not exists concept_embeddings(node_id text primary key, vector blob not null);
"""


def _schema_ready(conn: Connection) -> bool:
    """True if the schema is already in place, checked with reads only."""
    try:
        row = conn.execute("select value from meta where key = 'schema_version'").fetchone()
    except Exception:  # noqa: BLE001 — no meta table yet
        return False
    return row is not None and row[0] == SCHEMA_VERSION


def normalize_name(name: str) -> str:
    """Dedup key: lowercase + trim. Mechanical only — no NLP."""
    return name.strip().lower()


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None so output is valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}d", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 8}d", blob))


class SqliteGraphStore:
    def __init__(self, path: str | None = None, conn: Connection | None = None):
        self.path = path
        self._conn = conn if conn is not None else connect(path)
        if _schema_ready(self._conn):   # existing KB: open without writing (a build may hold the lock)
            return
        for stmt in _SCHEMA.split(";"):  # not executescript: it would commit an open transaction
            if stmt.strip():
                self._conn.execute(stmt)
        self._conn.execute("insert or ignore into meta(key, value) values ('schema_version', ?)",
                           (SCHEMA_VERSION,))

    @property
    def conn(self) -> Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------ transactions
    def begin(self) -> None:
        """Start the build transaction: nothing is visible to other readers until ``commit``."""
        self._conn.execute("begin immediate")

    def commit(self) -> None:
        if self._conn.in_transaction:
            self._conn.execute("commit")

    def rollback(self) -> None:
        if self._conn.in_transaction:
            self._conn.execute("rollback")

    # ---------------------------------------------------------------- origins
    def add_origin(self, kind: str, sha256: str, label: str | None = None, extra: dict | None = None,
                   created_at: str | None = None) -> str:
        """Register an ingested original. ``created_at`` = first ingested; set here."""
        if not kind or not sha256:
            raise ValueError("origin needs a kind and a sha256")
        oid = new_id(ORIGIN_PREFIX)
        self._conn.execute(
            "insert into origins(id, kind, created_at, sha256, label, extra) values (?, ?, ?, ?, ?, ?)",
            (oid, kind, created_at or now_iso(), sha256, label, json.dumps(extra or {})))
        return oid

    def get_origin(self, origin_id: str) -> dict | None:
        row = self._conn.execute("select id, kind, created_at, sha256, label, extra from origins where id = ?",
                                 (origin_id,)).fetchone()
        return self._origin_dict(row) if row else None

    def find_origin_by_sha(self, sha256: str) -> str | None:
        row = self._conn.execute("select id from origins where sha256 = ? order by rowid limit 1",
                                 (sha256,)).fetchone()
        return row[0] if row else None

    def list_origins(self) -> list[dict]:
        rows = self._conn.execute(
            "select id, kind, created_at, sha256, label, extra from origins order by rowid").fetchall()
        return [self._origin_dict(r) for r in rows]

    def find_origins(self, key: str, value: str) -> list[str]:
        """Origins whose ``extra[key]`` equals ``value`` or, for a list, contains it (e.g. a file
        path in ``extra.paths``, later a chat session id). Oldest first."""
        if not key.replace("_", "").isalnum():
            raise ValueError(f"invalid extra key: {key!r}")
        return self._ids(
            f"select id from origins o where exists(select 1 from json_each(o.extra, '$.{key}') "
            f"where value = ?) order by rowid", (value,))

    def update_origin_extra(self, origin_id: str, patch: dict) -> None:
        """Merge ``patch`` into the origin's ``extra`` (top-level keys replaced)."""
        origin = self.get_origin(origin_id)
        if origin is None:
            raise KeyError(f"Origin not found: {origin_id}")
        extra = {**origin["extra"], **patch}
        self._conn.execute("update origins set extra = ? where id = ?", (json.dumps(extra), origin_id))

    def add_origin_path(self, origin_id: str, path: str) -> None:
        """Record another location of the same content (a copy): appended to ``extra.paths``."""
        origin = self.get_origin(origin_id)
        if origin is None:
            raise KeyError(f"Origin not found: {origin_id}")
        paths = list(origin["extra"].get("paths") or [])
        if path not in paths:
            self.update_origin_extra(origin_id, {"paths": [*paths, path]})

    def sources_by_origin(self, origin_ids: list[str]) -> list[str]:
        rows = self._conn.execute(
            "select id from nodes where type = 'Source' and origin_id in (select value from json_each(?)) "
            "order by rowid", (json.dumps(list(origin_ids)),)).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _origin_dict(row) -> dict:
        oid, kind, created_at, sha, label, extra = row
        return {"id": oid, "kind": kind, "created_at": created_at, "sha256": sha, "label": label,
                "extra": json.loads(extra) if extra else {}}

    # ------------------------------------------------------------ node create
    def add_source(self, content: str, metadata: dict | None = None, is_new: bool = True,
                   idempotency: str = "content_hash", origin_id: str | None = None,
                   position: dict | None = None) -> str:
        """``idempotency="content_hash"`` (default): reuse the Source whose
        ``sha256(content + sorted(metadata))`` matches instead of creating a duplicate.
        ``"always_new"`` opts out. ``origin_id``/``position`` record provenance; the hash
        includes ``origin_id``, so each origin (e.g. each version of a file) has its own sources."""
        metadata = metadata or {}
        h = content_hash(content, metadata if origin_id is None else {**metadata, "__origin_id": origin_id})
        if idempotency == "content_hash":
            existing = self.find_source_by_hash(h)
            if existing is not None:
                return existing
        if origin_id is not None and self.get_origin(origin_id) is None:
            raise KeyError(f"Origin not found: {origin_id}")
        created_at = metadata.get("created_at") or now_iso()
        node = SourceNode(is_new=is_new, created_at=created_at,
                          attributes={"content": content, "metadata": metadata, "content_hash": h})  # type: ignore[arg-type]
        self._insert_node(node.model_dump(), origin_id=origin_id, position=position)
        return node.id

    def find_source_by_hash(self, content_hash_value: str) -> str | None:
        row = self._conn.execute(
            "select id from nodes where type = 'Source' and content_hash = ? order by rowid limit 1",
            (content_hash_value,)).fetchone()
        return row[0] if row else None

    def add_entity(self, name: str, context: ContextLike | None = None,
                   inferred_type: str | None = None) -> str:
        """Create an entity, or reuse the one with the same normalized name (appending context).
        ``inferred_type`` is first-write-wins on reuse."""
        existing = self.find_entity_by_name(name)
        if existing is not None:
            with atomic(self._conn):
                if context is not None:
                    self.append_entity_context(existing, context)
                if inferred_type is not None:
                    attrs = self._attrs(existing)
                    if attrs.get(Attr.INFERRED_TYPE) is None:
                        attrs[Attr.INFERRED_TYPE] = inferred_type
                        self._save_attrs(existing, attrs)
            return existing
        ctx = [self._coerce_context(context)] if context is not None else []
        node = EntityNode(name=name, attributes={"context": ctx, "inferred_type": inferred_type})  # type: ignore[arg-type]
        self._insert_node(node.model_dump())
        return node.id

    def add_concept(self, name: str, embedding: list[float] | None = None) -> str:
        """Create an atomic concept, or reuse the one with the same name (its embedding is kept)."""
        existing = self.find_concept_by_name(name)
        if existing is not None:
            return existing
        node = ConceptNode(name=name)
        with atomic(self._conn):
            self._insert_node(node.model_dump())
            if embedding is not None:
                self._write_embedding(node.id, embedding)
        return node.id

    def add_centroid(self, name: str, members: list[str] | None = None) -> str:
        node = CentroidNode(name=name, attributes={"members": list(members or [])})  # type: ignore[arg-type]
        self._insert_node(node.model_dump())
        return node.id

    def _insert_node(self, d: dict, origin_id: str | None = None, position: dict | None = None) -> None:
        attrs = dict(d.get("attributes") or {})
        attrs.pop(Attr.EMBEDDING, None)
        ntype = d["type"]
        name = d.get("name")
        try:
            self._conn.execute(
                "insert into nodes(id, type, name, name_key, content_hash, is_new, has_tags, origin_id, "
                "position, created_at, updated_at, data) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (d["id"], ntype, name, normalize_name(name) if name is not None else None,
                 attrs.get("content_hash") if ntype == NodeType.SOURCE.value else None,
                 (1 if d.get(Attr.IS_NEW) else 0) if ntype == NodeType.SOURCE.value else None,
                 1 if attrs.get(Attr.TAGS) is not None else 0, origin_id,
                 json.dumps(position) if position is not None else None,
                 d.get("created_at"), d.get("updated_at"), json.dumps(_sanitize_for_json(attrs))))
        except Exception as exc:  # sqlite3.IntegrityError from either sqlite module
            if "UNIQUE" in str(exc) and "nodes.id" in str(exc):
                raise ValueError(f"Node id collision: {d['id']}") from exc
            raise

    # ------------------------------------------------------------ edge create
    def link_mentions(self, source_id: str, entity_id: str) -> str:
        return self._add_edge(source_id, entity_id, EdgeType.MENTIONS)

    def link_is_concept(self, entity_id: str, concept_id: str, original_trigger: str | None = None) -> str:
        attrs = {"original_trigger": original_trigger} if original_trigger else {}
        return self._add_edge(entity_id, concept_id, EdgeType.IS_CONCEPT, attrs)

    def link_belongs_to(self, concept_id: str, centroid_id: str) -> str:
        return self._add_edge(concept_id, centroid_id, EdgeType.BELONGS_TO)

    def _add_edge(self, u: str, v: str, etype: EdgeType, attributes: dict | None = None) -> str:
        if not self._exists(u):
            raise KeyError(f"Source node not found: {u}")
        if not self._exists(v):
            raise KeyError(f"Target node not found: {v}")
        row = self._conn.execute("select id from edges where src = ? and dst = ?", (u, v)).fetchone()
        if row:
            return row[0]  # one edge per ordered pair
        edge = Edge(type=etype, attributes=attributes or {})
        self._conn.execute("insert into edges(id, src, dst, type, data) values (?, ?, ?, ?, ?)",
                           (edge.id, u, v, edge.type, json.dumps(edge.attributes)))
        return edge.id

    def edge_data(self, u: str, v: str) -> dict | None:
        """The edge ``u → v`` as ``{"id", "type", "attributes"}``, or None."""
        row = self._conn.execute("select id, type, data from edges where src = ? and dst = ?", (u, v)).fetchone()
        return {"id": row[0], "type": row[1], "attributes": json.loads(row[2] or "{}")} if row else None

    # --------------------------------------------------------------- mutation
    def append_entity_context(self, entity_id: str, context: ContextLike) -> None:
        attrs = self._attrs(entity_id)
        attrs.setdefault("context", []).append(self._coerce_context(context))
        self._save_attrs(entity_id, attrs)

    def set_entity_tags(self, entity_id: str, tags: list[str]) -> None:
        attrs = self._attrs(entity_id)
        attrs[Attr.TAGS] = list(tags)
        self._save_attrs(entity_id, attrs)

    def mark_source_processed(self, source_id: str) -> None:
        self._conn.execute("update nodes set is_new = 0, updated_at = ? where id = ?", (now_iso(), source_id))

    def reset_sources_for_reprocessing(self, source_ids: list[str]) -> None:
        """Flip existing sources back to is_new so the next NoD run re-extracts them (resume)."""
        self._conn.execute(
            "update nodes set is_new = 1, updated_at = ? where type = 'Source' "
            "and id in (select value from json_each(?))", (now_iso(), json.dumps(list(source_ids))))

    @staticmethod
    def _coerce_context(context: ContextLike) -> dict:
        if isinstance(context, ContextEntry):
            return context.model_dump()
        return ContextEntry(**context).model_dump()  # validates required keys

    def _attrs(self, node_id: str) -> dict:
        row = self._conn.execute("select data from nodes where id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"Node not found: {node_id}")
        return json.loads(row[0])

    def _save_attrs(self, node_id: str, attrs: dict) -> None:
        self._conn.execute(
            "update nodes set data = ?, has_tags = ?, updated_at = ? where id = ?",
            (json.dumps(_sanitize_for_json(attrs)), 1 if attrs.get(Attr.TAGS) is not None else 0,
             now_iso(), node_id))

    # ---------------------------------------------------------------- lookups
    _NODE_COLS = "id, type, name, is_new, origin_id, position, created_at, updated_at, data"

    def get_node(self, node_id: str) -> dict | None:
        row = self._conn.execute(f"select {self._NODE_COLS} from nodes where id = ?", (node_id,)).fetchone()
        return self._node_dict(row) if row else None

    @staticmethod
    def _node_dict(row) -> dict:
        nid, ntype, name, is_new, origin_id, position, created_at, updated_at, data = row
        d: dict = {"id": nid, "type": ntype}
        if name is not None:
            d["name"] = name
        d["created_at"] = created_at
        d["updated_at"] = updated_at
        d["attributes"] = json.loads(data)
        if ntype == NodeType.SOURCE.value:
            d[Attr.IS_NEW] = bool(is_new)
            d["origin_id"] = origin_id
            d["position"] = json.loads(position) if position else None
        return d

    def _exists(self, node_id: str) -> bool:
        return self._conn.execute("select 1 from nodes where id = ?", (node_id,)).fetchone() is not None

    def __contains__(self, node_id: str) -> bool:
        return self._exists(node_id)

    def _ids(self, sql: str, params: tuple = ()) -> list[str]:
        return [r[0] for r in self._conn.execute(sql, params).fetchall()]

    def nodes_by_type(self, node_type: NodeType | str) -> list[str]:
        nt = node_type.value if isinstance(node_type, NodeType) else node_type
        return self._ids("select id from nodes where type = ? order by rowid", (nt,))

    def find_entity_by_name(self, name: str) -> str | None:
        return self._find_named(NodeType.ENTITY, name)

    def find_concept_by_name(self, name: str) -> str | None:
        return self._find_named(NodeType.CONCEPT, name)

    def _find_named(self, ntype: NodeType, name: str) -> str | None:
        row = self._conn.execute("select id from nodes where type = ? and name_key = ?",
                                 (ntype.value, normalize_name(name))).fetchone()
        return row[0] if row else None

    def neighbors(self, node_id: str, direction: str = "out") -> list[str]:
        if not self._exists(node_id):
            return []
        if direction == "out":
            return self._ids("select dst from edges where src = ? order by rowid", (node_id,))
        if direction == "in":
            return self._ids("select src from edges where dst = ? order by rowid", (node_id,))
        if direction == "both":
            return list(set(self.neighbors(node_id, "out")) | set(self.neighbors(node_id, "in")))
        raise ValueError(f"Invalid direction '{direction}'. Use 'out', 'in', or 'both'.")

    def ancestors(self, node_id: str) -> list[str]:
        """All nodes with a path *to* node_id (any type/depth). ``union`` makes it cycle-safe."""
        return self._ids(
            "with recursive anc(id) as (select src from edges where dst = ? "
            "union select e.src from edges e join anc a on e.dst = a.id) "
            "select id from anc where id != ?", (node_id, node_id))

    def trace_sources(self, start_id: str) -> list[str]:
        """Source nodes upstream of ``start_id`` — works from any tier because all edges point
        upward (Source→Entity→Concept→Centroid)."""
        if not self._exists(start_id):
            raise KeyError(f"Node not found: {start_id}")
        return self._ids(
            "with recursive anc(id) as (select src from edges where dst = ? "
            "union select e.src from edges e join anc a on e.dst = a.id) "
            "select n.id from anc join nodes n on n.id = anc.id where n.type = 'Source'", (start_id,))

    def subgraph_edges(self, node_ids: list[str]) -> list[dict]:
        """Edges with both endpoints in ``node_ids``: ``{"source", "target", "type"}``."""
        ids = json.dumps(list(set(node_ids)))
        rows = self._conn.execute(
            "select src, dst, type from edges where src in (select value from json_each(?)) "
            "and dst in (select value from json_each(?)) order by rowid", (ids, ids)).fetchall()
        return [{"source": u, "target": v, "type": t} for u, v, t in rows]

    def shortest_path(self, from_id: str, to_id: str) -> list[str] | None:
        """Directed shortest path following forward edges, or None if unreachable."""
        if not (self._exists(from_id) and self._exists(to_id)):
            return None
        parent: dict[str, str | None] = {from_id: None}
        queue = deque([from_id])
        while queue:
            current = queue.popleft()
            if current == to_id:
                path = [current]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])  # type: ignore[arg-type]
                return path[::-1]
            for nxt in self._ids("select dst from edges where src = ? order by rowid", (current,)):
                if nxt not in parent:
                    parent[nxt] = current
                    queue.append(nxt)
        return None

    def match_entities_by_keyword(self, keyword: str) -> list[dict]:
        """Case-insensitive substring match on Entity name/tags. ``{"id", "name", "type"}`` per hit."""
        kw = keyword.lower()
        rows = self._conn.execute(
            "select id, name, type from nodes n where type = ? and (instr(lower(name), ?) > 0 or exists("
            "select 1 from json_each(n.data, '$.tags') where instr(lower(value), ?) > 0)) order by rowid",
            (NodeType.ENTITY.value, kw, kw)).fetchall()
        return [{"id": i, "name": n, "type": t} for i, n, t in rows]

    # ------------------------------------------------------------------ counts
    def node_count(self) -> int:
        return self._conn.execute("select count(*) from nodes").fetchone()[0]

    def edge_count(self) -> int:
        return self._conn.execute("select count(*) from edges").fetchone()[0]

    def source_node_count(self) -> int:
        return self._conn.execute("select count(*) from nodes where type = 'Source'").fetchone()[0]

    # ----------------------------------------------------- incremental queues
    def sources_to_process(self) -> list[str]:
        """Sources with is_new (awaiting NoD)."""
        return self._ids("select id from nodes where type = 'Source' and is_new = 1 order by rowid")

    def entities_to_tag(self) -> list[str]:
        """Entities with no tags yet (an empty tag list counts as tagged)."""
        return self._ids("select id from nodes where type = ? and has_tags = 0 order by rowid",
                         (NodeType.ENTITY.value,))

    def concepts_unconsolidated(self) -> list[str]:
        """Concepts with no outgoing BELONGS_TO edge (awaiting centroid assignment)."""
        return self._ids(
            "select id from nodes n where type = 'Concept' and not exists("
            "select 1 from edges e where e.src = n.id and e.type = ?) order by rowid",
            (EdgeType.BELONGS_TO.value,))

    # ----------------------------------------------------- centroid helpers
    def concept_embedding(self, concept_id: str) -> list[float] | None:
        row = self._conn.execute("select vector from concept_embeddings where node_id = ?",
                                 (concept_id,)).fetchone()
        return _unpack(row[0]) if row else None

    def set_concept_embedding(self, concept_id: str, embedding: list[float]) -> None:
        with atomic(self._conn):
            self._write_embedding(concept_id, embedding)
            self._conn.execute("update nodes set updated_at = ? where id = ?", (now_iso(), concept_id))

    def _write_embedding(self, concept_id: str, embedding: list[float]) -> None:
        self._conn.execute("insert or replace into concept_embeddings(node_id, vector) values (?, ?)",
                           (concept_id, _pack(list(embedding))))

    def concepts_missing_embedding(self) -> list[str]:
        return self._ids(
            "select id from nodes n where type = 'Concept' and not exists("
            "select 1 from concept_embeddings c where c.node_id = n.id) order by rowid")

    def centroid_members(self, centroid_id: str) -> list[str]:
        """Member concept ids (authoritative: incoming BELONGS_TO edges)."""
        return self._ids("select src from edges where dst = ? and type = ? order by rowid",
                         (centroid_id, EdgeType.BELONGS_TO.value))

    def assign_concept_to_centroid(self, concept_id: str, centroid_id: str) -> None:
        """Link concept→centroid and keep the centroid's members list in sync."""
        with atomic(self._conn):
            self.link_belongs_to(concept_id, centroid_id)
            attrs = self._attrs(centroid_id)
            members = attrs.setdefault(Attr.MEMBERS, [])
            if concept_id not in members:
                members.append(concept_id)
                self._save_attrs(centroid_id, attrs)

    def clear_centroids(self) -> None:
        """Remove all Centroid nodes and their edges; Concepts and their entity edges stay."""
        with atomic(self._conn):
            self._conn.execute(
                "delete from edges where src in (select id from nodes where type = 'Centroid') "
                "or dst in (select id from nodes where type = 'Centroid')")
            self._conn.execute("delete from nodes where type = 'Centroid'")

    # ------------------------------------------------------------------ export
    def export_json(self, path: str | None = None) -> dict:
        """The whole graph as node-link JSON (plus origins), without embeddings. Written
        atomically to ``path`` when given (debug checkpoints, ``lum export``)."""
        rows = self._conn.execute(f"select {self._NODE_COLS} from nodes order by rowid").fetchall()
        edges = self._conn.execute("select id, src, dst, type, data from edges order by rowid").fetchall()
        data = {
            "directed": True, "multigraph": False, "graph": {},
            "nodes": [self._node_dict(r) for r in rows],
            "links": [{"source": s, "target": t, "id": i, "type": ty, "attributes": json.loads(a or "{}")}
                      for i, s, t, ty, a in edges],
            "origins": self.list_origins(),
        }
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        return data

    # ------------------------------------------------------------------ import
    def import_node_link(self, data: dict) -> dict:
        """Load a node-link graph (an old ``graph.json``) into this store, keeping ids. Concept
        embeddings move from node attributes to ``concept_embeddings``; a Source missing its
        ``content_hash`` (saved before it existed) gets it computed. Returns counts."""
        counts = {"nodes": 0, "edges": 0, "embeddings": 0}
        with atomic(self._conn):
            for n in data.get("nodes", []):
                d = dict(n)
                attrs = dict(d.get("attributes") or {})
                if d.get("type") == NodeType.SOURCE.value and not attrs.get("content_hash"):
                    attrs["content_hash"] = content_hash(attrs.get("content", ""), attrs.get("metadata", {}))
                emb = attrs.get(Attr.EMBEDDING)
                d["attributes"] = attrs
                self._insert_node(d, origin_id=d.get("origin_id"), position=d.get("position"))
                counts["nodes"] += 1
                if d.get("type") == NodeType.CONCEPT.value and emb:
                    self._write_embedding(d["id"], emb)
                    counts["embeddings"] += 1
            links = data.get("links", data.get("edges", []))
            for e in links:
                self._conn.execute(
                    "insert or ignore into edges(id, src, dst, type, data) values (?, ?, ?, ?, ?)",
                    (e.get("id") or new_id("EDGE_"), e["source"], e["target"], e.get("type"),
                     json.dumps(e.get("attributes") or {})))
                counts["edges"] += 1
            for o in data.get("origins", []):
                self._conn.execute(
                    "insert or ignore into origins(id, kind, created_at, sha256, label, extra) "
                    "values (?, ?, ?, ?, ?, ?)",
                    (o["id"], o["kind"], o["created_at"], o["sha256"], o.get("label"),
                     json.dumps(o.get("extra") or {})))
        return counts


def load_graph_json(path: str, target: str | None = None) -> SqliteGraphStore:
    """Read an old node-link ``graph.json`` into a new store (in memory, or the ``target`` file)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    store = SqliteGraphStore(path=target)
    store.import_node_link(data)
    return store
