"""Ingestor — faithfully record already-prepared source units as Source nodes.

Pre-processing (deciding what a unit is, choosing content vs metadata) happens *outside*
the library. This stage just records units, immutably, marked is_new
for downstream NoD. A unit is either a plain string (content only) or a dict
``{"content": str, "metadata": dict, "origin_id"?: str, "position"?: dict}`` (provenance).

Idempotency: default ``idempotency="content_hash"`` reuses the
existing Source node whose ``sha256(content + sorted(metadata))`` matches one already in
``kg`` — safe against a caller retrying the same ingest call with the same existing-graph
slice fed back in. ``"always_new"`` opts out, for a caller that wants literal duplicates.
"""

from __future__ import annotations

from collections.abc import Iterable

from lumintelligence.graph import GraphStore

Unit = str | dict


class Ingestor:
    def run(self, kg: GraphStore, units: Iterable[Unit], idempotency: str = "content_hash") -> list[str]:
        ids: list[str] = []
        for unit in units:
            if isinstance(unit, str):
                ids.append(kg.add_source(unit, idempotency=idempotency))
                continue
            ids.append(kg.add_source(unit["content"], metadata=unit.get("metadata", {}),
                                     idempotency=idempotency, origin_id=unit.get("origin_id"),
                                     position=unit.get("position")))
        return ids
