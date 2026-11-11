"""JobStatus — where an ingest job is, for local agents.

A plain object fed by the pipeline's ``on_event`` stream, optionally persisted through a
``StatusStore`` connector so another process (e.g. an agent's ``status`` command) can read
progress while a build runs. The store is a JSON file (``JsonFileStatusStore``).

Tracks job state (pending/running/completed/failed), per-source status seeded at
ingest (sources deduped onto already-processed ones start as completed), progress counts, the
error, and the lazy stall watchdog (a "running" job with no update for STALL_TIMEOUT_SECONDS
reads as failed). Resume = ``sources_needing_resume()`` then re-run what is still pending.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

STALL_TIMEOUT_SECONDS = 30 * 60


@runtime_checkable
class StatusStore(Protocol):
    """Where job state is persisted (connector seam). Default: a JSON file."""

    def save(self, data: dict) -> None: ...
    def load(self) -> dict | None: ...


class JsonFileStatusStore:
    def __init__(self, path: str):
        self.path = path

    def save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)   # atomic: a reader never sees a partial file

    def load(self) -> dict | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobStatus:
    def __init__(self, store: StatusStore | None = None, job_id: str | None = None):
        self.store = store
        self.data: dict = {
            "job_id": job_id or uuid.uuid4().hex,
            "state": "pending",
            "stage": None,
            "stages": {},     # stage -> {"phase", "processed", "total", ...}
            "sources": {},    # source_id -> {"status", "error"}
            "counts": {},     # node_count / edge_count / source_node_count
            "error": None,
            "started_at": None,
            "updated_at": _now(),
        }

    # ------------------------------------------------------------------ writes
    def start(self) -> None:
        self.data["state"] = "running"
        self.data["started_at"] = _now()
        self._touch()

    def seed_sources(self, source_ids: list[str], already_completed: set[str] | None = None) -> None:
        done = already_completed or set()
        for sid in source_ids:
            self.data["sources"][sid] = {"status": "completed" if sid in done else "pending", "error": None}
        self._touch()

    def on_event(self, event: dict) -> None:
        """``on_event`` callback for the pipeline stages."""
        stage = event.get("stage")
        if stage is None:
            return
        self.data["stage"] = stage
        info = {k: v for k, v in event.items() if k not in ("stage", "source_id", "success", "error")}
        self.data["stages"].setdefault(stage, {}).update(info)
        if stage == "nod" and event.get("phase") == "progress" and "source_id" in event:
            self.data["sources"][event["source_id"]] = {
                "status": "completed" if event.get("success") else "failed", "error": event.get("error"),
            }
        self._touch()

    def set_counts(self, *, node_count: int, edge_count: int, source_node_count: int) -> None:
        self.data["counts"] = {"node_count": node_count, "edge_count": edge_count,
                               "source_node_count": source_node_count}
        self._touch()

    def set_files(self, files: list[dict]) -> None:
        """Per-file results of a file ingest: path, plan status, result, origin, chunks."""
        self.data["files"] = files
        self._touch()

    def complete(self) -> None:
        self.data["state"] = "completed"
        self._touch()

    def fail(self, error: str) -> None:
        self.data["state"] = "failed"
        self.data["error"] = error
        self._touch()

    # ------------------------------------------------------------------- reads
    def sources_needing_resume(self) -> list[str]:
        return [sid for sid, s in self.data["sources"].items() if s["status"] != "completed"]

    def summary(self) -> dict:
        """Snapshot with the stall watchdog applied and per-source totals."""
        out = json.loads(json.dumps(self.data))
        if out["state"] == "running":
            age = (datetime.now(UTC) - datetime.fromisoformat(out["updated_at"])).total_seconds()
            if age > STALL_TIMEOUT_SECONDS:
                out["state"] = "failed"
                out["error"] = f"stalled — no progress in over {STALL_TIMEOUT_SECONDS // 60} minutes"
        totals: dict[str, int] = {}
        for s in out["sources"].values():
            totals[s["status"]] = totals.get(s["status"], 0) + 1
        out["source_totals"] = totals
        return out

    @classmethod
    def read(cls, store: StatusStore) -> JobStatus | None:
        data = store.load()
        if data is None:
            return None
        js = cls(store=store)
        js.data = data
        return js

    # ----------------------------------------------------------------- private
    def _touch(self) -> None:
        self.data["updated_at"] = _now()
        if self.store is not None:
            self.store.save(self.data)
