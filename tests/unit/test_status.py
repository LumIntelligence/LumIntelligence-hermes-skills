"""JobStatus: per-source tracking from NoD events, StatusStore persistence, stall watchdog."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lumintelligence import status as status_mod
from lumintelligence.status import JobStatus, JsonFileStatusStore, StatusStore


def test_tracks_sources_and_persists(tmp_path):
    store = JsonFileStatusStore(str(tmp_path / "status.json"))
    assert isinstance(store, StatusStore)
    assert JobStatus.read(store) is None
    job = JobStatus(store=store)
    job.start()
    job.seed_sources(["a", "b", "c"], already_completed={"c"})
    job.on_event({"stage": "nod", "phase": "progress", "processed": 1, "total": 2,
                  "source_id": "a", "success": True, "error": None})
    job.on_event({"stage": "nod", "phase": "progress", "processed": 2, "total": 2,
                  "source_id": "b", "success": False, "error": "boom"})

    read = JobStatus.read(store).summary()
    assert read["state"] == "running" and read["stage"] == "nod"
    assert read["stages"]["nod"]["processed"] == 2
    assert read["source_totals"] == {"completed": 2, "failed": 1}
    assert JobStatus.read(store).sources_needing_resume() == ["b"]


def test_stalled_running_job_reads_as_failed():
    job = JobStatus()
    job.start()
    old = datetime.now(UTC) - timedelta(seconds=status_mod.STALL_TIMEOUT_SECONDS + 5)
    job.data["updated_at"] = old.isoformat()
    s = job.summary()
    assert s["state"] == "failed" and "stalled" in s["error"]
