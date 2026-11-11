"""Job queue and background worker.

``queue.json`` holds waiting jobs; edits happen under ``.queue.lock``. One worker at a time holds
``.lock`` (an OS file lock, released automatically if the process dies), runs jobs in order and
exits when the queue is empty. The check "queue empty → release the worker lock" happens under the
queue lock, so a job added at that moment always finds either the running worker or no worker (and
starts one).
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

from lumintelligence.constants import now_iso
from lumintelligence.home import path


@contextmanager
def _queue_lock() -> Iterator[None]:
    path(".queue.lock").parent.mkdir(parents=True, exist_ok=True)
    with open(path(".queue.lock"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read() -> list[dict]:
    q = path("queue.json")
    return json.loads(q.read_text(encoding="utf-8")).get("jobs", []) if q.exists() else []


def _write(jobs: list[dict]) -> None:
    q = path("queue.json")
    tmp = q.with_suffix(".tmp")
    tmp.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")
    os.replace(tmp, q)


def enqueue(kind: str, paths: list[str] | None = None) -> tuple[dict, int]:
    """Add a job; returns it and its 1-based position in the queue."""
    job = {"id": f"JOB_{uuid.uuid4().hex[:12]}", "kind": kind, "paths": list(paths or []), "queued_at": now_iso()}
    with _queue_lock():
        jobs = _read()
        jobs.append(job)
        _write(jobs)
        return job, len(jobs)


def queued() -> list[dict]:
    with _queue_lock():
        return _read()


def worker_running() -> bool:
    lock = path(".lock")
    if not lock.exists():
        return False
    with open(lock, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def worker_pid() -> int | None:
    """The running worker's process id (it writes it into ``.lock``), or None."""
    if not worker_running():
        return None
    try:
        return int(path(".lock").read_text().strip())
    except (OSError, ValueError):
        return None


def stop(clear_queue: bool = True, timeout: float = 10.0) -> dict:
    """Stop the running worker. The queue is emptied first (unless ``clear_queue`` is False) so the
    worker can't move on to the next job. The job's open transaction is never committed, so the KB
    stays as it was before the job; model answers already received stay in the journal."""
    removed: list[dict] = []
    if clear_queue:
        with _queue_lock():
            removed = _read()
            _write([])
    pid = worker_pid()
    if worker_running() and pid is None:
        raise RuntimeError("A worker is running but its process id is unknown.")
    killed = False
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while worker_running() and time.monotonic() < deadline:
            time.sleep(0.1)
        if worker_running():
            os.kill(pid, signal.SIGKILL)
            while worker_running() and time.monotonic() < deadline + 5:
                time.sleep(0.1)
        killed = not worker_running()
        if killed:
            with open(path("worker.log"), "a") as log:
                log.write(f"{now_iso()} stopped worker {pid} (lum stop)\n")
    return {"stopped": killed, "pid": pid, "removed_from_queue": removed}


def start_worker() -> bool:
    """Start a detached worker unless one is running. Returns True if one was started."""
    if worker_running():
        return False
    log = open(path("worker.log"), "a")  # noqa: SIM115 — handed to the child process
    subprocess.Popen([sys.executable, "-m", "lumintelligence.cli", "_worker"], stdin=subprocess.DEVNULL,
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
    log.close()
    return True


@contextmanager
def worker_lock(block: bool = False) -> Iterator[IO | None]:
    """Hold ``.lock``; yields None if another worker has it (and ``block`` is False)."""
    path(".lock").parent.mkdir(parents=True, exist_ok=True)
    f = open(path(".lock"), "a")  # noqa: SIM115
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None
            return
        f.truncate(0)
        f.write(str(os.getpid()))
        f.flush()
        yield f
    finally:
        f.close()   # closing releases the lock


def run_job(job: dict) -> dict:
    from lumintelligence.home import open_kb

    lum = open_kb()
    try:
        if job["kind"] == "ingest":
            return lum.ingest(job["paths"])
        if job["kind"] == "resume":
            return lum.resume()
        raise ValueError(f"unknown job kind {job['kind']!r}")
    finally:
        lum.graph.conn.close()  # type: ignore[attr-defined]


def run_worker() -> int:
    """Drain the queue. Returns the number of jobs run (0 if another worker holds the lock)."""
    done = 0
    with worker_lock() as held:
        if held is None:
            return 0
        while True:
            with _queue_lock():
                jobs = _read()
                if not jobs:
                    held.close()   # release the worker lock while the queue lock is still held
                    return done
                job = jobs.pop(0)
                _write(jobs)
            print(f"{now_iso()} start {job['id']} {job['kind']} {job['paths']}", file=sys.stderr, flush=True)
            try:
                summary = run_job(job)
                print(f"{now_iso()} done {job['id']} {summary.get('state')}", file=sys.stderr, flush=True)
            except Exception:  # noqa: BLE001 — log and continue with the next job
                print(f"{now_iso()} failed {job['id']}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            done += 1
