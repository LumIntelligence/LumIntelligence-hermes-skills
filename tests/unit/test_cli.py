"""The `lum` CLI, offline: mock provider, a temporary home folder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from lumintelligence import cli, jobs
from lumintelligence import home as H


@pytest.fixture
def lum_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMINTELLIGENCE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LUM_PROVIDER", "mock")
    monkeypatch.setenv("LUM_ENTRY_CUTOFF", "2.0")   # mock embeddings carry no meaning
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "board.md").write_text("# Board\nMaria Lopez chairs the board.\nTom Reyes is the treasurer.\n")
    (docs / "grant.txt").write_text("The Harbor Fund proposed a grant to Maria Lopez.\n")
    return tmp_path


def run(capsys, *argv: str) -> tuple[int, dict]:
    code = cli.main(["--json", *argv])
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip() else {})


def _init(capsys) -> dict:
    code, out = run(capsys, "init", "--yes", "--mode", "offline")
    assert code == 0, out
    return out


def test_commands_need_init(lum_home, capsys):
    code, out = run(capsys, "status")
    assert code == 1 and "lum init" in out["hint"]


def test_init_writes_env_checks_models_and_creates_the_kb(lum_home, capsys):
    out = _init(capsys)
    env = H.path(".env")
    assert oct(env.stat().st_mode)[-3:] == "600"
    env_values = H.read_env()
    assert env_values["LUM_MODE"] == "offline" and env_values["LUM_OFFLINE_CHAT_MODEL"] == "hermes3"
    assert env_values["LUM_ONLINE_EMBED_MODEL"] == "text-embedding-3-small"   # room for online
    assert "LUM_EMBED_DIM" not in env_values
    assert out["check"]["chat"]["ok"] and out["check"]["embedding"]["dim"] == 16
    assert out["ready"] and H.path("kb.db").exists()
    assert (H.Path(out["skill"]["installed"]) / "SKILL.md").exists()          # skill folded into init


def test_set_env_value_keeps_comments(lum_home, capsys):
    _init(capsys)
    H.set_env_value("LUM_MAX_IN_FLIGHT", "4")
    text = H.path(".env").read_text()
    assert "LUM_MAX_IN_FLIGHT=4" in text and "# model calls in parallel" in text


def test_other_embedding_model_is_refused(lum_home, capsys, monkeypatch):
    _init(capsys)
    monkeypatch.setenv("LUM_MODE", "online")       # online mode uses a different embedding model
    code, out = run(capsys, "files")
    assert code == 1 and "kb rebuild" in out["hint"] and "LUM_MODE" in out["hint"]
    assert "nomic-embed-text" in out["error"]


def test_plan_ingest_wait_status_files_query_export(lum_home, capsys):
    _init(capsys)
    docs = str(lum_home / "docs")
    assert run(capsys, "plan", docs)[1]["counts"] == {"new": 2}

    code, out = run(capsys, "ingest", docs, "--wait")
    assert code == 0 and out["result"]["state"] == "completed"
    assert {f["result"] for f in out["result"]["files"]} == {"ingested"}

    status = run(capsys, "status")[1]
    assert status["job"]["state"] == "completed" and status["queue"] == []

    files = run(capsys, "files")[1]["files"]
    assert sorted(f["label"] for f in files) == ["board.md", "grant.txt"]

    q = run(capsys, "query", "What does Maria Lopez do?")[1]
    assert q["answer"].startswith("Mock answer [1]") and q["cited"] == [1]
    first = q["sources"][0]
    assert first["origin"]["label"] in ("board.md", "grant.txt") and first["origin"]["ingested_at"]
    assert first["source"]["position"] and first["path"]

    calls = []
    real = cli.H.open_kb

    def spy():
        lum = real()
        original = lum.llm.complete_json

        def counting(system_prompt, content, schema):
            calls.append(schema)
            return original(system_prompt, content, schema)
        lum.llm.complete_json = counting
        return lum

    cli.H.open_kb = spy
    try:
        q2 = run(capsys, "query", "What does Maria Lopez do?", "--no-answer")[1]
    finally:
        cli.H.open_kb = real
    assert q2["answer"] is None and q2["sources"]
    assert not any("answer" in (s.get("properties") or {}) for s in calls)

    out_file = lum_home / "graph.json"
    assert run(capsys, "export", "--out", str(out_file))[1]["nodes"] > 0
    assert json.loads(out_file.read_text())["origins"]


def test_answer_text_lists_numbered_sources(lum_home, capsys):
    text = cli._answer_text({"answer": "Yes [1].", "note": None, "sources": [
        {"n": 1, "path": ["Maria Lopez (entity)"],
         "source": {"content": "Maria Lopez chairs the board.", "position": {"page": 3, "line_start": 1, "line_end": 5}},
         "origin": {"label": "board.pdf", "ingested_at": "2026-11-06T10:00:00"}}]})
    assert "[1] board.pdf, p. 3, lines 1-5 — ingested 2026-11-06" in text
    assert "found via: Maria Lopez (entity)" in text


def test_queue_runs_jobs_in_order(lum_home, capsys):
    _init(capsys)
    docs = lum_home / "docs"
    first, _ = jobs.enqueue("ingest", [str(docs / "board.md")])
    second, pos = jobs.enqueue("ingest", [str(docs / "grant.txt")])
    assert pos == 2 and [j["id"] for j in jobs.queued()] == [first["id"], second["id"]]
    assert jobs.run_worker() == 2
    assert jobs.queued() == [] and not jobs.worker_running()
    assert len(run(capsys, "files")[1]["files"]) == 2


def test_background_ingest_finishes_and_second_worker_is_not_started(lum_home, capsys):
    _init(capsys)
    code, out = run(capsys, "ingest", str(lum_home / "docs"))
    assert code == 0 and out["queue_position"] == 1 and out["worker_started"]
    deadline = time.time() + 30
    while time.time() < deadline:
        status = run(capsys, "status")[1]
        if not status["worker_running"] and not status["queue"] and status["job"]:
            break
        time.sleep(0.2)
    assert status["job"]["state"] == "completed", H.path("worker.log").read_text()


def test_worker_lock_blocks_a_second_worker(lum_home, capsys):
    _init(capsys)
    with jobs.worker_lock() as held:
        assert held is not None
        assert jobs.worker_running()
        assert jobs.run_worker() == 0            # another worker holds the lock
        code, out = run(capsys, "ingest", str(lum_home / "docs"), "--wait")
        assert code == 1 and "--wait" in out["hint"]
    assert not jobs.worker_running()


def test_status_reports_a_dead_worker(lum_home, capsys):
    _init(capsys)
    H.path("status.json").write_text(json.dumps({
        "job_id": "x", "state": "running", "stage": "nod", "stages": {}, "sources": {},
        "started_at": "2026-11-06T00:00:00+00:00", "updated_at": "2099-01-01T00:00:00+00:00"}))
    status = run(capsys, "status")[1]
    assert status["job"]["state"] == "failed" and "ingest" in status["job"]["hint"]


def test_read_commands_work_while_a_build_holds_the_write_lock(lum_home, capsys, monkeypatch):
    from lumintelligence.storage import sqlite_conn

    _init(capsys)
    docs = str(lum_home / "docs")
    run(capsys, "ingest", docs, "--wait")
    monkeypatch.setattr(sqlite_conn, "BUSY_TIMEOUT_MS", 200)   # fail fast instead of after 10 s
    build = sqlite_conn.connect(str(H.path("kb.db")))
    build.execute("begin immediate")    # what a running build holds for its whole job
    try:
        start = time.monotonic()
        assert run(capsys, "status")[1]["job"]["state"] == "completed"
        assert len(run(capsys, "files")[1]["files"]) == 2
        assert run(capsys, "query", "Who chairs the board?", "--no-answer")[0] == 0
        assert run(capsys, "plan", docs)[1]["counts"] == {"unchanged": 2}
        assert time.monotonic() - start < 5
    finally:
        build.rollback()
        build.close()


def _fake_worker() -> subprocess.Popen:
    """A process that holds the worker lock like a real worker mid-job."""
    code = ("import time\nfrom lumintelligence import jobs\n"
            "with jobs.worker_lock() as held:\n    print('held', flush=True)\n    time.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    assert p.stdout is not None and p.stdout.readline().strip() == "held"
    return p


def test_stop_kills_the_worker_empties_the_queue_and_marks_the_job(lum_home, capsys):
    _init(capsys)
    worker = _fake_worker()
    try:
        H.path("status.json").write_text(json.dumps({
            "job_id": "J1", "state": "running", "stage": "nod", "stages": {}, "sources": {},
            "started_at": "2026-11-11T00:00:00+00:00", "updated_at": "2099-01-01T00:00:00+00:00"}))
        jobs.enqueue("ingest", ["/tmp/x"])
        assert jobs.worker_running() and jobs.worker_pid() == worker.pid

        code, out = run(capsys, "stop")
        assert code == 0 and out["stopped"] and out["pid"] == worker.pid and out["job_id"] == "J1"
        assert len(out["removed_from_queue"]) == 1 and "lum ingest" in out["hint"]
        assert worker.wait(timeout=5) != 0 and not jobs.worker_running() and jobs.queued() == []
        status = run(capsys, "status")[1]
        assert status["job"]["state"] == "stopped" and not status["worker_running"]
        assert "stopped worker" in H.path("worker.log").read_text()
    finally:
        if worker.poll() is None:
            worker.kill()


def test_stop_with_nothing_running(lum_home, capsys):
    _init(capsys)
    jobs.enqueue("ingest", ["/tmp/x"])
    code, out = run(capsys, "stop", "--keep-queue")
    assert code == 0 and not out["stopped"] and out["pid"] is None and len(jobs.queued()) == 1


def test_kb_rebuild_never_deletes_without_a_terminal(lum_home, capsys):
    _init(capsys)
    run(capsys, "ingest", str(lum_home / "docs"), "--wait")
    code, out = run(capsys, "kb", "rebuild")
    assert code == 0 and out["rebuilt"] is False and out["impact"]["files"] == 2
    assert len(run(capsys, "files")[1]["files"]) == 2


def test_kb_rebuild_needs_the_typed_word(lum_home, capsys, monkeypatch):
    _init(capsys)
    run(capsys, "ingest", str(lum_home / "docs"), "--wait")
    monkeypatch.setattr(cli, "_tty_in", lambda: True)

    monkeypatch.setattr("builtins.input", lambda _: "yes")
    cli.main(["kb", "rebuild"])
    assert "Cancelled" in capsys.readouterr().out
    assert len(run(capsys, "files")[1]["files"]) == 2

    monkeypatch.setattr("builtins.input", lambda _: "rebuild")
    cli.main(["kb", "rebuild"])
    assert "Done" in capsys.readouterr().out
    assert run(capsys, "files")[1]["files"] == []


def test_kb_rebuild_refused_while_jobs_wait(lum_home, capsys, monkeypatch):
    _init(capsys)
    jobs.enqueue("ingest", [str(lum_home / "docs")])
    code, out = run(capsys, "kb", "rebuild")
    assert code == 1 and "lum status" in out["hint"]


def test_read_env_handles_quotes_and_comments(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# comment\nA=1  # trailing\nB="x # not a comment"\nC=\n')
    assert H.read_env(f) == {"A": "1", "B": "x # not a comment", "C": ""}
    assert os.environ.get("A") is None
