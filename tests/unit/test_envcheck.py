"""Environment check and install (user-managed Python): requirements, fixes, installer choice, and the
clear error when SQLite can't load extensions."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from lumintelligence import cli, envcheck
from lumintelligence.storage import sqlite_conn


def test_this_environment_passes():
    env = envcheck.check_environment()
    assert env["ok"], env
    assert {i["name"] for i in env["items"]} >= {"python", "sqlite_extensions", "sqlite_vec", "openai"}


def test_missing_package_is_reported_with_a_fix(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "pdfplumber" else real(name, *a))
    env = envcheck.check_environment()
    item = next(i for i in env["items"] if i["name"] == "pdfplumber")
    assert not env["ok"] and not item["ok"] and "PDF" in item["detail"]
    assert env["installable"] == ["pdfplumber"]


def test_installer_prefers_pip_then_uv_then_none(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object() if name == "pip" else real(name, *a))
    assert envcheck.installer_command(["x"]) == [sys.executable, "-m", "pip", "install", "x"]
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "pip" else real(name, *a))
    monkeypatch.setattr(envcheck.shutil, "which", lambda name: "/opt/uv")
    assert envcheck.installer_command(["x"]) == ["/opt/uv", "pip", "install", "--python", sys.executable, "x"]
    monkeypatch.setattr(envcheck.shutil, "which", lambda name: None)
    assert envcheck.installer_command(["x"]) is None
    assert envcheck.install(["x"])["ok"] is False


def test_sqlite_without_extensions_gives_a_clear_error(monkeypatch):
    class NoExt:
        class Connection:   # no enable_load_extension
            pass

    monkeypatch.setattr(sqlite_conn, "_stdlib_sqlite3", NoExt)
    monkeypatch.setitem(sys.modules, "pysqlite3", None)        # fallback not installed
    monkeypatch.setitem(sys.modules, "pysqlite3.dbapi2", None)
    with pytest.raises(sqlite_conn.SQLiteExtensionError) as err:
        sqlite_conn.connect()
    assert "can't load SQLite extensions" in str(err.value) and "lum check --fix" in err.value.hint


def test_check_before_init_reports_environment(capsys, monkeypatch):
    assert cli.main(["--json", "check"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["environment"]["ok"] and "lum init" in out["hint"]


def test_agents_never_install_without_yes(capsys, monkeypatch):
    monkeypatch.setattr(envcheck, "check_environment", lambda: {
        "ok": False, "python": sys.executable, "installable": ["pdfplumber"],
        "items": [{"name": "pdfplumber", "label": "PDF reader", "ok": False, "detail": "missing", "fix": "Install"}]})
    monkeypatch.setattr(envcheck, "install", lambda pkgs: pytest.fail("must not install"))
    cli.main(["--json", "check", "--fix"])                      # no terminal, no --yes: report only
    out = json.loads(capsys.readouterr().out)
    assert out["environment"]["ok"] is False and "install" not in out["environment"]
