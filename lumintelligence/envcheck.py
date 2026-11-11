"""Environment check: does this Python have what LumIntelligence needs, and install what's missing.

The user manages their own Python. ``check_environment()`` reports each requirement (Python version,
SQLite extension loading, sqlite-vec, the model client, PDF/Word readers) with a fix.
``install(packages)`` adds missing packages to *this* environment (pip if present, else ``uv pip``);
the CLI only calls it after the user agrees in a terminal. A Python built without SQLite extension
support can't be changed; the ``pysqlite3`` fallback package is the only installable fix for that.
"""

from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import subprocess
import sys

MIN_PYTHON = (3, 11)


def _fallback_package() -> str:
    # pysqlite3-binary ships ready-made wheels for Linux; elsewhere pysqlite3 builds from source.
    return "pysqlite3-binary" if sys.platform.startswith("linux") else "pysqlite3"


def _sqlite_extensions() -> dict:
    item = {"name": "sqlite_extensions", "label": "SQLite extension loading", "ok": False}
    if hasattr(sqlite3.Connection, "enable_load_extension"):
        try:
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            conn.close()
            return {**item, "ok": True, "detail": "built into this Python"}
        except Exception as exc:  # noqa: BLE001
            item["detail"] = f"enable_load_extension failed: {exc}"
    if importlib.util.find_spec("pysqlite3") is not None:
        return {**item, "ok": True, "detail": "via the pysqlite3 fallback"}
    return {**item, "detail": "this Python's sqlite3 can't load extensions",
            "package": _fallback_package(),
            "fix": "Install the pysqlite3 fallback, or use a Python with SQLite extension support "
                   "(uv, Homebrew, conda, or pyenv built with --enable-loadable-sqlite-extensions)."}


def _sqlite_vec(ext_ok: bool) -> dict:
    item = {"name": "sqlite_vec", "label": "sqlite-vec (vector search)", "ok": False}
    if importlib.util.find_spec("sqlite_vec") is None:
        return {**item, "detail": "not installed", "package": "sqlite-vec", "fix": "Install sqlite-vec."}
    if not ext_ok:
        return {**item, "detail": "installed, but can't load until SQLite extension loading works"}
    try:
        from lumintelligence.storage.sqlite_conn import connect

        connect().close()
        return {**item, "ok": True, "detail": "loads"}
    except Exception as exc:  # noqa: BLE001
        return {**item, "detail": f"failed to load: {exc}"[:300]}


def _package(module: str, package: str, label: str, needed_for: str) -> dict:
    ok = importlib.util.find_spec(module) is not None
    out = {"name": module, "label": label, "ok": ok, "detail": "installed" if ok else f"missing ({needed_for})"}
    if not ok:
        out.update(package=package, fix=f"Install {package}.")
    return out


def check_environment() -> dict:
    py_ok = sys.version_info[:2] >= MIN_PYTHON
    items = [{"name": "python", "label": "Python 3.11 or newer", "ok": py_ok,
              "detail": f"{sys.version.split()[0]} at {sys.executable}",
              **({} if py_ok else {"fix": "Install LumIntelligence with Python 3.11 or newer."})}]
    ext = _sqlite_extensions()
    items += [ext, _sqlite_vec(ext["ok"]),
              _package("openai", "openai", "Model client (openai)", "needed to talk to Ollama or online models"),
              _package("pdfplumber", "pdfplumber", "PDF reader (pdfplumber)", "needed to add PDF files"),
              _package("docx", "python-docx", "Word reader (python-docx)", "needed to add .docx files")]
    return {"ok": all(i["ok"] for i in items), "python": sys.executable, "items": items,
            "installable": [i["package"] for i in items if not i["ok"] and i.get("package")]}


def installer_command(packages: list[str]) -> list[str] | None:
    """How to install into this environment: pip if available, else uv; None if neither."""
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", *packages]
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, *packages]
    return None


def install(packages: list[str]) -> dict:
    """Install ``packages`` into this environment. ``{"ok", "command", "output"}``."""
    cmd = installer_command(packages)
    if cmd is None:
        return {"ok": False, "command": None,
                "output": f"Neither pip nor uv is available for {sys.executable}; install these yourself: "
                          + ", ".join(packages)}
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {"ok": proc.returncode == 0, "command": " ".join(cmd), "output": (proc.stdout + proc.stderr)[-2000:]}
