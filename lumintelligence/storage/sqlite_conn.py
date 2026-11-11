"""One SQLite connection per knowledge base: the graph tables and the sqlite-vec tables
share it, so one transaction covers a whole build.

Some sqlite3 builds (notably python.org's macOS installer) lack ``enable_load_extension``; then
``pysqlite3`` (the ``sqlite-ext`` extra) is used instead.
"""

from __future__ import annotations

import sqlite3 as _stdlib_sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

Connection = Any  # either stdlib sqlite3.Connection or pysqlite3's — same interface

EXTENSION_HINT = (
    "Use a Python whose sqlite3 can load extensions (for example from uv, Homebrew or conda, or pyenv built "
    "with --enable-loadable-sqlite-extensions), or run `lum check --fix` to install the pysqlite3 fallback "
    "into this environment.")


class SQLiteExtensionError(RuntimeError):
    """This Python's SQLite can't load extensions and no fallback is installed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.hint = EXTENSION_HINT


def sqlite_module() -> Any:
    """The sqlite3 module to use: the standard one if it can load extensions, else ``pysqlite3``."""
    if hasattr(_stdlib_sqlite3.Connection, "enable_load_extension"):
        return _stdlib_sqlite3
    try:
        import pysqlite3.dbapi2 as fallback  # type: ignore[import-not-found]
    except ImportError as exc:
        import sys
        raise SQLiteExtensionError(
            f"This Python ({sys.version.split()[0]} at {sys.executable}) can't load SQLite extensions, "
            "which LumIntelligence needs for search.") from exc
    return fallback

BUSY_TIMEOUT_MS = 10_000


def connect(target: str | None = None) -> Connection:
    """Open ``target`` (a file path) or an in-memory db, with sqlite-vec loaded.

    Autocommit mode (``isolation_level=None``): transactions are explicit (``atomic``). File dbs use
    WAL, so readers in other processes see the last committed state while a build is writing."""
    import sqlite_vec

    path = target or ":memory:"
    conn = sqlite_module().connect(path, isolation_level=None, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(f"pragma busy_timeout = {BUSY_TIMEOUT_MS}")
    if path != ":memory:":
        conn.execute("pragma journal_mode = wal")
        conn.execute("pragma synchronous = normal")
    return conn


@contextmanager
def atomic(conn: Connection) -> Iterator[None]:
    """All-or-nothing block. Opens a transaction, or a savepoint inside an already open one (so a
    method's writes join the surrounding build transaction)."""
    if conn.in_transaction:
        sp = f"sp_{uuid.uuid4().hex}"
        conn.execute(f"savepoint {sp}")
        try:
            yield
        except BaseException:
            conn.execute(f"rollback to {sp}")
            conn.execute(f"release {sp}")
            raise
        conn.execute(f"release {sp}")
    else:
        conn.execute("begin immediate")
        try:
            yield
        except BaseException:
            conn.execute("rollback")
            raise
        conn.execute("commit")
