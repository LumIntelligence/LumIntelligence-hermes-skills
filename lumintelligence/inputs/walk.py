"""Folder walk and ingest plan: which files are new, changed, copies or unchanged.

Decisions use the origins table: a file's sha256 says whether its content is known, and its
path (``extra.paths``) whether the file is known. Nothing is ever deleted: origins whose file is
gone are only reported.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from datetime import UTC, datetime

from lumintelligence.graph.protocol import GraphStore
from lumintelligence.inputs.types import (
    CHANGED,
    COPY,
    NEW,
    TOO_LARGE,
    UNCHANGED,
    UNREADABLE,
    UNSUPPORTED,
    FileItem,
    IngestPlan,
)

SUPPORTED: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".json": "records",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hidden(name: str) -> bool:
    return name.startswith((".", "~$"))   # dotfiles, Office lock files


def expand_paths(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    """Files under ``paths`` (absolute, sorted per folder) and the folders walked. Recursive; hidden
    files and folders skipped; symlinks not followed."""
    files: list[str] = []
    folders: list[str] = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isdir(p):
            folders.append(p)
            for root, dirs, names in os.walk(p, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not _hidden(d))
                for n in sorted(names):
                    full = os.path.join(root, n)
                    if not _hidden(n) and not os.path.islink(full):
                        files.append(full)
        else:
            files.append(p)
    return files, folders


def plan_paths(paths: Iterable[str], kg: GraphStore, max_file_mb: float | None = None) -> IngestPlan:
    files, folders = expand_paths(paths)
    plan = IngestPlan()
    pending: dict[str, str] = {}   # sha -> path of a file earlier in this plan that will be ingested
    seen_paths: set[str] = set()
    for path in files:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        ftype = SUPPORTED.get(os.path.splitext(path)[1].lower())
        if not os.path.isfile(path):
            plan.items.append(FileItem(path, UNREADABLE, reason="not a file or does not exist"))
            continue
        if ftype is None:
            plan.items.append(FileItem(path, UNSUPPORTED, reason="file type not supported"))
            continue
        try:
            st = os.stat(path)
            if max_file_mb is not None and st.st_size > max_file_mb * 1024 * 1024:
                plan.items.append(FileItem(path, TOO_LARGE, ftype, size=st.st_size,
                                           reason=f"larger than {max_file_mb} MB"))
                continue
            sha = sha256_file(path)
        except OSError as exc:
            plan.items.append(FileItem(path, UNREADABLE, ftype, reason=str(exc)))
            continue
        item = FileItem(path, NEW, ftype, sha, st.st_size,
                        datetime.fromtimestamp(st.st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%S"))
        known = kg.find_origin_by_sha(sha)
        if known is not None:
            origin = kg.get_origin(known) or {}
            item.origin_id = known
            item.status = UNCHANGED if path in (origin.get("extra", {}).get("paths") or []) else COPY
        elif sha in pending:
            item.status, item.copy_of = COPY, pending[sha]
        else:
            prior = kg.find_origins("paths", path)
            if prior:
                item.status, item.origin_id = CHANGED, prior[-1]
            pending[sha] = path
        plan.items.append(item)
    plan.missing = _missing(kg, folders, seen_paths)
    return plan


def _missing(kg: GraphStore, folders: list[str], seen: set[str]) -> list[dict]:
    """Known file origins under the walked folders whose path no longer exists."""
    if not folders:
        return []
    out = []
    for o in kg.list_origins():
        if o["kind"] != "file":
            continue
        for p in o["extra"].get("paths") or []:
            under = any(p.startswith(f.rstrip(os.sep) + os.sep) for f in folders)
            if under and p not in seen and not os.path.exists(p):
                out.append({"origin_id": o["id"], "path": p, "label": o.get("label")})
    return out
