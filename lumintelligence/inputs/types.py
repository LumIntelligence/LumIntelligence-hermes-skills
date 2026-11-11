"""Shared types for the input layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Plan statuses for one file.
NEW = "new"                  # never seen: ingest
CHANGED = "changed"          # path known, content differs: ingest as a new origin
COPY = "copy"                # content known at another path: skip, record the path
UNCHANGED = "unchanged"      # content and path known: skip
UNSUPPORTED = "unsupported"  # extension not handled
TOO_LARGE = "too_large"      # over Config.max_file_mb
UNREADABLE = "unreadable"    # could not be read or hashed

INGEST_STATUSES = (NEW, CHANGED)


@dataclass
class FileItem:
    path: str                       # absolute
    status: str
    file_type: str | None = None    # pdf | docx | md | txt | records
    sha256: str | None = None
    size: int | None = None
    mtime: str | None = None        # ISO, UTC
    origin_id: str | None = None    # copy/unchanged: the origin holding this content; changed: latest prior version
    copy_of: str | None = None      # copy of a file earlier in the same plan (not yet ingested)
    reason: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class IngestPlan:
    items: list[FileItem] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)   # known origins whose path is gone (report only)

    def to_ingest(self) -> list[FileItem]:
        return [i for i in self.items if i.status in INGEST_STATUSES]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.items:
            out[i.status] = out.get(i.status, 0) + 1
        if self.missing:
            out["missing"] = len(self.missing)
        return out

    def to_dict(self) -> dict:
        return {"counts": self.counts(), "files": [i.to_dict() for i in self.items], "missing": self.missing}


@dataclass
class ParsedFile:
    """One file turned into source units. ``units`` are ``{"content", "metadata", "position"}``;
    ``extra`` is merged into the origin (e.g. ``pages``); ``note`` explains an empty result."""

    units: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    note: str | None = None
