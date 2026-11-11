"""JSON records: a file holding the engine's own units, one origin per file.

Accepted shapes: a list of units (strings or ``{"content", "metadata"}``), or an object with a
``"records"`` list. Each record's own metadata is kept; position is ``{"record": n}`` (1-based).
"""

from __future__ import annotations

import json

from lumintelligence.inputs.types import ParsedFile


def parse_records(path: str) -> ParsedFile:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise TypeError('expected a JSON list of records, or an object with a "records" list')
    units = []
    for n, rec in enumerate(records, start=1):
        if isinstance(rec, str):
            content, metadata = rec, {}
        elif isinstance(rec, dict) and isinstance(rec.get("content"), str):
            content, metadata = rec["content"], rec.get("metadata") or {}
        else:
            raise TypeError(f'record {n}: expected a string or an object with "content"')
        if content.strip():
            units.append({"content": content, "metadata": metadata, "position": {"record": n}})
    return ParsedFile(units=units, note=None if units else "no records with content")
