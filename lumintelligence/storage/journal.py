"""Journal — durable record of raw LLM results during a build (connector).

Each raw answer is recorded as soon as it arrives, keyed by a hash of (stage, model, prompt,
input). A rerun after a crash replays recorded answers instead of calling the model again; the
same records serve as a record/replay cache for the eval harness. The journal is cleared after a
successful build.

Connector seam: ``Journal`` is the protocol; ``JsonlJournal`` (one JSON line per result in the
project directory) is the offline implementation, ``NullJournal`` records nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Protocol, runtime_checkable


def journal_key(stage: str, model: str, prompt: str, payload: Any) -> str:
    raw = json.dumps([stage, model, prompt, payload], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@runtime_checkable
class Journal(Protocol):
    def get(self, key: str) -> Any | None: ...
    def put(self, key: str, stage: str, result: Any) -> None: ...
    def clear(self) -> None: ...


class NullJournal:
    def get(self, key: str) -> Any | None:
        return None

    def put(self, key: str, stage: str, result: Any) -> None:
        return None

    def clear(self) -> None:
        return None


class JsonlJournal:
    """Append-only JSON lines; loaded into memory on open. Thread-safe ``put``."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._entries: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue   # a torn last line from a crash: ignore it
                    self._entries[rec["key"]] = rec["result"]

    def get(self, key: str) -> Any | None:
        return self._entries.get(key)

    def put(self, key: str, stage: str, result: Any) -> None:
        line = json.dumps({"key": key, "stage": stage, "result": result}, ensure_ascii=False)
        with self._lock:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            self._entries[key] = result

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            if os.path.exists(self.path):
                os.remove(self.path)

    def __len__(self) -> int:
        return len(self._entries)
