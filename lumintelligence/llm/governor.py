"""Governor — bounds how many provider calls run at once.

``map_ordered(fn, items)`` runs ``fn`` over ``items`` with at most ``max_in_flight`` calls in
flight and yields results **in input order**, each as soon as it and all earlier ones are done.
Callers apply results to the graph on their own thread as they are yielded (the single ordered
writer), so workers never touch the graph and the output matches a serial run.

An exception raised by ``fn`` is yielded as the result for that item rather than raised, so one
failed source or batch does not stop the rest.

Offline policy only: a fixed cap, sized to the local server's parallel slots (e.g.
``OLLAMA_NUM_PARALLEL``). An online policy (adaptive cap, request/token budgets) can be added
here later behind the same interface.
"""

from __future__ import annotations

import concurrent.futures as cf
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


class Governor:
    def __init__(self, max_in_flight: int = 1):
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        self.max_in_flight = max_in_flight

    def map_ordered(self, fn: Callable[[_T], _R], items: Iterable[_T]) -> Iterator[_R | Exception]:
        items = list(items)
        if self.max_in_flight == 1 or len(items) <= 1:
            for item in items:
                yield _call(fn, item)
            return
        with cf.ThreadPoolExecutor(max_workers=self.max_in_flight) as pool:
            futures = [pool.submit(_call, fn, item) for item in items]
            for f in futures:   # in input order; each waits only for its own result
                yield f.result()


def _call(fn: Callable[[_T], _R], item: _T) -> _R | Exception:
    try:
        return fn(item)
    except Exception as exc:  # noqa: BLE001 — returned to the caller as that item's result
        return exc
