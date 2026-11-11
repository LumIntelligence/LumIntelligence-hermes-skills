"""Governor: input-order results, bounded concurrency, exceptions returned as results."""

from __future__ import annotations

import threading
import time

import pytest

from lumintelligence.llm.governor import Governor


def test_results_come_back_in_input_order_even_when_finishing_out_of_order():
    delays = [0.05, 0.0, 0.03, 0.0, 0.01]

    def fn(i):
        time.sleep(delays[i])
        return i * 10

    assert list(Governor(3).map_ordered(fn, range(5))) == [0, 10, 20, 30, 40]


def test_never_more_than_max_in_flight():
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def fn(_):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return True

    list(Governor(3).map_ordered(fn, range(12)))
    assert state["peak"] == 3


def test_exceptions_are_returned_as_results():
    def fn(i):
        if i == 1:
            raise RuntimeError("boom")
        return i

    out = list(Governor(2).map_ordered(fn, range(3)))
    assert out[0] == 0 and out[2] == 2
    assert isinstance(out[1], RuntimeError)


def test_serial_governor_and_validation():
    assert list(Governor(1).map_ordered(lambda x: x + 1, [1, 2])) == [2, 3]
    with pytest.raises(ValueError):
        Governor(0)
