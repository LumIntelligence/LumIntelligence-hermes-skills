"""Test-wide guard: no test may touch the real ~/.hermes or ~/.lumintelligence."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_homes(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("homes")
    monkeypatch.setenv("HERMES_HOME", str(root / "hermes"))
    monkeypatch.setenv("LUMINTELLIGENCE_HOME", str(root / "lumintelligence"))
