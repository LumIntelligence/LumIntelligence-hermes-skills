"""`_find_entry_points`'s tuning params (vector search vs keyword fallback, max_results/threshold).
Uses a minimal fake vector store — `_find_entry_points`
only calls `.query(keyword, embedder, top_k=...)` on it.
"""

from __future__ import annotations

from lumintelligence.query import _find_entry_points


class _FakeVectorStore:
    def __init__(self, hits: list[dict]):
        self._hits = hits
        self.last_top_k: int | None = None

    def query(self, keyword, embedder, top_k=5):
        self.last_top_k = top_k
        return self._hits[:top_k]


def _hit(node_id: str, score: float) -> dict:
    return {"node_id": node_id, "score": score, "metadata": {"node_type": "Discovered_entity", "name": node_id}}


def test_find_entry_points_passes_max_results_through_as_top_k():
    vs = _FakeVectorStore([_hit("a", 0.1), _hit("b", 0.2), _hit("c", 0.3)])
    hits = _find_entry_points(None, vs, None, "kw", top_k=2, threshold=None)
    assert vs.last_top_k == 2
    assert [h["node_id"] for h in hits] == ["a", "b"]


def test_find_entry_points_filters_by_threshold_lower_is_stricter():
    vs = _FakeVectorStore([_hit("a", 0.1), _hit("b", 0.5), _hit("c", 0.9)])
    hits = _find_entry_points(None, vs, None, "kw", top_k=5, threshold=0.5)
    assert [h["node_id"] for h in hits] == ["a", "b"]   # 0.9 excluded, stricter than cutoff


def test_find_entry_points_threshold_has_no_effect_on_keyword_fallback():
    class _FakeStore:
        def match_entities_by_keyword(self, keyword):
            return [{"id": "a", "type": "Discovered_entity", "name": "a"}]

    hits = _find_entry_points(_FakeStore(), None, None, "kw", top_k=5, threshold=0.0)
    assert [h["node_id"] for h in hits] == ["a"]   # no distance signal to filter by
    assert hits[0]["distance"] is None
