"""Consolidation — grouping concepts under Centroid nodes.

The default is **incremental nearest-centroid** assignment (stable, live-friendly).
``rebuild`` re-derives groupings globally with agglomerative clustering and is
**non-destructive**: it clears only Centroid nodes/edges, never Concepts or their entity
edges. Both reuse the cached concept embeddings (no re-embedding) and
the same graph-write path. This is the grouping half of what used to be called ADHD;
turning entities into concepts in the first place is the separate tagging stage
(`lumintelligence.adhd`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from lumintelligence.constants import NodeType
from lumintelligence.graph import GraphStore


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


class Consolidator:
    def __init__(self, similarity_threshold: float):
        self.threshold = similarity_threshold

    # -------------------------------------------------- incremental (default)
    def run(self, kg: GraphStore, on_event: Callable[[dict], None] | None = None) -> int:
        """Assign each unconsolidated concept to the nearest existing centroid (cosine
        >= threshold) or spin up a new centroid. Returns concepts assigned."""
        emit = on_event or (lambda e: None)
        pending = kg.concepts_unconsolidated()
        emit({"stage": "consolidate", "phase": "start", "total": len(pending)})
        if not pending:
            emit({"stage": "consolidate", "phase": "done", "centroids": len(kg.nodes_by_type(NodeType.CENTROID))})
            return 0

        # Running-mean centroid vectors, seeded from existing members (in-memory, temporary).
        centroids: dict[str, dict] = {}
        for cen in kg.nodes_by_type("Centroid"):
            vecs = [kg.concept_embedding(m) for m in kg.centroid_members(cen)]
            vecs = [v for v in vecs if v]
            if vecs:
                arr = np.asarray(vecs, dtype=float)
                centroids[cen] = {"sum": arr.sum(axis=0), "count": len(vecs)}

        assigned = 0
        for cid in pending:
            emb = kg.concept_embedding(cid)
            name = kg.get_node(cid)["name"]  # type: ignore[index]
            if emb is None:
                # Can't compare without a vector — give it its own centroid.
                cen = kg.add_centroid(name)
                kg.assign_concept_to_centroid(cid, cen)
                assigned += 1
                continue
            v = np.asarray(emb, dtype=float)
            best_id, best_sim = None, -1.0
            for cen, data in centroids.items():
                sim = _cosine(v, data["sum"] / data["count"])
                if sim > best_sim:
                    best_sim, best_id = sim, cen
            if best_id is not None and best_sim >= self.threshold:
                kg.assign_concept_to_centroid(cid, best_id)
                centroids[best_id]["sum"] += v
                centroids[best_id]["count"] += 1
            else:
                cen = kg.add_centroid(name)
                kg.assign_concept_to_centroid(cid, cen)
                centroids[cen] = {"sum": v.copy(), "count": 1}
            assigned += 1
        emit({"stage": "consolidate", "phase": "done", "assigned": assigned,
              "centroids": len(kg.nodes_by_type(NodeType.CENTROID))})
        return assigned

    # ----------------------------------------------- full rebuild (non-destructive)
    def rebuild(self, kg: GraphStore, on_event: Callable[[dict], None] | None = None) -> int:
        """Globally re-derive centroids via agglomerative clustering. Clears only
        Centroid nodes/edges; Concepts and their IS_CONCEPT edges are preserved.
        Returns the number of centroids created."""
        from sklearn.cluster import AgglomerativeClustering

        emit = on_event or (lambda e: None)
        kg.clear_centroids()
        concepts = kg.nodes_by_type("Concept")
        emit({"stage": "consolidate", "phase": "start", "total": len(concepts)})
        valid = [(c, kg.concept_embedding(c)) for c in concepts]
        valid = [(c, e) for c, e in valid if e]
        if not valid:
            emit({"stage": "consolidate", "phase": "done", "centroids": 0})
            return 0

        ids = [c for c, _ in valid]
        X = np.asarray([e for _, e in valid], dtype=float)

        if len(valid) == 1:
            labels = [0]
        else:
            model = AgglomerativeClustering(
                n_clusters=None,  # type: ignore[arg-type]
                distance_threshold=1.0 - self.threshold,
                linkage="average",
                metric="cosine",
            )
            labels = model.fit_predict(X)

        clusters: dict[int, list[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            clusters[int(lab)].append(idx)

        for idxs in clusters.values():
            sub = X[idxs]
            centroid_vec = sub.mean(axis=0)
            # medoid = member closest to the cluster centroid
            medoid_local = idxs[int(np.argmin([_cosine(X[i], centroid_vec) * -1 for i in idxs]))]
            cen = kg.add_centroid(kg.get_node(ids[medoid_local])["name"])  # type: ignore[index]
            for i in idxs:
                kg.assign_concept_to_centroid(ids[i], cen)
        emit({"stage": "consolidate", "phase": "done", "centroids": len(clusters)})
        return len(clusters)
