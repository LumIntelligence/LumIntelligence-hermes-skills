"""Direct Consolidator coverage for both algorithms: incremental
nearest-centroid `run()` and agglomerative `rebuild()`."""

from __future__ import annotations

from lumintelligence.consolidation import Consolidator
from lumintelligence.graph import SqliteGraphStore


def test_run_on_empty_graph_returns_zero():
    kg = SqliteGraphStore()
    assert Consolidator(0.8).run(kg) == 0


def test_run_gives_concept_without_embedding_its_own_centroid_unconditionally():
    kg = SqliteGraphStore()
    concept = kg.add_concept("Company")  # no embedding

    assigned = Consolidator(0.8).run(kg)

    assert assigned == 1
    centroids = kg.nodes_by_type("Centroid")
    assert len(centroids) == 1
    assert kg.centroid_members(centroids[0]) == [concept]


def test_run_joins_existing_centroid_exactly_at_threshold_boundary():
    """The comparison is `best_sim >= threshold` — a concept whose cosine similarity to
    the nearest centroid exactly equals the threshold must join it, not spin up a new one."""
    kg = SqliteGraphStore()
    first = kg.add_concept("First", embedding=[1.0, 0.0])
    second = kg.add_concept("Second", embedding=[1.0, 0.0])  # identical vector -> cosine == 1.0

    consolidator = Consolidator(similarity_threshold=1.0)
    consolidator.run(kg)  # assigns `first` its own centroid (nothing to compare against yet)
    consolidator.run(kg)  # now assigns `second`, comparing against `first`'s centroid

    centroids = kg.nodes_by_type("Centroid")
    assert len(centroids) == 1  # joined, did not spin up a second centroid
    assert set(kg.centroid_members(centroids[0])) == {first, second}


def test_run_spins_up_new_centroid_when_below_threshold():
    kg = SqliteGraphStore()
    kg.add_concept("First", embedding=[1.0, 0.0])
    kg.add_concept("Second", embedding=[0.0, 1.0])  # orthogonal -> cosine == 0.0

    consolidator = Consolidator(similarity_threshold=0.5)
    consolidator.run(kg)
    consolidator.run(kg)

    assert len(kg.nodes_by_type("Centroid")) == 2


def test_run_running_mean_keeps_third_identical_vector_joining_same_centroid():
    kg = SqliteGraphStore()
    kg.add_concept("First", embedding=[1.0, 0.0])
    kg.add_concept("Second", embedding=[1.0, 0.0])
    kg.add_concept("Third", embedding=[1.0, 0.0])

    consolidator = Consolidator(similarity_threshold=1.0)
    for _ in range(3):
        consolidator.run(kg)

    centroids = kg.nodes_by_type("Centroid")
    assert len(centroids) == 1
    assert len(kg.centroid_members(centroids[0])) == 3


def test_rebuild_on_empty_graph_returns_zero():
    kg = SqliteGraphStore()
    assert Consolidator(0.8).rebuild(kg) == 0


def test_rebuild_returns_zero_when_no_concepts_have_embeddings():
    kg = SqliteGraphStore()
    kg.add_concept("Company")  # no embedding
    assert Consolidator(0.8).rebuild(kg) == 0


def test_rebuild_single_concept_gets_its_own_cluster():
    kg = SqliteGraphStore()
    concept = kg.add_concept("Company", embedding=[1.0, 0.0])

    clusters = Consolidator(0.8).rebuild(kg)

    assert clusters == 1
    centroids = kg.nodes_by_type("Centroid")
    assert kg.centroid_members(centroids[0]) == [concept]


def test_rebuild_groups_similar_concepts_and_separates_dissimilar_ones():
    kg = SqliteGraphStore()
    kg.add_concept("A1", embedding=[1.0, 0.0])
    kg.add_concept("A2", embedding=[0.99, 0.01])   # near-identical to A1
    kg.add_concept("B1", embedding=[0.0, 1.0])
    kg.add_concept("B2", embedding=[0.01, 0.99])   # near-identical to B1, orthogonal to A-cluster

    clusters = Consolidator(similarity_threshold=0.7).rebuild(kg)

    assert clusters == 2
    centroids = kg.nodes_by_type("Centroid")
    sizes = sorted(len(kg.centroid_members(c)) for c in centroids)
    assert sizes == [2, 2]


def test_rebuild_picks_the_member_closest_to_cluster_mean_as_medoid():
    """The medoid (used as the resulting centroid's name) is the cluster member closest
    to the cluster's arithmetic mean — not simply the first concept added."""
    kg = SqliteGraphStore()
    kg.add_concept("First", embedding=[1.0, 0.0])
    kg.add_concept("Second", embedding=[1.0, 0.6])
    kg.add_concept("Third", embedding=[1.0, 0.3])  # equals the arithmetic mean of the three

    Consolidator(similarity_threshold=0.5).rebuild(kg)

    centroids = kg.nodes_by_type("Centroid")
    assert len(centroids) == 1  # all three are close enough to cluster together
    assert kg.get_node(centroids[0])["name"] == "Third"


def test_rebuild_clears_stale_centroids_before_regrouping():
    kg = SqliteGraphStore()
    concept = kg.add_concept("Company", embedding=[1.0, 0.0])
    stale_centroid = kg.add_centroid("Stale cluster")
    kg.assign_concept_to_centroid(concept, stale_centroid)

    Consolidator(0.8).rebuild(kg)

    assert stale_centroid not in kg.nodes_by_type("Centroid")
    new_centroids = kg.nodes_by_type("Centroid")
    assert len(new_centroids) == 1
    assert new_centroids[0] != stale_centroid


def test_rebuild_is_non_destructive_to_concepts_and_their_entity_edges():
    kg = SqliteGraphStore()
    entity = kg.add_entity("Acme")
    concept = kg.add_concept("Company", embedding=[1.0, 0.0])
    kg.link_is_concept(entity, concept)

    Consolidator(0.8).rebuild(kg)

    assert concept in kg.nodes_by_type("Concept")
    assert kg.neighbors(entity, "out") == [concept]  # IS_CONCEPT edge survives


def test_rebuild_emits_start_and_done_events():
    kg = SqliteGraphStore()
    kg.add_concept("Company", embedding=[1.0, 0.0])
    events: list[dict] = []

    Consolidator(0.8).rebuild(kg, on_event=events.append)

    phases = [e["phase"] for e in events]
    assert phases[0] == "start"
    assert phases[-1] == "done"
