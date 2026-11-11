"""Direct NodeDiscovery coverage: success path, the 5-attempt
retry loop, all-attempts-exhausted silent degrade, blank-entity-name skipping, and
inferred_type canonicalization via embedding similarity."""

from __future__ import annotations

from lumintelligence.graph import SqliteGraphStore
from lumintelligence.llm.mock import MockProvider
from lumintelligence.nod import NodeDiscovery


class _FixedVectorEmbedder(MockProvider):
    """Test-only embedder returning caller-controlled vectors for known strings, so
    similarity-based canonicalization can be tested deterministically (MockProvider's
    real hash-based vectors have no notion of semantic similarity between strings)."""

    def __init__(self, vectors: dict[str, list[float]]):
        super().__init__()
        self._vectors = vectors

    def embed(self, texts, purpose=None):
        return [self._vectors[t] for t in texts]


def test_run_extracts_entities_and_marks_source_processed():
    kg = SqliteGraphStore()
    src = kg.add_source("Acme Corp met Globex.")

    def handler(system_prompt, content, schema):
        return {"Acme Corp": "Acme Corp met Globex", "Globex": "Acme Corp met Globex"}

    processed = NodeDiscovery(MockProvider(json_handler=handler), MockProvider(), "prompt").run(kg)

    assert processed == 1
    assert src not in kg.sources_to_process()
    assert kg.find_entity_by_name("Acme Corp") is not None
    assert kg.find_entity_by_name("Globex") is not None
    assert len(kg.neighbors(src, "out")) == 2  # MENTIONS edges to both entities


def test_run_retries_and_succeeds_after_transient_failures():
    kg = SqliteGraphStore()
    kg.add_source("Acme Corp is here.")
    calls = {"n": 0}

    def handler(system_prompt, content, schema):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient provider error")
        return {"Acme Corp": "Acme Corp is here"}

    processed = NodeDiscovery(MockProvider(json_handler=handler), MockProvider(), "prompt").run(kg)

    assert processed == 1
    assert calls["n"] == 3
    assert kg.find_entity_by_name("Acme Corp") is not None


def test_run_all_attempts_exhausted_still_marks_source_processed_with_no_entities():
    """A source whose LLM call fails all 5 attempts is not an error — it's a silent
    degrade: the source is still marked processed (removed from the queue) with zero
    entities extracted, rather than being retried forever on the next run() call."""
    kg = SqliteGraphStore()
    src = kg.add_source("Unparseable content.")

    def always_fails(system_prompt, content, schema):
        raise RuntimeError("provider down")

    processed = NodeDiscovery(MockProvider(json_handler=always_fails), MockProvider(), "prompt").run(kg)

    assert processed == 1
    assert src not in kg.sources_to_process()
    assert kg.nodes_by_type("Discovered_entity") == []


def test_run_skips_blank_entity_names_from_flaky_llm_output():
    kg = SqliteGraphStore()
    kg.add_source("content")

    def handler(system_prompt, content, schema):
        return {"   ": "blank name", "Acme Corp": "valid name"}

    NodeDiscovery(MockProvider(json_handler=handler), MockProvider(), "prompt").run(kg)

    assert kg.find_entity_by_name("Acme Corp") is not None
    assert kg.nodes_by_type("Discovered_entity") == [kg.find_entity_by_name("Acme Corp")]


def test_run_emits_start_progress_and_done_events():
    kg = SqliteGraphStore()
    kg.add_source("content")
    events: list[dict] = []

    def handler(system_prompt, content, schema):
        return {"Acme Corp": "content"}

    NodeDiscovery(MockProvider(json_handler=handler), MockProvider(), "prompt").run(kg, on_event=events.append)

    phases = [e["phase"] for e in events]
    assert phases[0] == "start"
    assert phases[-1] == "done"
    assert "progress" in phases


def test_run_on_empty_graph_processes_nothing():
    kg = SqliteGraphStore()
    processed = NodeDiscovery(MockProvider(), MockProvider(), "prompt").run(kg)
    assert processed == 0


def test_run_backward_compatible_plain_string_value_has_no_inferred_type():
    """The legacy {name: "context string"} shape (no dict, no inferred_type) must keep
    working — treated as context-only."""
    kg = SqliteGraphStore()
    kg.add_source("content")

    def handler(system_prompt, content, schema):
        return {"Acme Corp": "plain context string"}

    NodeDiscovery(MockProvider(json_handler=handler), MockProvider(), "prompt").run(kg)

    entity = kg.find_entity_by_name("Acme Corp")
    node = kg.get_node(entity)
    assert node["attributes"]["context"][0]["text"] == "plain context string"
    assert node["attributes"]["inferred_type"] is None


def test_run_dict_shaped_value_sets_context_and_inferred_type():
    kg = SqliteGraphStore()
    kg.add_source("content")

    def handler(system_prompt, content, schema):
        return {"Acme Corp": {"context": "Acme is a company", "inferred_type": "Organization"}}

    embedder = _FixedVectorEmbedder({"Organization": [1.0, 0.0]})
    NodeDiscovery(MockProvider(json_handler=handler), embedder, "prompt").run(kg)

    entity = kg.find_entity_by_name("Acme Corp")
    node = kg.get_node(entity)
    assert node["attributes"]["context"][0]["text"] == "Acme is a company"
    assert node["attributes"]["inferred_type"] == "Organization"


def test_run_decodes_object_value_sent_as_a_json_string():
    """Some models encode each value as JSON text; it is decoded, not stored as the context."""
    kg = SqliteGraphStore()
    kg.add_source("content")

    def handler(system_prompt, content, schema):
        return {"Acme Corp": '{"context": "Acme is a company", "inferred_type": "Organization"}',
                "Globex": "{not json, kept as plain context",
                "Initech": '{"name": "no context key"}'}

    embedder = _FixedVectorEmbedder({"Organization": [1.0, 0.0]})
    NodeDiscovery(MockProvider(json_handler=handler), embedder, "prompt").run(kg)

    acme = kg.get_node(kg.find_entity_by_name("Acme Corp"))
    assert acme["attributes"]["context"][0]["text"] == "Acme is a company"
    assert acme["attributes"]["inferred_type"] == "Organization"
    globex = kg.get_node(kg.find_entity_by_name("Globex"))
    assert globex["attributes"]["context"][0]["text"] == "{not json, kept as plain context"
    assert globex["attributes"]["inferred_type"] is None
    initech = kg.get_node(kg.find_entity_by_name("Initech"))
    assert initech["attributes"]["context"][0]["text"] == '{"name": "no context key"}'


def test_run_canonicalizes_similar_inferred_types_via_embedding_similarity():
    """"corporation" and "company" are textually different but semantically the same
    type — the second must collapse into the first via embedding similarity, not be
    stored as a separate, fragmenting label."""
    kg = SqliteGraphStore()
    kg.add_source("Acme is a company.")
    kg.add_source("Globex is a corporation.")
    embedder = _FixedVectorEmbedder({"company": [1.0, 0.0], "corporation": [0.99, 0.01]})

    def handler(system_prompt, content, schema):
        if "Acme" in content:
            return {"Acme": {"context": "Acme is a company.", "inferred_type": "company"}}
        return {"Globex": {"context": "Globex is a corporation.", "inferred_type": "corporation"}}

    NodeDiscovery(MockProvider(json_handler=handler), embedder, "prompt",
                 type_similarity_threshold=0.9).run(kg)

    acme_type = kg.get_node(kg.find_entity_by_name("Acme"))["attributes"]["inferred_type"]
    globex_type = kg.get_node(kg.find_entity_by_name("Globex"))["attributes"]["inferred_type"]
    assert acme_type == "company"
    assert globex_type == "company"  # canonicalized to the first-seen label, not "corporation"


def test_run_keeps_genuinely_different_inferred_types_separate():
    kg = SqliteGraphStore()
    kg.add_source("Acme is a company.")
    kg.add_source("This apple is a fruit.")
    embedder = _FixedVectorEmbedder({"company": [1.0, 0.0], "fruit": [0.0, 1.0]})

    def handler(system_prompt, content, schema):
        if "Acme" in content:
            return {"Acme": {"context": "Acme is a company.", "inferred_type": "company"}}
        return {"Apple": {"context": "This apple is a fruit.", "inferred_type": "fruit"}}

    NodeDiscovery(MockProvider(json_handler=handler), embedder, "prompt",
                 type_similarity_threshold=0.9).run(kg)

    acme_type = kg.get_node(kg.find_entity_by_name("Acme"))["attributes"]["inferred_type"]
    apple_type = kg.get_node(kg.find_entity_by_name("Apple"))["attributes"]["inferred_type"]
    assert acme_type == "company"
    assert apple_type == "fruit"


def test_run_seeds_registry_from_existing_graph_state():
    """A second, later run() call must canonicalize against types already recorded on
    the graph from a prior call — not just types discovered within the same call."""
    kg = SqliteGraphStore()
    kg.add_entity("Acme", inferred_type="company")  # simulates a prior NoD run's result
    kg.add_source("Globex is a corporation.")
    embedder = _FixedVectorEmbedder({"company": [1.0, 0.0], "corporation": [0.99, 0.01]})

    def handler(system_prompt, content, schema):
        return {"Globex": {"context": "Globex is a corporation.", "inferred_type": "corporation"}}

    NodeDiscovery(MockProvider(json_handler=handler), embedder, "prompt",
                 type_similarity_threshold=0.9).run(kg)

    globex_type = kg.get_node(kg.find_entity_by_name("Globex"))["attributes"]["inferred_type"]
    assert globex_type == "company"  # canonicalized against the pre-existing type, not a fresh label


def test_run_first_write_wins_inferred_type_on_entity_reuse():
    kg = SqliteGraphStore()
    kg.add_entity("Acme", inferred_type="company")

    def handler(system_prompt, content, schema):
        return {"Acme": {"context": "more context", "inferred_type": "nonprofit"}}

    kg.add_source("more context about Acme")
    embedder = _FixedVectorEmbedder({"company": [1.0, 0.0], "nonprofit": [0.0, 1.0]})
    NodeDiscovery(MockProvider(json_handler=handler), embedder,
                 "prompt", type_similarity_threshold=0.9).run(kg)

    assert kg.get_node(kg.find_entity_by_name("Acme"))["attributes"]["inferred_type"] == "company"


# --------------------------------------------------------------- parallel + journal
def _graph_signature(kg: SqliteGraphStore) -> list:
    """Order-sensitive content of a graph, ignoring generated ids and timestamps."""
    data = kg.export_json()
    names = {d["id"]: (d.get("name") or d["attributes"].get("content")) for d in data["nodes"]}
    nodes = [(d["type"], names[d["id"]], d["attributes"].get("inferred_type"),
              [c["text"] for c in d["attributes"].get("context", [])])
             for d in data["nodes"]]
    edges = sorted((names[e["source"]], names[e["target"]], e["type"]) for e in data["links"])
    return [nodes, edges]


def _varied_handler(system_prompt, content, schema):
    import random
    import time
    time.sleep(random.random() * 0.01)                       # finish out of order
    n = int(content.split()[-1])
    return {"Shared Org": {"context": f"mention {n}", "inferred_type": "Organization" if n % 2 else "Company"},
            f"Person {n}": {"context": f"person {n}", "inferred_type": "Person"}}


def test_parallel_nod_builds_the_same_graph_as_serial():
    from lumintelligence.llm.governor import Governor

    graphs = []
    for gov in (None, Governor(4)):
        kg = SqliteGraphStore()
        for i in range(12):
            kg.add_source(f"source number {i}")
        NodeDiscovery(MockProvider(json_handler=_varied_handler), MockProvider(), "prompt").run(kg, governor=gov)
        graphs.append(_graph_signature(kg))
    assert graphs[0] == graphs[1]


def test_progress_events_arrive_in_source_order_with_parallel_calls():
    from lumintelligence.llm.governor import Governor

    kg = SqliteGraphStore()
    sids = [kg.add_source(f"source number {i}") for i in range(8)]
    events: list[dict] = []
    NodeDiscovery(MockProvider(json_handler=_varied_handler), MockProvider(), "prompt").run(
        kg, on_event=events.append, governor=Governor(4))
    assert [e["source_id"] for e in events if e["phase"] == "progress"] == sids


def test_journal_replays_recorded_answers_without_calling_the_model(tmp_path):
    from lumintelligence.storage.journal import JsonlJournal

    calls = {"n": 0}

    def counting(system_prompt, content, schema):
        calls["n"] += 1
        return {"Acme": {"context": "Acme", "inferred_type": "Organization"}}

    path = str(tmp_path / "journal.jsonl")
    kg1 = SqliteGraphStore()
    kg1.add_source("Acme signed a deal.")
    NodeDiscovery(MockProvider(json_handler=counting), MockProvider(), "prompt").run(kg1, journal=JsonlJournal(path))
    assert calls["n"] == 1

    kg2 = SqliteGraphStore()                                   # e.g. a rerun after a crash
    kg2.add_source("Acme signed a deal.")
    NodeDiscovery(MockProvider(json_handler=counting), MockProvider(), "prompt").run(kg2, journal=JsonlJournal(path))
    assert calls["n"] == 1                                   # replayed, not called
    assert kg2.find_entity_by_name("Acme") is not None

    kg3 = SqliteGraphStore()                                   # a different prompt does not match
    kg3.add_source("Acme signed a deal.")
    NodeDiscovery(MockProvider(json_handler=counting), MockProvider(), "other prompt").run(
        kg3, journal=JsonlJournal(path))
    assert calls["n"] == 2


def test_failed_sources_are_not_journaled(tmp_path):
    from lumintelligence.storage.journal import JsonlJournal

    def always_fails(system_prompt, content, schema):
        raise RuntimeError("down")

    j = JsonlJournal(str(tmp_path / "journal.jsonl"))
    kg = SqliteGraphStore()
    kg.add_source("text")
    NodeDiscovery(MockProvider(json_handler=always_fails), MockProvider(), "prompt").run(kg, journal=j)
    assert len(j) == 0


def test_valid_empty_answer_is_accepted_without_retry():
    kg = SqliteGraphStore()
    src = kg.add_source("Nothing to see here.")
    calls = {"n": 0}

    def empty(system_prompt, content, schema):
        calls["n"] += 1
        return {}

    events: list[dict] = []
    NodeDiscovery(MockProvider(json_handler=empty), MockProvider(), "prompt").run(kg, on_event=events.append)
    assert calls["n"] == 1
    assert src not in kg.sources_to_process()
    assert [e["success"] for e in events if e["phase"] == "progress"] == [True]


def test_non_object_answer_is_retried_then_reported_as_failure():
    class ListAnswer(MockProvider):
        def complete_json(self, system_prompt, content, schema):
            self.n = getattr(self, "n", 0) + 1
            return ["not", "an", "object"]

    kg = SqliteGraphStore()
    kg.add_source("text")
    llm = ListAnswer()
    events: list[dict] = []
    NodeDiscovery(llm, MockProvider(), "prompt").run(kg, on_event=events.append)
    assert llm.n == 5
    progress = next(e for e in events if e["phase"] == "progress")
    assert progress["success"] is False and "JSON object" in progress["error"]
