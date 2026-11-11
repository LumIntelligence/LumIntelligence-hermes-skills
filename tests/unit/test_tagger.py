"""Tagger coverage, batched: tagging + concept creation + the batched embedding phase,
retries, the empty/echo-only retry round, concept dedup across entities and batches, batch
planning, the engine-owned frame and schema, truncation splitting, parallel == serial, and that an
embedder exception (unlike the LLM call) propagates uncaught."""

from __future__ import annotations

import json
import random
import time

import pytest

from lumintelligence.adhd import Tagger
from lumintelligence.adhd.tagger import clean_tags, plan_batches
from lumintelligence.graph import SqliteGraphStore
from lumintelligence.llm.base import OutputTruncated
from lumintelligence.llm.governor import Governor
from lumintelligence.llm.mock import MockProvider


def _kg_with_entities(*names: str) -> tuple[SqliteGraphStore, list[str]]:
    kg = SqliteGraphStore()
    src = kg.add_source("content")
    return kg, [kg.add_entity(n, context={"source_id": src, "text": f"mention of {n}"}) for n in names]


def _batch(fn):
    """Handler adapter: call fn(name, entity_input) for every entity in the batch."""
    def handler(system_prompt, content, schema):
        return {name: fn(name, data) for name, data in json.loads(content).items()}
    return handler


def _tags(kg, eid):
    return kg.get_node(eid)["attributes"]["tags"]


# ------------------------------------------------------------------ core behaviour
def test_run_tags_entity_creates_concepts_and_embeds_them():
    kg, (entity,) = _kg_with_entities("Acme Corp")
    tagged = Tagger(MockProvider(json_handler=_batch(lambda n, d: ["company", "tech"])), MockProvider(),
                    "prompt").run(kg)

    assert tagged == 1
    assert _tags(kg, entity) == ["company", "tech"]
    concepts = kg.nodes_by_type("Concept")
    assert {kg.get_node(c)["name"] for c in concepts} == {"company", "tech"}
    assert all(kg.concept_embedding(c) is not None for c in concepts)


def test_run_dedups_concepts_by_name_across_entities_and_batches():
    kg, ids = _kg_with_entities(*[f"Org {i}" for i in range(12)])
    Tagger(MockProvider(json_handler=_batch(lambda n, d: ["company"])), MockProvider(), "prompt",
           batch_max=4, batch_min=1).run(kg, governor=Governor(3))
    assert len(kg.nodes_by_type("Concept")) == 1
    assert all(_tags(kg, e) == ["company"] for e in ids)


def test_run_retries_and_succeeds_after_transient_failures():
    kg, (entity,) = _kg_with_entities("Acme Corp")
    attempts = {"n": 0}

    def handler(system_prompt, content, schema):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return {"Acme Corp": ["company"]}

    Tagger(MockProvider(json_handler=handler), MockProvider(), "prompt").run(kg)
    assert _tags(kg, entity) == ["company"]


def test_run_all_attempts_exhausted_tags_entity_with_empty_list():
    kg, (entity,) = _kg_with_entities("Acme Corp")

    def always_fails(system_prompt, content, schema):
        raise RuntimeError("down")

    Tagger(MockProvider(json_handler=always_fails), MockProvider(), "prompt").run(kg)
    assert _tags(kg, entity) == []          # tagged-as-empty, so it is not retried forever
    assert kg.entities_to_tag() == []


def test_clean_tags_drops_blank_duplicate_and_name_echo_tags():
    assert clean_tags("Acme Corp", ["  ", "Company", "company", "acme  corp", "Tech ", ""]) == ["Company", "Tech"]
    assert clean_tags("Acme Corp", None) == []


def test_empty_or_echo_only_entities_are_retried_once_then_accepted():
    kg, (a, b, c) = _kg_with_entities("Alpha", "Beta", "Gamma")
    seen: list[list[str]] = []

    def handler(system_prompt, content, schema):
        names = list(json.loads(content))
        seen.append(names)
        if len(seen) == 1:   # first pass: Beta echo-only, Gamma missing
            return {"Alpha": ["Letter"], "Beta": ["Beta"]}
        return {n: (["Second Letter"] if n == "Beta" else []) for n in names}

    Tagger(MockProvider(json_handler=handler), MockProvider(), "prompt").run(kg)
    assert seen[1] == ["Beta", "Gamma"]                   # one retry round with only those
    assert len(seen) == 2
    assert _tags(kg, a) == ["Letter"] and _tags(kg, b) == ["Second Letter"] and _tags(kg, c) == []


def test_answer_keys_matched_by_normalized_name():
    kg, (entity,) = _kg_with_entities("Acme Corp")
    Tagger(MockProvider(json_handler=lambda s, c, sc: {"acme  CORP": ["company"]}), MockProvider(),
           "prompt").run(kg)
    assert _tags(kg, entity) == ["company"]


def test_run_propagates_embedder_exception_unlike_llm_retry_loop():
    kg, _ = _kg_with_entities("Acme Corp")

    class FailingEmbedder(MockProvider):
        def embed(self, texts, purpose=None):
            raise RuntimeError("embedding service down")

    with pytest.raises(RuntimeError, match="embedding service down"):
        Tagger(MockProvider(json_handler=_batch(lambda n, d: ["company"])), FailingEmbedder(), "prompt").run(kg)


def test_run_emits_start_progress_embedding_and_done_events():
    kg, _ = _kg_with_entities("Acme Corp", "Globex")
    events: list[dict] = []
    Tagger(MockProvider(json_handler=_batch(lambda n, d: ["company"])), MockProvider(), "prompt").run(
        kg, on_event=events.append)
    phases = [e["phase"] for e in events]
    assert phases[0] == "start" and phases[-1] == "done"
    assert "progress" in phases and "embedding" in phases


def test_run_on_empty_graph_processes_nothing():
    assert Tagger(MockProvider(), MockProvider(), "prompt").run(SqliteGraphStore()) == 0


# ------------------------------------------------------------------ batch format
def test_batch_input_includes_contexts_and_inferred_type():
    kg = SqliteGraphStore()
    src = kg.add_source("content")
    kg.add_entity("Acme Corp", context={"source_id": src, "text": "m1"}, inferred_type="Organization")
    kg.add_entity("Globex", context={"source_id": src, "text": "m2"})
    seen: dict = {}

    def handler(system_prompt, content, schema):
        seen.update(json.loads(content))
        return {n: ["tag"] for n in json.loads(content)}

    Tagger(MockProvider(json_handler=handler), MockProvider(), "prompt").run(kg)
    assert seen["Acme Corp"] == {"inferred_type": "Organization", "context": ["m1"]}
    assert seen["Globex"]["inferred_type"] is None


def test_prompt_is_guidance_plus_frame_and_schema_requires_every_name():
    kg, _ = _kg_with_entities("Acme Corp", "Globex")
    seen = {}

    def handler(system_prompt, content, schema):
        seen["prompt"], seen["schema"] = system_prompt, schema
        return {n: ["tag"] for n in json.loads(content)}

    Tagger(MockProvider(json_handler=handler), MockProvider(), "MY GUIDANCE").run(kg)
    assert seen["prompt"].startswith("MY GUIDANCE")
    assert "Batch format" in seen["prompt"]
    assert sorted(seen["schema"]["required"]) == ["Acme Corp", "Globex"]
    assert seen["schema"]["additionalProperties"] is False


@pytest.mark.parametrize("n,slots,bmax,bmin,sizes", [
    (178, 3, 50, 5, [30, 30, 30, 30, 30, 28]),
    (97, 3, 50, 5, [33, 33, 31]),
    (12, 3, 50, 5, [6, 6]),
    (3, 3, 50, 5, [3]),
    (120, 2, 50, 5, [30, 30, 30, 30]),
    (5, 2, 1, 5, [1, 1, 1, 1, 1]),
])
def test_plan_batches(n, slots, bmax, bmin, sizes):
    assert [len(b) for b in plan_batches([str(i) for i in range(n)], slots, bmax, bmin)] == sizes


def test_batch_max_one_tags_one_entity_per_call():
    kg, _ = _kg_with_entities("A1", "B2", "C3")
    sizes: list[int] = []

    def handler(system_prompt, content, schema):
        sizes.append(len(json.loads(content)))
        return {n: ["tag"] for n in json.loads(content)}

    Tagger(MockProvider(json_handler=handler), MockProvider(), "prompt", batch_max=1).run(kg)
    assert sizes == [1, 1, 1]


def test_truncated_batch_is_split_in_half():
    kg, ids = _kg_with_entities(*[f"Org {i}" for i in range(8)])
    sizes: list[int] = []

    def handler(system_prompt, content, schema):
        names = list(json.loads(content))
        sizes.append(len(names))
        if len(names) > 2:
            raise OutputTruncated("cut off")
        return {n: ["company"] for n in names}

    Tagger(MockProvider(json_handler=handler), MockProvider(), "prompt", batch_min=1).run(kg)
    assert sizes == [8, 4, 2, 2, 4, 2, 2]
    assert all(_tags(kg, e) == ["company"] for e in ids)


def test_parallel_tagging_builds_the_same_graph_as_serial():
    def handler(system_prompt, content, schema):
        time.sleep(random.random() * 0.01)
        return {n: [f"T{len(n) % 3}", f"U{n[-1]}"] for n in json.loads(content)}

    sigs = []
    for gov in (Governor(1), Governor(4)):
        kg, ids = _kg_with_entities(*[f"Entity {i}" for i in range(40)])
        Tagger(MockProvider(json_handler=handler), MockProvider(), "prompt", batch_max=5, batch_min=1).run(
            kg, governor=gov)
        concepts = [kg.get_node(c)["name"] for c in kg.nodes_by_type("Concept")]   # creation order
        sigs.append((concepts, [_tags(kg, e) for e in ids]))
    assert sigs[0] == sigs[1]
