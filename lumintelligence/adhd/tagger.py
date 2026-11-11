"""Tagging — abstract concept discovery (entity → concepts), batched.

Runs on net-new entities only (those with no tags yet), after NoD has finished, so every entity
carries all of its contexts. Entities are split by count into batches (first-mention order; at
most ``batch_max``, spread over the governor's slots, at least ``batch_min`` each); each batch is
one LLM call, several in flight, and results are applied in batch order on this thread.

The batch format is engine-owned (``sp_ADHD_frame``): the caller's prompt supplies only the
tagging guidance, the frame adds the input/output contract, and the call carries a JSON schema
with every entity name required. Safeguards: blank, duplicate and name-echo tags are dropped;
entities that come back missing, empty or echo-only are retried once; a batch whose answer is cut
off is split in half. Each tag becomes an atomic Concept (deduped by name, embedded once in
clustering space) linked from the entity via IS_CONCEPT; grouping concepts into centroids is the
separate consolidation stage.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

from lumintelligence.adhd.prompts import load_prompt
from lumintelligence.constants import Attr, NodeType
from lumintelligence.graph import GraphStore
from lumintelligence.llm.base import EmbeddingProvider, EmbedPurpose, LLMProvider, OutputTruncated
from lumintelligence.llm.governor import Governor
from lumintelligence.storage.journal import Journal, NullJournal, journal_key

_ATTEMPTS = 3   # per batch, for errors other than truncation (transport retries happen in the provider)


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def plan_batches(entity_ids: list[str], slots: int, batch_max: int, batch_min: int) -> list[list[str]]:
    """Split by count: at most ``batch_max`` per batch; enough batches to fill the slots (rounded
    up to a multiple of ``slots``) unless that would make batches smaller than ``batch_min``."""
    n = len(entity_ids)
    if n == 0:
        return []
    batch_max = max(1, batch_max)
    n_batches = math.ceil(n / batch_max)
    n_batches = slots * math.ceil(n_batches / slots)                 # fill every slot in each round
    n_batches = min(n_batches, max(math.ceil(n / batch_max), n // max(1, batch_min), 1))
    size = math.ceil(n / n_batches)
    return [entity_ids[i:i + size] for i in range(0, n, size)]


def tag_schema(names: list[str]) -> dict:
    return {"type": "object",
            "properties": {n: {"type": "array", "items": {"type": "string"}} for n in names},
            "required": list(names), "additionalProperties": False}


def clean_tags(name: str, raw: object) -> list[str]:
    """Strip, drop blanks, the entity's own name, and case-insensitive duplicates."""
    if not isinstance(raw, list):
        return []
    own, seen, out = _norm(name), set(), []
    for t in raw:
        s = str(t).strip()
        k = _norm(s)
        if s and k != own and k not in seen:
            seen.add(k)
            out.append(s)
    return out


class Tagger:
    def __init__(self, llm: LLMProvider, embedder: EmbeddingProvider, system_prompt: str,
                 batch_max: int = 50, batch_min: int = 5):
        self.llm = llm
        self.embedder = embedder
        self.guidance = system_prompt
        self.prompt = f"{system_prompt.rstrip()}\n\n{load_prompt('sp_ADHD_frame')}"
        self.batch_max = batch_max
        self.batch_min = batch_min
        self._model_id = str(getattr(getattr(llm, "config", None), "llm_model", type(llm).__name__))

    def run(self, kg: GraphStore, on_event: Callable[[dict], None] | None = None,
            governor: Governor | None = None, journal: Journal | None = None) -> int:
        """Tag all untagged entities; create/link their concepts. Returns entities tagged.

        Two phases on purpose: (1) all LLM tagging and concept/edge creation, then (2) one batched
        embedding of all new concepts (keeps a local server from swapping between its chat and
        embedding models). ``on_event`` receives start / progress (per batch) / embedding / done."""
        emit = on_event or (lambda e: None)
        gov = governor or Governor(1)
        jrn = journal if journal is not None else NullJournal()
        entities = kg.entities_to_tag()
        total = len(entities)
        emit({"stage": "tag", "phase": "start", "total": total})
        batches = plan_batches(entities, gov.max_in_flight, self.batch_max, self.batch_min)

        # Phase 1a — batches in flight, applied in batch order.
        tagged = 0
        retry: list[str] = []
        payloads = [self._payload(kg, b) for b in batches]
        for (ids, names, user), answer in zip(payloads, gov.map_ordered(lambda p: self._fetch(*p, jrn), payloads)):
            answer = answer if isinstance(answer, dict) else {}
            for eid, name in zip(ids, names):
                tags = clean_tags(name, answer.get(name))
                if tags:
                    self._apply(kg, eid, tags)
                    tagged += 1
                else:
                    retry.append(eid)
            emit({"stage": "tag", "phase": "progress", "processed": tagged, "total": total,
                  "batch_size": len(ids), "concepts": len(kg.nodes_by_type(NodeType.CONCEPT))})

        # Phase 1b — one retry round for missing / empty / echo-only entities; then accept empty.
        if retry:
            emit({"stage": "tag", "phase": "retry", "entities": len(retry)})
            retry_batches = plan_batches(retry, gov.max_in_flight, self.batch_max, self.batch_min)
            payloads = [self._payload(kg, b) for b in retry_batches]
            for (ids, names, user), answer in zip(payloads, gov.map_ordered(lambda p: self._fetch(*p, jrn), payloads)):
                answer = answer if isinstance(answer, dict) else {}
                for eid, name in zip(ids, names):
                    self._apply(kg, eid, clean_tags(name, answer.get(name)))
                    tagged += 1
            emit({"stage": "tag", "phase": "progress", "processed": tagged, "total": total,
                  "concepts": len(kg.nodes_by_type(NodeType.CONCEPT))})

        # Phase 2 — one batched embedding call for all concepts still missing a vector.
        pending = kg.concepts_missing_embedding()
        if pending:
            emit({"stage": "tag", "phase": "embedding", "concepts": len(pending)})
            names = [kg.get_node(c)["name"] for c in pending]  # type: ignore[index]
            vectors = self.embedder.embed(names, EmbedPurpose.CLUSTERING)
            for cid, vector in zip(pending, vectors):
                kg.set_concept_embedding(cid, vector)
        emit({"stage": "tag", "phase": "done", "processed": tagged, "total": total})
        return tagged

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _payload(kg: GraphStore, batch: list[str]) -> tuple[list[str], list[str], dict]:
        """Read a batch's input from the graph (main thread) before it is sent."""
        names, user = [], {}
        for eid in batch:
            node = kg.get_node(eid)
            assert node is not None
            names.append(node["name"])
            user[node["name"]] = {"inferred_type": node["attributes"].get(Attr.INFERRED_TYPE),
                                  "context": [c["text"] for c in node["attributes"].get(Attr.CONTEXT, [])]}
        return batch, names, user

    def _fetch(self, ids: list[str], names: list[str], user: dict, journal: Journal) -> dict:
        """One batch call (worker thread; never touches the graph). Returns {name: raw tags}."""
        key = journal_key("tag", self._model_id, self.prompt, user)
        cached = journal.get(key)
        if cached is not None:
            return cached
        content = json.dumps(user, ensure_ascii=False, indent=1)
        for _attempt in range(_ATTEMPTS):
            try:
                result = self.llm.complete_json(self.prompt, content, tag_schema(names))
            except OutputTruncated:
                if len(names) == 1:
                    return {}
                half = len(names) // 2
                out = self._fetch(ids[:half], names[:half], {n: user[n] for n in names[:half]}, journal)
                out.update(self._fetch(ids[half:], names[half:], {n: user[n] for n in names[half:]}, journal))
                return out
            except Exception:  # noqa: BLE001, S112 — retry loop, any failure just tries again
                continue
            answer = self._by_name(names, result)
            journal.put(key, "tag", answer)
            return answer
        return {}

    @staticmethod
    def _by_name(names: list[str], result: object) -> dict:
        """Map the answer back to the batch's names (exact key, else normalized key)."""
        if not isinstance(result, dict):
            return {}
        by_norm = {_norm(k): v for k, v in result.items()}
        return {n: result[n] if n in result else by_norm.get(_norm(n)) for n in names}

    @staticmethod
    def _apply(kg: GraphStore, eid: str, tags: list[str]) -> None:
        kg.set_entity_tags(eid, tags)
        for tag in tags:
            cid = kg.add_concept(tag)  # dedups by name; new concepts have no embedding yet
            kg.link_is_concept(eid, cid, original_trigger=tag)
