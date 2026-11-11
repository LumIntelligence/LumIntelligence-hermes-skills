"""NodeDiscovery (NoD) — extract entities from new sources, inline.

For each source still marked is_new, a single LLM call returns per-entity data: a
context snippet and a free-form ``inferred_type`` classification (e.g. "Person",
"Organization") — not a fixed taxonomy, and not compared by raw string equality
downstream. Each entity is created (or reused by normalized name) and a
MENTIONS edge is added at extraction time, then the source is marked processed. The
source itself is never modified beyond flipping is_new.

Raw ``inferred_type`` values are canonicalized against a registry of already-seen types
via embedding similarity before being stored — "corporation" and "company" should
collapse to one label, not fragment the type vocabulary just because the LLM phrased
them differently on different calls. The registry is seeded once per `run()` call from
the graph's existing distinct inferred_type values (cheap: bounded by vocabulary size,
not entity count) and extended in-memory as new sources in the same call introduce
genuinely new types — never rebuilt from scratch mid-batch, and each source's newly-seen
raw types are embedded in one batched call rather than one call per entity.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

from lumintelligence.constants import Attr, NodeType
from lumintelligence.graph import GraphStore
from lumintelligence.llm.base import EmbeddingProvider, EmbedPurpose, LLMProvider, json_object_schema
from lumintelligence.llm.governor import Governor
from lumintelligence.storage.journal import Journal, NullJournal, journal_key


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


class NodeDiscovery:
    def __init__(self, llm: LLMProvider, embedder: EmbeddingProvider, system_prompt: str,
                type_similarity_threshold: float = 0.85):
        self.llm = llm
        self.embedder = embedder
        self.prompt = system_prompt
        self.type_threshold = type_similarity_threshold
        self._model_id = str(getattr(getattr(llm, "config", None), "llm_model", type(llm).__name__))

    def run(self, kg: GraphStore, on_event: Callable[[dict], None] | None = None,
            governor: Governor | None = None, journal: Journal | None = None) -> int:
        """Process all is_new sources. Returns the number of sources processed.
        ``on_event`` (optional) receives progress dicts for live monitoring.

        LLM calls go through ``governor`` (several in flight; default: one at a time) and are
        recorded in ``journal``; results are applied to the graph on this thread **in source
        order**, so the graph matches a serial run for the same model answers."""
        emit = on_event or (lambda e: None)
        gov = governor or Governor(1)
        jrn = journal if journal is not None else NullJournal()
        sources = kg.sources_to_process()
        total = len(sources)
        emit({"stage": "nod", "phase": "start", "total": total})
        registry = self._seed_type_registry(kg)
        contents = [kg.get_node(sid)["attributes"][Attr.CONTENT] for sid in sources]  # type: ignore[index]
        processed = 0
        for sid, fetched in zip(sources, gov.map_ordered(lambda c: self._fetch(c, jrn), contents)):
            if isinstance(fetched, Exception):
                fetched = {"extracted": None, "success": False, "error": str(fetched)}
            self._apply(kg, sid, fetched, registry)
            processed += 1
            emit({"stage": "nod", "phase": "progress", "processed": processed, "total": total,
                  "entities": len(kg.nodes_by_type(NodeType.ENTITY)),
                  "source_id": sid, "success": fetched["success"], "error": fetched["error"]})
        emit({"stage": "nod", "phase": "done", "processed": processed, "total": total})
        return processed

    def _fetch(self, content: str, journal: Journal) -> dict:
        """The LLM call for one source (worker thread; never touches the graph)."""
        key = journal_key("nod", self._model_id, self.prompt, content)
        cached = journal.get(key)
        if cached is not None:
            return cached
        # Retries are for errors only (an exception, or an answer that is not a JSON object). A
        # valid empty object means the source has no entities and is accepted at once.
        extracted: dict | None = None
        last_error: str | None = None
        for _attempt in range(5):
            try:
                result = self.llm.complete_json(self.prompt, content, json_object_schema("string"))
            except Exception as exc:  # noqa: BLE001 — retry loop, any failure just tries again
                last_error = str(exc)
                continue
            if isinstance(result, dict):
                extracted = result
                break
            last_error = f"expected a JSON object, got {type(result).__name__}"
        success = extracted is not None
        out = {"extracted": extracted, "success": success,
               "error": None if success else (last_error or "extraction failed after 5 attempts")}
        if success:
            journal.put(key, "nod", out)
        return out

    def _apply(self, kg: GraphStore, sid: str, fetched: dict, registry: dict[str, np.ndarray]) -> None:
        """Write one source's result to the graph (main thread, source order)."""
        extracted = fetched["extracted"]
        if isinstance(extracted, dict):
            parsed = [
                (stripped_name, *self._split_value(value))
                for name, value in extracted.items()
                if (stripped_name := str(name).strip())  # skip blank names from flaky LLM output
            ]
            canonical_map = self._canonicalize_batch(parsed, registry)
            for name, context_text, raw_type in parsed:
                canonical_type = canonical_map.get(raw_type) if raw_type is not None else None
                eid = kg.add_entity(name, context={"source_id": sid, "text": context_text},
                                    inferred_type=canonical_type)
                kg.link_mentions(sid, eid)
        kg.mark_source_processed(sid)

    # ------------------------------------------------------- output-shape parsing
    @staticmethod
    def _split_value(value: object) -> tuple[str, str | None]:
        """Accepts both the current {"context": ..., "inferred_type": ...} shape and
        the legacy plain-string shape (context-only, no type) — keeps older prompts and
        existing tests working unchanged. Some models return the object shape encoded as a
        JSON string (``"{\\"context\\": ..., \\"inferred_type\\": ...}"``); that string is
        decoded first, so its text is not stored as the context."""
        if isinstance(value, str) and value.lstrip().startswith("{"):
            try:
                decoded = json.loads(value)
            except ValueError:
                decoded = None
            if isinstance(decoded, dict) and "context" in decoded:
                value = decoded
        if isinstance(value, dict):
            context_text = str(value.get("context", ""))
            raw_type = value.get("inferred_type")
            raw_type = (str(raw_type).strip() or None) if raw_type is not None else None
            return context_text, raw_type
        return str(value), None

    # ------------------------------------------------- inferred_type canonicalization
    def _seed_type_registry(self, kg: GraphStore) -> dict[str, np.ndarray]:
        existing = {
            t for eid in kg.nodes_by_type(NodeType.ENTITY)
            if (t := kg.get_node(eid)["attributes"].get(Attr.INFERRED_TYPE)) is not None  # type: ignore[index]
        }
        if not existing:
            return {}
        names = list(existing)
        vectors = self.embedder.embed(names, EmbedPurpose.CLUSTERING)
        return {name: np.asarray(vec, dtype=float) for name, vec in zip(names, vectors)}

    def _canonicalize_batch(self, parsed: list[tuple[str, str, str | None]],
                            registry: dict[str, np.ndarray]) -> dict[str, str]:
        """Batch-embed this source's newly-seen distinct raw types in one call, then
        assign/register each sequentially against the (growing) registry in-memory — no
        further embedding calls after this point. Returns raw_type -> canonical_type
        for every distinct raw type in `parsed` (already-registered ones included), so
        the caller never needs to re-look-up a type this method already resolved."""
        distinct = {rt for _, _, rt in parsed if rt is not None}
        if not distinct:
            return {}
        to_embed = [rt for rt in distinct if rt not in registry]
        vectors = self.embedder.embed(to_embed, EmbedPurpose.CLUSTERING) if to_embed else []
        precomputed = {rt: np.asarray(vec, dtype=float) for rt, vec in zip(to_embed, vectors)}
        mapping: dict[str, str] = {}
        for rt in distinct:
            if rt in registry:
                mapping[rt] = rt
            else:
                mapping[rt] = self._nearest_registered(rt, registry, vector=precomputed[rt])
        return mapping

    def _nearest_registered(self, raw_type: str, registry: dict[str, np.ndarray],
                            vector: np.ndarray | None = None) -> str:
        """Return an existing canonical type if `raw_type`'s embedding is within
        threshold of one already in the registry; otherwise register `raw_type` itself
        as a new canonical entry and return it."""
        if raw_type in registry:
            return raw_type
        v = vector if vector is not None else np.asarray(
            self.embedder.embed_one(raw_type, EmbedPurpose.CLUSTERING), dtype=float)
        best_name, best_sim = None, -1.0
        for name, emb in registry.items():
            sim = _cosine(v, emb)
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_name is not None and best_sim >= self.type_threshold:
            return best_name
        registry[raw_type] = v
        return raw_type
