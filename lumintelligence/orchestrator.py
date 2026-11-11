"""Pipeline — orchestrates graph construction end to end.

Ingest → NoD → tag → consolidate → (optional) vectorize, writing through one GraphStore, with
optional per-stage JSON checkpoints and an ``on_event`` progress stream. Single-call and
incremental: calling ``build_graph`` again with the same graph and a new batch processes only
what's new (new sources, untagged entities, unconsolidated concepts). Prompts arrive fully resolved (see the facade).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable

from lumintelligence.adhd import Tagger
from lumintelligence.config import Config, default_config
from lumintelligence.consolidation import Consolidator
from lumintelligence.graph import GraphStore, SqliteGraphStore
from lumintelligence.ingestion import Ingestor, Unit
from lumintelligence.llm.base import EmbeddingProvider, LLMProvider
from lumintelligence.llm.governor import Governor
from lumintelligence.nod import NodeDiscovery
from lumintelligence.storage.journal import Journal
from lumintelligence.storage.protocol import VectorStore
from lumintelligence.storage.records import build_records

OnEvent = Callable[[dict], None]


class Pipeline:
    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        nod_prompt: str,
        adhd_prompt: str,
        config: Config | None = None,
    ):
        self.config = (config or default_config).resolved()
        self.embedder = embedder
        self.ingestor = Ingestor()
        self.nod = NodeDiscovery(llm, embedder, nod_prompt)
        self.tagger = Tagger(llm, embedder, adhd_prompt, batch_max=self.config.tag_batch_max,
                             batch_min=self.config.tag_batch_min)
        self.consolidator = Consolidator(float(self.config.consolidation_threshold))
        # One governor shared by NoD and tagging: at most max_in_flight provider calls.
        self.governor = Governor(self.config.max_in_flight)

    def build_graph(
        self,
        units: Iterable[Unit],
        kg: GraphStore | None = None,
        checkpoint_dir: str | None = None,
        on_event: OnEvent | None = None,
        vector_store: VectorStore | None = None,
        idempotency: str = "content_hash",
        journal: Journal | None = None,
    ) -> GraphStore:
        kg = kg if kg is not None else SqliteGraphStore()
        emit = on_event or (lambda e: None)
        source_ids = self.ingestor.run(kg, units, idempotency=idempotency)  # type: ignore[arg-type]
        emit({"stage": "ingest", "phase": "done", "source_ids": source_ids,
              "pending": kg.sources_to_process()})
        self._checkpoint(kg, checkpoint_dir, "1_ingest")
        self.process_pending(kg, checkpoint_dir=checkpoint_dir, on_event=on_event, vector_store=vector_store,
                             journal=journal)
        return kg

    def process_pending(
        self,
        kg: GraphStore,
        checkpoint_dir: str | None = None,
        on_event: OnEvent | None = None,
        vector_store: VectorStore | None = None,
        journal: Journal | None = None,
        full_vectorize: bool = False,
    ) -> GraphStore:
        """Run NoD → tag → consolidate → vectorize on whatever is pending. Shared by a fresh
        build (after ingest) and a resume (after ``reset_sources_for_reprocessing``). NoD and
        tagging calls run through the shared governor; results are applied in order. Vectorize syncs
        the index (only new or changed records are embedded); ``full_vectorize`` re-embeds all."""
        self.nod.run(kg, on_event=on_event, governor=self.governor, journal=journal)  # type: ignore[arg-type]
        self._checkpoint(kg, checkpoint_dir, "2_nod")
        self.tagger.run(kg, on_event=on_event, governor=self.governor, journal=journal)  # type: ignore[arg-type]
        self._checkpoint(kg, checkpoint_dir, "3_tagged")
        self.consolidator.run(kg, on_event=on_event)  # type: ignore[arg-type]
        self._checkpoint(kg, checkpoint_dir, "4_concepts")
        if vector_store is not None:
            emit = on_event or (lambda e: None)
            emit({"stage": "vectorize", "phase": "start"})
            t0 = time.perf_counter()
            counts = vector_store.sync(build_records(kg), self.embedder, full=full_vectorize)
            emit({"stage": "vectorize", "phase": "done", "records": vector_store.count(),
                  "seconds": round(time.perf_counter() - t0, 3), **counts})
        return kg

    @staticmethod
    def _checkpoint(kg: GraphStore, checkpoint_dir: str | None, name: str) -> None:
        if checkpoint_dir:
            kg.export_json(os.path.join(checkpoint_dir, f"{name}.json"))
