"""LumIntelligence — the library's single entry point.

Ties providers, the pipeline, a graph store, a vector store and job status together. Every
adapter (CLI for the Hermes skill, MCP server, memory plugin) is meant to call only this:

    lum = LumIntelligence.open("my_project", llm=provider, embedder=provider)
    lum.build(units)                       # ingest → NoD → tag → consolidate → vectorize
    lum.status()                           # stage, per-source status, counts (also on disk)
    lum.query("who funded the west wing?")

``open(kb_dir)`` persists to ``kb.db`` (graph, origins and sqlite-vec vectors in one SQLite file), ``status.json`` (last job) and ``journal.jsonl`` (raw LLM answers of an unfinished build) in
that directory; without a directory everything stays in an in-memory db. A build is one
transaction: other readers see the last complete version until it commits, and a failed
build rolls back. Re-calling ``build`` processes only what's new (incremental); ``resume`` re-runs the
sources the last job did not complete.

Custom prompts: ``Config.nod_prompt``/``adhd_prompt``/``intention_prompt`` take a bundled name,
a file path, or a name in ``Config.prompts_dir``; ``build``/``query`` also accept full prompt
text per call. ``{{ATTENTION_HINTS}}`` and ``{{DATASET_DESCRIPTION}}`` are filled either way.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable

from lumintelligence.config import Config, default_config
from lumintelligence.graph import GraphStore, SqliteGraphStore
from lumintelligence.ingestion import Unit
from lumintelligence.inputs import PageFilter, parse_file, plan_paths
from lumintelligence.inputs.types import COPY, INGEST_STATUSES, IngestPlan
from lumintelligence.llm.base import EmbeddingProvider, LLMProvider
from lumintelligence.orchestrator import Pipeline
from lumintelligence.prompts import fill_placeholders, resolve_prompt
from lumintelligence.query import run_query
from lumintelligence.status import JobStatus, JsonFileStatusStore
from lumintelligence.storage import SqliteVecStore, VectorStore
from lumintelligence.storage.journal import Journal, JsonlJournal, NullJournal
from lumintelligence.storage.sqlite_conn import connect

DB_FILE = "kb.db"
_TIER = {"Discovered_entity": "entity", "Concept": "concept", "Centroid": "theme", "Source": "source"}


def describe_position(pos: dict | None) -> str | None:
    """Human-readable location: "p. 3, lines 12-26", "Policies > Travel, lines 4-9", "record 7"."""
    if not pos:
        return None
    parts = []
    if pos.get("heading_path"):
        parts.append(" > ".join(pos["heading_path"]))
    if "page" in pos:
        parts.append(f"p. {pos['page']}")
    for unit in ("line", "paragraph"):
        if f"{unit}_start" in pos:
            a, b = pos[f"{unit}_start"], pos.get(f"{unit}_end")
            parts.append(f"{unit}s {a}-{b}" if b and b != a else f"{unit} {a}")
    if pos.get("parts"):
        p, q = pos.get("part"), pos.get("part_end")
        parts.append(f"parts {p}-{q} of {pos['parts']}" if q and q != p else f"part {p} of {pos['parts']}")
    if "record" in pos:
        parts.append(f"record {pos['record']}")
    return ", ".join(parts) or None
STATUS_FILE = "status.json"
JOURNAL_FILE = "journal.jsonl"


class LumIntelligence:
    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        config: Config | None = None,
        graph: GraphStore | None = None,
        vector_store: VectorStore | None = None,
        project_dir: str | None = None,
    ):
        self.config = (config or default_config).resolved()
        self.llm = llm
        self.embedder = embedder
        if graph is None:
            graph = SqliteGraphStore()
            if vector_store is None:  # share the in-memory db, so a build's rollback covers vectors too
                vector_store = SqliteVecStore(dimension=embedder.dimension, conn=graph.conn)
        self.graph: GraphStore = graph
        self.vector_store: VectorStore = (
            vector_store if vector_store is not None else SqliteVecStore(dimension=embedder.dimension)
        )
        self.project_dir = project_dir
        self._last_status: JobStatus | None = None
        # Raw LLM results of the current build; replayed after a crash, cleared on success.
        self.journal: Journal = (JsonlJournal(os.path.join(project_dir, JOURNAL_FILE)) if project_dir
                                 else NullJournal())

    # ----------------------------------------------------------- constructors
    @classmethod
    def open(cls, project_dir: str, llm: LLMProvider, embedder: EmbeddingProvider,
             config: Config | None = None) -> LumIntelligence:
        """Open (or create) a knowledge-base directory: one ``kb.db`` holds graph and vectors."""
        os.makedirs(project_dir, exist_ok=True)
        conn = connect(os.path.join(project_dir, DB_FILE))
        graph = SqliteGraphStore(conn=conn)
        vectors = SqliteVecStore(dimension=embedder.dimension, conn=conn)
        return cls(llm, embedder, config=config, graph=graph, vector_store=vectors, project_dir=project_dir)

    @classmethod
    def with_openai(cls, api_key: str | None = None, config: Config | None = None,
                    project_dir: str | None = None) -> LumIntelligence:
        """OpenAI, or any OpenAI-compatible endpoint via ``config.openai.base_url``."""
        from lumintelligence.llm.openai import OpenAIProvider

        cfg = config or Config()
        provider = OpenAIProvider(api_key=api_key, config=cfg.openai)
        if project_dir:
            return cls.open(project_dir, provider, provider, config=cfg)
        return cls(provider, provider, config=cfg)

    # ------------------------------------------------------------------ build
    def build(
        self,
        units: Iterable[Unit],
        *,
        nod_prompt: str | None = None,
        adhd_prompt: str | None = None,
        on_event: Callable[[dict], None] | None = None,
        checkpoint_dir: str | None = None,
        idempotency: str = "content_hash",
    ) -> dict:
        """Ingest a batch and run every stage on what's pending. ``nod_prompt``/``adhd_prompt``
        are full prompt text for this call only. Returns the job status summary."""
        job = self._new_job()
        pipeline = self._pipeline(nod_prompt, adhd_prompt)
        events = self._job_events(job, on_event)
        return self._run_job(job, lambda: pipeline.build_graph(
            units, kg=self.graph, checkpoint_dir=checkpoint_dir, on_event=events,
            vector_store=self.vector_store, idempotency=idempotency, journal=self.journal))

    # ---------------------------------------------------------------- files
    def plan_ingest(self, paths: Iterable[str]) -> dict:
        """What ``ingest(paths)`` would do, without changing anything: per-file status (new,
        changed, copy, unchanged, unsupported, ...), counts, and known files now missing."""
        return plan_paths(paths, self.graph, self.config.max_file_mb).to_dict()

    def ingest(
        self,
        paths: Iterable[str],
        *,
        nod_prompt: str | None = None,
        adhd_prompt: str | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> dict:
        """Ingest files and folders: plan, parse new and changed files into sources (one origin
        per file), then build. One job and one transaction. Returns the job summary with
        ``plan`` (counts) and ``files`` (per-file results)."""
        plan = plan_paths(paths, self.graph, self.config.max_file_mb)
        job = self._new_job()
        pipeline = self._pipeline(nod_prompt, adhd_prompt)
        events = self._job_events(job, on_event)
        page_filter = None
        if self.config.page_filter:
            prompt = resolve_prompt(self.config.page_filter_prompt, self.config.prompts_dir)
            page_filter = PageFilter(self.llm, prompt, pipeline.governor, self.journal)
        files: list[dict] = []
        from lumintelligence.prompts import prompt_source

        used = ["sp_NoD", "sp_ADHD"] + (["sp_PageFilter"] if self.config.page_filter else [])
        custom = (nod_prompt is not None or adhd_prompt is not None
                  or any(prompt_source(n, self.config.prompts_dir) == "custom" for n in used))
        prompts_signal = "custom" if custom else "default"

        def work() -> None:
            units = self._parse_plan(plan, page_filter, files, events, prompts_signal)
            job.set_files(files)
            pipeline.build_graph(units, kg=self.graph, on_event=events,
                                 vector_store=self.vector_store, journal=self.journal)

        summary = self._run_job(job, work)
        summary["plan"] = plan.counts()
        summary["files"] = files
        return summary

    def _parse_plan(self, plan: IngestPlan, page_filter: PageFilter | None, files: list[dict],
                    emit: Callable[[dict], None], prompts_signal: str = "default") -> list[dict]:
        """Create origins for new/changed files and return their units; record copies' paths.
        A file that fails to parse is reported and gets no origin (it is retried next time)."""
        c = self.config
        created: dict[str, str] = {}      # path -> origin id, for copies within this plan
        units: list[dict] = []
        todo = plan.to_ingest()
        emit({"stage": "parse", "phase": "start", "total": len(todo)})
        done = 0
        for item in plan.items:
            entry = item.to_dict()
            files.append(entry)
            if item.status == COPY:
                target = item.origin_id or created.get(item.copy_of or "")
                if target:
                    self.graph.add_origin_path(target, item.path)
                    entry.update(result="recorded_copy", origin_id=target)
                else:
                    entry["result"] = "skipped"
                continue
            if item.status not in INGEST_STATUSES:
                entry["result"] = "skipped"
                continue
            try:
                parsed = parse_file(item, c.chunk_window_lines, c.chunk_overlap_lines, page_filter, c.chunk_max_chars)
            except Exception as exc:  # noqa: BLE001 — one bad file never stops the others
                entry.update(result="failed", error=f"{type(exc).__name__}: {exc}")
                parsed = None
            done += 1
            emit({"stage": "parse", "phase": "progress", "processed": done, "total": len(todo),
                  "path": item.path})
            if parsed is None:
                continue
            if parsed.note and parsed.note.startswith("no extractable text"):
                entry.update(result="skipped", reason=parsed.note)
                continue
            extra = {"path": item.path, "paths": [item.path], "size": item.size, "mtime": item.mtime,
                     "file_type": item.file_type, "prompts": prompts_signal, **parsed.extra}
            if item.status == "changed":
                extra["previous_version"] = item.origin_id
            oid = self.graph.add_origin("file", item.sha256 or "", label=os.path.basename(item.path),
                                        extra=extra)
            created[item.path] = oid
            for u in parsed.units:
                units.append({**u, "origin_id": oid})
            entry.update(result="ingested", origin_id=oid, chunks=len(parsed.units))
            if "pages" in parsed.extra:
                entry["pages"] = parsed.extra["pages"]
            if parsed.note:
                entry["note"] = parsed.note
        emit({"stage": "parse", "phase": "done", "processed": done, "total": len(todo)})
        return units

    def resume(self, *, nod_prompt: str | None = None, adhd_prompt: str | None = None,
               on_event: Callable[[dict], None] | None = None) -> dict:
        """Re-run the sources the last job did not complete, then the downstream stages."""
        last = self._load_status()
        retry = last.sources_needing_resume() if last else []
        job = self._new_job()
        job.seed_sources(retry)
        self.graph.reset_sources_for_reprocessing(retry)
        pipeline = self._pipeline(nod_prompt, adhd_prompt)

        def events(event: dict) -> None:
            job.on_event(event)
            if on_event:
                on_event(event)

        return self._run_job(job, lambda: pipeline.process_pending(
            self.graph, on_event=events, vector_store=self.vector_store, journal=self.journal))

    def rebuild_concepts(self) -> LumIntelligence:
        """Globally re-derive centroids (non-destructive to concepts/edges), then re-vectorize."""
        pipeline = self._pipeline(None, None)
        self.graph.begin()
        try:
            pipeline.consolidator.rebuild(self.graph)
            pipeline.process_pending(self.graph, vector_store=self.vector_store)
        except BaseException:
            self.graph.rollback()
            raise
        self.graph.commit()
        return self

    # ------------------------------------------------------------------ query
    def query(self, text: str, *, intention_prompt: str | None = None, detail: str = "standard",
              max_results: int | None = None, threshold: float | None = None,
              union_or_intersection: str = "union", dataset_description: str | None = None,
              use_vectors: bool = True) -> dict:
        """Intention extraction → entry points → lineage to sources (see ``lumintelligence.query``).
        ``detail``: "summary" | "standard" (adds subgraph) | "full" (adds per-source paths)."""
        vectors = self.vector_store if use_vectors and self.vector_store.count() > 0 else None
        return run_query(self.graph, vectors, self.llm, self.config, text, embedder=self.embedder,
                         dataset_description=dataset_description, detail=detail, max_results=max_results,
                         threshold=threshold, union_or_intersection=union_or_intersection,
                         intention_prompt_override=intention_prompt)

    def ask(self, question: str, *, max_results: int = 10, answer: bool = True,
            answer_prompt: str | None = None) -> dict:
        """Query, then (unless ``answer=False``) a synthesized answer citing the numbered sources. Each source: ``path`` (how it was found in the network), ``source`` (chunk and
        position), ``origin`` (file name, path, ingested date)."""
        from lumintelligence.answer import synthesize

        res = self.query(question, detail="full", max_results=max_results)
        sources = []
        for n, row in enumerate(res["detail_results"][:max_results], start=1):
            origin = row.get("origin") or {}
            extra = origin.get("extra") or {}
            sources.append({
                "n": n,
                "path": [f"{p['name']} ({_TIER.get(p['type'], p['type'])})" for p in row.get("path", [])[1:]],
                "source": {"content": row.get("content"), "position": row.get("position"),
                           "matched": row.get("matched_keywords", [])},
                "origin": {"label": origin.get("label"), "kind": origin.get("kind"), "path": extra.get("path"),
                           "ingested_at": origin.get("created_at")} if origin else None,
            })
        out: dict = {"question": question, "keywords": res["exact_keywords"], "answer": None, "cited": [],
                     "sources": sources}
        if answer:
            prompt = answer_prompt or resolve_prompt(self.config.answer_prompt, self.config.prompts_dir)
            excerpts = [{"n": s["n"], "content": s["source"]["content"] or "",
                         "label": (s["origin"] or {}).get("label"), "where": describe_position(s["source"]["position"])}
                        for s in sources]
            out.update(synthesize(self.llm, prompt, question, excerpts))
        return out

    # ------------------------------------------------------- status / persistence
    def status(self) -> dict | None:
        """Last job's status summary (from disk when a project dir is used)."""
        job = self._load_status()
        return job.summary() if job else None

    def export(self, path: str | None = None) -> dict:
        """The whole graph as node-link JSON (nodes, edges with lineage, origins); no embeddings."""
        return self.graph.export_json(path)

    # ---------------------------------------------------------------- private
    def _pipeline(self, nod_override: str | None, adhd_override: str | None) -> Pipeline:
        c = self.config
        nod = fill_placeholders(nod_override if nod_override is not None
                                else resolve_prompt(c.nod_prompt, c.prompts_dir),
                                attention_hints=c.nod_attention_hints)
        adhd = fill_placeholders(adhd_override if adhd_override is not None
                                 else resolve_prompt(c.adhd_prompt, c.prompts_dir),
                                 attention_hints=c.adhd_attention_hints)
        return Pipeline(self.llm, self.embedder, nod, adhd, c)

    @staticmethod
    def _job_events(job: JobStatus, on_event: Callable[[dict], None] | None) -> Callable[[dict], None]:
        def events(event: dict) -> None:
            if event.get("stage") == "ingest":
                pending = set(event["pending"])
                job.seed_sources(event["source_ids"], {s for s in event["source_ids"] if s not in pending})
            else:
                job.on_event(event)
            if on_event:
                on_event(event)
        return events

    def _status_store(self) -> JsonFileStatusStore | None:
        return JsonFileStatusStore(os.path.join(self.project_dir, STATUS_FILE)) if self.project_dir else None

    def _new_job(self) -> JobStatus:
        self._last_status = JobStatus(store=self._status_store())
        return self._last_status

    def _load_status(self) -> JobStatus | None:
        store = self._status_store()
        if store is not None:
            return JobStatus.read(store)
        return self._last_status

    def _run_job(self, job: JobStatus, work: Callable[[], object]) -> dict:
        """Run one job as one transaction. On success it commits and the journal is cleared. On an
        exception it rolls back: the KB keeps its previous state, the journal keeps every LLM answer
        received so far, and re-running ``build`` replays them."""
        job.start()
        self.graph.begin()
        try:
            work()
            job.set_counts(node_count=self.graph.node_count(), edge_count=self.graph.edge_count(),
                           source_node_count=self.graph.source_node_count())
        except BaseException as exc:
            self.graph.rollback()
            job.fail(str(exc))
            raise
        self.graph.commit()
        self.journal.clear()
        job.complete()
        return job.summary()
