"""Query — intention extraction, entry-point search, lineage tracing, union/intersection and
path assembly over any GraphStore (+ optional VectorStore).

Works against the GraphStore/VectorStore protocols. Entry points come from
vector search when a vector store is given, else from ``match_entities_by_keyword`` (a name/tag
substring match with no distance signal).
"""

from __future__ import annotations

from collections import defaultdict

from lumintelligence.config import Config
from lumintelligence.constants import NodeType
from lumintelligence.graph.protocol import GraphStore
from lumintelligence.prompts import fill_placeholders, resolve_prompt
from lumintelligence.storage.protocol import VectorStore

IE_SCHEMA = {
    "type": "object",
    "properties": {
        "exact_keywords": {"type": "array", "items": {"type": "string"}},
        "expansion_keywords": {"type": "array", "items": {"type": "string"}},
    },
}

_ENTRY_POINT_TOP_K = 5  # matches SqliteVecStore.query()'s default top_k


def _node_title(node: dict | None) -> str | None:
    if node is None:
        return None
    return node.get("name") or (node.get("attributes", {}).get("metadata") or {}).get("title")


def _distance_rank(node_distance: dict[str, float | None], node_id: str) -> float:
    """Sort key for picking the best-matching entry point: lower distance first, with an
    unranked (None — the keyword-match fallback path) entry point sorting last."""
    d = node_distance.get(node_id)
    return d if d is not None else float("inf")


def _find_entry_points(store: GraphStore | None, vector_store: VectorStore | None,
                       embedder, keyword: str, *, top_k: int, threshold: float | None) -> list[dict]:
    """One entry-point-discovery hit per candidate node: {"node_id", "type", "name",
    "distance"}. Real vector search when vector_store is configured; the original keyword/
    name-match stand-in (with distance always None — no ranking signal exists there)
    otherwise. `threshold` filters vector-search hits by distance (lower = stricter match,
    the same lower-is-better convention used everywhere else in this codebase) — it has no
    effect on the keyword-match fallback, which carries no distance signal to filter by."""
    if vector_store is not None:
        hits = [
            {"node_id": hit["node_id"], "type": hit["metadata"].get("node_type"),
             "name": hit["metadata"].get("name"), "distance": hit["score"]}
            for hit in vector_store.query(keyword, embedder, top_k=top_k)
        ]
        if threshold is not None:
            hits = [h for h in hits if h["distance"] is not None and h["distance"] <= threshold]
        return hits
    assert store is not None
    return [
        {"node_id": hit["id"], "type": hit["type"], "name": hit["name"], "distance": None}
        for hit in store.match_entities_by_keyword(keyword)
    ]


def run_query(store: GraphStore, vector_store: VectorStore | None, provider, cfg: Config,
              query_text: str, *, dataset_description: str | None = None, detail: str = "standard",
              max_results: int | None = None, threshold: float | None = None,
              union_or_intersection: str = "union", intention_prompt_override: str | None = None,
              embedder=None) -> dict:
    """`intention_prompt_override`, when supplied, is used verbatim (full prompt text) for this
    call only, bypassing `resolve_prompt(cfg.intention_prompt, ...)`. `provider` does the intention
    LLM call; `embedder` (default: `provider`) embeds keywords for vector search."""
    embedder = embedder if embedder is not None else provider
    desc = dataset_description or cfg.dataset_description
    ie_prompt = fill_placeholders(
        intention_prompt_override if intention_prompt_override is not None
        else resolve_prompt(cfg.intention_prompt, cfg.prompts_dir),
        dataset_description=desc)
    top_k = max_results if max_results is not None else (cfg.n_per_keyword or _ENTRY_POINT_TOP_K)
    cutoff = threshold if threshold is not None else (
        None if cfg.entry_cutoff in (None, "auto") else float(cfg.entry_cutoff)
    )

    parsed = provider.complete_json(ie_prompt, query_text, IE_SCHEMA)
    exact = [str(k) for k in parsed.get("exact_keywords", [])] if isinstance(parsed, dict) else []
    expansion = [str(k) for k in parsed.get("expansion_keywords", [])] if isinstance(parsed, dict) else []

    entry_points: list[dict] = []
    entry_by_kw: dict[str, list[str]] = {}
    node_distance: dict[str, float | None] = {}
    for kw in exact:
        ids = []
        for hit in _find_entry_points(store, vector_store, embedder, kw, top_k=top_k, threshold=cutoff):
            ids.append(hit["node_id"])
            entry_points.append({"keyword": kw, "node_id": hit["node_id"], "type": hit["type"],
                                 "name": hit["name"], "distance": hit["distance"]})
            prior = node_distance.get(hit["node_id"])
            if prior is None or (hit["distance"] is not None and hit["distance"] < prior):
                node_distance[hit["node_id"]] = hit["distance"]
        entry_by_kw[kw] = ids

    src_kw: dict[str, set] = defaultdict(set)
    for kw, ids in entry_by_kw.items():
        for nid in ids:
            for s in store.trace_sources(nid):
                src_kw[s].add(kw)

    def _src_title(node_id: str):
        return _node_title(store.get_node(node_id))

    def _src_created_at(node_id: str):
        n = store.get_node(node_id)
        return n.get("created_at") if n is not None else None

    sources = [
        {"source_id": s, "title": _src_title(s), "created_at": _src_created_at(s),
         "matched_keywords": sorted(kws), "n_matched": len(kws)}
        for s, kws in src_kw.items()
    ]
    sources.sort(key=lambda r: r["n_matched"], reverse=True)
    intersection = [r for r in sources if exact and r["n_matched"] == len(exact)]
    # `sources`/`intersection` are always both computed (union_count/intersection_count are
    # informational regardless of the caller's choice) — `result_rows` is which one actually
    # feeds detail_results and the top-level `sources` field in the response.
    result_rows = intersection if union_or_intersection == "intersection" else sources

    sub_nodes: list[dict] = []
    sub_edges: list[dict] = []
    if detail != "summary":
        entry_ids = {ep["node_id"] for ep in entry_points}
        path_nodes: set = set(entry_ids)
        for nid in entry_ids:
            path_nodes |= set(store.ancestors(nid))
        for n in path_nodes:
            node = store.get_node(n)
            if node is None:
                continue
            sub_nodes.append({
                "id": n, "type": node.get("type"), "name": _node_title(node),
                "created_at": node.get("created_at"), "updated_at": node.get("updated_at"),
                "entry": n in entry_ids, "is_source": node.get("type") == NodeType.SOURCE.value,
            })
        sub_edges = store.subgraph_edges(list(path_nodes))

    detail_results: list[dict] = []
    if detail == "full":
        src_to_eps: dict[str, list[str]] = defaultdict(list)
        for kw, ids in entry_by_kw.items():
            for nid in ids:
                for s in store.trace_sources(nid):
                    src_to_eps[s].append(nid)

        n_exact = len(exact)
        for row in result_rows:
            sid = row["source_id"]
            node = store.get_node(sid) or {}
            attrs = node.get("attributes", {})

            linked_eps = src_to_eps.get(sid, [])
            best_ep = (
                min(linked_eps, key=lambda nid: _distance_rank(node_distance, nid)) if linked_eps else None
            )

            chain: list[dict] = []
            if best_ep:
                path_ids = store.shortest_path(sid, best_ep)
                for pid in (path_ids or []):
                    pn = store.get_node(pid) or {}
                    chain.append({"id": pid, "type": pn.get("type", ""), "name": _node_title(pn)})

            origin_id = node.get("origin_id")
            detail_results.append({
                "source_id": sid, "content": attrs.get("content"), "metadata": attrs.get("metadata") or {},
                "position": node.get("position"),
                "origin": store.get_origin(origin_id) if origin_id else None,
                "created_at": node.get("created_at"), "updated_at": node.get("updated_at"),
                "matched_keywords": row["matched_keywords"], "n_matched": row["n_matched"],
                "match_mode": "intersection" if n_exact and row["n_matched"] == n_exact else "union",
                "path": chain, "relevance_score": node_distance.get(best_ep) if best_ep else None,
            })

    return {"query": query_text, "exact_keywords": exact, "expansion_keywords": expansion,
            "entry_points": entry_points, "sources": result_rows,
            "union_count": len(sources), "intersection_count": len(intersection),
            "subgraph": {"nodes": sub_nodes, "edges": sub_edges},
            "detail_results": detail_results}
