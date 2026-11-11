"""``lum`` — the LumIntelligence command line.

For people in a terminal and for agents (the Hermes skill). Output is JSON when stdout is not a
terminal (or with ``--json``), short readable text otherwise. Errors: ``{"error", "hint"}`` and
exit code 1 (2 for usage errors).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable

from lumintelligence import home as H
from lumintelligence import jobs
from lumintelligence.constants import now_iso
from lumintelligence.llm.base import EmbedPurpose
from lumintelligence.storage.sqlite_conn import SQLiteExtensionError

_JSON = False


# ---------------------------------------------------------------------- output
def _out(data: dict, text: Callable[[dict], str] | None = None) -> None:
    if _JSON or text is None:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    else:
        print(text(data))


def _tty_in() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ask(label: str, default: str) -> str:
    shown = f" [{default}]" if default else ""
    value = input(f"{label}{shown}: ").strip()
    return value or default


# --------------------------------------------------------------------- commands
def _env_text(env: dict) -> str:
    lines = [f"Environment ({env['python']}):"]
    for i in env["items"]:
        lines.append(f"  {'ok  ' if i['ok'] else 'MISSING'} {i['label']}: {i['detail']}")
        if not i["ok"] and i.get("fix"):
            lines.append(f"          → {i['fix']}")
    return "\n".join(lines)


def _env_with_fix(interactive: bool, assume_yes: bool, offer: bool = True) -> dict:
    """Check the environment; if packages are missing, install them after a yes (terminal) or with
    ``assume_yes``. Agents (no terminal, no --yes) only get the report."""
    from lumintelligence import envcheck

    env = envcheck.check_environment()
    if env["ok"] or not env["installable"] or not offer:
        return env
    pkgs = env["installable"]
    if interactive and not assume_yes:
        print(_env_text(env))
        if _ask(f"Install {', '.join(pkgs)} into this Python environment? [Y/n]", "y").lower() not in ("y", "yes"):
            return env
    elif not assume_yes:
        return env
    result = envcheck.install(pkgs)
    env = envcheck.check_environment()
    env["install"] = result
    return env


def _offline_ready(values: dict, prefix: str, interactive: bool, report: dict) -> bool:
    """Ollama reachable and both models installed; in a terminal, re-check until ready or skip."""
    if os.environ.get(H.PROVIDER_KEY) == "mock":
        return True
    wanted = [values[prefix + "CHAT_MODEL"], values[prefix + "EMBED_MODEL"]]
    while True:
        st = H.ollama_status(values[prefix + "BASE_URL"])
        if not st["reachable"]:
            missing, msg = ["Ollama"], (f"Ollama is not running at {values[prefix + 'BASE_URL']}. Install Ollama "
                                        "from https://ollama.com and open it.")
        else:
            missing = H.missing_models(st["models"], wanted)
            if not missing:
                return True
            msg = f"Install these models in Ollama: {', '.join(missing)}."
        report.update(missing=missing, message=msg)
        if not interactive:
            return False
        print(msg)
        if input("Press Enter to check again, or type skip: ").strip().lower() == "skip":
            report["skipped"] = True
            return False


def cmd_init(a: argparse.Namespace) -> None:
    import getpass

    values = {**{k: d for k, (d, _) in H.SETTINGS.items()}, **H.read_env()}
    interactive = _tty_in() and not a.yes
    if interactive:
        print(f"Setting up LumIntelligence in {H.home()} (Enter keeps the value in brackets).\n")
    env = _env_with_fix(interactive, assume_yes=False)
    if interactive and not env["ok"]:
        print(_env_text(env) + "\nSome requirements are missing; setup continues, but fix these before use.\n")
    m = (a.mode or values.get("LUM_MODE") or "offline").lower()
    if interactive and not a.mode:
        print("Models: offline uses Ollama on this computer; online uses a hosted API (e.g. OpenAI).")
        print(H.MODE_NOTE + "\n")
        m = ""
        while m not in H.MODES:
            m = _ask("Mode (offline/online)", values.get("LUM_MODE") or "offline").lower()
    if m not in H.MODES:
        raise H.SetupError(f"Unknown mode {m!r}.", "Use --mode offline or --mode online.")
    values["LUM_MODE"] = m
    pre = f"LUM_{m.upper()}_"
    for field, flag in (("BASE_URL", a.base_url), ("API_KEY", a.api_key), ("CHAT_MODEL", a.chat_model),
                        ("EMBED_MODEL", a.embed_model)):
        if flag is not None:
            values[pre + field] = flag
    report: dict = {"mode": m, "env": str(H.path(".env")), "environment": env}

    if m == "online":
        if interactive:
            if a.api_key is None:
                key = getpass.getpass("API key (hidden; Enter keeps the current one): ").strip()
                values[pre + "API_KEY"] = key or values[pre + "API_KEY"]
            for field, label in (("BASE_URL", "Base URL (empty = OpenAI)"), ("CHAT_MODEL", "Chat model"),
                                 ("EMBED_MODEL", "Embedding model")):
                values[pre + field] = _ask(label, values[pre + field])
        ready = bool(values[pre + "API_KEY"]) or os.environ.get(H.PROVIDER_KEY) == "mock"
        if not ready:
            report.update(missing=["API key"], message=f"Add your API key as {pre}API_KEY in {H.path('.env')}.")
    else:
        if interactive:
            for field, label in (("CHAT_MODEL", "Chat model"), ("EMBED_MODEL", "Embedding model")):
                values[pre + field] = _ask(label, values[pre + field])
        ready = _offline_ready(values, pre, interactive, report)

    H.write_env(values)
    if not env["ok"]:
        ready = False
        report["message"] = "This Python environment is missing requirements; run `lum check --fix`."
    if ready and not a.skip_check:
        report["check"] = _check()
        ready = bool(report["check"]["chat"].get("ok") and report["check"]["embedding"].get("ok"))
        if not ready:
            report["message"] = "The model check failed; fix the settings, then run `lum check`."
    if ready:
        H.open_kb().graph.conn.close()  # type: ignore[attr-defined]
        report["kb"] = str(H.path("kb.db"))
    else:
        report["kb"] = None
    report["ready"] = ready

    if not a.no_skill and (not interactive or _ask("Install the Hermes skill? [Y/n]", "y").lower() in ("y", "yes")):
        report["skill"] = _install_skill(a.hermes_home)
    _out(report, _init_text)


def _init_text(r: dict) -> str:
    lines = [f"Settings written to {r['env']} (mode: {r['mode']})."]
    if not r.get("environment", {}).get("ok", True):
        lines.append(_env_text(r["environment"]))
    if "check" in r:
        lines.append(_check_text(r["check"]))
    if r["ready"]:
        lines.append(f"Knowledge base ready: {r['kb']}")
    else:
        lines.append(f"Not ready yet: {r.get('message', '')}")
        lines.append("When that's done, run `lum check`; the knowledge base is created on first use.")
    sk = r.get("skill")
    if sk:
        lines.append(f"Hermes skill installed to {sk['installed']}." + ("" if sk["lum_found"] else f" Warning: {sk['hint']}"))
    if r["ready"]:
        lines.append("Next: add files with `lum ingest <folder or files>`.")
    return "\n".join(lines)


def _check() -> dict:
    s = H.require_init()
    r = H.resolve_models(s)
    out: dict = {"mode": r["mode"], "chat_model": r["chat_model"], "embed_model": r["embed_model"]}
    t = time.time()
    try:
        chat, embed = H.make_providers(s)
    except H.SetupError as exc:
        out["chat"] = {"ok": False, "error": str(exc)}
        out["embedding"] = {"ok": False, "error": str(exc)}
        return out
    try:
        res = chat.complete_json("Reply with JSON {\"ok\": true}.", "ping",
                                 {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                  "required": ["ok"], "additionalProperties": False})
        out["chat"] = {"ok": isinstance(res, dict), "seconds": round(time.time() - t, 2)}
    except Exception as exc:  # noqa: BLE001 — reported, not raised
        out["chat"] = {"ok": False, "error": str(exc)[:300]}
    t = time.time()
    try:
        dim = len(embed.embed_one("ping", EmbedPurpose.QUERY))
        out["embedding"] = {"ok": True, "dim": dim, "seconds": round(time.time() - t, 2)}
    except Exception as exc:  # noqa: BLE001
        out["embedding"] = {"ok": False, "error": str(exc)[:300]}
    return out


def _check_text(c: dict) -> str:
    def line(name: str, r: dict) -> str:
        if r.get("ok"):
            extra = f", dim {r['dim']}" if "dim" in r else ""
            return f"  {name}: ok ({r['seconds']}s{extra})"
        return f"  {name}: FAILED — {r.get('error')}"
    return f"Model check ({c['mode']}):\n" + line(f"chat {c['chat_model']}", c["chat"]) + "\n" + line(
        f"embedding {c['embed_model']}", c["embedding"])


def cmd_check(a: argparse.Namespace) -> None:
    env = _env_with_fix(_tty_in(), assume_yes=a.yes, offer=a.fix)
    if not H.initialized():
        _out({"environment": env, "hint": "Run `lum init` to set up models."},
             lambda r: _env_text(env) + "\nNot set up yet: run `lum init`.")
        sys.exit(0 if env["ok"] else 1)
    c = _check()
    c["environment"] = env
    _out(c, lambda r: _env_text(env) + ("\nInstall: " + ("ok" if env["install"]["ok"] else "FAILED\n"
                                       + env["install"]["output"]) if "install" in env else "")
         + "\n" + _check_text(r)
         + ("" if env["ok"] else "\nFix the missing requirements (`lum check --fix` offers to install them)."))
    if not (env["ok"] and c["chat"].get("ok") and c["embedding"].get("ok")):
        sys.exit(1)


def _plan_text(p: dict) -> str:
    counts = ", ".join(f"{v} {k}" for k, v in p["counts"].items()) or "no files"
    lines = [f"Plan: {counts}"]
    for f in p["files"]:
        lines.append(f"  {f['status']:<11} {f['path']}" + (f"  ({f['reason']})" if f.get("reason") else ""))
    for m in p.get("missing", []):
        lines.append(f"  {'missing':<11} {m['path']}  (known, no longer on disk; kept in the KB)")
    return "\n".join(lines)


def cmd_plan(a: argparse.Namespace) -> None:
    lum = H.open_kb()
    _out(lum.plan_ingest(a.paths), _plan_text)


def cmd_ingest(a: argparse.Namespace) -> None:
    paths = [os.path.abspath(os.path.expanduser(p)) for p in a.paths]
    s = H.require_init()
    lum = H.open_kb(s)
    plan = lum.plan_ingest(paths)
    to_add = [f["path"] for f in plan["files"] if f["status"] in ("new", "changed")]
    limit = H.int_setting(s, "LUM_CONFIRM_ABOVE")
    if len(to_add) > limit and not a.yes:
        if _tty_in():
            print(_plan_text(plan))
            if input(f"\nIngest {len(to_add)} files? [y/N]: ").strip().lower() not in ("y", "yes"):
                print("Cancelled; nothing was added.")
                return
        else:
            _out({"needs_confirmation": True, "plan": plan["counts"], "files": to_add,
                  "hint": f"{len(to_add)} files would be added (more than {limit}). Show this plan to the "
                          "user and, once they agree, run the same ingest again with --yes."})
            return
    if a.wait:
        with jobs.worker_lock() as held:
            if held is None:
                raise H.SetupError("Jobs are already running.", "Drop --wait to add this to the queue.")
            lum.graph.conn.close()  # type: ignore[attr-defined]
            summary = jobs.run_job({"kind": "ingest", "paths": paths})
        _out({"plan": plan["counts"], "result": _summary(summary)}, lambda r: _job_text(r["result"]))
        return
    lum.graph.conn.close()  # type: ignore[attr-defined]
    job, pos = jobs.enqueue("ingest", paths)
    started = jobs.start_worker()
    _out({"job_id": job["id"], "queue_position": pos, "worker_started": started, "plan": plan["counts"],
          "files": [f["path"] for f in plan["files"] if f["status"] in ("new", "changed")],
          "next": "lum status"},
         lambda r: _plan_text(plan) + f"\nQueued as {r['job_id']} (position {r['queue_position']}). "
                                      "Follow it with `lum status`.")


def _summary(s: dict | None) -> dict | None:
    if not s:
        return None
    keep = ("job_id", "state", "stage", "started_at", "updated_at", "error", "counts", "source_totals")
    out = {k: s.get(k) for k in keep if k in s}
    if "stages" in s:
        out["progress"] = {k: {kk: vv for kk, vv in v.items() if kk in ("phase", "processed", "total")}
                           for k, v in s["stages"].items()}
    if s.get("files"):
        out["files"] = [{k: f.get(k) for k in ("path", "status", "result", "chunks", "error", "reason", "note")
                         if f.get(k) is not None} for f in s["files"]]
    return out


def _job_text(j: dict | None) -> str:
    if not j:
        return "No jobs yet."
    lines = [f"Job: {j.get('state')}" + (f" (stage {j['stage']})" if j.get("state") == "running" else "")]
    for stage, p in (j.get("progress") or {}).items():
        if "total" in p:
            lines.append(f"  {stage}: {p.get('processed', 0)}/{p['total']} {p.get('phase', '')}")
    for f in j.get("files") or []:
        lines.append(f"  {f.get('result', f.get('status')):<14} {f['path']}"
                     + (f"  ({f.get('error') or f.get('reason') or f.get('note')})"
                        if f.get("error") or f.get("reason") or f.get("note") else ""))
    if j.get("error"):
        lines.append(f"  error: {j['error']}")
    return "\n".join(lines)


def cmd_status(a: argparse.Namespace) -> None:
    # Reads status.json directly, never kb.db: a running build holds the database's write lock.
    from lumintelligence.facade import STATUS_FILE
    from lumintelligence.status import JobStatus, JsonFileStatusStore

    H.require_init()
    job = JobStatus.read(JsonFileStatusStore(str(H.path(STATUS_FILE))))
    last = _summary(job.summary() if job else None)
    running = jobs.worker_running()
    if last and last.get("state") == "running" and not running:
        last["state"] = "failed"
        last["error"] = "the background worker stopped before finishing"
        last["hint"] = "Run the same `lum ingest` again; model answers already received are reused."
    q = jobs.queued()
    _out({"worker_running": running, "job": last, "queue": q},
         lambda r: _job_text(r["job"]) + (f"\nQueued: {len(q)} job(s)" + "".join(
             f"\n  {j['id']} {j['kind']} {' '.join(j['paths'])}" for j in q) if q else ""))


def cmd_stop(a: argparse.Namespace) -> None:
    from lumintelligence.facade import STATUS_FILE
    from lumintelligence.status import JsonFileStatusStore

    H.require_init()
    try:
        r = jobs.stop(clear_queue=not a.keep_queue)
    except RuntimeError as exc:
        raise H.SetupError(str(exc), "Find it with `pgrep -fl 'lumintelligence.cli _worker'` and `kill` it.") from exc
    store = JsonFileStatusStore(str(H.path(STATUS_FILE)))
    data = store.load()
    if r["stopped"] and data and data.get("state") == "running":
        data["state"] = "stopped"
        data["error"] = "stopped with `lum stop`"
        data["updated_at"] = now_iso()
        store.save(data)
    r["job_id"] = data.get("job_id") if data and r["stopped"] else None
    r["hint"] = ("The knowledge base is as it was before the job. To continue later, run the same "
                 "`lum ingest`; model answers already received are reused.") if r["stopped"] else None

    def text(r: dict) -> str:
        n = len(r["removed_from_queue"])
        head = (f"Stopped the running job (worker {r['pid']})." if r["stopped"]
                else "No job was running.")
        return head + (f" Removed {n} queued job(s)." if n else "") + (f"\n{r['hint']}" if r["hint"] else "")
    _out(r, text)


def cmd_resume(a: argparse.Namespace) -> None:
    job, pos = jobs.enqueue("resume")
    started = jobs.start_worker()
    _out({"job_id": job["id"], "queue_position": pos, "worker_started": started, "next": "lum status"},
         lambda r: f"Resume queued as {r['job_id']}. Follow it with `lum status`.")


def _answer_text(r: dict) -> str:
    from lumintelligence.facade import describe_position

    lines = []
    if r.get("note"):
        lines.append(f"Note: {r['note']}\n")
    if r.get("answer"):
        lines += [r["answer"], ""]
    lines.append("Sources:" if r["sources"] else "No sources found.")
    for s in r["sources"]:
        o = s.get("origin") or {}
        where = describe_position(s["source"].get("position"))
        lines.append(f"[{s['n']}] {o.get('label') or 'unknown origin'}" + (f", {where}" if where else "")
                     + (f" — ingested {o['ingested_at'][:10]}" if o.get("ingested_at") else ""))
        if s.get("path"):
            lines.append(f"     found via: {' → '.join(s['path'])}")
        snippet = " ".join((s["source"].get("content") or "").split())
        lines.append(f"     \"{snippet[:200]}{'…' if len(snippet) > 200 else ''}\"")
    return "\n".join(lines)


def cmd_query(a: argparse.Namespace) -> None:
    lum = H.open_kb()
    r = lum.ask(a.question, max_results=a.max_results, answer=not a.no_answer)
    r["note"] = ("A build is running; results reflect the knowledge base before it."
                 if jobs.worker_running() else None)
    _out(r, _answer_text)


def _origin_stats(lum) -> list[dict]:
    out = []
    for o in lum.graph.list_origins():
        pages = o["extra"].get("pages") or {}
        out.append({"origin_id": o["id"], "label": o["label"], "kind": o["kind"],
                    "paths": o["extra"].get("paths") or [], "ingested_at": o["created_at"],
                    "chunks": len(lum.graph.sources_by_origin([o["id"]])),
                    "skipped_pages": pages.get("skipped") or [],
                    "prompts": o["extra"].get("prompts"),
                    "previous_version": o["extra"].get("previous_version")})
    return out


def cmd_files(a: argparse.Namespace) -> None:
    lum = H.open_kb()
    files = _origin_stats(lum)
    _out({"files": files}, lambda r: "\n".join(
        [f"{len(files)} file(s) in the knowledge base"]
        + [f"  {f['ingested_at'][:10]}  {f['label']}  ({f['chunks']} chunks"
           + (f", {len(f['skipped_pages'])} pages skipped" if f["skipped_pages"] else "")
           + (f", {f['prompts']} prompts" if f.get("prompts") else "") + ")"
           + "".join(f"\n      {p}" for p in f["paths"]) for f in files]))


def cmd_export(a: argparse.Namespace) -> None:
    lum = H.open_kb()
    data = lum.export(a.out)
    if a.out:
        _out({"written": a.out, "nodes": len(data["nodes"]), "links": len(data["links"])},
             lambda r: f"Wrote {r['nodes']} nodes and {r['links']} links to {r['written']}.")
    else:
        print(json.dumps(data, ensure_ascii=False, default=str))


def cmd_kb_rebuild(a: argparse.Namespace) -> None:
    lum = H.open_kb()
    files = _origin_stats(lum)
    g = lum.graph
    dates = sorted(f["ingested_at"] for f in files)
    db = H.path("kb.db")
    impact = {"files": len(files), "oldest": dates[0][:10] if dates else None,
              "newest": dates[-1][:10] if dates else None, "sources": g.source_node_count(),
              "entities": len(g.nodes_by_type("Discovered_entity")), "concepts": len(g.nodes_by_type("Concept")),
              "size_mb": round(sum(p.stat().st_size for p in db.parent.glob("kb.db*")) / 1e6, 1)}
    g.conn.close()  # type: ignore[attr-defined]
    text = ("This deletes everything in your knowledge base and starts empty:\n"
            f"  - {impact['files']} files"
            + (f" (ingested {impact['oldest']} to {impact['newest']})" if impact["files"] else "") + "\n"
            f"  - {impact['sources']} sources, {impact['entities']} entities, {impact['concepts']} concepts\n"
            f"  - kb.db, {impact['size_mb']} MB\n"
            "Your files on disk are not touched; you can ingest them again afterwards.\n"
            "This cannot be undone.")
    if jobs.worker_running() or jobs.queued():
        raise H.SetupError("Jobs are running or queued.", "Wait until `lum status` shows no jobs, then rebuild.")
    if not _tty_in():
        _out({"rebuilt": False, "impact": impact,
              "message": "Not rebuilt. The user must run `lum kb rebuild` themselves in a terminal and confirm."})
        return
    print(text)
    if input('\nType "rebuild" to confirm: ').strip() != "rebuild":
        print("Cancelled; nothing was changed.")
        return
    for name in ("kb.db", "kb.db-wal", "kb.db-shm", "journal.jsonl", "status.json"):
        p = H.path(name)
        if p.exists():
            p.unlink()
    H.open_kb().graph.conn.close()  # type: ignore[attr-defined]
    print("Done. The knowledge base is empty; ingest your files again with `lum ingest <paths>`.")


def _current_lum() -> str | None:
    """The full path of the `lum` command running now, if this process was started as `lum`."""
    exe = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if os.path.basename(exe) == "lum" and os.path.isfile(exe) and os.access(exe, os.X_OK):
        return exe
    candidate = os.path.join(os.path.dirname(sys.executable), "lum")   # same environment's bin folder
    return candidate if os.path.isfile(candidate) and os.access(candidate, os.X_OK) else None


def cmd_prompts_export(a: argparse.Namespace) -> None:
    from lumintelligence.prompts import CUSTOMIZABLE, load_prompt

    folder = os.path.abspath(os.path.expanduser(a.folder))
    os.makedirs(folder, exist_ok=True)
    existing = [n for n in CUSTOMIZABLE if os.path.exists(os.path.join(folder, f"{n}.txt"))]
    if existing and not a.force:
        raise H.SetupError(f"{folder} already has {', '.join(f'{n}.txt' for n in existing)}.",
                           "Choose another folder, or add --force to overwrite them.")
    written = []
    for name in CUSTOMIZABLE:
        target = os.path.join(folder, f"{name}.txt")
        with open(target, "w", encoding="utf-8") as f:
            f.write(load_prompt(name) + "\n")
        written.append(target)
    _out({"folder": folder, "written": written, "env_line": f"LUM_PROMPTS_DIR={folder}"},
         lambda r: f"Copied {len(written)} default prompts to {folder}.\n"
                   f"Keep only the files you change (the others fall back to the defaults), then set\n"
                   f"  {r['env_line']}\nin {H.path('.env')} and run `lum prompts check`.")


def prompts_report(folder: str | None) -> dict:
    """Custom/default per prompt, plus errors and warnings for the custom folder."""
    from lumintelligence.prompts import CUSTOMIZABLE, ENGINE_OWNED, load_prompt, output_section, prompt_source

    items, errors, warnings = [], [], []
    for name in CUSTOMIZABLE:
        source = prompt_source(name, folder)
        item = {"name": name, "source": source}
        if source == "custom" and folder:
            with open(os.path.join(folder, f"{name}.txt"), encoding="utf-8") as fh:
                text = fh.read()
            if not text.strip():
                errors.append(f"{name}.txt is empty.")
            else:
                default_out = output_section(load_prompt(name))
                custom_out = output_section(text)
                if default_out and custom_out is None:
                    warnings.append(f"{name}.txt has no \"Output format\" section; answers may not parse.")
                elif default_out and custom_out != default_out:
                    warnings.append(f"{name}.txt changes the \"Output format\" section; keep it as in the default "
                                    "unless you know the code accepts the new format.")
        items.append(item)
    if folder:
        known = {f"{n}.txt" for n in CUSTOMIZABLE}
        for entry in sorted(os.listdir(folder)):
            if entry.startswith(".") or entry in known:
                continue
            if entry.removesuffix(".txt") in ENGINE_OWNED:
                errors.append(f"{entry} can't be overridden (the engine owns the tagging output frame); remove it.")
            elif entry.endswith(".txt"):
                errors.append(f"{entry} is not a prompt name LumIntelligence uses (typo?). "
                              f"Expected: {', '.join(sorted(known))}.")
    return {"folder": folder, "prompts": items, "errors": errors, "warnings": warnings, "ok": not errors}


def cmd_prompts_check(a: argparse.Namespace) -> None:
    s = H.require_init()
    r = prompts_report(H.prompts_dir(s))

    def text(r: dict) -> str:
        lines = [f"Prompts folder: {r['folder'] or '(none: using the default prompts)'}"]
        lines += [f"  {p['source']:<8} {p['name']}" for p in r["prompts"]]
        lines += [f"ERROR: {e}" for e in r["errors"]] + [f"Warning: {w}" for w in r["warnings"]]
        if r["ok"] and not r["warnings"]:
            lines.append("All good.")
        return "\n".join(lines)
    _out(r, text)
    if not r["ok"]:
        sys.exit(1)


def _lum_found_by_skill(recorded: str | None = None) -> str | None:
    """Where the skill's lum.sh would find `lum` (same order as the script)."""
    import shutil

    for c in (os.environ.get("LUM_BIN"), recorded,
              os.path.join(os.environ["UV_TOOL_BIN_DIR"], "lum") if os.environ.get("UV_TOOL_BIN_DIR") else None,
              os.path.join(os.environ["XDG_BIN_HOME"], "lum") if os.environ.get("XDG_BIN_HOME") else None,
              os.path.expanduser("~/.local/bin/lum"), shutil.which("lum")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _install_skill(hermes_home: str | None = None) -> dict:
    import shutil
    from importlib.resources import as_file, files

    hermes = os.path.expanduser(hermes_home or os.environ.get("HERMES_HOME") or "~/.hermes")
    dest = os.path.join(hermes, "skills", "research", "lumintelligence")
    with as_file(files("lumintelligence") / "skill") as src:
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
    os.chmod(os.path.join(dest, "scripts", "lum.sh"), 0o755)
    current = _current_lum()
    if current:   # the skill runs the same lum that installed it, whatever environment it lives in
        with open(os.path.join(dest, "scripts", ".lum_bin"), "w") as f:
            f.write(current)
    found = _lum_found_by_skill(current)
    return {"installed": dest, "lum_found": found,
            "hint": None if found else "The skill can't find `lum`. Install it with `uv tool install` (it goes to "
                                       "~/.local/bin), or set LUM_BIN to its full path."}


def cmd_skill_install(a: argparse.Namespace) -> None:
    _out(_install_skill(a.hermes_home),
         lambda r: f"Skill installed to {r['installed']}." + (
             f"\nIt will run {r['lum_found']}." if r["lum_found"] else f"\nWarning: {r['hint']}")
         + "\nRestart Hermes (or start a new chat) to load it.")


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lum", description="LumIntelligence: build a knowledge network from "
                                "your files and ask it questions with cited sources.")
    p.add_argument("--json", action="store_true", help="always print JSON")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="guided setup: models, knowledge base, Hermes skill")
    s.add_argument("--mode", choices=["offline", "online"])
    for flag in ("base-url", "api-key", "chat-model", "embed-model", "hermes-home"):
        s.add_argument(f"--{flag}")
    s.add_argument("--yes", action="store_true", help="don't ask; use flags, existing values and presets")
    s.add_argument("--skip-check", action="store_true", help="don't call the models")
    s.add_argument("--no-skill", action="store_true", help="don't install the Hermes skill")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("check", help="check this Python environment and test the models")
    s.add_argument("--fix", action="store_true", help="offer to install missing packages into this environment")
    s.add_argument("--yes", action="store_true", help="with --fix: install without asking")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("plan", help="show what ingest would do, without changing anything")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("ingest", help="add files or folders (runs in the background)")
    s.add_argument("paths", nargs="+")
    s.add_argument("--wait", action="store_true", help="run now in the foreground")
    s.add_argument("--yes", action="store_true", help="don't ask to confirm a large ingest")
    s.set_defaults(fn=cmd_ingest)

    sub.add_parser("status", help="the running job, the queue, the last job").set_defaults(fn=cmd_status)
    s = sub.add_parser("stop", help="stop the running job and empty the queue")
    s.add_argument("--keep-queue", action="store_true", help="leave queued jobs (they start with the next ingest)")
    s.set_defaults(fn=cmd_stop)
    sub.add_parser("resume", help="re-run sources the last job did not finish").set_defaults(fn=cmd_resume)

    s = sub.add_parser("query", help="ask a question; answer with cited sources")
    s.add_argument("question")
    s.add_argument("--no-answer", action="store_true", help="sources only, no synthesized answer")
    s.add_argument("--max-results", type=int, default=10)
    s.set_defaults(fn=cmd_query)

    sub.add_parser("files", help="list the files in the knowledge base").set_defaults(fn=cmd_files)

    s = sub.add_parser("export", help="the graph as JSON")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_export)

    kb = sub.add_parser("kb", help="knowledge base maintenance")
    kbsub = kb.add_subparsers(dest="kb_command", required=True)
    kbsub.add_parser("rebuild", help="delete everything and start empty (asks to confirm)").set_defaults(
        fn=cmd_kb_rebuild)

    pr = sub.add_parser("prompts", help="custom prompts (guide: references/custom_prompts.md in the installed skill)")
    prsub = pr.add_subparsers(dest="prompts_command", required=True)
    s = prsub.add_parser("export", help="copy the default prompts into a folder as a starting point")
    s.add_argument("folder")
    s.add_argument("--force", action="store_true", help="overwrite prompt files already in the folder")
    s.set_defaults(fn=cmd_prompts_export)
    prsub.add_parser("check", help="check the custom prompts folder (LUM_PROMPTS_DIR)").set_defaults(
        fn=cmd_prompts_check)

    sk = sub.add_parser("skill", help="the Hermes skill")
    sksub = sk.add_subparsers(dest="skill_command", required=True)
    s = sksub.add_parser("install", help="copy the skill into Hermes (~/.hermes or HERMES_HOME)")
    s.add_argument("--hermes-home", help="Hermes home folder (default: $HERMES_HOME or ~/.hermes)")
    s.set_defaults(fn=cmd_skill_install)
    return p


def main(argv: list[str] | None = None) -> int:
    global _JSON
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["_worker"]:              # internal: the detached background worker
        jobs.run_worker()
        return 0
    args = build_parser().parse_args(argv)
    _JSON = args.json or not sys.stdout.isatty()
    try:
        args.fn(args)
    except (H.SetupError, SQLiteExtensionError) as exc:
        _out({"error": str(exc), "hint": exc.hint}, lambda r: f"Error: {r['error']}\n{r['hint'] or ''}".strip())
        return 1
    except SystemExit as exc:            # commands that set an exit status (e.g. a failed check)
        return int(exc.code or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 — one JSON error shape for agents
        _out({"error": f"{type(exc).__name__}: {exc}", "hint": None}, lambda r: f"Error: {r['error']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
