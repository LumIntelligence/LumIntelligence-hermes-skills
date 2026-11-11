# LumIntelligence command output (JSON)

## query
```json
{"question": "...", "keywords": ["Harbor Fund"],
 "answer": "The Harbor Fund proposed a grant ... [1][3]", "cited": [1, 3],
 "sources": [
   {"n": 1,
    "path": ["Harbor Fund (entity)", "Funding Source (concept)"],
    "source": {"content": "<the passage>", "position": {"page": 3, "line_start": 12, "line_end": 26},
               "matched": ["Harbor Fund"]},
    "origin": {"label": "grant_notes.pdf", "kind": "file", "path": "/Users/.../grant_notes.pdf",
               "ingested_at": "2026-11-05T14:31:00"}}],
 "note": null}
```
- `path`: how the passage was found in the knowledge network.
- `position` keys by file type (plus `part`/`parts` when a long paragraph was split): PDF `page`, `line_start`, `line_end`; Markdown `heading_path`,
  `line_start`, `line_end`; Word `heading_path`, `paragraph_start`, `paragraph_end`; text
  `line_start`, `line_end`; JSON records `record`.
- `answer` is null with `--no-answer`.
- `note` is set while a build is running.

## plan
```json
{"counts": {"new": 12, "changed": 1, "unchanged": 3, "copy": 1, "unsupported": 2},
 "files": [{"path": "...", "status": "new", "file_type": "pdf", "reason": null}],
 "missing": [{"path": "...", "label": "..."}]}
```
- `changed`: a new version of a file already added (both versions are kept).
- `copy`: the same content already exists at another path (recorded, not re-added).
- `missing`: files added before that are no longer on disk (kept in the knowledge base).

## ingest
- Queued: `{"job_id": "JOB_…", "queue_position": 1, "worker_started": true, "plan": {...}, "files": [...]}`
- Needs confirmation: `{"needs_confirmation": true, "plan": {...}, "files": [...], "hint": "..."}`.
  Ask the user, then run again with `--yes`.

## status
```json
{"worker_running": true,
 "job": {"state": "running", "stage": "nod",
         "progress": {"parse": {"processed": 3, "total": 3}, "nod": {"processed": 20, "total": 57}},
         "files": [{"path": "...", "result": "ingested", "chunks": 19},
                   {"path": "...", "result": "failed", "error": "..."}],
         "error": null, "hint": null},
 "queue": [{"id": "JOB_…", "kind": "ingest", "paths": ["..."]}]}
```
- `job.state`: running, completed or failed.
- `files[].result`: ingested, recorded_copy, skipped or failed.

## files
`{"files": [{"label": "...", "kind": "file", "paths": ["..."], "ingested_at": "...", "chunks": 14,
"skipped_pages": [{"page": 1, "reason": "table of contents"}], "prompts": "default"}]}`
- `prompts`: whether the file was added with the default prompts or the user's custom prompts
  (`LUM_PROMPTS_DIR`). For customization questions, see `customizing.md` and `custom_prompts.md`;
  never edit prompts or `.env` yourself.

## init (when run with --yes, e.g. by a script)
`{"mode": "offline", "ready": false, "missing": ["nomic-embed-text"], "message": "Install these models in
Ollama: nomic-embed-text.", "kb": null, "skill": {...}}`. With `"ready": true`, `kb` is the knowledge
base path.

## Models and modes
- `~/.lumintelligence/.env` holds two sets of models: `LUM_OFFLINE_*` (Ollama; presets hermes3 and
  nomic-embed-text) and `LUM_ONLINE_*` (empty base URL = OpenAI; any OpenAI-compatible provider).
  `LUM_MODE=offline|online` picks the set.
- Switching mode switches both the chat and the embedding model. A knowledge base only works with
  the embedding model it was built with: switch back to use it again, or run `lum kb rebuild`
  (deletes everything) and add the files again.
- Changing only the chat model (same embedding model) needs no rebuild.

## Common errors
| error / hint mentions | Tell the user |
|---|---|
| `lum init` | LumIntelligence isn't set up: run `lum init` in a terminal. |
| `lum` not found | Install it (see the hint), then run `lum init` and `lum skill install`. |
| built with embedding model … | The active mode uses a different embedding model than the knowledge base was built with. Switch `LUM_MODE` back, or run `lum kb rebuild` in a terminal (deletes everything) and add the files again. |
| Ollama is not running / install these models | Open the Ollama app, or install the named models in Ollama, then `lum check`. |
| model check failed / connection | The chat or embedding model isn't reachable: check the local server or API key, then `lum check`. |
| jobs are running or queued | Wait, or ask for progress. |
