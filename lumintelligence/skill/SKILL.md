---
name: lumintelligence
description: Answer questions from the user's own files (PDF, Word, Markdown, text) with cited sources, and add files or folders to their personal knowledge base.
version: 0.1.2
platforms: [macos, linux]
metadata:
  hermes:
    tags: [knowledge-base, documents, search, citations, personal-files]
    category: research
    requires_toolsets: [terminal]
---

# LumIntelligence: the user's personal knowledge base

LumIntelligence builds a knowledge network from files the user adds (PDF, DOCX, Markdown, TXT, JSON
records) and answers questions from them, citing each statement with the file, location and the date
the file was added. Everything runs locally through one command:

    ${HERMES_SKILL_DIR}/scripts/lum.sh <command> [arguments]

Every command prints JSON. On failure it prints `{"error": "...", "hint": "..."}`; relay both to the
user in plain words.

## When to Use

- The user asks about their own documents or knowledge: "what do my notes say about…", "according to
  my files…", "in the board documents, who…", "search my knowledge base for…".
- The user asks to add, ingest, import or remember files or folders: "add this folder to my knowledge
  base", "remember this PDF".
- The user asks about progress or contents: "is the import done?", "how's the ingest going?", "did
  anything fail?", "what's in my knowledge base?", "which files have I added?".

- The user asks how to customize, configure, tune or change it: "how do I customize this?", "can I
  use a different model?", "how do I teach it our terms?", "can answers be shorter / in bullets?",
  "how do I change the prompts?".

Do not use it for general knowledge, the web, or files the user has not asked about.

## Quick Reference

| Goal | Command |
|---|---|
| Answer a question with cited sources | `lum.sh query "<question>"` |
| Sources only, no answer | `lum.sh query "<question>" --no-answer` |
| Preview adding files or folders | `lum.sh plan <absolute paths…>` |
| Add files or folders (background) | `lum.sh ingest <absolute paths…>` |
| Add after the user confirmed a large plan | `lum.sh ingest <absolute paths…> --yes` |
| Progress, failures, queue | `lum.sh status` |
| Files in the knowledge base | `lum.sh files` |
| Re-run what the last job didn't finish | `lum.sh resume` |
| Stop the running job and empty the queue (only when the user asks) | `lum.sh stop` |

## Procedures

### 1. Answer a question
1. Run `lum.sh query "<the user's question>"`.
2. Show `answer` exactly as returned. Keep its citation numbers like [1] [2] unchanged; they point to
   the numbered sources.
3. Then list the sources that were cited (`cited`), one line each:
   `[n] <origin.label>, <location from source.position> (added <origin.ingested_at date>)`.
   Location: page and lines for PDFs, the heading path for Markdown and Word, line numbers for text.
4. If `note` is set (a build is running), tell the user that the newest files are not searchable yet.
5. If the answer says the knowledge base has nothing relevant, say so. Do not answer from general
   knowledge instead unless the user asks you to.

Use `--no-answer` when the user only wants to see where something is mentioned, or when you will
combine these sources with other information yourself; then cite the sources by their numbers.

### 2. Add files or folders
1. Use absolute paths (expand `~`). Commands may run from any folder, so relative paths are wrong.
2. For a folder, or several files, run `lum.sh plan <paths…>` first and tell the user the counts:
   new, changed (new versions of files already added), unchanged, copies, unsupported.
3. Run `lum.sh ingest <paths…>`.
   - If it returns `job_id`, tell the user it is queued and that they can ask for progress any time.
     Do not check progress on your own.
   - If it returns `"needs_confirmation": true`, show the user the plan counts and ask whether to go
     ahead. Run `lum.sh ingest <same paths…> --yes` only after the user agrees in the chat.
4. Unchanged files and copies are skipped automatically; nothing is ever deleted.

### 3. Progress and failures (only when the user asks)
Run `lum.sh status` and summarise: whether a job is running or finished, progress per stage
(parse, nod, tag, consolidate, vectorize), files that failed and why (`files[].error`), and jobs
waiting in `queue`. If `job.hint` is set, pass it on.

### 4. What's in the knowledge base
Run `lum.sh files` and list the files with the date each was added. Mention skipped PDF pages
(contents, cover, blank) if the user asks why something is missing.

### 5. First-time setup and models
- If a command says LumIntelligence is not set up, tell the user to run `lum init` in a terminal. It
  guides them through offline models (Ollama on their computer: hermes3 and nomic-embed-text by
  default) or online models (a hosted API such as OpenAI, with their API key), and installs this skill.
- If `lum.sh` says `lum` was not found, relay the install instructions from the hint.
- If the user asks about changing models, or an error says the knowledge base was built with another
  embedding model: the setting `LUM_MODE` (offline/online) in `~/.lumintelligence/.env` switches both
  the chat and the embedding model. A knowledge base only works with the embedding model it was built
  with, so after a switch the user either switches back, or runs `lum kb rebuild` in a terminal
  (deletes everything) and adds their files again. Explain this; never edit `.env` yourself.

### 6. Explaining customization
When the user asks how to customize LumIntelligence (in general or one part of it):
1. Read `${HERMES_SKILL_DIR}/references/customizing.md`. For prompt questions, also read
   `${HERMES_SKILL_DIR}/references/custom_prompts.md`.
2. Answer with **all** the ways to customize, one or two lines each: models (offline/online), how
   files are read, speed, prompts, where data lives and test setups, the skill itself. Then go into
   detail on the area the user asked about, with the exact steps: commands to run, the `.env` lines to
   change (in `~/.lumintelligence/.env`), and when the change takes effect.
3. For prompts, name the example set that fits their case (in
   `${HERMES_SKILL_DIR}/examples/custom_prompts/`: small-model tuning, custom glossary, formatting
   specs, meeting notes) and show the relevant part of it if helpful.
4. Mention the consequences that matter: changing the embedding model or `LUM_MODE` needs a rebuild;
   extraction prompts and file settings apply only to files added afterwards.
5. You may run `lum.sh prompts check` to show which prompts are currently custom. Do not edit `.env`
   or prompt files and do not run commands that change settings; the user does that.

## Pitfalls

- Never reword or renumber the answer's citations: each number must keep pointing at its source.
- Always pass absolute paths: the working folder of a command is not the user's folder.
- Only add files the user asked for; never add folders on your own initiative.
- Never handle API keys or settings yourself. If the error says setup is missing or a model fails,
  tell the user to run `lum init` (or `lum check`) themselves in a terminal.
- Never run `lum kb rebuild`. It deletes the whole knowledge base. If a rebuild is needed (for example
  after changing the embedding model), explain that and tell the user to run it in a terminal, where
  it asks for confirmation.
- If the user attaches a file in the chat and its path on disk is not visible to you, ask where the
  file is saved instead of guessing.

## References

- `${HERMES_SKILL_DIR}/references/output.md`: the JSON each command returns, and common errors.
- `${HERMES_SKILL_DIR}/references/customizing.md`: every way to customize, with steps.
- `${HERMES_SKILL_DIR}/references/custom_prompts.md`: the full custom prompts guide.
- `${HERMES_SKILL_DIR}/examples/custom_prompts/`: four example prompt sets.
