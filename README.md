# LumIntelligence for Hermes

Turn your own files into a personal knowledge base that answers questions with cited sources, from
[Hermes](https://hermes-agent.nousresearch.com) or from the command line.

LumIntelligence reads your PDFs, Word documents, Markdown and text files and builds a knowledge network
of the people, organizations, places and ideas in them. When you ask a question, it finds the relevant
passages through that network and writes an answer that cites each one: which file, where in the
file, and when you added it.

- **Local first:** your knowledge base is a single file on your computer. Use offline models through
  [Ollama](https://ollama.com), or an online API such as OpenAI.
- **Works inside Hermes:** a bundled skill lets Hermes add files and answer from your knowledge base
  when you ask in plain language.
- **Traceable answers:** every statement points back to a passage in one of your files.

> Status: early release (0.1.2). It works end to end, but expect rough edges; see
> [Current limits](#current-limits).

## Requirements

- macOS or Linux
- **Python 3.11 or newer whose SQLite can load extensions** (needed for search). Python from
  [uv](https://docs.astral.sh/uv/), Homebrew or conda works. The python.org macOS installer and pyenv
  without `--enable-loadable-sqlite-extensions` don't, but `lum check --fix` can add a fallback
  package for them (see [Check your environment](#check-your-environment)).
- One of:
  - **Offline:** [Ollama](https://ollama.com) with the models `hermes3` and `nomic-embed-text`
    (or others you choose)
  - **Online:** an API key for OpenAI or another OpenAI-compatible provider
- [Hermes](https://hermes-agent.nousresearch.com) (Desktop or CLI) to use it as a skill. The
  command line works without Hermes.

## Install

Pick whichever way you manage Python tools. Install from GitHub, or from a downloaded copy of this
folder:

| Tool | From GitHub | From a downloaded folder |
|---|---|---|
| **uv** (recommended) | `uv tool install "lumintelligence-engine[openai,files] @ git+https://github.com/LumIntelligence/LumIntelligence-hermes-skills"` | `uv tool install "/path/to/LumIntelligence-hermes-skills[openai,files]"` |
| **pipx** | `pipx install "lumintelligence-engine[openai,files] @ git+https://github.com/LumIntelligence/LumIntelligence-hermes-skills"` | `pipx install "/path/to/LumIntelligence-hermes-skills[openai,files]"` |
| **pip**, in a virtual environment you manage | `pip install "lumintelligence-engine[openai,files] @ git+https://github.com/LumIntelligence/LumIntelligence-hermes-skills"` | `pip install "/path/to/LumIntelligence-hermes-skills[openai,files]"` |

The Hermes skill runs the same `lum` that installed it (`lum init` or `lum skill install` records
its full path), so any of these works, including your own virtual environment. If you later move or
reinstall `lum`, run `lum skill install` again.

Then run:

```sh
lum init
```

`lum init` sets everything up and asks before each step:

1. **Environment:** checks your Python (see below) and offers to install anything missing.
2. **Mode:** `offline` (Ollama on this computer, the default) or `online` (a hosted API).
3. **Offline:** checks that Ollama is running and that the models are installed. If something is
   missing, it tells you what to install. Install it, then press Enter to check again, or type
   `skip` to finish later.
4. **Online:** asks for your API key (typed hidden). The base URL and model names are pre-filled
   for OpenAI and can be changed.
5. Tests the chat and embedding models, then creates your knowledge base.
6. Offers to install the Hermes skill.

Then restart Hermes (or start a new chat) so it loads the skill.

If you skipped a step, finish it (for example, install the missing model), then run `lum check`.
Your knowledge base is created the first time the models work.

### Check your environment

`lum check` reports whether this Python has everything LumIntelligence needs, then tests your models:

```
Environment (/Users/you/.local/share/uv/tools/lumintelligence-engine/bin/python):
  ok      Python 3.11 or newer: 3.11.16
  ok      SQLite extension loading: built into this Python
  ok      sqlite-vec (vector search): loads
  ok      Model client (openai): installed
  ok      PDF reader (pdfplumber): installed
  ok      Word reader (python-docx): installed
```

`lum check --fix` offers to install missing packages into the same environment (with pip, or with
uv if pip isn't there) and asks first. If your Python can't load SQLite extensions, it installs the
`pysqlite3` fallback. If that isn't possible on your system, install LumIntelligence with a Python
that supports extensions (uv, Homebrew or conda). It never changes anything without asking, and
Hermes never runs it.

## Using it in Hermes

Just ask. Hermes uses the skill when you talk about your own files or knowledge base:

| You say | What happens |
|---|---|
| "Add my ~/Documents/Board folder to my knowledge base" | Hermes previews the folder (new, changed, already added, unsupported), then starts adding the files in the background |
| "What do my documents say about the Harbor Fund grant?" | An answer with citations like [1] [2], followed by the sources |
| "Is the import done?" / "Did anything fail?" | Progress per stage, failed files and why, queued jobs |
| "What's in my knowledge base?" | Your files, with the date each was added |
| "Just show me where Maria Lopez is mentioned" | The sources only, without a written answer |

You can also call the skill directly: `/lumintelligence what do my notes say about the budget?`

**Large folders:** when more than 20 files would be added, Hermes first shows you the plan and asks.
Adding starts only after you agree.

**An answer looks like this:**

```
The Harbor Fund proposed a grant for the west wing; the amount has not been confirmed [1][2].

Sources:
[1] grant_notes.pdf, p. 3, lines 12-26 — ingested 2026-11-05
     found via: Harbor Fund (entity) → Funding Source (concept)
     "The board is reviewing a proposed grant from the Harbor Fund; the amount..."
[2] board_minutes.md, Meetings > October, lines 4-9 — ingested 2026-11-06
```

If your knowledge base has nothing relevant, it says so instead of guessing.

## Using the command line

Everything Hermes does is available as `lum` commands. In a terminal they print readable text;
add `--json` for JSON (agents always get JSON).

| Command | What it does |
|---|---|
| `lum init` | Guided setup (models, knowledge base, Hermes skill) |
| `lum check` | Check this Python environment and test the models (`--fix` offers to install what's missing) |
| `lum plan <paths…>` | Show what adding these files or folders would do, without changing anything |
| `lum ingest <paths…>` | Add files or folders in the background (`--wait` to run in the foreground, `--yes` to skip the large-folder question) |
| `lum status` | The running job, queued jobs, and the last result |
| `lum resume` | Re-run the parts of the last job that didn't finish |
| `lum query "<question>"` | Answer with cited sources (`--no-answer` for sources only, `--max-results N`) |
| `lum files` | List the files in your knowledge base |
| `lum export --out graph.json` | Export the knowledge network as JSON |
| `lum prompts export <folder>` | Copy the default prompts into a folder as a starting point for your own |
| `lum prompts check` | Check your custom prompts (see [Customizing](#customizing)) |
| `lum skill install` | Install or update the Hermes skill |
| `lum kb rebuild` | Delete everything and start empty (asks you to confirm) |

Adding files runs in the background, one job at a time. Further `lum ingest` calls wait in a queue,
and `lum status` shows them. You can keep asking questions while a job runs. The answers use the
knowledge base as it was before the job, until the job finishes.

## Settings

All settings live in `~/.lumintelligence/.env`, written by `lum init` with a comment on each line.
Edit it any time. Environment variables with the same names override it.

| Setting | Default | Meaning |
|---|---|---|
| `LUM_MODE` | `offline` | Which set of models to use: `offline` or `online` |
| `LUM_OFFLINE_BASE_URL` | `http://localhost:11434/v1` | Ollama's address |
| `LUM_OFFLINE_CHAT_MODEL` | `hermes3` | Model that reads your files and writes answers |
| `LUM_OFFLINE_EMBED_MODEL` | `nomic-embed-text` | Model that powers search |
| `LUM_ONLINE_BASE_URL` | empty (OpenAI) | Any OpenAI-compatible endpoint |
| `LUM_ONLINE_API_KEY` | empty | Your API key |
| `LUM_ONLINE_CHAT_MODEL` | `gpt-6-luna` | |
| `LUM_ONLINE_EMBED_MODEL` | `text-embedding-3-small` | |
| `LUM_MAX_IN_FLIGHT` | `2` | Model calls in parallel; match your server's capacity |
| `LUM_PAGE_FILTER` | `true` | Skip contents, cover and blank pages in PDFs |
| `LUM_CHUNK_WINDOW_LINES` / `LUM_CHUNK_MAX_CHARS` | `15` / `1500` | Passage size: up to 15 lines and 1,500 characters (long paragraphs are split at sentences) |
| `LUM_CHUNK_OVERLAP_LINES` | `3` | Lines shared by consecutive passages (at most 300 characters) |
| `LUM_CONFIRM_ABOVE` | `20` | Ask before adding more files than this |
| `LUM_PROMPTS_DIR` | empty | Folder with your own prompts; empty uses the prompts that come with the skill (see [Customizing](#customizing)) |

Each mode also has `_API_KEY` and `_REASONING_EFFORT` settings.

## Customizing

You can use your own prompts so LumIntelligence ingests your information better: for example, to
teach it your product codes and acronyms, to focus on decisions and action items in meeting notes, or
to get answers in a fixed format.

- Put your prompt files in a folder and set `LUM_PROMPTS_DIR` in `~/.lumintelligence/.env` to it. This
  is the only place prompts are configured. Leave it empty to use the default prompts that come with
  the skill.
- Each file in the folder replaces the default of the same name; prompts you don't provide keep their
  default.
- Extraction, tagging and page-filter prompts apply to files added afterwards; question and answer
  prompts apply immediately. To use specific prompts for specific files, change `LUM_PROMPTS_DIR`
  before adding them. `lum files` shows which files were added with custom prompts.
- Start with `lum prompts export <folder>`, and check your work with `lum prompts check`.

In Hermes, ask "how do I customize this?": the skill explains every option (models, how files are
read, speed, prompts, data location) with exact steps. The same overview is in
[`lumintelligence/skill/references/customizing.md`](lumintelligence/skill/references/customizing.md).

The [Custom prompts guide](CUSTOM_PROMPTS_GUIDE.md) explains how each prompt works, best practices
from our own tuning, troubleshooting, and four example sets in
[`examples/custom_prompts/`](examples/custom_prompts/) (small-model tuning, a custom glossary,
formatting specs, meeting notes).

## Offline and online, and switching between them

`LUM_MODE` switches **both** the chat model and the embedding model. A knowledge base only works with
the embedding model it was built with. So after switching modes you either:

- switch back to keep using your existing knowledge base, or
- run `lum kb rebuild` (deletes everything) and add your files again.

If you switch without rebuilding, `lum` stops with a message explaining this. Changing only the chat
model, with the same embedding model, needs no rebuild.

## Files

- **Supported:** PDF, Word (`.docx`), Markdown (`.md`), text (`.txt`), and JSON records
  (`[{"content": "...", "metadata": {...}}]`). Other files are listed as unsupported.
- **Folders** are read recursively. Hidden files and folders are skipped, and symbolic links aren't
  followed.
- **PDF pages** such as contents, cover or blank pages are skipped using the chat model
  (`LUM_PAGE_FILTER`). `lum files` shows which pages were skipped and why. Scanned PDFs without a
  text layer aren't supported (no OCR).
- **Adding the same folder again** only adds what's new:
  - **Unchanged files** are skipped.
  - **Changed files** are added as a new version; the earlier version stays, so answers can show
    which version a passage came from.
  - **Copies** of a file at another path are recorded, not added twice.
  - **Files you deleted** stay in the knowledge base. Nothing is ever removed automatically.

## Where your data lives

Everything is in `~/.lumintelligence/`:

| File | Contents |
|---|---|
| `.env` | Settings and API keys (readable only by you) |
| `kb.db` | Your knowledge base: one SQLite file |
| `status.json`, `queue.json`, `worker.log` | Background job state and log |

For command-line testing, `LUMINTELLIGENCE_HOME=<folder>` uses a completely separate folder (its own
settings and knowledge base). Hermes always uses `~/.lumintelligence/`. To move your main knowledge
base to another disk, move the folder and leave a symbolic link at `~/.lumintelligence`.

Your original files are never changed or copied. The knowledge base stores the extracted passages
and where they came from.

**Starting over:** `lum kb rebuild` shows what will be deleted and asks you to type `rebuild`. It
only works in a terminal: Hermes can explain it but never runs it.

## Troubleshooting

| Problem | What to do |
|---|---|
| Hermes says `lum` was not found | Install it (see [Install](#install)), then run `lum skill install` again so the skill records where `lum` is. |
| "This Python … can't load SQLite extensions" | Run `lum check --fix`, or reinstall with a Python from uv, Homebrew or conda. |
| A package is missing (e.g. PDF or Word files can't be read) | Run `lum check --fix`. |
| "LumIntelligence is not set up" | Run `lum init` in a terminal. |
| "Ollama is not running" / "Install these models in Ollama" | Open the Ollama app and install the named models, then `lum check`. |
| "This knowledge base was built with embedding model …" | You switched modes or embedding models. See [switching](#offline-and-online-and-switching-between-them). |
| A job shows as failed: "the background worker stopped" | Run the same `lum ingest` again. Model answers already received are reused. |
| `lum check` fails for the chat or embedding model | Check the model names, the base URL, and (online) your API key in `~/.lumintelligence/.env`. |
| Hermes doesn't use the skill | Restart Hermes after `lum skill install`, or call it directly with `/lumintelligence`. |
| Details of a background job | See `~/.lumintelligence/worker.log`. |

## Current limits

- One knowledge base per user (`~/.lumintelligence/kb.db`).
- No OCR for scanned PDFs. On some PDFs, rotated margin text (such as an arXiv stamp) comes out
  reversed in a page's first lines.
- Chats aren't added to the knowledge base (Hermes keeps its own memory); only files are.
- The default prompts were tuned on larger models than `hermes3` (8B). Expect better extraction
  with larger local models or online models.
- Search is rebuilt after every job, so adding files to a large knowledge base takes longer than
  the new files alone would suggest.

## License

LumIntelligence Engine Noncommercial License, based on PolyForm Noncommercial 1.0.0. Personal,
individual, non-commercial use and local evaluation only. Commercial use, organizational use and
hosting require a separate license from LumIntelligence, Inc. See LICENSE.md and
NOTICE.
