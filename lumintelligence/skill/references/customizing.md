# Customizing LumIntelligence

Everything about LumIntelligence can be adjusted by the user: which models it uses, how files are read,
how fast it works, the prompts that decide what is extracted and how answers are written, and where
data lives. All changes are made by the user, in a terminal or a text editor. The agent explains them
and never edits settings or prompt files itself.

Settings live in one file: `~/.lumintelligence/.env`. It is written by `lum init` with a comment on
every line. Open it in any text editor. Changes take effect with the next `lum` command; there is
nothing to restart. Real environment variables with the same names override the file.

| # | Area | Where | Takes effect |
|---|---|---|---|
| 1 | [Models: offline or online](#1-models-offline-or-online) | `.env` | Next command (embedding model: see the rebuild note) |
| 2 | [How files are read](#2-how-files-are-read) | `.env` | Files added afterwards |
| 3 | [Speed](#3-speed) | `.env` | Next job |
| 4 | [Prompts](#4-prompts) | `.env` (`LUM_PROMPTS_DIR`) + your prompt files | Extraction: files added afterwards; answers: immediately |
| 5 | [Where data lives, and test setups](#5-where-data-lives-and-test-setups) | Terminal | Advanced |
| 6 | [The Hermes skill itself](#6-the-hermes-skill-itself) | The installed skill folder | After restarting Hermes |

## 1. Models: offline or online

`.env` holds two complete sets of models, and `LUM_MODE` picks one:

```
LUM_MODE=offline                                   # or: online

LUM_OFFLINE_BASE_URL=http://localhost:11434/v1     # Ollama
LUM_OFFLINE_API_KEY=
LUM_OFFLINE_CHAT_MODEL=hermes3
LUM_OFFLINE_EMBED_MODEL=nomic-embed-text
LUM_OFFLINE_REASONING_EFFORT=

LUM_ONLINE_BASE_URL=                               # empty = OpenAI; any OpenAI-compatible URL works
LUM_ONLINE_API_KEY=sk-...
LUM_ONLINE_CHAT_MODEL=gpt-6-luna
LUM_ONLINE_EMBED_MODEL=text-embedding-3-small
LUM_ONLINE_REASONING_EFFORT=none
```

- **Chat model** (`_CHAT_MODEL`): reads your files, tags them, filters PDF pages and writes answers.
  A larger model extracts more and follows instructions better; a smaller one is faster.
- **Embedding model** (`_EMBED_MODEL`): powers search.
- **Steps to change a model:**
  1. For offline models, install the model in Ollama first.
  2. Edit the line in `.env`.
  3. Run `lum check` to test it.
- **Changing the chat model** is safe at any time.
- **Changing the embedding model, or switching `LUM_MODE`** (which switches both models), needs a
  fresh knowledge base. A knowledge base only works with the embedding model it was built with.
  Either switch back, or run `lum kb rebuild` (deletes everything) and add your files again. Until
  then, `lum` stops with a message that says so.
- `lum init` can also be run again; it keeps your current values as defaults.

## 2. How files are read

```
LUM_CHUNK_WINDOW_LINES=15     # lines per passage
LUM_CHUNK_MAX_CHARS=1500      # characters per passage; longer paragraphs are split at sentences
LUM_CHUNK_OVERLAP_LINES=3     # lines shared by consecutive passages (at most a fifth of the size)
LUM_PAGE_FILTER=true          # PDFs: skip contents, cover and blank pages using the chat model
LUM_CONFIRM_ABOVE=20          # ask before adding more new or changed files than this
```

- **Smaller passages** (`LUM_CHUNK_MAX_CHARS=800–1000`) give more precise citations and help small
  models find more in dense text, but mean more model calls.
- **Larger passages** (2000–3000) mean fewer calls; they suit large models and loosely written text.
- **`LUM_PAGE_FILTER=false`** keeps every PDF page and saves one model call per page. Use it if
  pages you need are being skipped (`lum files` lists skipped pages and why), or tune which pages
  count as content with a custom page-filter prompt (see 4).
- **`LUM_CONFIRM_ABOVE`** sets when adding a folder asks first.
- These settings apply to files added afterwards. To re-read files already in the knowledge base with
  new settings, run `lum kb rebuild` and add them again.

## 3. Speed

```
LUM_MAX_IN_FLIGHT=2           # model calls running at the same time
```

Raise it if your model server handles parallel requests: for Ollama, set `OLLAMA_NUM_PARALLEL` to
the same number and restart Ollama; online APIs usually handle 4–8. Lower it to 1 if the computer
becomes unresponsive or the server reports errors during adding files.

## 4. Prompts

Prompts decide what is extracted from your files, how it is tagged, which PDF pages are skipped, how
a question becomes search keywords, and how answers are written. Your own prompts can teach
LumIntelligence your vocabulary (product codes, acronyms), focus it on what matters in your documents
(decisions in meeting notes), or change the answer style.

**Steps:**
1. Copy the default prompts into a folder as a starting point:
   ```sh
   lum prompts export ~/lum-prompts/mine
   ```
2. Delete the files you don't want to change; they keep their default.
3. Edit the rest in a text editor. Keep each "Output format" section exactly as it is.
4. Point LumIntelligence to the folder, in `.env`:
   ```
   LUM_PROMPTS_DIR=/Users/you/lum-prompts/mine
   ```
   Leave it empty to use the default prompts that come with the skill.
5. Check:
   ```sh
   lum prompts check
   ```
6. Add two or three test files and compare the results (`lum files`, `lum query`).

| Prompt file | Controls | Takes effect |
|---|---|---|
| `sp_NoD.txt` | What is extracted from each passage | Files added afterwards |
| `sp_ADHD.txt` | Tags that link what was extracted | Files added afterwards |
| `sp_PageFilter.txt` | Which PDF pages are skipped | Files added afterwards |
| `sp_Intention_Engine.txt` | How a question becomes search keywords; a description of your collection | Immediately |
| `sp_Answer.txt` | How answers are written (style, length, language) | Immediately |

**Four ready-made example sets** come with the skill in `examples/custom_prompts/` (installed at
`~/.hermes/skills/research/lumintelligence/examples/custom_prompts/`):

| Set | For |
|---|---|
| `1-small-model-tuning` | Small local models that skip parts of long instructions |
| `2-custom-glossary` | Your own codes, document numbers and acronyms (example: a fictional manufacturer) |
| `3-formatting-specs` | Bullet-point answers with a length limit in a fixed language; keeping appendices in PDFs |
| `4-meeting-notes` | Decisions, action items, owners and due dates |

Copy a set to your own folder (don't edit it in the skill folder, which is replaced on updates),
adapt it, and point `LUM_PROMPTS_DIR` at your copy. Sets can be combined: one file per prompt name.

**Different prompts for different files:** `LUM_PROMPTS_DIR` applies to everything added while it
is set. Change it before adding files that need different prompts, then change it back. `lum files`
shows whether each file was added with custom or default prompts.

Full details, the anatomy of a prompt, best practices and troubleshooting:
[custom_prompts.md](custom_prompts.md).

## 5. Where data lives, and test setups

By default everything is in `~/.lumintelligence/`: the settings (`.env`), the knowledge base
(`kb.db`), and job status and logs.

- **A separate test setup** is useful for trying models or prompts on a few files without touching
  your main knowledge base. In a terminal:
  ```sh
  export LUMINTELLIGENCE_HOME=~/lum-test
  lum init
  lum ingest ~/some-test-files
  lum query "…"
  ```
  This uses a completely separate folder with its own settings and knowledge base, for that terminal
  session only. Hermes always uses `~/.lumintelligence/` (it doesn't see variables set in a
  terminal), so this is for command-line testing.
- **Moving your main knowledge base** to another disk: move the folder and leave a link behind, so
  Hermes still finds it:
  ```sh
  mv ~/.lumintelligence /Volumes/Data/lumintelligence
  ln -s /Volumes/Data/lumintelligence ~/.lumintelligence
  ```
- **Starting over:** `lum kb rebuild` (deletes the knowledge base; asks you to type `rebuild`).

## 6. The Hermes skill itself

The skill's instructions are in `~/.hermes/skills/research/lumintelligence/SKILL.md`. You can edit
them (for example, to change when Hermes uses the skill), but `lum skill install` and `lum init`
replace the folder with the version that comes with LumIntelligence, so your edits are lost on the
next update. To change what LumIntelligence extracts or how it answers, use custom prompts (4); they
live in your own folder and survive updates.

After changing the skill, restart Hermes or start a new chat. If Hermes can't find `lum` after you
reinstalled it, run `lum skill install` again so the skill records where `lum` is.
