# Custom prompts guide

LumIntelligence uses a few prompts to turn your files into a knowledge network and to answer your
questions. The default prompts are written for general documents. If your files have their own
vocabulary (product codes, internal acronyms, project names), a particular structure (meeting notes,
research papers), or you want answers in a particular style, your own prompts can do noticeably better.

This guide covers how custom prompts work, how to write them, and four example sets to start from.

- [When to customize](#when-to-customize)
- [How it works](#how-it-works)
- [Start from the defaults](#start-from-the-defaults)
- [Anatomy of a prompt](#anatomy-of-a-prompt)
- [Best practices](#best-practices)
- [Example sets](#example-sets)
- [Troubleshooting](#troubleshooting)

## When to customize

**Keep the defaults** when your files are ordinary prose (articles, reports, notes) and answers
already find what you ask about.

**Customize** when:
- answers miss things that are obviously in your files, because the extraction doesn't recognise your
  terms (codes, acronyms, jargon)
- the same thing appears under several names (an acronym and its full name)
- your documents have a structure that matters (decisions and action items in meeting notes)
- you want answers in a fixed format, length or language
- you use a small local model that follows long instructions poorly

## How it works

### The prompts

| File | What it controls | Takes effect |
|---|---|---|
| `sp_NoD.txt` | Extraction: which entities and concepts are found in each passage, with their supporting text | Files added afterwards |
| `sp_ADHD.txt` | Tagging: the tags that link entities into concepts | Files added afterwards |
| `sp_PageFilter.txt` | Which PDF pages are skipped (contents, cover, blank…) | Files added afterwards |
| `sp_Intention_Engine.txt` | How a question becomes search keywords; holds the description of your collection | Every question, immediately |
| `sp_Answer.txt` | How answers are written | Every question, immediately |

The output format of tagging is fixed by LumIntelligence itself and can't be replaced; `sp_ADHD.txt`
only holds the tagging guidance.

### Pointing to your prompts: `LUM_PROMPTS_DIR`

Custom prompts are configured in one place: the `LUM_PROMPTS_DIR` line in
`~/.lumintelligence/.env`.

- **Empty** (the default): the prompts that come with the skill are used.
- **A folder**: every prompt file in that folder replaces the default of the same name. Files you
  don't put in the folder keep their default, so a folder can hold just one prompt.

```
LUM_PROMPTS_DIR=/Users/you/lum-prompts/glossary
```

If the line points to a folder that doesn't exist, every `lum` command stops and tells you, so a typo
never silently falls back to the defaults.

### Hints and the collection description

There are no separate settings for hints or for a description of your collection: write them into
the prompt text. The default prompts contain placeholders, `{{ATTENTION_HINTS}}` (extraction and
tagging) and `{{DATASET_DESCRIPTION}}` (question keywords). In your copy you can replace them with your
own text, as the example sets do, or delete them.

### What changes when

- **Extraction, tagging and page filter** shape what's stored when a file is added. Changing them
  affects only files you add afterwards; files already in the knowledge base keep what they were
  built with. To apply new prompts to everything, run `lum kb rebuild` (deletes everything) and add
  your files again.
- **Question keywords and answers** run on every question, so a change applies to your next
  question at once.

### Different prompts for different files

The prompts in `LUM_PROMPTS_DIR` apply to every file added while it's set. To use specific prompts for
specific files, point `LUM_PROMPTS_DIR` at a different folder (or swap the files in it) before adding
those files, then switch back. A job waiting in the queue uses whatever prompts are in place when it
starts.

`lum files` shows, for each file, whether it was added with **custom** or **default** prompts.

## Start from the defaults

Don't write prompts from scratch: copy the defaults and change what you need.

```sh
lum prompts export ~/lum-prompts/mine      # copies the five default prompts
```

1. Delete the files you don't want to change (they'll keep their default).
2. Edit the rest.
3. Set `LUM_PROMPTS_DIR=~/lum-prompts/mine` in `~/.lumintelligence/.env`.
4. Check them:
   ```sh
   lum prompts check
   ```
   It lists which prompts are custom and which are default. It reports **errors** for file names
   LumIntelligence doesn't use (usually a typo), empty files, and an attempt to replace the tagging
   frame, and **warnings** when an "Output format" section was changed or removed.
5. Add two or three representative files and compare the results (below).

Updates to LumIntelligence never touch your folder. After updating, `lum prompts export` into a new
folder shows you the new defaults to compare against.

## Anatomy of a prompt

The default prompts share a structure:

| Section | What it does | Safe to change? |
|---|---|---|
| **Purpose** | Tells the model its job and why it matters | Yes: adapt it to your collection |
| **Guidance** | How to decide what to extract or tag, with short lists of what *not* to produce | Yes: this is where most tuning happens |
| **Rules** | Hard constraints | Carefully: keep the "return only JSON" rule |
| **Output format** | The exact JSON the code reads | **No: keep it exactly as in the default** |
| **Examples** | Worked inputs and outputs | Yes: replace with examples from your kind of material |
| `{{ATTENTION_HINTS}}` / `{{DATASET_DESCRIPTION}}` | Placeholders | Replace with your text, or delete |

The code reads the model's answer as JSON in the shape the Output format describes. If that section
changes, extraction or answers may fail to parse (you'll see failed sources in `lum status`).

The answer prompt has one more requirement: citations must stay in the `[1]`, `[2]` form, because the
sources list under each answer is numbered to match.

## Best practices

These come from tuning the default prompts on real documents with several models.

1. **State what good looks like, then list a few things not to produce.** "Tags are reusable terms
   such as 'Funding Source'", followed by a short list of ineffective outputs ("a sentence", "a name
   inside a tag"), works better than a long list of prohibitions.
2. **Prefer mechanical rules to judgement calls.** "Product codes look like NF- followed by four
   digits" is followed far more reliably than "extract important products".
3. **Every example must follow the prompt's own rules.** Models copy examples more than they follow
   instructions; an example that breaks a rule teaches the model to break it.
4. **Keep examples generic.** Use invented names and situations that resemble your material, not
   passages from your actual files. Copying real passages makes the model over-fit to them.
5. **Shorter prompts for small local models.** Models around 8B parameters skip parts of long
   instructions. Fewer, numbered rules and one example do better (see example set 1).
6. **Change one thing at a time,** so you know which change helped.
7. **Test on two or three files first.** Use a separate test setup if you want to keep your main
   knowledge base clean: `LUMINTELLIGENCE_HOME=~/lum-test lum init`, then compare
   - `lum files`: chunks per file, and whether custom prompts were used
   - `lum query "…"`: whether questions you care about now find the right passages
   - `lum export --out graph.json`: the extracted entities and tags, if you want to look closely

## Example sets

Four ready-made sets are in [`examples/custom_prompts/`](examples/custom_prompts/) (after `lum skill install`, also
in `~/.hermes/skills/research/lumintelligence/examples/custom_prompts/`). Each folder holds
only the prompts it changes, plus a README explaining what changed and why. Copy a folder, adapt it,
and point `LUM_PROMPTS_DIR` at your copy.

| Set | Changes | Use it for |
|---|---|---|
| [`1-small-model-tuning`](examples/custom_prompts/1-small-model-tuning/) | Extraction, tagging | Small local models (around 8B) that skip parts of the long default instructions |
| [`2-custom-glossary`](examples/custom_prompts/2-custom-glossary/) | Extraction, tagging, question keywords | A collection with its own codes and acronyms (a fictional company, *Northwind Fabrication*) |
| [`3-formatting-specs`](examples/custom_prompts/3-formatting-specs/) | Answers, page filter | Bullet-point answers under 150 words in a fixed language; keeping appendices and glossaries in PDFs |
| [`4-meeting-notes`](examples/custom_prompts/4-meeting-notes/) | Extraction, tagging | Meeting notes: decisions, action items, owners and due dates |

Sets 2 and 4 show the safest way to customize: the default prompt with one added section where the
`{{ATTENTION_HINTS}}` placeholder was. Sets 1 and 3 rewrite their prompts more fully.

Sets can be combined: copy the files you want from several sets into one folder (one file per prompt
name).

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `lum status` shows failed sources after you changed a prompt | The model's answer no longer parses. Run `lum prompts check`; restore the "Output format" section and the "return only JSON" rule from the default. |
| You edited a prompt but nothing changed | Extraction, tagging and page-filter prompts only affect files added afterwards. Add a new file, or rebuild. Also check the file name with `lum prompts check`. |
| `lum prompts check` reports an unknown file | A typo in the file name; it must be exactly one of `sp_NoD.txt`, `sp_ADHD.txt`, `sp_PageFilter.txt`, `sp_Intention_Engine.txt`, `sp_Answer.txt`. |
| "LUM_PROMPTS_DIR points to … which is not a folder" | The folder moved or the path has a typo. Fix the path, or empty the line to use the defaults. |
| Answers are worse for older files than for new ones | Files were added with different prompts. `lum files` shows which used custom prompts; rebuild if you want everything consistent. |
| A small model ignores your additions | Shorten the prompt (see set 1) and turn additions into short, mechanical rules. |
