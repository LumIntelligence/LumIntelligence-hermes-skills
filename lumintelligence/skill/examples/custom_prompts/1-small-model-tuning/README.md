# Batch 1: small-model tuning

**For:** local models around 8B parameters (for example `hermes3`) that skip parts of the longer
default instructions.

**Overrides:** `sp_NoD.txt` (extraction) and `sp_ADHD.txt` (tagging). Everything else stays default.

**What changed from the defaults:**
- About a third of the length; numbered rules instead of guidance paragraphs.
- One short example each, which follows every rule in its prompt.
- The extraction output format is unchanged (copied from the default).

**Trade-off:** fewer nuances (for example, no summary contexts and no list of ineffective items), so
a large or online model will usually do better with the defaults.

**Use:** copy this folder, set `LUM_PROMPTS_DIR` to it, run `lum prompts check`, then add a few test
files and compare `lum files` and `lum query` with a knowledge base built on the defaults.
