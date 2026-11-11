# Batch 2: custom glossary

**For:** a collection with its own vocabulary: product codes, document numbers, internal acronyms. The
example is a fictional company, *Northwind Fabrication*; replace its terms with yours.

**Overrides:**
- `sp_NoD.txt`: the default prompt plus a "Glossary and conventions" section (where the default has
  `{{ATTENTION_HINTS}}`): which codes count as items, how acronyms are written, and when the company
  name itself is not worth extracting.
- `sp_ADHD.txt`: the default plus tag conventions, i.e. acronyms written out and a preferred shared
  vocabulary so related entities link up.
- `sp_Intention_Engine.txt`: the default with the dataset description written in, and one rule:
  questions that use an acronym search for the full name.

**Why start from the defaults:** the added section is the only change, so the tested guidance and the
output format stay intact.

**Use:** edit the glossary sections for your own terms, set `LUM_PROMPTS_DIR` to the folder, run
`lum prompts check`. The extraction and tagging changes apply to files you add afterwards; the
question prompt applies immediately.
