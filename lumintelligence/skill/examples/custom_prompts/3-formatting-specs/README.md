# Batch 3: formatting specs

**For:** controlling how answers look, and which PDF pages count as content.

**Overrides:**
- `sp_Answer.txt`: answers as 2–6 bullet points, at most 150 words, always in English, with a
  citation at the end of each bullet. Change the length, language or style to taste.
- `sp_PageFilter.txt`: keeps appendices, glossaries, reference lists and annexes (the default may
  skip some of them as boilerplate) and still skips covers, contents, blank and index pages.

**Keep:** the "Output format" sections exactly as they are. The answer must stay JSON with `answer`
and `cited`, and citations must stay as [n], because the sources list is numbered to match.

**When it applies:** the answer prompt applies to your next question immediately. The page filter
applies to PDFs you add afterwards.
