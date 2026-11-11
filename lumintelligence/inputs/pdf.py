"""PDF parsing:
column-aware line reflow, flattened tables, per-page line windows. Needs the ``files`` extra
(pdfplumber).

Position per chunk: ``{"page", "line_start", "line_end"}`` (1-based lines within the page's
reflowed line list). Skipped pages and the reasons are returned for the origin's ``extra.pages``.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from lumintelligence.inputs.chunking import DEFAULT_MAX_CHARS, chunk_lines
from lumintelligence.inputs.page_filter import PageFilter
from lumintelligence.inputs.types import ParsedFile


def _pdfplumber() -> Any:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("PDF ingestion needs the 'files' extra: pip install 'lumintelligence-engine[files]'") from exc
    return pdfplumber


def extract_reflowed_lines(page: Any) -> list[str]:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return []

    # Group words into lines by rounded top (y0) position, merging close tops.
    tol = 3.0
    raw_lines: list[dict] = []
    for w in sorted(words, key=lambda w: w["top"]):
        if raw_lines and abs(w["top"] - raw_lines[-1]["top"]) <= tol:
            raw_lines[-1]["words"].append(w)
            raw_lines[-1]["top"] = (raw_lines[-1]["top"] + w["top"]) / 2
        else:
            raw_lines.append({"top": w["top"], "words": [w]})

    lines = []
    for line in raw_lines:
        ws = sorted(line["words"], key=lambda w: w["x0"])
        lines.append({"top": line["top"], "x0": ws[0]["x0"], "text": " ".join(w["text"] for w in ws)})

    # Column bands: gaps in sorted line starts wider than 8% of the page width mark a boundary.
    x0s = sorted(ln["x0"] for ln in lines)
    gap_threshold = page.width * 0.08
    band_edges = [x0s[0]]
    prev = x0s[0]
    for x in x0s[1:]:
        if x - prev > gap_threshold:
            band_edges.append(x)
        prev = x
    band_edges.append(page.width)
    bands = list(pairwise(band_edges))

    def band_index(x0: float) -> int:
        for i, (lo, hi) in enumerate(bands):
            if lo - 1 <= x0 < hi + 1:
                return i
        return len(bands) - 1

    for ln in lines:
        ln["band"] = band_index(ln["x0"])
    # Left column top to bottom, then the next column, and so on.
    lines.sort(key=lambda ln: (ln["band"], ln["top"]))
    return [ln["text"] for ln in lines]


def extract_flattened_tables(page: Any) -> list[str]:
    try:
        tables = page.extract_tables()
    except Exception:  # noqa: BLE001 — a broken table never fails the page
        tables = []
    flattened = []
    for table in tables:
        for row in table:
            cells = [(c or "").strip() for c in row]
            if any(cells):
                flattened.append(" | ".join(cells))
    return flattened


def parse_pdf(path: str, window: int, overlap: int, page_filter: PageFilter | None = None,
              max_chars: int = DEFAULT_MAX_CHARS) -> ParsedFile:
    pdfplumber = _pdfplumber()
    with pdfplumber.open(path) as pdf:
        texts = [page.extract_text() or "" for page in pdf.pages]
        total = len(texts)
        if not any(t.strip() for t in texts):
            return ParsedFile(extra={"pages": {"total": total, "skipped": []}},
                              note="no extractable text (scanned PDF? OCR is not supported)")
        verdicts = page_filter.classify(texts) if page_filter else [(True, "")] * total
        skipped = []
        units: list[dict] = []
        for number, (page, (keep, reason)) in enumerate(zip(pdf.pages, verdicts), start=1):
            if not keep:
                skipped.append({"page": number, "reason": reason})
                continue
            lines = extract_reflowed_lines(page) + extract_flattened_tables(page)
            for ch in chunk_lines(list(enumerate(lines, start=1)), window, overlap, max_chars):
                position = {"page": number, "line_start": ch["start"], "line_end": ch["end"]}
                for key in ("part", "part_end", "parts"):
                    if key in ch:
                        position[key] = ch[key]
                units.append({"content": ch["text"], "metadata": {}, "position": position})
    note = None if units else "no content pages"
    return ParsedFile(units=units, extra={"pages": {"total": total, "skipped": skipped}}, note=note)
