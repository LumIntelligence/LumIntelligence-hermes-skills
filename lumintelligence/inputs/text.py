"""TXT, Markdown and DOCX parsing.

- TXT: non-empty lines in line windows; position ``{"line_start", "line_end"}`` = line numbers in
  the file (1-based), so an answer can point to them.
- MD: split at headings, then line windows per section. Position adds ``heading_path``, which is
  also put in front of the chunk text so NoD sees the section.
- DOCX: paragraphs and tables in document order (``python-docx``, ``files`` extra). Heading styles
  give the heading path, tables are flattened like the PDF code, table-of-contents paragraphs are
  dropped. Position: ``{"heading_path", "paragraph_start", "paragraph_end"}`` (1-based body blocks).
"""

from __future__ import annotations

import re
from typing import Any

from lumintelligence.inputs.chunking import DEFAULT_MAX_CHARS, chunk_lines
from lumintelligence.inputs.types import ParsedFile

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _section_units(lines: list[tuple[int, str]], heading_path: list[str], window: int, overlap: int,
                   unit: str = "line", max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    """Chunk ``(number, text)`` lines of one section into units."""
    out = []
    head = " > ".join(heading_path)
    for ch in chunk_lines(lines, window, overlap, max_chars):
        body = ch["text"]
        position: dict = {f"{unit}_start": ch["start"], f"{unit}_end": ch["end"]}
        for key in ("part", "part_end", "parts"):
            if key in ch:
                position[key] = ch[key]
        if heading_path:
            position = {"heading_path": list(heading_path), **position}
            body = f"{head}\n{body}"
        out.append({"content": body, "metadata": {}, "position": position})
    return out


def parse_txt(path: str, window: int, overlap: int, max_chars: int = DEFAULT_MAX_CHARS) -> ParsedFile:
    lines = [(n, ln.rstrip()) for n, ln in enumerate(_read_text(path).splitlines(), start=1) if ln.strip()]
    units = _section_units(lines, [], window, overlap, max_chars=max_chars)
    return ParsedFile(units=units, note=None if units else "empty file")


def _sectioned(blocks: list[tuple[int, int | None, str]], window: int, overlap: int, unit: str,
               max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    """``blocks`` = ``(number, heading_level or None, text)``; split at headings, window each section."""
    units: list[dict] = []
    path: list[tuple[int, str]] = []
    section: list[tuple[int, str]] = []

    def flush() -> None:
        if section:
            units.extend(_section_units(section, [t for _, t in path], window, overlap, unit, max_chars))
            section.clear()

    for number, level, text in blocks:
        if level is not None:
            flush()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, text))
        else:
            section.append((number, text))
    flush()
    return units


def parse_md(path: str, window: int, overlap: int, max_chars: int = DEFAULT_MAX_CHARS) -> ParsedFile:
    blocks: list[tuple[int, int | None, str]] = []
    in_fence = False
    for n, raw in enumerate(_read_text(path).splitlines(), start=1):
        line = raw.rstrip()
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _MD_HEADING.match(line)
        if m:
            blocks.append((n, len(m.group(1)), m.group(2).strip()))
        elif line.strip():
            blocks.append((n, None, line))
    units = _sectioned(blocks, window, overlap, "line", max_chars)
    return ParsedFile(units=units, note=None if units else "empty file")


def _docx() -> Any:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("DOCX ingestion needs the 'files' extra: pip install 'lumintelligence-engine[files]'") from exc
    return docx


def _heading_level(style_name: str) -> int | None:
    if style_name == "Title":
        return 1
    m = re.match(r"Heading (\d)", style_name)
    return int(m.group(1)) if m else None


def parse_docx(path: str, window: int, overlap: int, max_chars: int = DEFAULT_MAX_CHARS) -> ParsedFile:
    docx = _docx()
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    blocks: list[tuple[int, int | None, str]] = []
    for number, el in enumerate(document.element.body.iterchildren(), start=1):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(el, document)
            text = para.text.strip()
            style = (para.style.name if para.style is not None else None) or ""
            if not text or style.upper().startswith("TOC"):
                continue
            blocks.append((number, _heading_level(style), text))
        elif tag == "tbl":
            for row in Table(el, document).rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    blocks.append((number, None, " | ".join(cells)))
    units = _sectioned(blocks, window, overlap, "paragraph", max_chars)
    return ParsedFile(units=units, note=None if units else "no text")
