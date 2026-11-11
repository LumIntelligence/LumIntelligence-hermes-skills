"""Chunking shared by every file type: windows of whole lines, capped by lines and by characters.

A chunk takes whole lines until it has ``window`` lines or ``max_chars`` characters. A line longer
than ``max_chars`` (a paragraph-per-line Markdown file, a wide table row) is first split at sentence
boundaries, or at spaces for a single over-long sentence; its pieces carry ``part``/``parts``.
Consecutive chunks share up to ``overlap`` trailing lines, limited to a fifth of ``max_chars``, so
with paragraph-sized lines the overlap is about one sentence rather than whole paragraphs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 1500

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"”’)\]]*\s+")


@dataclass
class Piece:
    number: int              # line (or paragraph) number in the file
    text: str
    part: int | None = None  # set when the line was split
    parts: int | None = None


def _split_long(text: str, max_chars: int) -> list[str]:
    """Pieces of at most ``max_chars``: whole sentences where possible, spaces otherwise."""
    sentences = [s for s in _SENTENCE_END.split(text) if s]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:                       # one sentence longer than the cap
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].rstrip())
            sentence = sentence[cut:].lstrip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


def to_pieces(lines: list[tuple[int, str]], max_chars: int) -> list[Piece]:
    out: list[Piece] = []
    for number, text in lines:
        if len(text) <= max_chars:
            out.append(Piece(number, text))
            continue
        split = _split_long(text, max_chars)
        out.extend(Piece(number, s, i, len(split)) for i, s in enumerate(split, start=1))
    return out


def chunk_lines(lines: list[tuple[int, str]], window: int, overlap: int,
                max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    """Chunks of ``(number, text)`` lines: ``{"text", "start", "end", "part"?, "parts"?}`` where
    ``start``/``end`` are line numbers; ``part``/``parts`` appear when a chunk lies within one split
    line."""
    pieces = to_pieces(lines, max_chars)
    chunks: list[dict] = []
    overlap_chars = max_chars // 5
    i = 0
    while i < len(pieces):
        j, size = i, 0
        while j < len(pieces) and j - i < window and (j == i or size + 1 + len(pieces[j].text) <= max_chars):
            size += len(pieces[j].text) + (1 if j > i else 0)
            j += 1
        group = pieces[i:j]
        chunk: dict = {"text": "\n".join(p.text for p in group), "start": group[0].number, "end": group[-1].number}
        if group[0].number == group[-1].number and group[0].part is not None:
            chunk["part"], chunk["parts"] = group[0].part, group[0].parts
            if group[-1].part != group[0].part:
                chunk["part_end"] = group[-1].part
        chunks.append(chunk)
        if j >= len(pieces):
            break
        back, kept = 0, 0                                   # overlap: trailing pieces within the limits
        while back < overlap and j - back - 1 > i:
            n = len(pieces[j - back - 1].text)
            if kept + n > overlap_chars:
                break
            kept += n
            back += 1
        i = j - back
    return chunks


def chunk_line_ranges(num_lines: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) 0-indexed, end-exclusive line ranges, by line count only (kept for callers that
    need plain ranges)."""
    if num_lines == 0:
        return []
    step = max(window - overlap, 1)
    ranges = []
    start = 0
    while start < num_lines:
        end = min(start + window, num_lines)
        ranges.append((start, end))
        if end == num_lines:
            break
        start += step
    return ranges
