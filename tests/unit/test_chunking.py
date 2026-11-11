"""Chunking: whole lines capped by count and characters, long lines split at sentences, bounded
overlap, positions with parts."""

from __future__ import annotations

from lumintelligence.inputs.chunking import chunk_lines


def _numbered(texts: list[str]) -> list[tuple[int, str]]:
    return list(enumerate(texts, start=1))


def test_short_lines_behave_as_line_windows():
    lines = _numbered([f"line {i}" for i in range(1, 41)])
    chunks = chunk_lines(lines, window=15, overlap=3, max_chars=1500)
    assert [(c["start"], c["end"]) for c in chunks] == [(1, 15), (13, 27), (25, 39), (37, 40)]


def test_paragraph_lines_respect_the_character_cap():
    para = "A sentence about the museum and its history. " * 20          # ~900 chars per line
    chunks = chunk_lines(_numbered([para.strip()] * 6), window=15, overlap=3, max_chars=1500)
    assert all(len(c["text"]) <= 1500 for c in chunks)
    assert len(chunks) == 6                                         # one paragraph per chunk


def test_long_paragraph_splits_at_sentences_with_parts():
    sentences = [f"Sentence number {i} says something specific about the board." for i in range(60)]
    long_line = " ".join(sentences)                                  # ~3,700 chars on one line
    chunks = chunk_lines([(7, long_line)], window=15, overlap=3, max_chars=1000)
    assert len(chunks) >= 4 and all(len(c["text"]) <= 1000 for c in chunks)
    assert all(c["start"] == c["end"] == 7 for c in chunks)
    assert chunks[0]["part"] == 1 and chunks[0]["parts"] >= 4
    assert all(c["text"].rstrip().endswith(".") for c in chunks)    # cut between sentences
    joined = " ".join(c["text"] for c in chunks)
    assert all(s in joined for s in sentences)                       # nothing lost


def test_overlong_sentence_splits_at_spaces():
    word_soup = " ".join(["word"] * 600)                             # 3,000 chars, no sentence end
    chunks = chunk_lines([(1, word_soup)], window=15, overlap=0, max_chars=500)
    assert all(len(c["text"]) <= 500 for c in chunks)
    assert all(not c["text"].startswith(" ") for c in chunks)


def test_overlap_is_limited_to_a_fifth_of_the_cap():
    para = "x" * 700
    chunks = chunk_lines(_numbered([para] * 5), window=15, overlap=3, max_chars=1500)
    # 700-char lines exceed the 300-char overlap limit, so consecutive chunks share nothing.
    assert [(c["start"], c["end"]) for c in chunks] == [(1, 2), (3, 4), (5, 5)]
