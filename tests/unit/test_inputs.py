"""File input layer: folder walk and plan, per-type parsing and positions, the PDF page
filter, and facade ingest with origins (new, changed, copy, unchanged, failures, rollback)."""

from __future__ import annotations

import json
import os
import re

import pytest

from lumintelligence import Config, LumIntelligence
from lumintelligence.graph import SqliteGraphStore
from lumintelligence.inputs import PageFilter, plan_paths
from lumintelligence.inputs.pdf import parse_pdf
from lumintelligence.inputs.records import parse_records
from lumintelligence.inputs.text import parse_docx, parse_md, parse_txt
from lumintelligence.llm.mock import MockProvider
from lumintelligence.storage.journal import JsonlJournal

_NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")


def _handler(system_prompt: str, content: str, schema: dict):
    props = schema.get("properties", {})
    if "has_content" in props:                            # page filter
        toc = "contents" in content.lower()
        return {"has_content": not toc, "reason": "table of contents" if toc else "body text"}
    if "exact_keywords" in json.dumps(schema):
        return {"exact_keywords": ["Maria Lopez"], "expansion_keywords": []}
    if schema.get("additionalProperties") is False:      # batched tagging
        return {n: ["Board Member"] for n in schema["required"]}
    names = list(dict.fromkeys(_NAME.findall(content)))[:5]
    return {n: {"context": n, "inferred_type": "Person"} for n in names}


@pytest.fixture
def provider():
    return MockProvider(json_handler=_handler, embed_dim=16)


def _write(path, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)


def _make_pdf(path) -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 720, "Table of Contents")
    c.drawString(72, 700, "1. Introduction ........ 2")
    c.showPage()
    for i in range(12):                                  # two columns
        c.drawString(72, 720 - i * 14, f"Left column line {i} about Maria Lopez")
        c.drawString(330, 720 - i * 14, f"Right column line {i} about Tom Reyes")
    c.showPage()
    c.showPage()                                         # blank page
    c.save()
    return str(path)


# ----------------------------------------------------------------------- walk
def test_walk_skips_hidden_and_reports_unsupported(tmp_path):
    _write(tmp_path / "a.txt", "hello")
    _write(tmp_path / ".hidden.txt", "secret")
    _write(tmp_path / ".git" / "x.txt", "repo")
    _write(tmp_path / "sub" / "b.md", "# T\nbody")
    _write(tmp_path / "image.png", "png")

    plan = plan_paths([str(tmp_path)], SqliteGraphStore())
    by_name = {os.path.basename(i.path): i for i in plan.items}
    assert set(by_name) == {"a.txt", "b.md", "image.png"}
    assert by_name["image.png"].status == "unsupported"
    assert plan.counts() == {"new": 2, "unsupported": 1}


def test_walk_too_large(tmp_path):
    _write(tmp_path / "a.txt", "x" * 2048)
    plan = plan_paths([str(tmp_path)], SqliteGraphStore(), max_file_mb=0.001)
    assert plan.items[0].status == "too_large"


def test_two_identical_new_files_in_one_plan_make_one_copy(tmp_path):
    _write(tmp_path / "a.txt", "same text")
    _write(tmp_path / "b.txt", "same text")
    plan = plan_paths([str(tmp_path)], SqliteGraphStore())
    assert [i.status for i in plan.items] == ["new", "copy"]
    assert plan.items[1].copy_of == str(tmp_path / "a.txt")


# -------------------------------------------------------------------- parsing
def test_txt_windows_use_file_line_numbers(tmp_path):
    lines = [f"line {i}" if i % 5 else "" for i in range(1, 41)]   # blank every 5th line
    p = _write(tmp_path / "a.txt", "\n".join(lines))
    units = parse_txt(p, window=15, overlap=3).units
    assert units[0]["position"] == {"line_start": 1, "line_end": 18}   # 15 non-empty lines span 1..18
    assert units[1]["position"]["line_start"] < units[0]["position"]["line_end"]   # overlap


def test_md_sections_carry_heading_path(tmp_path):
    p = _write(tmp_path / "a.md", "# Policies\nIntro line\n## Travel\nBook early.\n```\n# not a heading\n```\n")
    units = parse_md(p, 15, 3).units
    assert units[0]["position"] == {"heading_path": ["Policies"], "line_start": 2, "line_end": 2}
    assert units[1]["position"]["heading_path"] == ["Policies", "Travel"]
    assert units[1]["content"].startswith("Policies > Travel\nBook early.")
    assert "# not a heading" in units[1]["content"]


def test_docx_headings_tables_and_toc(tmp_path):
    import docx

    d = docx.Document()
    d.add_paragraph("Contents entry 1", style="TOC Heading" if "TOC Heading" in [s.name for s in d.styles] else None)
    d.add_heading("Board", level=1)
    d.add_paragraph("Maria Lopez chairs the board.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Name", "Role"
    t.cell(1, 0).text, t.cell(1, 1).text = "Tom Reyes", "Treasurer"
    path = str(tmp_path / "a.docx")
    d.save(path)

    units = parse_docx(path, 15, 3).units
    board = next(u for u in units if u["position"].get("heading_path") == ["Board"])
    assert "Maria Lopez chairs the board." in board["content"]
    assert "Tom Reyes | Treasurer" in board["content"]
    assert "paragraph_start" in board["position"]


def test_records_file(tmp_path):
    p = _write(tmp_path / "r.json", json.dumps([{"content": "A", "metadata": {"k": 1}}, "B", {"content": " "}]))
    units = parse_records(p).units
    assert [(u["content"], u["position"]) for u in units] == [("A", {"record": 1}), ("B", {"record": 2})]
    assert units[0]["metadata"] == {"k": 1}


def test_pdf_columns_and_page_filter(tmp_path, provider):
    path = _make_pdf(tmp_path / "a.pdf")
    calls: list[str] = []

    class Spy(MockProvider):
        def complete_json(self, system_prompt, content, schema):
            calls.append(content)
            return super().complete_json(system_prompt, content, schema)

    parsed = parse_pdf(path, 15, 3, PageFilter(Spy(json_handler=_handler, embed_dim=16), "filter"))
    assert parsed.extra["pages"]["total"] == 3
    assert [s["page"] for s in parsed.extra["pages"]["skipped"]] == [1, 3]
    assert len(calls) == 2                                   # blank page: no call
    first = parsed.units[0]
    assert first["position"] == {"page": 2, "line_start": 1, "line_end": 12}
    assert first["content"].splitlines()[0].startswith("Left column line 0")   # left column first


def test_page_filter_error_keeps_the_page():
    class Broken(MockProvider):
        def complete_json(self, system_prompt, content, schema):
            raise RuntimeError("model down")

    verdict = PageFilter(Broken(), "filter").classify(["A page with plenty of real text on it."])
    assert verdict[0][0] is True and "page kept" in verdict[0][1]


def test_page_filter_answers_are_journaled_and_replayed(tmp_path):
    journal = JsonlJournal(str(tmp_path / "journal.jsonl"))
    live = {"on": True}

    class Model(MockProvider):          # same model id for both runs, as in a real rerun
        def complete_json(self, system_prompt, content, schema):
            assert live["on"], "should replay from the journal"
            return _handler(system_prompt, content, schema)

    PageFilter(Model(), "filter", journal=journal).classify(["Contents: chapter one, two"])
    live["on"] = False
    verdict = PageFilter(Model(), "filter", journal=journal).classify(["Contents: chapter one, two"])
    assert verdict == [(False, "table of contents")]


# --------------------------------------------------------------------- ingest
def test_ingest_new_unchanged_copy_changed_and_missing(tmp_path, provider):
    docs = tmp_path / "docs"
    a = _write(docs / "a.txt", "Maria Lopez chairs the board.")
    lum = LumIntelligence.open(str(tmp_path / "kb"), provider, provider, config=Config())

    first = lum.ingest([str(docs)])
    assert first["state"] == "completed" and first["plan"] == {"new": 1}
    oid = first["files"][0]["origin_id"]
    origin = lum.graph.get_origin(oid)
    assert origin["kind"] == "file" and origin["label"] == "a.txt" and origin["extra"]["path"] == a
    src = lum.graph.sources_by_origin([oid])
    assert src and lum.graph.get_node(src[0])["position"] == {"line_start": 1, "line_end": 1}

    _write(docs / "copy" / "a_copy.txt", "Maria Lopez chairs the board.")
    second = lum.ingest([str(docs)])
    assert second["plan"] == {"unchanged": 1, "copy": 1}
    assert lum.graph.get_origin(oid)["extra"]["paths"] == [a, str(docs / "copy" / "a_copy.txt")]
    assert lum.plan_ingest([str(docs)])["counts"] == {"unchanged": 2}

    _write(docs / "a.txt", "Maria Lopez chairs the board.\nTom Reyes is the treasurer.")
    third = lum.ingest([a])
    assert third["plan"] == {"changed": 1}
    new_oid = third["files"][0]["origin_id"]
    assert new_oid != oid and lum.graph.get_origin(new_oid)["extra"]["previous_version"] == oid
    # Old version keeps its source; the new version has its own, including the identical line.
    assert lum.graph.sources_by_origin([oid]) == src
    new_src = lum.graph.sources_by_origin([new_oid])
    assert len(new_src) == 1 and set(new_src).isdisjoint(src)

    os.remove(str(docs / "copy" / "a_copy.txt"))
    assert lum.plan_ingest([str(docs)])["missing"][0]["path"].endswith("a_copy.txt")


def test_ingest_reports_a_bad_file_and_continues(tmp_path, provider):
    _write(tmp_path / "bad.json", "{not json")
    _write(tmp_path / "good.txt", "Maria Lopez chairs the board.")
    lum = LumIntelligence.open(str(tmp_path / "kb"), provider, provider)

    res = lum.ingest([str(tmp_path / "bad.json"), str(tmp_path / "good.txt")])
    by_name = {os.path.basename(f["path"]): f for f in res["files"]}
    assert by_name["bad.json"]["result"] == "failed" and "JSONDecodeError" in by_name["bad.json"]["error"]
    assert by_name["good.txt"]["result"] == "ingested"
    assert len(lum.graph.list_origins()) == 1                 # the bad file is retried next time


def test_failed_build_leaves_no_origins(tmp_path):
    class Exploding(MockProvider):
        def embed(self, texts, purpose=None):
            raise RuntimeError("embedder down")

    p = Exploding(json_handler=_handler, embed_dim=16)
    _write(tmp_path / "a.txt", "Maria Lopez chairs the board.")
    lum = LumIntelligence.open(str(tmp_path / "kb"), p, p)
    with pytest.raises(RuntimeError):
        lum.ingest([str(tmp_path / "a.txt")])
    assert lum.graph.list_origins() == [] and lum.graph.source_node_count() == 0
    assert lum.plan_ingest([str(tmp_path / "a.txt")])["counts"] == {"new": 1}


def test_ingest_pdf_records_skipped_pages_on_the_origin(tmp_path, provider):
    path = _make_pdf(tmp_path / "a.pdf")
    lum = LumIntelligence.open(str(tmp_path / "kb"), provider, provider)
    res = lum.ingest([path])
    origin = lum.graph.get_origin(res["files"][0]["origin_id"])
    assert origin["extra"]["pages"]["skipped"] == [
        {"page": 1, "reason": "table of contents"}, {"page": 3, "reason": "blank or near-empty page"}]
    assert lum.graph.find_entity_by_name("Maria Lopez") is not None
