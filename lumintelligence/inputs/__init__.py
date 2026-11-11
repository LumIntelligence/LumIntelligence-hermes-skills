"""Input layer: files and folders → an ingest plan → source units with provenance.

``plan_paths(paths, kg)`` decides per file (new, changed, copy, unchanged, unsupported, ...) using the
origins table; ``parse_file(item, ...)`` turns one file into units with positions. The facade's
``ingest`` creates one origin per new/changed file and runs the build on the units.
"""

from __future__ import annotations

from lumintelligence.inputs.page_filter import PageFilter
from lumintelligence.inputs.types import FileItem, IngestPlan, ParsedFile
from lumintelligence.inputs.walk import SUPPORTED, expand_paths, plan_paths, sha256_file


def parse_file(item: FileItem, window: int, overlap: int, page_filter: PageFilter | None = None,
               max_chars: int = 1500) -> ParsedFile:
    """Parse one planned file by type. Raises on unreadable or malformed files."""
    if item.file_type == "pdf":
        from lumintelligence.inputs.pdf import parse_pdf
        return parse_pdf(item.path, window, overlap, page_filter, max_chars)
    from lumintelligence.inputs.text import parse_docx, parse_md, parse_txt
    if item.file_type == "docx":
        return parse_docx(item.path, window, overlap, max_chars)
    if item.file_type == "md":
        return parse_md(item.path, window, overlap, max_chars)
    if item.file_type == "txt":
        return parse_txt(item.path, window, overlap, max_chars)
    if item.file_type == "records":
        from lumintelligence.inputs.records import parse_records
        return parse_records(item.path)
    raise ValueError(f"unsupported file type: {item.file_type}")


__all__ = ["SUPPORTED", "FileItem", "IngestPlan", "PageFilter", "ParsedFile", "expand_paths",
           "parse_file", "plan_paths", "sha256_file"]
