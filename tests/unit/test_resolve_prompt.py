"""Custom-prompt intake: file path, prompts_dir, bundled default (URL fetching dropped)."""

from __future__ import annotations

from lumintelligence.prompts import fill_placeholders, load_prompt, resolve_prompt


def test_resolve_prompt_falls_back_to_bundled_default():
    assert resolve_prompt("sp_NoD") == load_prompt("sp_NoD")
    assert resolve_prompt("sp_NoD", "./no-such-dir") == load_prompt("sp_NoD")


def test_resolve_prompt_reads_explicit_path(tmp_path):
    p = tmp_path / "my_nod.txt"
    p.write_text("custom nod prompt", encoding="utf-8")
    assert resolve_prompt(str(p)) == "custom nod prompt"


def test_resolve_prompt_prefers_prompts_dir_over_bundled(tmp_path):
    (tmp_path / "sp_NoD.txt").write_text("project nod prompt", encoding="utf-8")
    assert resolve_prompt("sp_NoD", str(tmp_path)) == "project nod prompt"


def test_fill_placeholders():
    text = "a {{ATTENTION_HINTS}} b {{DATASET_DESCRIPTION}}"
    assert fill_placeholders(text, attention_hints="hint", dataset_description="docs") == "a hint b docs"
    assert fill_placeholders(text) == text
