"""Custom prompts: LUM_PROMPTS_DIR with per-file fallback, the custom/default signal on files,
`lum prompts export|check`, question-time prompts applying at once, and the shipped example sets."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest

from lumintelligence import Config, LumIntelligence, cli
from lumintelligence import home as H
from lumintelligence.llm.mock import MockProvider, scripted_handler
from lumintelligence.prompts import CUSTOMIZABLE, load_prompt, resolve_prompt

EXAMPLES = Path(str(files("lumintelligence") / "skill" / "examples" / "custom_prompts"))


@pytest.fixture
def lum_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LUM_PROVIDER", "mock")
    monkeypatch.setenv("LUM_ENTRY_CUTOFF", "2.0")
    assert cli.main(["--json", "init", "--yes", "--no-skill"]) == 0
    capsys.readouterr()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Maria Lopez chairs the board.\n")
    return tmp_path


def _json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_empty_setting_uses_defaults_and_missing_folder_is_an_error(tmp_path):
    s = {k: d for k, (d, _) in H.SETTINGS.items()}
    assert H.prompts_dir(s) is None and H.make_config(s).prompts_dir is None
    s["LUM_PROMPTS_DIR"] = str(tmp_path / "nope")
    with pytest.raises(H.SetupError, match="not a folder"):
        H.make_config(s)


def test_folder_overrides_per_file_and_falls_back(tmp_path):
    (tmp_path / "sp_Answer.txt").write_text("MY ANSWER PROMPT")
    assert resolve_prompt("sp_Answer", str(tmp_path)) == "MY ANSWER PROMPT"
    assert resolve_prompt("sp_NoD", str(tmp_path)) == load_prompt("sp_NoD")       # not in the folder


def test_export_copies_defaults_and_refuses_to_overwrite(lum_home, capsys):
    folder = lum_home / "mine"
    assert cli.main(["--json", "prompts", "export", str(folder)]) == 0
    out = _json(capsys)
    assert sorted(p.stem for p in folder.iterdir()) == sorted(CUSTOMIZABLE)
    assert out["env_line"] == f"LUM_PROMPTS_DIR={folder}"
    assert cli.main(["--json", "prompts", "export", str(folder)]) == 1
    assert "--force" in _json(capsys)["hint"]
    assert cli.main(["--json", "prompts", "export", str(folder), "--force"]) == 0


def test_check_reports_sources_errors_and_warnings(lum_home, capsys, monkeypatch):
    assert cli.main(["--json", "prompts", "check"]) == 0
    out = _json(capsys)
    assert out["folder"] is None and {p["source"] for p in out["prompts"]} == {"default"}

    folder = lum_home / "custom"
    folder.mkdir()
    monkeypatch.setenv("LUM_PROMPTS_DIR", str(folder))
    (folder / "sp_Answer.txt").write_text(load_prompt("sp_Answer").replace('"cited"', '"refs"'))
    (folder / "sp_NoD.txt").write_text("Extract things. No output section here.")
    assert cli.main(["--json", "prompts", "check"]) == 0                 # warnings only
    out = _json(capsys)
    assert {p["name"]: p["source"] for p in out["prompts"]}["sp_Answer"] == "custom"
    assert len(out["warnings"]) == 2 and not out["errors"]

    (folder / "sp_Answers.txt").write_text("typo")
    (folder / "sp_ADHD_frame.txt").write_text("frame")
    (folder / "sp_ADHD.txt").write_text("   ")
    assert cli.main(["--json", "prompts", "check"]) == 1
    errors = " ".join(_json(capsys)["errors"])
    assert "sp_Answers.txt" in errors and "can't be overridden" in errors and "empty" in errors


def test_ingest_records_custom_or_default(lum_home, capsys, monkeypatch):
    docs = lum_home / "docs"
    assert cli.main(["--json", "ingest", str(docs / "a.txt"), "--wait"]) == 0
    capsys.readouterr()
    folder = lum_home / "custom"
    folder.mkdir()
    (folder / "sp_NoD.txt").write_text(load_prompt("sp_NoD"))
    monkeypatch.setenv("LUM_PROMPTS_DIR", str(folder))
    (docs / "b.txt").write_text("Tom Reyes is the treasurer.\n")
    assert cli.main(["--json", "ingest", str(docs / "b.txt"), "--wait"]) == 0
    capsys.readouterr()
    assert cli.main(["--json", "files"]) == 0
    by_label = {f["label"]: f["prompts"] for f in _json(capsys)["files"]}
    assert by_label == {"a.txt": "default", "b.txt": "custom"}


def test_custom_answer_prompt_applies_to_the_next_question(tmp_path):
    seen: list[str] = []

    def spy(system_prompt, content, schema):
        seen.append(system_prompt)
        return scripted_handler(system_prompt, content, schema)

    p = MockProvider(json_handler=spy, embed_dim=16)
    lum = LumIntelligence(p, p, config=Config(entry_cutoff=2.0))
    lum.build(["Maria Lopez chairs the board."])
    folder = tmp_path / "prompts"
    folder.mkdir()
    (folder / "sp_Answer.txt").write_text(load_prompt("sp_Answer") + "\nANSWER IN BULLETS")
    lum.config.prompts_dir = str(folder)
    lum.ask("Who is Maria Lopez?")
    assert any("ANSWER IN BULLETS" in s for s in seen)


def test_example_sets_pass_the_check():
    sets = sorted(d for d in EXAMPLES.iterdir() if d.is_dir())
    assert len(sets) == 4
    for d in sets:
        r = cli.prompts_report(str(d))
        assert r["ok"] and not r["warnings"], (d.name, r)
        assert (d / "README.md").exists()
        assert any(p["source"] == "custom" for p in r["prompts"])
