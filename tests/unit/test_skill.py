"""The Hermes skill: SKILL.md shape, its commands exist, lum.sh lookup, install, ingest guard."""

from __future__ import annotations

import json
import os
import re
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest

from lumintelligence import cli

SKILL = Path(str(files("lumintelligence") / "skill"))


def _frontmatter() -> dict[str, str]:
    text = (SKILL / "SKILL.md").read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md must start with a frontmatter block"
    return {k: v.strip() for k, v in re.findall(r"^(\w+):\s*(.*)$", m.group(1), re.MULTILINE)}


def test_frontmatter_has_the_required_fields():
    fm = _frontmatter()
    assert fm["name"] == "lumintelligence"
    assert fm["description"] and len(fm["description"]) < 200
    text = (SKILL / "SKILL.md").read_text()
    assert "requires_toolsets: [terminal]" in text
    assert "${HERMES_SKILL_DIR}/scripts/lum.sh" in text


def test_every_command_the_skill_mentions_exists():
    text = (SKILL / "SKILL.md").read_text()
    used = set(re.findall(r"lum\.sh (\w+)", text))
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    assert used and used <= set(sub.choices), used - set(sub.choices)
    for flag in ("--no-answer", "--yes"):
        assert flag in text


def _run_script(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(SKILL / "scripts" / "lum.sh"), *args], capture_output=True, text=True,
                          env=env, check=False)


def test_lum_sh_uses_lum_bin_and_adds_json(tmp_path):
    fake = tmp_path / "lum"
    fake.write_text('#!/bin/sh\necho "args: $*"\n')
    fake.chmod(0o755)
    r = _run_script({"LUM_BIN": str(fake), "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, "status")
    assert r.stdout.strip() == "args: --json status"


def test_lum_sh_finds_local_bin(tmp_path):
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "lum").write_text('#!/bin/sh\necho found\n')
    (bin_dir / "lum").chmod(0o755)
    r = _run_script({"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, "files")
    assert r.stdout.strip() == "found"


def test_lum_sh_reports_missing_lum_as_json(tmp_path):
    r = _run_script({"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, "status")
    assert r.returncode == 127
    err = json.loads(r.stdout)
    assert "not found" in err["error"] and "uv tool install" in err["hint"]


def test_skill_install_copies_and_replaces(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    assert cli.main(["--json", "skill", "install"]) == 0
    out = json.loads(capsys.readouterr().out)
    dest = Path(out["installed"])
    assert dest == tmp_path / "hermes" / "skills" / "research" / "lumintelligence"
    assert (dest / "SKILL.md").exists() and (dest / "references" / "output.md").exists()
    assert os.access(dest / "scripts" / "lum.sh", os.X_OK)

    (dest / "stale.txt").write_text("old")
    assert cli.main(["--json", "skill", "install"]) == 0
    capsys.readouterr()
    assert not (dest / "stale.txt").exists()


@pytest.fixture
def home_with_docs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LUMINTELLIGENCE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LUM_PROVIDER", "mock")
    monkeypatch.setenv("LUM_CONFIRM_ABOVE", "2")
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(3):
        (docs / f"note{i}.txt").write_text(f"Maria Lopez wrote note {i}.\n")
    assert cli.main(["--json", "init", "--yes", "--skip-check"]) == 0
    capsys.readouterr()
    return docs


def test_large_ingest_needs_confirmation_from_an_agent(home_with_docs, capsys):
    assert cli.main(["--json", "ingest", str(home_with_docs), "--wait"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["needs_confirmation"] is True and len(out["files"]) == 3 and "--yes" in out["hint"]

    assert cli.main(["--json", "ingest", str(home_with_docs), "--wait", "--yes"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["result"]["state"] == "completed"


def test_large_ingest_asks_in_a_terminal(home_with_docs, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_tty_in", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    cli.main(["ingest", str(home_with_docs), "--wait"])
    assert "Cancelled" in capsys.readouterr().out


def test_lum_sh_uses_the_path_recorded_at_install(tmp_path, monkeypatch, capsys):
    fake = tmp_path / "venv" / "bin" / "lum"
    fake.parent.mkdir(parents=True)
    fake.write_text('#!/bin/sh\necho "recorded: $*"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(cli.sys, "argv", [str(fake)])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    assert cli.main(["--json", "skill", "install"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["lum_found"] == str(fake)
    script = Path(out["installed"]) / "scripts" / "lum.sh"
    r = subprocess.run([str(script), "files"], capture_output=True, text=True, check=False,
                       env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert r.stdout.strip() == "recorded: --json files"


def test_skill_references_exist_and_install_with_the_skill(tmp_path, capsys, monkeypatch):
    text = (SKILL / "SKILL.md").read_text()
    for ref in re.findall(r"\$\{HERMES_SKILL_DIR\}/([\w./-]+[\w/])", text):
        assert (SKILL / ref).exists(), ref
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    assert cli.main(["--json", "skill", "install"]) == 0
    dest = Path(json.loads(capsys.readouterr().out)["installed"])
    assert (dest / "references" / "customizing.md").exists()
    assert (dest / "references" / "custom_prompts.md").exists()
    sets = sorted(p.name for p in (dest / "examples" / "custom_prompts").iterdir())
    assert sets == ["1-small-model-tuning", "2-custom-glossary", "3-formatting-specs", "4-meeting-notes"]
    assert (dest / "examples" / "custom_prompts" / "2-custom-glossary" / "sp_NoD.txt").exists()


def test_customizing_guide_covers_every_setting():
    from lumintelligence.home import SETTINGS

    guide = (SKILL / "references" / "customizing.md").read_text()
    missing = [k for k in SETTINGS if k not in guide and not k.startswith(("LUM_OFFLINE_", "LUM_ONLINE_"))]
    assert not missing, missing
    for mode in ("OFFLINE", "ONLINE"):
        assert f"LUM_{mode}_CHAT_MODEL" in guide and f"LUM_{mode}_EMBED_MODEL" in guide
